module
public import NeuralReversi.PvsParallel.NativeFrameCache
public import NeuralReversi.PvsParallel.FrameTableReads
public import NeuralReversi.TranspositionTable.WriterMir

@[expose] public section

namespace NeuralReversi.PvsParallel.NativeTableWrites.Journal

open TranspositionTable
open SeqLock (Word Location Field Memory)
open FrameTableReads (Address Mark)

inductive Origin where
  | clear
  | store (frame : Option Nat) (board : PvsBoardTree.Position) (entry : EntryData)

/-- Arguments of a single invocation of TTEntry::seqlock_write, including its CAS outcome. -/
structure Attempt where
  address : Address
  expected : {v : Word // v &&& 1 = 0}
  origin : Origin
  success : Bool
  event : Nat → Nat

def Attempt.args (w : Attempt) (endian : Bool) : Word × Word × EntryData :=
  match w.origin with
  | .clear => (0,0,Packing.unpack endian 0)
  | .store _ b d => (b.val.player,b.val.opponent,d)

def Attempt.program (w : Attempt) (endian : Bool) :=
  let (p,o,d) := w.args endian
  WriterMir.writeStable endian w.expected p o d w.success

inductive Pc where
  | cas | fence | field (f : Field) | finish
  deriving DecidableEq

def Pc.index : Pc → Nat
  | .cas => 0 | .fence => 1 | .field .player => 2 | .field .opponent => 3
  | .field .metadata => 4 | .finish => 5

def Pc.location : Pc → Option Location
  | .cas | .finish => some .sequence
  | .field f => some (.payload f)
  | .fence => none

def Attempt.call (w : Attempt) (endian : Bool) : WriterTrace.Call Nat :=
  let (p,o,d) := w.args endian
  ⟨w.expected.val,⟨p,o,Packing.pack endian d⟩,w.event 0,w.event 1,
    fun f => w.event (Pc.field f).index,w.event 5⟩

theorem Attempt.publication (w : Attempt) (endian : Bool) (success : w.success = true) :
    WriterMir.publication w.event (w.program endian) = some (w.call endian) := by
  simp only [Attempt.program,success,WriterMir.published,Attempt.call]
  congr 2
  funext f
  cases f <;> rfl

def Attempt.published (w : Attempt) (endian : Bool) : ConcurrentReplay.Event :=
  ⟨w.event 5,match w.origin with
    | .clear => .cleared (w.call endian)
    | .store frame b d => .stored frame (w.call endian) b d⟩

theorem Attempt.published_call (w : Attempt) (endian : Bool) :
    (w.published endian).call = some (w.call endian) := by
  cases h : w.origin <;> simp [Attempt.published,ConcurrentReplay.Event.call,h]

theorem Attempt.words (w : Attempt) (endian : Bool) : (w.published endian).action.Words endian := by
  cases h : w.origin <;> simp [Attempt.published,Attempt.call,Attempt.args,h,ConcurrentReplay.Action.Words,
    SlotJournal.packed,Packing.pack_unpack]

structure Site where
  invocation : Nat
  pc : Pc
  deriving DecidableEq

inductive Record where
  | localCall (worker : Nat) (event : NativeFrames.Event)
  | writer (site : Site)

abbrev Calls := Nat → Attempt

def Site.event (c : Calls) (s : Site) := (c s.invocation).event s.pc.index

def Site.action (c : Calls) (endian : Bool) (s : Site) : WriterTrace.Action Nat :=
  if s.pc = .cas then .claim ((c s.invocation).call endian) else .publish ((c s.invocation).call endian)

def Site.word (c : Calls) (endian : Bool) (s : Site) : Word :=
  match s.pc with
  | .cas => (c s.invocation).expected.val ||| 1
  | .field f => WriterTrace.fieldWord ((c s.invocation).call endian).payload f
  | .finish => SeqLock.publish (c s.invocation).expected.val
  | .fence => 0

/-- A failed compare_exchange has no place in the modification order. -/
def writes (success : Bool) : Option WriterMir.Atom → Option (Location × Word)
  | some (.cas loc _ new _ _) => if success then some (loc,new) else none
  | some (.store loc value _) => some (loc,value)
  | _ => none

theorem Site.modification (c : Calls) (endian : Bool) (s : Site) :
    writes (c s.invocation).success ((c s.invocation).program endian).1[s.pc.index]? =
      if (c s.invocation).success then s.pc.location.map (fun loc => (loc,s.word c endian)) else none := by
  obtain ⟨i,pc⟩ := s
  cases h : (c i).success <;> cases pc with
  | field f => cases f <;> simp [Attempt.program,WriterMir.writeStable,h,Pc.index,Pc.location,
      Site.word,writes,Attempt.call,WriterTrace.fieldWord]
  | _ => simp [Attempt.program,WriterMir.writeStable,h,Pc.index,Pc.location,Site.word,writes,SeqLock.publish]

def select (c : Calls) (a : Address) (loc : Location) : Record → Option Site
  | .localCall .. => none
  | .writer s =>
    if (c s.invocation).address = a ∧ (c s.invocation).success = true ∧ s.pc.location = some loc
    then some s else none

def mods (c : Calls) (rs : List Record) (a : Address) (loc : Location) := rs.filterMap (select c a loc)

theorem mem_mods : s ∈ mods c rs a loc ↔ Record.writer s ∈ rs ∧
    (c s.invocation).address = a ∧ (c s.invocation).success = true ∧ s.pc.location = some loc := by
  constructor
  · intro mem
    obtain ⟨r,member,eq⟩ := List.mem_filterMap.mp mem
    cases r with
    | localCall => contradiction
    | writer t =>
      simp only [select] at eq
      split at eq <;> try contradiction
      cases eq
      exact ⟨member,by assumption⟩
  · rintro ⟨present,atSlot,success,location⟩
    exact List.mem_filterMap.mpr ⟨_,present,by simp [select,atSlot,success,location]⟩

def mark (c : Calls) (endian : Bool) : Record → Option Mark
  | .localCall .. => none
  | .writer s =>
    let w := c s.invocation
    if w.success then match s.pc with
      | .cas => some (.claim w.address (w.call endian))
      | .finish => some (.finish w.address (w.published endian))
      | _ => none
    else none

def frameRecord (c : Calls) (endian : Bool) : Record → Option NativeFrames.Record
  | .localCall w e => some (.localCall w e)
  | .writer s => ((mark c endian (.writer s)).bind Mark.event?).map .table

def sequence (c : Calls) (endian : Bool) (rs : List Record) (a : Address) :=
  (mods c rs a .sequence).map (Site.action c endian)

def fields (c : Calls) (endian : Bool) (rs : List Record) (a : Address) (f : Field) :=
  (mods c rs a (.payload f)).map (fun s => (c s.invocation).call endian)

theorem marks_sequence (c : Calls) (endian : Bool) (rs : List Record) (a : Address) :
    sequence c endian rs a = (rs.filterMap (mark c endian)).filterMap (Mark.actionAt a) := by
  simp only [sequence,mods,List.map_filterMap,List.filterMap_filterMap]
  congr 1
  funext r
  cases r with
  | localCall => rfl
  | writer s =>
    obtain ⟨i,pc⟩ := s
    cases success : (c i).success <;> cases pc <;> by_cases atSlot : (c i).address = a <;>
      simp [select,mark,Site.action,Pc.location,Mark.actionAt,success,atSlot,Attempt.published_call]
    rfl

theorem marks_table (c : Calls) (endian : Bool) (rs : List Record) :
    (rs.filterMap (mark c endian)).filterMap Mark.event? =
      NativeFrames.tableEvents (rs.filterMap (frameRecord c endian)) := by
  unfold NativeFrames.tableEvents
  simp only [List.filterMap_filterMap]
  congr 1
  funext r
  cases r with
  | localCall => rfl
  | writer s =>
    change _ = (((mark c endian (.writer s)).bind Mark.event?).map NativeFrames.Record.table).bind _
    cases (mark c endian (.writer s)).bind Mark.event? <;> rfl

/-- Every projected table event is the final release store of a particular
    successful invocation. A failed CAS or unfinished writer cannot supply it. -/
theorem publishing_site (c : Calls) (endian : Bool) (rs : List Record)
    (member : e ∈ NativeFrames.tableEvents (rs.filterMap (frameRecord c endian))) :
    ∃ k, Record.writer ⟨k,.finish⟩ ∈ rs ∧ (c k).success = true ∧ e = (c k).published endian := by
  rw [←marks_table] at member
  obtain ⟨marked,hit,kept⟩ := List.mem_filterMap.mp member
  obtain ⟨record,inside,found⟩ := List.mem_filterMap.mp hit
  cases record with
  | localCall => contradiction
  | writer site =>
    obtain ⟨k,pc⟩ := site
    cases success : (c k).success <;> cases pc <;> simp [mark,success] at found
    all_goals subst marked
    · contradiction
    · exact ⟨k,inside,success,(Option.some.inj kept).symm⟩

theorem Attempt.stored_payload (w : Attempt) (endian : Bool) (query : ProbeRetry.Key) (d : EntryData)
    (equal : (w.call endian).payload = ⟨query.1,query.2,Packing.pack endian d⟩) (occupied : d.bound ≠ 0#8) :
    ∃ owner b, w.origin = .store owner b d ∧ (b.val.player,b.val.opponent) = query := by
  obtain ⟨qp,qo⟩ := query
  have metadata := congrArg (fun words => Packing.unpack endian words.metadata) equal
  cases h : w.origin with
  | clear =>
    have zero : Packing.unpack endian 0 = d := by simpa [call,args,h,Packing.pack_unpack,Packing.unpack_pack] using metadata
    rw [←zero] at occupied
    cases endian <;> contradiction
  | store owner b entry =>
    have data : entry = d := by simpa [call,args,h,Packing.unpack_pack] using metadata
    refine ⟨owner,b,data ▸ rfl,Prod.ext ?_ ?_⟩
    · simpa only [call,args,h] using congrArg SeqLock.Payload.player equal
    · simpa only [call,args,h] using congrArg SeqLock.Payload.opponent equal

end NeuralReversi.PvsParallel.NativeTableWrites.Journal
