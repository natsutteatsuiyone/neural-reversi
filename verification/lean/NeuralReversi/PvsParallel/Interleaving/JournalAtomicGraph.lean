module
public import NeuralReversi.PvsParallel.Interleaving.ModificationIndex
public import NeuralReversi.PvsParallel.Interleaving.WriterSteps
public import NeuralReversi.PvsParallel.Interleaving.AtomicReadback

@[expose] public section

namespace NeuralReversi.PvsParallel.Interleaving.JournalAtomicGraph

open TranspositionTable
open SeqLock (Word Location)
open FrameTableReads (Address)
open NativeTableWrites.Journal (Calls Record Site Pc mods)
open ModificationIndex (event word decode)

/-- The physical reads-from graph names the actual write event. The initial words and the
    happens-before relation come from the same execution. -/
structure Graph where
  initial : Address → Location → Word
  source : Address → Nat → Option Nat
  hb : Nat → Nat → Prop

private theorem timed (h : WriterSteps.Facts c es rs) : ModificationIndex.Timed c rs es.length :=
  ⟨fun _ mem => h.bound mem,fun _ _ m n same => h.injective m n same,h.distinct⟩

def SB (es : Joint.Tape Op) (u v : Nat) :=
  u < v ∧ v < es.length ∧ Joint.worker es u = Joint.worker es v

/-- The derived model projects HB to emitted events. Synthetic initial tokens and the absent
    modification sentinel are never instructions on a worker. -/
def HB (g : Graph) (es : Joint.Tape Op) (u v : Nat) := u < es.length ∧ v < es.length ∧ g.hb u v

def WriterAt (c : Calls) (rs : List Record) (a : Address) (pc : Pc) (e : Nat) (successful : Bool) :=
  ∃ k, Record.writer ⟨k,pc⟩ ∈ rs ∧ (c k).address = a ∧ (c k).event pc.index = e ∧
    (successful = true → (c k).success = true)

def Reads (c : Calls) (rs : List Record) (es : Joint.Tape Op) (a : Address) (l : Location) (r : Nat) :=
  (∃ mode, Joint.opcode es r = some (.load a l mode)) ∨ (l = .sequence ∧ WriterAt c rs a .cas r false)

def Release (c : Calls) (rs : List Record) (a : Address) (e : Nat) :=
  WriterAt c rs a .finish e true ∨ WriterAt c rs a .cas e true

def Acquire (c : Calls) (rs : List Record) (es : Joint.Tape Op) (a : Address) (e : Nat) :=
  (∃ l, Joint.opcode es e = some (.load a l .acquire)) ∨ WriterAt c rs a .cas e true

/-- A graph edge enters the model only at a read instruction and from a decoded write to
    its precise object. Raw labels on unrelated events carry no reads-from meaning. -/
def RF (g : Graph) (c : Calls) (rs : List Record) (es : Joint.Tape Op) (a : Address) (w r : Nat) :=
  g.source a r = some w ∧ ∃ l i, decode c rs es.length a w = some (l,i) ∧ Reads c rs es a l r

def Graph.value (g : Graph) (c : Calls) (endian : Bool) (rs : List Record) (n : Nat)
    (a : Address) (r : Nat) : Word :=
  (((g.source a r >>= decode c rs n a).map fun (l,i) => word c endian rs g.initial a l i)).getD 0

/-- The strong CAS compares the word identified by its raw source with the actual expected
    operand; a successful RMW is the next modification. A failed invocation has no later PC. -/
def WriterOK (g : Graph) (c : Calls) (endian : Bool) (rs : List Record) (n : Nat) (s : Site) : Prop :=
  let k := s.invocation
  let a := (c k).address
  match s.pc with
  | .cas => ∃ w i, g.source a ((c k).event 0) = some w ∧ decode c rs n a w = some (.sequence,i) ∧
      ((c k).success = true ↔ word c endian rs g.initial a .sequence i = (c k).expected.val) ∧
      ((c k).success = true → (mods c rs a .sequence)[i]? = some s)
  | _ => (c k).success = true

/-- These physical memory laws compare only identified writes to the same atomic object. The
    per-location modification order is the writer projection of the input journal. An unordered
    relaxed read can observe any allowed source, independently of timestamp order across workers. -/
structure Rules (g : Graph) (c : Calls) (endian : Bool) (rs : List Record) (es : Joint.Tape Op) : Prop where
  trans : ∀ {u v w}, HB g es u v → HB g es v w → HB g es u w
  irrefl : ∀ u, ¬ HB g es u u
  program : ∀ {u v}, SB es u v → HB g es u v
  sync : ∀ {a w r}, Release c rs a w → Acquire c rs es a r → RF g c rs es a w r → HB g es w r
  fences : ∀ {a f w r b}, WriterAt c rs a .fence f true → Joint.opcode es b = some (.fence .acquire) →
    SB es f w → RF g c rs es a w r → SB es r b → HB g es f b
  ww : ∀ {a l u v i j}, decode c rs es.length a u = some (l,i) → decode c rs es.length a v = some (l,j) →
    HB g es u v → i < j
  wr : ∀ {a l u w r i j}, decode c rs es.length a u = some (l,i) → decode c rs es.length a w = some (l,j) →
    HB g es u r → RF g c rs es a w r → i ≤ j
  rr : ∀ {a l u v p r i j}, decode c rs es.length a u = some (l,i) → decode c rs es.length a v = some (l,j) →
    HB g es p r → RF g c rs es a u p → RF g c rs es a v r → i ≤ j
  rw : ∀ {a l u v r i j}, decode c rs es.length a u = some (l,i) → decode c rs es.length a v = some (l,j) →
    RF g c rs es a u r → HB g es r v → i < j
  writer : ∀ s, Record.writer s ∈ rs → WriterOK g c endian rs es.length s
  load : ∀ r a l mode, Joint.opcode es r = some (.load a l mode) →
    ∃ w i, g.source a r = some w ∧ decode c rs es.length a w = some (l,i) ∧
      word c endian rs g.initial a l i = Joint.word es r

theorem emitted (h : WriterSteps.Facts c es rs) (real : event c rs es.length a l i < es.length) :
    decode c rs es.length a (event c rs es.length a l i) = some (l,i) := by
  apply (ModificationIndex.decode_eq_some_iff (timed h)).mpr
  refine ⟨?_,rfl⟩
  apply Nat.le_of_not_gt
  intro outside
  rw [ModificationIndex.event_outside outside] at real
  omega

theorem origin (h : WriterSteps.Facts c es rs) (read : RF g c rs es a (event c rs es.length a l i) r) :
    decode c rs es.length a (event c rs es.length a l i) = some (l,i) := by
  obtain ⟨_,⟨_,_,found,_⟩⟩ := read
  obtain ⟨bound,same⟩ := (ModificationIndex.decode_eq_some_iff (timed h)).mp found
  obtain ⟨rfl,rfl⟩ := ModificationIndex.event_injective (timed h) bound same
  exact found

def memory (h : WriterSteps.Facts c es rs) (laws : Rules g c endian rs es) (a : Address) : SeqLock.Memory Nat where
  sb := SB es
  hb := HB g es
  rf := RF g c rs es a
  write := event c rs es.length a
  writeValue := word c endian rs g.initial a
  readValue := g.value c endian rs es.length a
  rf_value := by intro l i r read; simp [Graph.value,read.1,origin h read]
  release := Release c rs a
  acquire := Acquire c rs es a
  releaseFence := fun e => WriterAt c rs a .fence e true
  acquireFence := fun e => Joint.opcode es e = some (.fence .acquire)
  sb_hb := laws.program
  sb_trans := fun x y => ⟨Nat.lt_trans x.1 y.1,y.2.1,x.2.2.trans y.2.2⟩
  hb_trans := laws.trans
  hb_irrefl := laws.irrefl
  release_acquire := laws.sync
  fence_fence := laws.fences
  write_write := fun edge => laws.ww (emitted h edge.1) (emitted h edge.2.1) edge
  write_read := fun edge read => laws.wr (emitted h edge.1) (origin h read) edge read
  read_read := fun edge x y => laws.rr (origin h x) (origin h y) edge x y

private theorem casAt (present : Record.writer ⟨k,Pc.cas⟩ ∈ rs) :
    WriterAt c rs (c k).address .cas ((c k).event 0) false :=
  ⟨k,present,rfl,rfl,by simp⟩

def access (h : WriterSteps.Facts c es rs) (laws : Rules g c endian rs es) : WriterSteps.Access c endian rs where
  memory := memory h laws
  source := by
    intro a l i r read
    exact ((ModificationIndex.decode_eq_some_iff (timed h)).mp (origin h read)).1
  modified := fun _ _ _ _ hit => ModificationIndex.modified hit
  rmw := by
    intro a j k hit
    obtain ⟨present,addr,success,_⟩ := NativeTableWrites.Journal.mem_mods.mp (List.mem_of_getElem? hit)
    change (c k).address = a at addr
    subst a
    obtain ⟨w,i,raw,found,_,next⟩ := laws.writer _ present
    have same := next success
    have eq : i = j := (h.distinct _ .sequence).eq_of_getElem?_eq (by simp; exact (List.getElem?_eq_some_iff.mp same).1)
      (by simp [List.getElem?_map,same,hit])
    subst i
    obtain ⟨_,stamp⟩ := (ModificationIndex.decode_eq_some_iff (timed h)).mp found
    subst w
    exact ⟨raw,_,_,found,Or.inr ⟨rfl,casAt present⟩⟩
  ordering := by
    rintro ⟨k,pc⟩ present
    have ok := laws.writer _ present
    cases pc with
    | cas =>
      obtain ⟨w,i,raw,found,compared,_⟩ := ok
      cases success : (c k).success <;>
        simp only [NativeTableWrites.Journal.Attempt.program,WriterMir.writeStable,success,ite_true,
          Pc.index,List.getElem?_cons_zero,NativeTableWrites.Journal.mode] <;> intro yes
      · contradiction
      · refine ⟨Or.inr ⟨k,present,rfl,rfl,fun _ => success⟩,?_⟩
        change g.value c endian rs es.length (c k).address ((c k).event 0) = _
        simpa [Graph.value,raw,found] using compared.mp success
    | fence =>
      change (c k).success = true at ok
      simpa [NativeTableWrites.Journal.Attempt.program,WriterMir.writeStable,ok,
        NativeTableWrites.Journal.mode,Pc.index,memory,Site.event] using
        (show WriterAt c rs (c k).address .fence ((c k).event 1) true from ⟨k,present,rfl,rfl,fun _ => ok⟩)
    | finish =>
      change (c k).success = true at ok
      simpa [NativeTableWrites.Journal.Attempt.program,WriterMir.writeStable,ok,
        NativeTableWrites.Journal.mode,Pc.index,memory,Site.event] using
        (show Release c rs (c k).address ((c k).event 5) from Or.inl ⟨k,present,rfl,rfl,fun _ => ok⟩)
    | field f =>
      change (c k).success = true at ok
      cases f <;> simp [NativeTableWrites.Journal.Attempt.program,WriterMir.writeStable,ok,
        NativeTableWrites.Journal.mode,Pc.index]

def execution (h : WriterSteps.Facts c es rs) (laws : Rules g c endian rs es) :
    NativeTableWrites.Journal.Execution c endian rs :=
  h.execution (access h laws) (fun _ _ _ lt inside same => ⟨lt,inside,same⟩)

theorem observed (h : WriterSteps.Facts c es rs) (laws : Rules g c endian rs es) :
    AtomicReadback.Observed (execution h laws) es where
  reads r a l mode load := by
    obtain ⟨w,i,raw,found,eq⟩ := laws.load _ _ _ _ load
    obtain ⟨_,stamp⟩ := (ModificationIndex.decode_eq_some_iff (timed h)).mp found
    subst w
    refine ⟨⟨i,raw,l,i,found,Or.inl ⟨mode,load⟩⟩,?_⟩
    change g.value c endian rs es.length a r = _
    simpa [Graph.value,raw,found] using eq
  acquire := fun _ _ l load => Or.inl ⟨l,load⟩
  fence := fun _ op _ => op
  sb := fun _ _ _ lt inside same => ⟨lt,inside,same⟩

theorem readWrite (h : WriterSteps.Facts c es rs) (laws : Rules g c endian rs es) :
    AtomicReadback.ReadWrite (execution h laws) := by
  intro a l r i j s hit read edge
  have pub := ModificationIndex.decode_modification (timed h) hit
  apply laws.rw (origin h read) pub read edge

namespace Regression

-- A successful CAS claims the slot at event 1. An unordered relaxed load at event 2
-- still observes the initial sequence word, independently of its dummy source annotation.
private def a : Address := ⟨4,1⟩
private def calls : Calls := fun _ => ⟨a,⟨0,by decide⟩,.clear,true,fun pc => pc+1⟩
private def es : Joint.Tape Unit := [.engine (.beginWrite 0 0 ()),.engine (.atomic 0 0),
  .issue 1 (.load a .sequence .relaxed) 99 0]
private def rs : List Record := [.writer ⟨0,.cas⟩]
private def actual : Graph :=
  ⟨fun _ _ => 0,fun addr r => if addr = a ∧ (r = 1 ∨ r = 2) then some 3 else none,
    fun u v => u = 0 ∧ v = 1⟩

private theorem sites : WriterSteps.Facts calls es rs := by
  let mc : Bitboard.Moves.X86Config := ⟨false,false⟩
  let fc : Cutoffs.FlipConfig := ⟨false,false,false⟩
  let events := es.map Joint.Event.engineEvent
  let state := (CallJournal.run mc fc calls false events).get (by decide)
  exact WriterSteps.run (t := state) (mc := mc) (fc := fc) (endian := false) (by rfl)

private theorem noZeroWrite (found : decode calls rs es.length addr 0 = some (l,i)) : False := by
  obtain ⟨inside,stamp⟩ := (ModificationIndex.decode_eq_some_iff (timed sites)).mp found
  cases i with
  | zero =>
    change 3 + ModificationIndex.tag l = 0 at stamp
    omega
  | succ j =>
    have hit := List.getElem?_eq_getElem (show j < _ from inside)
    have pair := ModificationIndex.modified (endian := false) (initial := actual.initial) (n := es.length) hit
    obtain ⟨present,_,_,_⟩ := NativeTableWrites.Journal.mem_mods.mp (List.mem_of_getElem? hit)
    have only : (mods calls rs addr l)[j] = ⟨0,.cas⟩ := Record.writer.inj (List.mem_singleton.mp present)
    rw [pair.1,only] at stamp
    contradiction

private theorem noReadZero {addr w} (read : RF actual calls rs es addr w 0) : False := by
  have raw := read.1
  simp [actual] at raw

private theorem lawful : Rules actual calls false rs es where
  trans := by
    rintro u v w ⟨_,_,_,ev⟩ ⟨_,_,ev',_⟩
    change v = 1 at ev
    change v = 0 at ev'
    omega
  irrefl := by rintro u ⟨_,_,x,y⟩; change u = 0 at x; change u = 1 at y; omega
  program := by
    rintro i j ⟨lt,inside,same⟩
    change j < 3 at inside
    have jCases : j = 1 ∨ j = 2 := by omega
    rcases jCases with rfl | rfl
    · obtain rfl : i = 0 := by omega
      exact ⟨by decide,by decide,rfl,rfl⟩
    · have iCases : i = 0 ∨ i = 1 := by omega
      rcases iCases with rfl | rfl <;> contradiction
  sync := by
    rintro addr w r (⟨k,member,_,_,_⟩ | ⟨k,member,_,stamp,_⟩) _ read
    · simp [rs] at member
    · obtain rfl : k = 0 := by simpa [rs] using member
      subst w
      have raw := read.1
      simp [actual,calls,Pc.index] at raw
  fences := by rintro addr f w r b ⟨k,member,_⟩; simp [rs] at member
  ww := by rintro addr l u v i j found _ ⟨_,_,rfl,_⟩; exact (noZeroWrite found).elim
  wr := by rintro addr l u w r i j found _ ⟨_,_,rfl,_⟩ _; exact (noZeroWrite found).elim
  rr := by rintro addr l u v p r i j _ _ ⟨_,_,rfl,_⟩ read _; exact (noReadZero read).elim
  rw := by rintro addr l u v r i j _ _ read ⟨_,_,rfl,_⟩; exact (noReadZero read).elim
  writer := by
    intro s mem
    obtain rfl : s = ⟨0,.cas⟩ := by simpa [rs] using mem
    exact ⟨3,0,rfl,rfl,by decide,fun _ => rfl⟩
  load := by
    intro r addr l mode load
    rcases r with _|(_|(_|r)) <;> try contradiction
    cases load
    exact ⟨3,0,rfl,rfl,rfl⟩

example : AtomicReadback.Observed (execution sites lawful) es := observed sites lawful
example : ((execution sites lawful).memory a).rf 3 2 := ⟨rfl,.sequence,0,rfl,Or.inl ⟨.relaxed,rfl⟩⟩
example : ¬ ((execution sites lawful).memory a).rf 1 2 := fun read => by cases read.1

end Regression

end NeuralReversi.PvsParallel.Interleaving.JournalAtomicGraph
