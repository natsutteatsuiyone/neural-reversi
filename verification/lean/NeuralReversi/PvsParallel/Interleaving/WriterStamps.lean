import NeuralReversi.PvsParallel.Interleaving.PhysicalReadback
import Std.Data.TreeMap.Lemmas

namespace NeuralReversi.PvsParallel.Interleaving.WriterStamps

open TranspositionTable
open NativeTableWrites.Journal (Attempt Calls Origin Pc Record Site)
open CallJournal (State Event Writer)

attribute [local instance] lexOrd

/-- Registers at writer entry, including the observed success/failure of its CAS.
    There are no caller-supplied event IDs. -/
structure Operands where
  address : FrameTableReads.Address
  expected : {v : SeqLock.Word // v &&& 1 = 0}
  origin : Origin
  success : Bool

def Operands.ofAttempt (a : Attempt) : Operands := ⟨a.address,a.expected,a.origin,a.success⟩

structure Index where
  active : Std.TreeMap Nat (Nat × Nat) := {}
  stamps : Std.TreeMap (Nat × Nat) Nat := {}

private def advance (old : Index) (time : Nat) : Event α → Index
  | .beginWrite w k _ => {old with active := old.active.insert w (k,0)}
  | .atomic w k =>
    match old.active[w]? with
    | some (key,pc) =>
      if key = k then ⟨old.active.insert w (k,pc+1),old.stamps.insert (k,pc) time⟩ else old
    | none => old
  | _ => old

def index (es : Joint.Tape α) : Index :=
  (es.map Joint.Event.engineEvent).zipIdx.foldl (fun a row => advance a row.2 row.1) {}

def restamp (data : Nat → Operands) (es : Joint.Tape α) : Calls :=
  let rows := (index es).stamps
  fun k =>
    let a := data k
    ⟨a.address,a.expected,a.origin,a.success,fun i => rows[(k,i)]?.getD es.length⟩

private theorem fetch_index {a : Attempt} (hit : a.fetch endian n = some pc) : pc.index = n := by
  have bound : n < (a.program endian).1.length := by
    obtain ⟨_,found,_⟩ := Option.map_eq_some_iff.mp hit
    exact (List.getElem?_eq_some_iff.mp found).1
  cases h : a.success <;> simp [Attempt.program,WriterMir.writeStable,h] at bound
  all_goals
    have ix : n = 0 ∨ n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 5 := by omega
    rcases ix with rfl | rfl | rfl | rfl | rfl | rfl <;>
      simp [Attempt.fetch,Attempt.program,WriterMir.writeStable,h] at hit <;> subst pc <;> rfl

private structure Matches (c : Calls) (a : Index) (s : State α) : Prop where
  active : ∀ w (v : Writer α), s.writing w = some v → a.active[w]? = some (v.invocation,v.next)
  recorded : ∀ site, Record.writer site ∈ s.records →
    a.stamps[(site.invocation,site.pc.index)]? = some (site.event c)
  emitted : ∀ k i t, a.stamps[(k,i)]? = some t →
    ∃ pc, Record.writer ⟨k,pc⟩ ∈ s.records ∧ pc.index = i ∧ (c k).event i = t

variable {α : Type} {s t : State α} {e : Event α} {es : Joint.Tape α}

private theorem preserve (m : Matches c a s) (writing : t.writing = s.writing)
    (writers : ∀ site, Record.writer site ∈ t.records ↔ Record.writer site ∈ s.records) : Matches c a t where
  active w v held := m.active w v (writing ▸ held)
  recorded site mem := m.recorded _ ((writers site).mp mem)
  emitted k i stamp ok := by
    obtain ⟨p,mem,idx,time⟩ := m.emitted k i stamp ok
    exact ⟨p,(writers _).mpr mem,idx,time⟩

private theorem preserved (old : Matches c a s) (run : CallJournal.step mc fc c endian s time e = some t) :
    Matches c (advance a time e) t := by
  cases e with
  | other => cases run; exact old
  | localCall w event args =>
    obtain ⟨writing,issued⟩ := CallJournal.localWriter run
    exact preserve old writing (by intro site; simp [State.records,issued,CallJournal.Issued.record])
  | boundaryScore =>
    obtain ⟨writing,issued⟩ := CallJournal.boundaryWriter run
    exact preserve old writing (by intro site; simp [State.records,issued])
  | beginWrite who k arg =>
    simp only [CallJournal.step,guard,Bind.bind,Option.bind,Pure.pure] at run
    split at run <;> try contradiction
    cases run
    refine ⟨?_,old.recorded,old.emitted⟩
    intro w v held
    change (if who = w then some _ else s.writing w) = some v at held
    by_cases same : who = w
    · subst w
      simp only [ite_true,Option.some.injEq] at held
      subst v
      exact Std.TreeMap.getElem?_insert_self
    · simpa [advance,Std.TreeMap.getElem?_insert,Std.LawfulEqOrd.compare_eq_iff_eq,same] using old.active _ _ (by simpa [same] using held)
  | atomic who k =>
    unfold CallJournal.step at run
    obtain ⟨v,held,step⟩ := Option.bind_eq_some_iff.mp run
    simp only [guard] at step
    split at step <;> try contradiction
    rename_i same
    have inv := beq_iff_eq.mp same
    subst k
    change (Option.bind ((c v.invocation).fetch endian v.next) _) = some t at step
    obtain ⟨pc,fetched,step⟩ := Option.bind_eq_some_iff.mp step
    split at step <;> try contradiction
    rename_i now
    have atTime := beq_iff_eq.mp now
    cases step
    have idx := fetch_index fetched
    have seen := old.active _ _ held
    simp only [advance,seen,↓reduceIte]
    constructor
    · intro w next stored
      change (if who = w then (if _ then some {v with next := v.next+1} else none) else s.writing w) = some next at stored
      by_cases same : who = w
      · subst w
        simp only [ite_true] at stored
        split at stored <;> cases stored
        exact Std.TreeMap.getElem?_insert_self
      · simpa [Std.TreeMap.getElem?_insert,Std.LawfulEqOrd.compare_eq_iff_eq,same] using old.active _ _ (by simpa [same] using stored)
    · intro site listed
      change Record.writer site ∈ (s.issued ++ [CallJournal.Issued.atomic v pc]).map CallJournal.Issued.record at listed
      simp only [List.map_append,List.map_cons,List.map_nil,CallJournal.Issued.record,List.mem_append,
        List.mem_singleton,Record.writer.injEq] at listed
      simp only [Std.TreeMap.getElem?_insert,Std.LawfulEqOrd.compare_eq_iff_eq]
      split
      · rename_i key
        obtain ⟨kEq,iEq⟩ := Prod.mk.inj key
        simp [Site.event,←kEq,←iEq,←idx,atTime]
      · rcases listed with prior | rfl
        · exact old.recorded _ prior
        · simp [idx] at *
    · intro k i tt hit
      simp only [Std.TreeMap.getElem?_insert,Std.LawfulEqOrd.compare_eq_iff_eq] at hit
      split at hit
      · rename_i key
        obtain ⟨rfl,rfl⟩ := Prod.mk.inj key
        cases hit
        exact ⟨pc,by simp [State.records,CallJournal.Issued.record],idx,idx ▸ atTime⟩
      · obtain ⟨p,mem,ix,stamp⟩ := old.emitted _ _ _ hit
        exact ⟨p,by simpa [State.records,CallJournal.Issued.record] using
          List.mem_append_left [Record.writer ⟨v.invocation,pc⟩] mem,ix,stamp⟩

private theorem matchesRun (good : CallJournal.run mc fc c endian (es.map Joint.Event.engineEvent) = some t) :
    Matches c (index es) t := by
  have fold (xs : List (Event α × Nat)) (a : Index) (s : State α) (m : Matches c a s)
      (run : xs.foldlM (fun st row => CallJournal.step mc fc c endian st row.2 row.1) s = some t) :
      Matches c (xs.foldl (fun x row => advance x row.2 row.1) a) t := by
    induction xs generalizing a s with
    | nil => cases run; exact m
    | cons row rest ih =>
      obtain ⟨s',step,tail⟩ := Option.bind_eq_some_iff.mp run
      exact ih _ _ (preserved m step) tail
  exact fold _ {} CallJournal.start
    ⟨by simp [CallJournal.start],by simp [CallJournal.start,State.records],by simp⟩ good

theorem stamp_eq (ran : CallJournal.run mc fc c endian (es.map Joint.Event.engineEvent) = some s)
    (member : Record.writer site ∈ s.records) :
    (index es).stamps[(site.invocation,site.pc.index)]? = some (site.event c) :=
  (matchesRun ran).recorded site member

theorem issued_iff {t : Nat} (ran : CallJournal.run mc fc c endian (es.map Joint.Event.engineEvent) = some s) :
    (index es).stamps[(k,i)]? = some t ↔
      ∃ pc, Record.writer ⟨k,pc⟩ ∈ s.records ∧ pc.index = i ∧ (c k).event i = t := by
  refine ⟨(matchesRun ran).emitted _ _ _,?_⟩
  rintro ⟨pc,mem,rfl,rfl⟩
  exact stamp_eq ran mem

theorem restamp_eq (ran : CallJournal.run mc fc c endian (es.map Joint.Event.engineEvent) = some s)
    (member : Record.writer site ∈ s.records) :
    site.event (restamp data es) = site.event c := by
  change ((index es).stamps[(site.invocation,site.pc.index)]?).getD es.length = _
  rw [stamp_eq ran member]
  rfl

private def Stamps (old fresh : Calls) (s : State α) :=
  ∀ site, Record.writer site ∈ s.records → site.event fresh = site.event old

private theorem transfer (fresh : Calls) (program : ∀ k i, (fresh k).fetch endian i = (c k).fetch endian i)
    (ok : CallJournal.step mc fc c endian s now e = some t) :
    (∀ site, Record.writer site ∈ s.records → Record.writer site ∈ t.records) ∧
    (Stamps c fresh t → CallJournal.step mc fc fresh endian s now e = some t) := by
  cases e with
  | other =>
    cases ok
    exact ⟨fun _ h => h,fun _ => rfl⟩
  | localCall =>
    have issued := (CallJournal.localWriter ok).2
    exact ⟨fun site member => by simpa [State.records,issued,CallJournal.Issued.record] using member,fun _ => ok⟩
  | boundaryScore =>
    have issued := (CallJournal.boundaryWriter ok).2
    exact ⟨fun site member => by simpa [State.records,issued] using member,fun _ => ok⟩
  | beginWrite =>
    refine ⟨?_,fun _ => ok⟩
    simp only [CallJournal.step,guard,Bind.bind,Option.bind,Pure.pure] at ok
    split at ok <;> cases ok
    exact fun _ member => member
  | atomic who k =>
    unfold CallJournal.step at ok
    obtain ⟨v,held,step⟩ := Option.bind_eq_some_iff.mp ok
    simp only [guard] at step
    split at step <;> try contradiction
    rename_i idEq
    have inv := beq_iff_eq.mp idEq
    subst k
    change (Option.bind ((c v.invocation).fetch endian v.next) _) = some t at step
    obtain ⟨pc,fetched,step⟩ := Option.bind_eq_some_iff.mp step
    split at step <;> try contradiction
    rename_i timeEq
    cases step
    refine ⟨fun site mem => by simpa [State.records,CallJournal.Issued.record] using
      (List.mem_append_left [Record.writer ⟨v.invocation,pc⟩] mem),?_⟩
    intro agree
    have stamp := agree ⟨v.invocation,pc⟩ (by simp [State.records,CallJournal.Issued.record])
    change (fresh v.invocation).event pc.index = (c v.invocation).event pc.index at stamp
    simp [CallJournal.step,held,program,fetched,guard,stamp,beq_iff_eq.mp timeEq]

/-- All previously accepted tapes remain accepted with the identical decoded state after the
    caller's arbitrary timestamp functions have been discarded. -/
theorem replay (ran : CallJournal.run mc fc c endian (es.map Joint.Event.engineEvent) = some t) :
    CallJournal.run mc fc (restamp (fun k => Operands.ofAttempt (c k)) es) endian (es.map Joint.Event.engineEvent) = some t := by
  let fresh := restamp (fun k => Operands.ofAttempt (c k)) es
  have same (k i) : (fresh k).fetch endian i = (c k).fetch endian i := rfl
  have fold (xs : List (Event α × Nat)) (s : State α)
      (ok : xs.foldlM (fun st row => CallJournal.step mc fc c endian st row.2 row.1) s = some t) :
      (∀ site, Record.writer site ∈ s.records → Record.writer site ∈ t.records) ∧
      (Stamps c fresh t → xs.foldlM (fun st row => CallJournal.step mc fc fresh endian st row.2 row.1) s = some t) := by
    induction xs generalizing s with
    | nil =>
      cases ok
      exact ⟨fun _ m => m,fun _ => rfl⟩
    | cons row rest ih =>
      obtain ⟨next,step,nextRun⟩ := Option.bind_eq_some_iff.mp ok
      obtain ⟨kept,runStep⟩ := transfer fresh same step
      obtain ⟨later,runTail⟩ := ih _ nextRun
      refine ⟨fun site m => later _ (kept site m),?_⟩
      intro agree
      exact Option.bind_eq_some_iff.mpr ⟨_,runStep (fun site m => agree _ (later site m)),runTail agree⟩
  exact (fold _ CallJournal.start ran).2 (fun _ m => restamp_eq ran m)

def Contract := ∀ (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (c : Calls)
    (endian : Bool) (α : Type) (es : Joint.Tape α) (s : State α),
    CallJournal.run mc fc c endian (es.map Joint.Event.engineEvent) = some s →
    CallJournal.run mc fc (restamp (fun k => Operands.ofAttempt (c k)) es) endian
        (es.map Joint.Event.engineEvent) = some s ∧
      ∀ k i t, (index es).stamps[(k,i)]? = some t ↔
        ∃ pc, Record.writer ⟨k,pc⟩ ∈ s.records ∧ pc.index = i ∧ (c k).event i = t

theorem reconstructed : Contract := fun _ _ _ _ _ _ _ ran =>
  ⟨replay ran,fun _ _ _ => issued_iff ran⟩

/-- Reconstruct the writer events before replaying frames or attaching memory evidence. -/
def run (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (data : Nat → Operands)
    (endian : Bool) (es : Joint.Tape α) :=
  CallJournal.run mc fc (restamp data es) endian (es.map Joint.Event.engineEvent)

def physical (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (data : Nat → Operands)
    (endian : Bool) (es : Joint.Tape Unit) (graph : JournalAtomicGraph.Graph) {s : State Unit}
    (accepted : run mc fc data endian es = some s)
    (rules : JournalAtomicGraph.Rules graph (restamp data es) endian s.records es) : PhysicalReadback.Prefix mc fc :=
  ⟨restamp data es,endian,es,s,accepted,graph,rules⟩

namespace Regression

private def mc : Bitboard.Moves.X86Config := ⟨false,false⟩
private def fc : Cutoffs.FlipConfig := ⟨false,false,false⟩
private def arg (k : Nat) : Operands := ⟨⟨0,if k = 2 then 1 else 0⟩,⟨0,by decide⟩,.clear,k != 0⟩

-- Two workers overlap. Writer 0 loses its CAS, writer 1 publishes, and
-- writer 2 reaches its fence on another slot before the capture ends.
private def moves : Joint.Tape Unit := ([
  .beginWrite 2 0 (),.beginWrite 5 1 (),.atomic 5 1,.atomic 2 0,.atomic 5 1,
  .beginWrite 2 2 (),.atomic 2 2,.atomic 5 1,.atomic 5 1,.atomic 5 1,
  .atomic 5 1,.atomic 2 2] : List (Event Unit)).map Joint.Event.engine

private def positions (k : Nat) := (List.range 6).map fun i => (index moves).stamps[(k,i)]?
example : positions 0 = [some 3,none,none,none,none,none] := by decide +kernel
example : positions 1 = [some 2,some 4,some 7,some 8,some 9,some 10] := by decide +kernel
example : positions 2 = [some 6,some 11,none,none,none,none] := by decide +kernel
example : ((run mc fc arg false moves).map fun s => s.records.length) = some 9 := by decide +kernel
example : (run mc fc arg false (moves ++ [.engine (.atomic 5 2)])).isNone := by decide +kernel
example : (run mc fc arg false (moves ++ [.engine (.beginWrite 5 1 ())])).isNone := by decide +kernel
example : (run mc fc arg false (moves ++ [.engine (.atomic 5 1)])).isNone := by decide +kernel
example : (restamp arg moves 2).event 5 = moves.length := by decide +kernel

end Regression

end NeuralReversi.PvsParallel.Interleaving.WriterStamps
