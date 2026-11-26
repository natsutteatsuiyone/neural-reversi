import NeuralReversi.PvsParallel.SlotFrameMir.Waiter

namespace NeuralReversi.PvsParallel.SlotFrameMir

open Threading
open AtomicWord (Write Read Event ReadWindow InReleaseSequence)
open CutoffVisibility (History HappensBefore)
open SplitCompletion (Action)

/-- A generation of the physical mask without any assumption about legal
    releases or about the location of a returned zero. -/
structure RawMask (ws : List Write) where
  owner : Worker
  actions : List Action
  mark : Fin ws.length
  bounded : mark.val + actions.length < ws.length
  initial : (ws.get mark).value = Bitboard.squareMask owner
  rmw : ∀ i : Fin actions.length,
    let pos : Fin ws.length := ⟨mark.val+i.val+1,by have:=i.isLt;omega⟩
    let prev : Fin ws.length := ⟨mark.val+i.val,by have:=i.isLt;omega⟩
    (ws.get pos).event = (actions.get i).event ∧ (ws.get pos).release = (actions.get i).release ∧
      (ws.get pos).rmw = true ∧ (ws.get pos).value = (actions.get i).updateBits (ws.get prev).value

def RawMask.at (g : RawMask ws) (i : Fin g.actions.length) : Fin ws.length :=
  ⟨g.mark.val+i.val+1,by have:=g.bounded;have:=i.isLt;omega⟩

def RawMask.before (g : RawMask ws) (i : Fin g.actions.length) : Fin ws.length :=
  ⟨g.mark.val+i.val,by have:=g.bounded;have:=i.isLt;omega⟩

private theorem raw_keeps_bit (a : Action) (bits : BitVec 64) (w : Worker) (on : bits.getLsbD w.val = true)
    (none : ∀ e, a ≠ .finish w e) : (a.updateBits bits).getLsbD w.val = true := by
  cases a with
  | book j e => simp only [Action.updateBits,BitVec.getLsbD_or,on,Bool.true_or]
  | finish j e =>
    have ne : w ≠ j := fun eq => none e (eq ▸ rfl)
    simp only [Action.updateBits,BitVec.getLsbD_and,BitVec.getLsbD_not,Bitboard.squareMask_bit,
      w.isLt,decide_true,Bool.true_and,Fin.val_inj,ne,decide_false,Bool.not_false,on]

private theorem raw_survives (g : RawMask ws) (start : Fin ws.length) (after : g.mark.val ≤ start.val)
    (bit : (ws.get start).value.getLsbD w.val = true)
    (n : Nat) (bounded : start.val+n ≤ g.mark.val+g.actions.length)
    (absent : ∀ i : Fin g.actions.length, start.val < (g.at i).val → (g.at i).val ≤ start.val+n →
      ∀ e, g.actions.get i ≠ .finish w e) :
    (ws.get ⟨start.val+n,by have:=g.bounded;omega⟩).value.getLsbD w.val = true := by
  induction n with
  | zero => exact bit
  | succ n ih =>
    let i : Fin g.actions.length := ⟨start.val+n-g.mark.val,by omega⟩
    have eq : g.mark.val + i.val = start.val+n := by dsimp [i];omega
    have idx : (g.at i).val = start.val+n+1 := by simp [RawMask.at,eq]
    have prev := ih (by omega) (fun j lo hi => absent j lo (by omega))
    have before : (ws.get ⟨g.mark.val+i.val,by have:=g.bounded;omega⟩).value.getLsbD w.val=true := by
      simpa [eq] using prev
    have ok := raw_keeps_bit (g.actions.get i) _ w before (absent i (by omega) (by omega))
    rw [← (g.rmw i).2.2.2] at ok
    simpa [eq,Nat.add_assoc] using ok

theorem RawMask.release_between (g : RawMask ws) (lo hi : Fin ws.length) (w : Worker)
    (base : g.mark.val ≤ lo.val) (order : lo.val ≤ hi.val) (top : hi.val ≤ g.mark.val+g.actions.length)
    (on : (ws.get lo).value.getLsbD w.val = true) (zero : (ws.get hi).value = 0) :
    ∃ i : Fin g.actions.length, ∃ e, g.actions.get i = .finish w e ∧
      lo.val < (g.at i).val ∧ (g.at i).val ≤ hi.val := by
  by_cases found : ∃ i : Fin g.actions.length, ∃ e, g.actions.get i = .finish w e ∧
      lo.val < (g.at i).val ∧ (g.at i).val ≤ hi.val
  · exact found
  · have survived := raw_survives g lo base on (hi.val-lo.val) (by omega)
      (fun i lower upper e eq => found ⟨i,e,eq,lower,by omega⟩)
    have idx : lo.val+(hi.val-lo.val)=hi.val := by omega
    have on2 : (ws.get hi).value.getLsbD w.val = true := by simpa [idx] using survived
    rw [zero] at on2; simp at on2

/-- Protected initial enrollment finishes before the first release. Further
    bookings require the still-open flag under the same slot lock. -/
def ClosedAfterFirst (g : RawMask ws) : Prop := ∀ i j : Fin g.actions.length,
  i.val < j.val → (g.actions.get i).release = true → (g.actions.get j).release = true

private theorem zero_suffix (g : RawMask ws) (src : Fin ws.length) (zero : (ws.get src).value=0)
    (all : ∀ i : Fin g.actions.length, src.val < (g.at i).val → (g.actions.get i).release=true)
    (base : g.mark.val ≤ src.val) (n : Nat) (bound : src.val+n ≤ g.mark.val+g.actions.length) :
    (ws.get ⟨src.val+n,by have:=g.bounded;omega⟩).value=0 := by
  induction n with
  | zero => exact zero
  | succ n ih =>
    let i : Fin g.actions.length := ⟨src.val+n-g.mark.val,by omega⟩
    have eq : g.mark.val+i.val=src.val+n := by dsimp [i];omega
    have idx : (g.at i).val = src.val+n+1 := by simp [RawMask.at,eq]
    have prev := ih (by omega)
    have old : (ws.get ⟨g.mark.val+i.val,by have:=g.bounded;omega⟩).value=0 := by simpa [eq] using prev
    have wrote := (g.rmw i).2.2.2
    have how := all i (by omega)
    rw [old] at wrote
    cases entry : g.actions.get i with
    | book w e => rw [entry] at how; cases how
    | finish w e =>
      rw [entry] at wrote
      simpa [eq,Nat.add_assoc,Action.updateBits] using wrote

/-- Erasing an already absent bit is allowed. Consequently the zero source may
    have later RMWs; every one of their values is still zero. -/
theorem RawMask.stays_zero (g : RawMask ws) (closed : ClosedAfterFirst g) (src later : Fin ws.length)
    (base : g.mark.val ≤ src.val) (order : src.val ≤ later.val)
    (within : later.val ≤ g.mark.val+g.actions.length) (zero : (ws.get src).value=0) :
    (ws.get later).value=0 := by
  have on : (ws.get g.mark).value.getLsbD g.owner.val = true := by
    rw [g.initial,Bitboard.squareMask_bit];simp
  obtain ⟨i,e,left,_,bound⟩ := g.release_between g.mark src g.owner (Nat.le_refl _) base (by omega) on zero
  have all : ∀ j : Fin g.actions.length, src.val < (g.at j).val → (g.actions.get j).release = true := by
    intro j lt
    change g.mark.val+i.val+1 ≤ src.val at bound
    change src.val < g.mark.val+j.val+1 at lt
    exact closed i j (by omega) (by rw [left];rfl)
  have gone := zero_suffix g src zero all base (later.val-src.val) (by omega)
  have idx : src.val+(later.val-src.val)=later.val := by omega
  simpa [idx] using gone

theorem RawMask.sync_reset (g : RawMask ws) (hist : History) (r : Read ws) (acquire : r.acquire=true)
    (i : Fin g.actions.length) (rel : (g.actions.get i).release=true) (within : r.source.val ≤ g.mark.val+g.actions.length)
    (ahead : (g.at i).val ≤ r.source.val)
    (sync : ∀ k : Fin ws.length, r.acquire=true → InReleaseSequence ws k r.source → hist.otherSync (ws.get k).event r.event) :
    HappensBefore hist (g.actions.get i).event r.event := by
  have h : InReleaseSequence ws (g.at i) r.source := by
    refine ⟨(g.rmw i).2.1.trans rel,ahead,?_⟩
    intro k lo hi
    change g.mark.val+i.val+1 < k.val at lo
    let j : Fin g.actions.length := ⟨k.val-g.mark.val-1,by omega⟩
    have idx : g.at j = k := Fin.ext (by change g.mark.val+(k.val-g.mark.val-1)+1=k.val;omega)
    exact idx ▸ (g.rmw j).2.2.1
  have joined := HappensBefore.other (sync _ acquire h)
  have ev : (ws.get (g.at i)).event = (g.actions.get i).event := (g.rmw i).1
  exact ev ▸ joined

/-- All prior normal workers that released a set bit synchronize with the
    exiting Acquire, even if unwinding issued an additional unset-bit RMW. -/
theorem RawMask.credited_release (g : RawMask ws) (closed : ClosedAfterFirst g)
    (hist : History) (r : Read ws) (acquire : r.acquire=true)
    (range : ReadWindow r (HappensBefore hist) g.mark (g.mark.val+g.actions.length))
    (zero : (ws.get r.source).value=0) (i : Fin g.actions.length) (w : Worker) (e : Event)
    (reset : g.actions.get i=.finish w e) (on : (ws.get (g.before i)).value.getLsbD w.val=true)
    (sync : ∀ k : Fin ws.length, r.acquire=true → InReleaseSequence ws k r.source →
      hist.otherSync (ws.get k).event r.event) : HappensBefore hist e r.event := by
  have ahead : (g.at i).val ≤ r.source.val := by
    by_cases late : r.source.val ≤ (g.before i).val
    · have within : (g.before i).val ≤ g.mark.val+g.actions.length := by
        change g.mark.val+i.val ≤ _;have:=i.isLt;omega
      have gone:=g.stays_zero closed r.source (g.before i) range.bounds.1 late within zero
      rw [gone] at on;simp at on
    · change ¬ r.source.val ≤ g.mark.val+i.val at late
      change g.mark.val+i.val+1 ≤ _;omega
  have found:=g.sync_reset hist r acquire i (by rw [reset];rfl) range.bounds.2 ahead sync
  rw [reset] at found
  exact found

/-- Last physical modification no later than the worker's capture. The monotone
    clock includes all slot-mutex mask updates. -/
structure CaptureCut (g : RawMask ws) (clock : Event → Nat) (capture : Event) where
  cut : Fin ws.length
  base : g.mark.val ≤ cut.val
  endAt : cut.val ≤ g.mark.val+g.actions.length
  after : ∀ i : Fin g.actions.length, cut.val < (g.at i).val → clock capture < clock (g.actions.get i).event

private theorem last_true (p : Nat → Prop) (first : p 0) (limit : Nat) :
    ∃ n, n ≤ limit ∧ p n ∧ ∀ k, n < k → k ≤ limit → ¬p k := by
  induction limit with
  | zero => exact ⟨0,Nat.le_refl _,first,fun _ lo hi _ => by omega⟩
  | succ limit ih =>
    by_cases found : p (limit+1)
    · exact ⟨_,Nat.le_refl _,found,fun _ lo hi _ => by omega⟩
    · obtain ⟨n,le,yes,final⟩ := ih
      exact ⟨n,by omega,yes,fun k lo hi pk => by
        by_cases atEnd : k=limit+1
        · exact found (atEnd ▸ pk)
        · exact final k lo (by omega) pk⟩

theorem capture_cut (g : RawMask ws) (clock : Event → Nat) (capture : Event)
    (published : clock (ws.get g.mark).event ≤ clock capture) :
    ∃ c : CaptureCut g clock capture, clock (ws.get c.cut).event ≤ clock capture := by
  let eligible := fun n => ∃ k : Fin ws.length, k.val=g.mark.val+n ∧ clock (ws.get k).event ≤ clock capture
  obtain ⟨n,bound,⟨k,eq,le⟩,last⟩ := last_true eligible ⟨g.mark,by simp,published⟩ g.actions.length
  refine ⟨⟨k,by omega,by omega,?_⟩,le⟩
  intro i later
  have event : (ws.get (g.at i)).event = (g.actions.get i).event := (g.rmw i).1
  by_cases bad : clock (ws.get (g.at i)).event ≤ clock capture
  · have excluded := last (i.val+1) (by change k.val<g.mark.val+i.val+1 at later;omega) i.isLt
    exact (excluded ⟨g.at i,by simp [RawMask.at,Nat.add_assoc],bad⟩).elim
  · rw [← event];omega

theorem RawMask.live_at_reuse (g : RawMask ws) (closed : ClosedAfterFirst g)
    (hist : History) (clock : Event → Nat) (capture : Event) (cut : CaptureCut g clock capture)
    (w : Worker) (r : Read ws) (range : ReadWindow r (HappensBefore hist) g.mark (g.mark.val+g.actions.length))
    (zero : (ws.get r.source).value=0) (acquire : r.acquire=true)
    (sync : ∀ k : Fin ws.length, r.acquire=true → InReleaseSequence ws k r.source → hist.otherSync (ws.get k).event r.event) :
    (ws.get cut.cut).value.getLsbD w.val = false ∨
    ∃ i : Fin g.actions.length, ∃ e, g.actions.get i = .finish w e ∧ clock capture < clock e ∧
      HappensBefore hist e r.event := by
  cases bit : (ws.get cut.cut).value.getLsbD w.val
  · exact Or.inl rfl
  · have bounds := range.bounds
    have before : cut.cut.val ≤ r.source.val := by
      by_cases ord : r.source.val ≤ cut.cut.val
      · have lost := g.stays_zero closed r.source cut.cut bounds.1 ord cut.endAt zero
        rw [lost] at bit;simp at bit
      · omega
    obtain ⟨i,e,release,lower,upper⟩ := g.release_between cut.cut r.source w cut.base before bounds.2 bit zero
    have ordered := cut.after i lower
    have join := g.sync_reset hist r acquire i (by rw [release];rfl) bounds.2 upper sync
    rw [release] at ordered join
    exact Or.inr ⟨i,e,release,ordered,join⟩

end NeuralReversi.PvsParallel.SlotFrameMir
