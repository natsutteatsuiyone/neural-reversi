module
public import NeuralReversi.TranspositionTable.WriterTrace

@[expose] public section

namespace NeuralReversi.TranspositionTable.Duplicate

open SeqLock (Field Location Memory Word)
open WriterTrace (Slot)

/-- Number of modifications after the initial write, on one atomic object. -/
def modifications (slot : Slot m) : Location → Nat
  | .sequence => slot.history.length
  | .payload f => (slot.fields f).history.length

/-- The visible side effect of a synchronized non-atomic copy of one word.
    Index zero denotes initialization, even if there are later modifications. -/
structure Visible (m : Memory Nat) (length : Location → Nat) (sample : Nat) where
  index : Location → Nat
  bounded : ∀ loc, index loc ≤ length loc
  before : ∀ loc, 0 < index loc → m.hb (m.write loc (index loc)) sample
  latest : ∀ loc j, j ≤ length loc → m.hb (m.write loc j) sample → j ≤ index loc

def lastBefore (p : Nat → Bool) : Nat → Nat
  | 0 => 0
  | n+1 => if p (n+1) then n+1 else lastBefore p n

theorem lastBefore_spec (p : Nat → Bool) (n : Nat) :
    lastBefore p n ≤ n ∧ (0 < lastBefore p n → p (lastBefore p n) = true) ∧
      ∀ i ≤ n, p i = true → i ≤ lastBefore p n := by
  induction n with
  | zero => simp [lastBefore]
  | succ n ih =>
    cases seen : p (n+1) with
    | true =>
      simp [lastBefore,seen]
      exact fun _ le _ => le
    | false =>
      simp only [lastBefore,seen,Bool.false_eq_true,ite_false]
      refine ⟨by omega,ih.2.1,?_⟩
      intro i le yes
      apply ih.2.2 _ _ yes
      by_cases eq : i = n+1
      · simp [eq,seen] at yes
      · omega

/-- Compute the last modification before the copy separately for each word.
    Unordered and future events are absent from that prefix. -/
noncomputable def Visible.atCopy (m : Memory Nat) (length : Location → Nat) (sample : Nat) :
    Visible m length sample := by
  classical
  let precedes := fun loc i => decide (m.hb (m.write loc i) sample)
  exact ⟨fun loc => lastBefore (precedes loc) (length loc),
    fun loc => (lastBefore_spec _ _).1,
    fun loc h => of_decide_eq_true ((lastBefore_spec (precedes loc) _).2.1 h),
    fun loc i bound earlier => (lastBefore_spec _ _).2.2 i bound (decide_eq_true earlier)⟩

theorem Visible.index_eq (v w : Visible m length sample) (loc : Location) : v.index loc = w.index loc := by
  have le (a b : Visible m length sample) : a.index loc ≤ b.index loc := by
    by_cases zero : a.index loc = 0
    · omega
    · exact b.latest _ _ (a.bounded loc) (a.before loc (by omega))
  exact Nat.le_antisymm (le v w) (le w v)

abbrev View {m : Memory Nat} (slot : Slot m) (sample : Nat) :=
  Visible m (modifications slot) sample

def Visible.word (v : Visible m length sample) (loc : Location) :=
  m.writeValue loc (v.index loc)

private theorem sequenceWord {m : Memory Nat} {slot : Slot m} (i : Nat) (inside : i ≤ slot.history.length) :
    m.writeValue .sequence i = SeqLock.seqWord (2*slot.toSequence.baseEpoch + i) := by
  cases i with
  | zero => exact slot.toSequence.initialWord
  | succ n =>
    have atIndex := List.getElem?_eq_getElem (show n < slot.history.length from inside)
    exact (slot.modified _ _ atIndex).2.trans (slot.toSequence.order.wordAt atIndex)

/-- The publication number is its position at the copy, independent of sequence-word wrap. -/
theorem View.evenIndex {m : Memory Nat} {slot : Slot m} (v : View slot sample)
    (stable : v.word .sequence &&& 1 = 0) : v.index .sequence = 2*(v.index .sequence / 2) := by
  unfold Visible.word at stable
  rw [sequenceWord _ (v.bounded .sequence),SeqLock.seqWord_even_iff] at stable
  omega

/-- A copy taken between whole writer invocations sees an even sequence. The
    source is allowed to receive more writes after the copy returns. -/
theorem View.evenOfQuiescent {m : Memory Nat} {slot : Slot m} (v : View slot sample)
    (completed : ∀ call, WriterTrace.Action.claim call ∈ slot.history → m.hb call.cas sample →
      WriterTrace.Action.publish call ∈ slot.history ∧ m.hb call.finish sample) :
    v.word .sequence &&& 1 = 0 := by
  let n := v.index .sequence
  have bounded : n ≤ slot.history.length := v.bounded .sequence
  have parity : n % 2 = 0 := by
    by_cases even : n % 2 = 0
    · exact even
    · let k := n/2
      have odd : n = 2*k+1 := by dsimp [k]; omega
      obtain ⟨call,atClaim,_⟩ := slot.toSequence.order.callAt (k+1) (by omega) (by omega)
      rw [show 2*(k+1)-2 = 2*k by omega] at atClaim
      have cas := (slot.modified _ _ atClaim).1
      change m.write .sequence (2*k+1) = call.cas at cas
      have before := v.before .sequence (show 0 < n by omega)
      change m.hb (m.write .sequence n) sample at before
      rw [odd,cas] at before
      obtain ⟨published,prior⟩ := completed call (List.mem_of_getElem? atClaim) before
      obtain ⟨j,atFinish⟩ := List.mem_iff_getElem?.mp published
      have release := (slot.modified _ _ atFinish).1
      change m.write .sequence (j+1) = call.finish at release
      have later := m.sb_hb (slot.publish _ _ atFinish).2.1
      rw [← cas,← release] at later
      have lt := m.write_write later
      have last := v.latest .sequence _ (List.getElem?_eq_some_iff.mp atFinish).1 (release ▸ prior)
      change _ ≤ n at last
      omega
  unfold Visible.word
  rw [sequenceWord _ bounded,SeqLock.seqWord_even_iff]
  change (2*_ + n) % 2 = 0
  omega

/-- The latest visible write to each payload field has the same rank as the
    published epoch, even when the entire slot history extends beyond the copy. -/
theorem View.fieldIndex {m : Memory Nat} {slot : Slot m} (v : View slot sample)
    (epoch : Nat) (even : v.index .sequence = 2*epoch) (f : Field) :
    v.index (.payload f) = epoch := by
  have bounded : v.index .sequence ≤ slot.history.length := v.bounded .sequence
  have le : v.index (.payload f) ≤ epoch := by
    cases value : v.index (.payload f) with
    | zero => omega
    | succ j =>
      have wordBefore := v.before (.payload f) (by omega)
      rw [value] at wordBefore
      have inside : j < (slot.fields f).history.length := by
        have bound := v.bounded (.payload f)
        rwa [value] at bound
      have stored := List.getElem?_eq_getElem inside
      have claimed := (slot.fields f).aligned stored
      have cas := (slot.modified _ _ claimed).1
      obtain ⟨write,_,_,_,casFence,fenceStore⟩ := (slot.fields f).store _ _ stored
      rw [write] at wordBefore
      have earlier := m.hb_trans (m.sb_hb (m.sb_trans casFence fenceStore)) wordBefore
      have latest := v.latest .sequence (2*j+1) (List.getElem?_eq_some_iff.mp claimed).1 (cas ▸ earlier)
      omega
  have ge : epoch ≤ v.index (.payload f) := by
    by_cases zero : epoch = 0
    · omega
    · obtain ⟨call,published,_⟩ := slot.publishedPayload epoch (by omega) (by omega)
      have field := (slot.fields f).atPublication (by omega) published
      have valid := (List.getElem?_eq_some_iff.mp field).1
      obtain ⟨write,_,_,_,_,_⟩ := (slot.fields f).store _ _ field
      rw [show epoch-1+1 = epoch by omega] at write
      have release := (slot.modified _ _ published).1
      rw [show 2*epoch-1+1 = 2*epoch by omega] at release
      have hb := v.before .sequence (by omega)
      rw [even,release] at hb
      have before := ((slot.fields f).finish call (List.mem_of_getElem? published)).2
      exact v.latest (.payload f) epoch (show epoch ≤ (slot.fields f).history.length by omega)
        (write ▸ m.hb_trans (m.sb_hb before) hb)
  omega

end NeuralReversi.TranspositionTable.Duplicate
