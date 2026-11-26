import NeuralReversi.PvsParallel.RootHeaderMir.Program
import NeuralReversi.PvsMoveLoop
import NeuralReversi.PvsPostpone

namespace NeuralReversi.PvsParallel

open PvsMoveLoop

def canHandoff (idx count depth minSplit : Nat) (canSplit : Bool) : Prop :=
  idx ≠ 0 ∧ minSplit ≤ depth ∧ 2 ≤ count - idx ∧ canSplit = true

instance : Decidable (canHandoff idx count depth minSplit free) :=
  inferInstanceAs (Decidable (_ ∧ _ ∧ _ ∧ _))

theorem headerAction_eq (idx count fresh depth minSplit : Nat) (canSplit busy : Bool)
    (alpha scout pv : Int) :
    headerAction idx count fresh depth minSplit canSplit busy alpha scout pv =
      if idx < count then
        if canHandoff idx count depth minSplit canSplit then .split
        else if PvsPostpone.shouldPostpone busy idx fresh then .postpone idx (fresh - 1)
        else if idx = 0 then .child (idx + 1) false true (-pv)
        else if alpha < -scout then .child (idx + 1) true true (-pv)
        else .child (idx + 1) true false (-scout)
      else .finish := by
  by_cases running : idx < count
  · by_cases first : idx = 0
    · simp [headerAction, canHandoff, PvsPostpone.shouldPostpone, first]
    · by_cases deep : minSplit ≤ depth <;> by_cases several : 2 ≤ count - idx <;>
        cases canSplit <;> cases busy <;> simp [headerAction, canHandoff, PvsPostpone.shouldPostpone, *]
  · simp [headerAction, running]

theorem header_child (search : ChildSearch Move) (done : ChildRelation Move)
    (idx count fresh depth minSplit : Nat) (free busy : Bool) (move : Move) (alpha beta : Int)
    (scoutUsed pvUsed : Bool) (score : Int)
    (action : headerAction idx count fresh depth minSplit free busy alpha
      (search move false (-(alpha + 1)) (-alpha)) (search move true (-beta) (-alpha)) =
      .child next scoutUsed pvUsed score)
    (scout : scoutUsed = true → done move false (-(alpha + 1)) (-alpha)
      (search move false (-(alpha + 1)) (-alpha)))
    (pv : pvUsed = true → done move true (-beta) (-alpha) (search move true (-beta) (-alpha))) :
    next = idx + 1 ∧ MoveCalls done search true (decide (idx = 0)) move alpha beta ∧
    score = moveScore true (decide (idx = 0)) search move alpha beta := by
  rw [headerAction_eq] at action
  split at action <;> try contradiction
  split at action <;> try contradiction
  split at action <;> try contradiction
  split at action
  · rename_i first
    cases action
    simpa [MoveCalls, moveScore, first] using pv rfl
  · rename_i later
    split at action
    · rename_i improved
      cases action
      simpa [MoveCalls, moveScore, later, improved] using And.intro (scout rfl) (pv rfl)
    · rename_i low
      cases action
      simpa [MoveCalls, moveScore, later, low] using scout rfl

theorem header_postpone (q : PvsPostpone.Queue Move) (depth minSplit : Nat)
    (free busy : Bool) (alpha scout pv : Int)
    (decision : headerAction q.idx (q.idx + q.pending.length) (q.idx + q.fresh.length)
      depth minSplit free busy alpha scout pv = .postpone next fresh) :
    ∃ after, PvsPostpone.step busy q = some (none, after) ∧ after.idx = next ∧
      after.idx + after.fresh.length = fresh ∧ after.budget < q.budget := by
  rw [headerAction_eq] at decision
  split at decision <;> try contradiction
  split at decision <;> try contradiction
  split at decision
  · rename_i postpone
    cases decision
    rcases q with ⟨idx, remaining, deferred⟩
    cases remaining with
    | nil =>
      simp [PvsPostpone.shouldPostpone] at postpone
      omega
    | cons move tail =>
      refine ⟨⟨idx, tail, deferred ++ [move]⟩, ?_, rfl, ?_, ?_⟩
      · simpa [PvsPostpone.step] using postpone
      · simp
      · simp [PvsPostpone.Queue.budget, PvsPostpone.Queue.pending]
        omega
  · split at decision <;> try contradiction
    split at decision <;> contradiction

theorem header_take (q : PvsPostpone.Queue Move) (depth minSplit : Nat)
    (free busy : Bool) (alpha scout pv : Int)
    (decision : headerAction q.idx (q.idx + q.pending.length) (q.idx + q.fresh.length)
      depth minSplit free busy alpha scout pv = .child next scoutUsed pvUsed value) :
    ∃ move after, PvsPostpone.step busy q = some (some move, after) ∧
      q.pending = move :: after.pending ∧ after.idx = next ∧ after.budget < q.budget := by
  rw [headerAction_eq] at decision
  split at decision <;> try contradiction
  rename_i running
  split at decision <;> try contradiction
  split at decision <;> try contradiction
  rename_i take
  have advance : next = q.idx + 1 := by
    split at decision
    · cases decision
      rfl
    · split at decision <;> cases decision <;> rfl
  subst next
  rcases q with ⟨idx, fresh, later⟩
  cases fresh with
  | nil =>
    cases later with
    | nil => simp [PvsPostpone.Queue.pending] at running
    | cons head tail =>
      refine ⟨head, ⟨idx+1, [], tail⟩, rfl, rfl, rfl, ?_⟩
      simp [PvsPostpone.Queue.budget, PvsPostpone.Queue.pending]
  | cons head tail =>
    refine ⟨head, ⟨idx+1, tail, later⟩, ?_, rfl, rfl, ?_⟩
    · simpa [PvsPostpone.step] using take
    · simp [PvsPostpone.Queue.budget, PvsPostpone.Queue.pending]
      omega

theorem header_finish_iff (q : PvsPostpone.Queue Move) (depth minSplit : Nat)
    (free busy : Bool) (alpha scout pv : Int) :
    headerAction q.idx (q.idx + q.pending.length) (q.idx + q.fresh.length)
      depth minSplit free busy alpha scout pv = .finish ↔ q.pending = [] := by
  rw [headerAction_eq]
  by_cases empty : q.pending = []
  · simp [empty]
  · have inside : q.idx < q.idx + q.pending.length :=
      Nat.lt_add_of_pos_right (List.length_pos_iff.mpr empty)
    simp only [ite_eq_left inside]
    grind

theorem header_split_iff (idx count fresh depth minSplit : Nat)
    (free busy : Bool) (alpha scout pv : Int) :
    headerAction idx count fresh depth minSplit free busy alpha scout pv = .split ↔
      idx < count ∧ canHandoff idx count depth minSplit free := by
  rw [headerAction_eq]
  grind

theorem header_arithmetic (idx count fresh depth : Nat) (a b : Int)
    (inside : idx < count) (capacity : count ≤ 64) (positive : 0 < depth)
    (window : Aspiration.Window a b) :
    idx + 1 ≤ 64 ∧ (count - idx) + idx = count ∧ (depth - 1) + 1 = depth ∧
    (idx + 1 < fresh → (fresh - 1) + 1 = fresh) ∧ -32767 ≤ a + 1 ∧ a + 1 ≤ 32767 := by
  have := window.lower
  have := window.ordered
  have := window.upper
  omega

theorem header_normalize (idx count raw depth minSplit : Nat) (free busy : Bool) (alpha scout pv : Int) :
    headerAction idx count raw depth minSplit free busy alpha scout pv =
      headerAction idx count (idx + (raw - idx)) depth minSplit free busy alpha scout pv := by
  by_cases stale : raw < idx
  · have normalized : idx + (raw - idx) = idx := by omega
    have noRaw : PvsPostpone.shouldPostpone busy idx raw = false := by
      simp [PvsPostpone.shouldPostpone, show ¬idx+1 < raw by omega]
    have noIdx : PvsPostpone.shouldPostpone busy idx idx = false := by
      simp [PvsPostpone.shouldPostpone]
    rw [normalized, headerAction_eq, headerAction_eq]
    simp [noRaw, noIdx]
  · rw [show idx + (raw - idx) = raw by omega]

theorem cursor_step (q after : PvsPostpone.Queue Move) (raw : Nat)
    (related : raw - q.idx = q.fresh.length)
    (advance : PvsPostpone.step busy q = some (taken, after)) :
    (if taken.isSome then raw else raw - 1) - after.idx = after.fresh.length := by
  rcases q with ⟨idx, fresh, postponed⟩
  cases fresh with
  | nil =>
    cases postponed with
    | nil => simp [PvsPostpone.step] at advance
    | cons head tail =>
      simp only [PvsPostpone.step, Option.some.injEq, Prod.mk.injEq] at advance
      rcases advance with ⟨rfl, rfl⟩
      simp at related ⊢
      omega
  | cons head tail =>
    simp at related
    cases delayed : PvsPostpone.shouldPostpone busy idx (idx + (tail.length + 1)) <;>
      simp [PvsPostpone.step, delayed] at advance <;>
      rcases advance with ⟨rfl, rfl⟩ <;> simp <;> omega

end NeuralReversi.PvsParallel

