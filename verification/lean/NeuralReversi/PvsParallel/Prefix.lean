module
public import NeuralReversi.Threading.SplitWindow
public import NeuralReversi.PvsPostpone

@[expose] public section

namespace NeuralReversi.PvsParallel.Prefix

open Aspiration (SoundResult Window)
open PvsMoveLoop (Bound ChildSearch ChildRelation ScanCalls bestValue moveScore scan scoreInf scoreMax)
open Threading.SplitWindow (State choosesMove marksCutoff publish raisesAlpha stateContract_proven)

theorem nextScanState_eq (pv : Bool) (beta : Int) (state : State) (score : Int) :
    PvsMoveLoop.nextScanState pv beta state.alpha state.best score =
      if marksCutoff pv beta state score then none
      else some ((publish pv beta state score).alpha, (publish pv beta state score).best) := by
  by_cases improved : state.best < score
  · by_cases raises : state.alpha < score
    · cases pv
      · simp [PvsMoveLoop.nextScanState, marksCutoff, improved, raises]
      · by_cases below : score < beta
        · by_cases maximum : score < scoreMax
          · simp [PvsMoveLoop.nextScanState, marksCutoff, publish, raisesAlpha, improved, raises, below,
              maximum, Int.not_le.mpr below, Int.not_le.mpr maximum, Int.max_eq_right (Int.le_of_lt improved)]
          · simp [PvsMoveLoop.nextScanState, marksCutoff, improved, raises, maximum, Int.le_of_not_gt maximum]
        · simp [PvsMoveLoop.nextScanState, marksCutoff, improved, raises, below, Int.le_of_not_gt below]
    · simp [PvsMoveLoop.nextScanState, marksCutoff, publish, raisesAlpha, improved, raises,
        Int.max_eq_right (Int.le_of_lt improved)]
  · simp [PvsMoveLoop.nextScanState, marksCutoff, publish, raisesAlpha, improved,
      Int.max_eq_left (Int.le_of_not_gt improved)]

def run (pv : Bool) (search : ChildSearch Move) (beta : Int) :
    List Move → Bool → State → Option State
  | [], _, state => some state
  | move :: rest, first, state =>
    let score := moveScore pv first search move state.alpha beta
    if marksCutoff pv beta state score then none
    else run pv search beta rest false (publish pv beta state score)

theorem scan_step (pv : Bool) (search : ChildSearch Move) (beta : Int)
    (move : Move) (rest : List Move) (first : Bool) (state : State) :
    let score := moveScore pv first search move state.alpha beta
    let next := publish pv beta state score
    scan pv search beta (move :: rest) first state.alpha state.best =
      if marksCutoff pv beta state score then score else scan pv search beta rest false next.alpha next.best := by
  let score := moveScore pv first search move state.alpha beta
  change scan pv search beta (move :: rest) first state.alpha state.best =
    if marksCutoff pv beta state score then score
    else scan pv search beta rest false (publish pv beta state score).alpha (publish pv beta state score).best
  rw [PvsMoveLoop.scan_step, nextScanState_eq pv beta state]
  change (match (if marksCutoff pv beta state score then none else
      some ((publish pv beta state score).alpha, (publish pv beta state score).best)) with
    | none => score
    | some (a, s) => scan pv search beta rest false a s) = _
  by_cases cut : marksCutoff pv beta state score = true <;> simp [cut]

theorem continued_scan (pv : Bool) (search : ChildSearch Move) (beta : Int)
    (moves : List Move) (first : Bool) (initial final : State)
    (continued : run pv search beta moves first initial = some final) :
    scan pv search beta moves first initial.alpha initial.best = final.best := by
  induction moves generalizing first initial with
  | nil =>
    have equal := Option.some.inj continued
    exact congrArg State.best equal
  | cons move rest ih =>
    let score := moveScore pv first search move initial.alpha beta
    cases cut : marksCutoff pv beta initial score
    · rw [scan_step]
      change marksCutoff pv beta initial (moveScore pv first search move initial.alpha beta) = false at cut
      simp only [cut, Bool.false_eq_true, ite_false]
      exact ih false _ (by simpa [run, score, cut] using continued)
    · simp [run, score, cut] at continued

theorem continued_window (pv : Bool) (search : ChildSearch Move) (original beta : Int)
    (moves : List Move) (first : Bool) (initial final : State)
    (window : Window initial.alpha beta) (cap : initial.alpha ≤ max original initial.best)
    (continued : run pv search beta moves first initial = some final) :
    Window final.alpha beta ∧ final.alpha ≤ max original final.best ∧
      (pv = false → final.alpha = initial.alpha) := by
  induction moves generalizing first initial with
  | nil => cases Option.some.inj continued; exact ⟨window, cap, fun _ => rfl⟩
  | cons move rest ih =>
    let score := moveScore pv first search move initial.alpha beta
    cases cut : marksCutoff pv beta initial score
    · let next := publish pv beta initial score
      have state : Window next.alpha beta ∧ next.alpha ≤ max original next.best ∧
          (pv = false → next.alpha = initial.alpha) :=
        (stateContract_proven pv original beta initial [score] window cap).2
      obtain ⟨valid, capped, fixed⟩ := ih false next state.1 state.2.1
        (by simpa [run, score, cut, next] using continued)
      exact ⟨valid, capped, fun noPv => (fixed noPv).trans (state.2.2 noPv)⟩
    · simp [run, score, cut] at continued

def SelectedContract : Prop :=
  ∀ (Move : Type) (pv : Bool) (search : ChildSearch Move) (value : Move → Int) (moves : List Move)
    (completed : ChildRelation Move) (original beta : Int) (final : State),
    Window original beta → (pv = false → beta = original + 1) →
    run pv search beta moves true ⟨original, -scoreInf⟩ = some final →
    (∀ move ∈ moves, value move ≤ scoreMax) →
    ScanCalls completed search pv beta moves true original (-scoreInf) →
    (∀ move ∈ moves, ∀ childPv a b s, Window a b → (childPv = false → b = a + 1) →
      completed move childPv a b s → SoundResult (-(value move)) a b s) →
    Window final.alpha beta ∧ final.alpha ≤ max original final.best ∧
      (pv = false → beta = final.alpha + 1) ∧
      Bound (bestValue value moves (-scoreInf)) original beta final.best

theorem selectedContract_proven : SelectedContract := by
  intro Move pv search value moves completed original beta final window null continued range calls children
  obtain ⟨valid, cap, fixed⟩ := continued_window pv search original beta moves true _
    final window (Int.le_max_left _ _) continued
  refine ⟨valid, cap, fun noPv => by rw [fixed noPv]; exact null noPv, ?_⟩
  have same := continued_scan pv search beta moves true _ final continued
  cases moves with
  | nil =>
    change -scoreInf = final.best at same
    rw [← same]
    exact ⟨fun _ => Int.le_refl _, fun _ => Int.le_refl _⟩
  | cons head tail =>
    rw [← same]
    exact Bound.ofSoundResult
      (PvsMoveLoop.selectedContract_proven Move pv search value completed (head :: tail) original beta
        (by simp) window null range calls children)

namespace Serial

open PvsPostpone (Queue step step_idx stepContract_proven)
open PvsMoveLoop (MoveCalls nextScanState)

inductive Outcome where
  | returned (score : Int)
  | handoff (state : State)

inductive Run (completed : ChildRelation Move) (search : ChildSearch Move) (pv : Bool) (beta : Int) :
    Queue Move → State → List Move → Queue Move → Outcome → Prop
  | exhausted : q.pending = [] → Run completed search pv beta q state [] q (.returned state.best)
  | handoff : Run completed search pv beta q state [] q (.handoff state)
  | postpone : step true q = some (none, next) →
      Run completed search pv beta next state visited remaining outcome →
      Run completed search pv beta q state visited remaining outcome
  | cutoff :
      step busy q = some (some move, next) →
      MoveCalls completed search pv (q.idx == 0) move state.alpha beta →
      marksCutoff pv beta state (moveScore pv (q.idx == 0) search move state.alpha beta) = true →
      Run completed search pv beta q state [move] next
        (.returned (moveScore pv (q.idx == 0) search move state.alpha beta))
  | searched :
      step busy q = some (some move, next) →
      MoveCalls completed search pv (q.idx == 0) move state.alpha beta →
      marksCutoff pv beta state (moveScore pv (q.idx == 0) search move state.alpha beta) = false →
      Run completed search pv beta next (publish pv beta state
        (moveScore pv (q.idx == 0) search move state.alpha beta)) visited remaining outcome →
      Run completed search pv beta q state (move :: visited) remaining outcome

def bestMove (pv : Bool) (search : ChildSearch Move) (beta : Int) :
    List Move → Bool → State → Option Move → Option Move
  | [], _, _, previous => previous
  | move :: rest, first, state, previous =>
    let score := moveScore pv first search move state.alpha beta
    let chosen := if choosesMove state score then some move else previous
    bestMove pv search beta rest false (publish pv beta state score) chosen

private theorem bestMove_member (found : bestMove pv search beta moves first state previous = some chosen) :
    previous = some chosen ∨ chosen ∈ moves := by
  induction moves generalizing first state previous with
  | nil => exact Or.inl found
  | cons move rest ih =>
    rw [bestMove] at found
    rcases ih found with kept | later
    · split at kept
      · exact Or.inr (List.mem_cons.mpr (Or.inl (Option.some.inj kept).symm))
      · exact Or.inl kept
    · exact Or.inr (List.mem_cons_of_mem _ later)

theorem Run.presearchContinues
    (move : Move) (rest : List Move) (narrow : beta = alpha + 1)
    (calls : MoveCalls completed search false true move alpha beta)
    (aboveSentinel : -scoreInf < moveScore false true search move alpha beta)
    (belowBeta : moveScore false true search move alpha beta < beta)
    (suffix : Run completed search false beta ⟨1, rest, []⟩
      ⟨alpha, moveScore false true search move alpha beta⟩ visited remaining outcome) :
    Run completed search false beta ⟨0, move :: rest, []⟩ ⟨alpha, -scoreInf⟩
      (move :: visited) remaining outcome := by
  have atMostAlpha : moveScore false true search move alpha beta ≤ alpha := by omega
  refine .searched (busy := false) rfl calls ?_ ?_
  · simpa [marksCutoff, aboveSentinel] using atMostAlpha
  · simpa [publish, raisesAlpha, Int.max_eq_right (Int.le_of_lt aboveSentinel)] using suffix

theorem Run.presearchCutoff
    (move : Move) (rest : List Move) (narrow : beta = alpha + 1)
    (calls : MoveCalls completed search false true move alpha beta)
    (aboveSentinel : -scoreInf < moveScore false true search move alpha beta)
    (reachedBeta : beta ≤ moveScore false true search move alpha beta) :
    Run completed search false beta ⟨0, move :: rest, []⟩ ⟨alpha, -scoreInf⟩
      [move] ⟨1, rest, []⟩ (.returned (moveScore false true search move alpha beta)) := by
  have aboveAlpha : alpha < moveScore false true search move alpha beta := by omega
  exact .cutoff (busy := false) rfl calls (by simp [marksCutoff, aboveSentinel, aboveAlpha])

def Reproduced (completed : ChildRelation Move) (search : ChildSearch Move) (pv : Bool) (beta : Int)
    (q : Queue Move) (state : State) (visited : List Move) (remaining : Queue Move) : Outcome → Prop
  | .returned score =>
    scan pv search beta (visited ++ remaining.pending) (q.idx == 0) state.alpha state.best = score ∧
    ScanCalls completed search pv beta (visited ++ remaining.pending) (q.idx == 0) state.alpha state.best
  | .handoff entry =>
    run pv search beta visited (q.idx == 0) state = some entry ∧
    ScanCalls completed search pv beta visited (q.idx == 0) state.alpha state.best

theorem Run.reproduced (trace : Run completed search pv beta q state visited remaining outcome) :
    (visited ++ remaining.pending).Perm q.pending ∧
    remaining.idx = q.idx + visited.length ∧
    Reproduced completed search pv beta q state visited remaining outcome := by
  induction trace with
  | exhausted empty =>
    simp [Reproduced, empty, scan, ScanCalls]
  | handoff =>
    exact ⟨by simp, by simp, rfl, trivial⟩
  | postpone advance _ ih =>
    have index := step_idx advance
    have pending := ((stepContract_proven _).2.2 _ _ _ _ advance).1
    simp only [Option.toList_none, List.nil_append] at pending
    obtain ⟨order, atEnd, reproduced⟩ := ih
    change _ = _ + 0 at index
    rw [Nat.add_zero] at index
    refine ⟨order.trans pending, atEnd.trans (congrArg (· + _) index), ?_⟩
    cases outcome <;> simpa only [Reproduced, index] using reproduced
  | @cutoff busy q move next state advance calls stops =>
    let score := moveScore pv (q.idx == 0) search move state.alpha beta
    have pending := ((stepContract_proven _).2.2 _ _ _ _ advance).1
    refine ⟨pending, step_idx advance, ?_, ⟨calls, ?_⟩⟩
    · change scan pv search beta (move :: next.pending) (q.idx == 0) state.alpha state.best = score
      rw [scan_step]
      simp only [stops, ↓reduceIte, score]
    · change (match nextScanState pv beta state.alpha state.best score with
        | none => True
        | some (a, b) => ScanCalls completed search pv beta next.pending false a b)
      rw [nextScanState_eq pv beta state, stops]
      trivial
  | @searched busy q move next state visited remaining outcome advance calls goes _ ih =>
    let score := moveScore pv (q.idx == 0) search move state.alpha beta
    let nextState := publish pv beta state score
    have pending := ((stepContract_proven _).2.2 _ _ _ _ advance).1
    obtain ⟨order, atEnd, reproduced⟩ := ih
    have index : next.idx = q.idx + 1 := step_idx advance
    have notFirst : (next.idx == 0) = false := by simp [index]
    refine ⟨(order.cons move).trans pending, by simp [atEnd, index, Nat.add_assoc, Nat.add_comm 1], ?_⟩
    cases outcome with
    | returned result =>
      obtain ⟨sameScore, nextCalls⟩ := reproduced
      refine ⟨?_, ⟨calls, ?_⟩⟩
      · change scan pv search beta (move :: (visited ++ remaining.pending))
          (q.idx == 0) state.alpha state.best = result
        rw [scan_step]
        simpa only [goes, Bool.false_eq_true, ite_false, notFirst, nextState] using sameScore
      · change (match nextScanState pv beta state.alpha state.best score with
          | none => True
          | some (a, b) => ScanCalls completed search pv beta (visited ++ remaining.pending) false a b)
        rw [nextScanState_eq pv beta state, goes]
        simpa only [Bool.false_eq_true, ite_false, notFirst, nextState] using nextCalls
    | handoff entry =>
      obtain ⟨sameState, nextCalls⟩ := reproduced
      refine ⟨?_, ⟨calls, ?_⟩⟩
      · change run pv search beta (move :: visited) (q.idx == 0) state = some entry
        simpa only [run, goes, Bool.false_eq_true, ite_false, notFirst, nextState] using sameState
      · change (match nextScanState pv beta state.alpha state.best score with
          | none => True
          | some (a, b) => ScanCalls completed search pv beta visited false a b)
        rw [nextScanState_eq pv beta state, goes]
        simpa only [Bool.false_eq_true, ite_false, notFirst, nextState] using nextCalls

theorem Run.bestMoveMember (trace : Run completed search pv beta q state visited remaining outcome)
    (found : bestMove pv search beta visited (q.idx == 0) state previous = some chosen) :
    previous = some chosen ∨ chosen ∈ q.pending := by
  rcases bestMove_member found with kept | member
  · exact Or.inl kept
  · exact Or.inr (trace.reproduced.1.mem_iff.mp (List.mem_append_left _ member))

private theorem reindex {source : ChildRelation Move} {target : ChildRelation Index}
    (label : Index → Move) (search : ChildSearch Move) (pv : Bool) (beta : Int)
    (indices : List Index) (first : Bool) (state : State)
    (lift : ∀ i ∈ indices, ∀ p a b s, source (label i) p a b s → target i p a b s) :
    run pv (fun i => search (label i)) beta indices first state =
        run pv search beta (indices.map label) first state ∧
      (ScanCalls source search pv beta (indices.map label) first state.alpha state.best →
        ScanCalls target (fun i => search (label i)) pv beta indices first state.alpha state.best) := by
  induction indices generalizing first state with
  | nil => exact ⟨rfl, id⟩
  | cons i tail ih =>
    let score := moveScore pv first search (label i) state.alpha beta
    have tailLift : ∀ j ∈ tail, ∀ p a b s, source (label j) p a b s → target j p a b s :=
      fun j mem => lift j (List.mem_cons_of_mem i mem)
    have rest := ih false (publish pv beta state score) tailLift
    refine ⟨?_, fun calls => ⟨?_, ?_⟩⟩
    · change (if marksCutoff pv beta state score then none else
        run pv (fun j => search (label j)) beta tail false (publish pv beta state score)) =
      (if marksCutoff pv beta state score then none else
        run pv search beta (tail.map label) false (publish pv beta state score))
      rw [rest.1]
    · have head : MoveCalls source search pv first (label i) state.alpha beta := calls.1
      have liftHead := lift i (by simp)
      unfold MoveCalls at head ⊢
      cases firstWide : pv && first
      · simp only [firstWide, Bool.false_eq_true, ite_false] at head ⊢
        exact ⟨liftHead _ _ _ _ head.1, fun retry => liftHead _ _ _ _ (head.2 retry)⟩
      · simp only [firstWide, ↓reduceIte] at head ⊢
        exact liftHead _ _ _ _ head
    · have after := calls.2
      change (match nextScanState pv beta state.alpha state.best score with
        | none => True
        | some (a, s) => ScanCalls target (fun j => search (label j)) pv beta tail false a s)
      change (match nextScanState pv beta state.alpha state.best score with
        | none => True
        | some (a, s) => ScanCalls source search pv beta (tail.map label) false a s) at after
      rw [nextScanState_eq] at after ⊢
      cases stopped : marksCutoff pv beta state score
      · simp only [stopped, Bool.false_eq_true, ite_false] at after ⊢
        exact rest.2 after
      · trivial

private theorem Run.handoffIndexed {source : ChildRelation Move} {target : ChildRelation Nat}
    (trace : Run source search pv beta q state visited remaining (.handoff entry))
    (first : q.idx = 0) (label : Nat → Move)
    (orderedPrefix : (List.range remaining.idx).map label = visited)
    (lift : ∀ i < remaining.idx, ∀ p a b s, source (label i) p a b s → target i p a b s) :
    run pv (fun i => search (label i)) beta (List.range remaining.idx) true state = some entry ∧
      ScanCalls target (fun i => search (label i)) pv beta
        (List.range remaining.idx) true state.alpha state.best := by
  have actual : run pv search beta visited true state = some entry ∧
      ScanCalls source search pv beta visited true state.alpha state.best := by
    simpa [Reproduced, first] using trace.reproduced.2.2
  have indexed := reindex label search pv beta (List.range remaining.idx) true state
    (fun i mem => lift i (List.mem_range.mp mem))
  rw [orderedPrefix] at indexed
  exact ⟨indexed.1.trans actual.1, indexed.2 actual.2⟩

theorem Run.handoffOrdered {source : ChildRelation Move} {target : ChildRelation Nat}
    (trace : Run source search pv beta q state visited remaining (.handoff entry))
    (first : q.idx = 0) (labels : List Move) (fallback : Move)
    (order : labels = visited ++ remaining.pending)
    (lift : ∀ i < labels.length, ∀ p a b s,
      source (labels[i]?.getD fallback) p a b s → target i p a b s) :
    let observe : ChildSearch Nat := fun i => search (labels[i]?.getD fallback)
    run pv observe beta (List.range remaining.idx) true state = some entry ∧
      ScanCalls target observe pv beta (List.range remaining.idx) true state.alpha state.best := by
  have count : remaining.idx = visited.length := by simpa [first] using trace.reproduced.2.1
  have bounded : remaining.idx ≤ labels.length := by simp [order, count]
  have orderedPrefix : (List.range remaining.idx).map (fun i => labels[i]?.getD fallback) = visited := by
    apply List.ext_getElem
    · simp [count]
    · intro i _ inside
      have inAll : i < labels.length := by omega
      simp [List.getElem?_eq_getElem inAll]
      subst labels
      exact List.getElem_append_left inside
  exact trace.handoffIndexed first _ orderedPrefix (fun i inside => lift i (by omega))

end Serial

example : (run true (fun (move : Int) _ _ _ => -move) 20 [4, 1] true
    ⟨0, -scoreInf⟩).map State.alpha = some 4 := rfl

example : run true (fun (move : Int) _ _ _ => -move) 6 [7] true ⟨0, -scoreInf⟩ = none := rfl

example : run false (fun (_ : Nat) _ _ _ => 0) 1 [] true ⟨0, -scoreInf⟩ =
    some ⟨0, -scoreInf⟩ := rfl

end NeuralReversi.PvsParallel.Prefix
