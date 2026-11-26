import NeuralReversi.PvsParallel.GeneralOwnerMir.Program
import NeuralReversi.PvsParallel.GeneralNonPvMir

namespace NeuralReversi.PvsParallel.GeneralOwnerMir

open PvsPostpone (Queue step shouldPostpone)
open PvsMoveLoop (ChildRelation ChildSearch MoveCalls moveScore)
open Threading.SplitWindow (State marksCutoff publish)
open Prefix.Serial (Run Outcome)

structure Schedule (Move : Type) where
  available : Nat → Queue Move → Bool
  busy : Nat → Queue Move → Bool

structure Receipt (Move : Type) where
  visited : List Move
  remaining : Queue Move
  outcome : Outcome
  bestMove : Option Move

def queueHeader (q : Queue Move) (depth : Nat) (cut available busy : Bool) (scout : Int) : HeaderAction :=
  if q.pending.isEmpty then .finish else
    if canSplit q.idx (q.idx + q.pending.length) depth cut available then .split else
      match step busy q with
      | none => .finish
      | some (none, next) => .postpone next.idx (next.idx + next.fresh.length)
      | some (some _, next) => .child next.idx true false (-scout)

theorem physicalHeader (q : Queue Move) (depth : Nat) (cut available busy : Bool) (scout : Int) :
    header q.idx (q.idx + q.pending.length) (q.idx + q.fresh.length) depth cut available busy scout =
      queueHeader q depth cut available busy scout := by
  rcases q with ⟨idx, fresh, postponed⟩
  cases fresh with
  | nil =>
    cases postponed with
    | nil => simp [header, queueHeader, Queue.pending]
    | cons m rest =>
      simp [header, queueHeader, Queue.pending, step, show ¬ idx + 1 < idx by omega]
      rfl
  | cons m rest =>
    simp [header, queueHeader, Queue.pending, step, shouldPostpone]
    split
    · rfl
    · split <;> rfl

theorem physicalUpdate (state : State) (score beta : Int) :
    (after state.alpha state.best score).1 = (publish false beta state score).best ∧
    (after state.alpha state.best score).2.1 = decide (Threading.SplitWindow.choosesMove state score) ∧
    (after state.alpha state.best score).2.2 = marksCutoff false beta state score ∧
    (publish false beta state score).alpha = state.alpha := by
  by_cases raises : state.best < score <;> by_cases takes : state.alpha < score <;>
    simp [after, raises, takes, publish, Threading.SplitWindow.raisesAlpha, Threading.SplitWindow.choosesMove,
      marksCutoff, Int.max_def] <;> omega

def run (schedule : Schedule Move) (depth : Nat) (cut : Bool) (search : ChildSearch Move) :
    Nat → Nat → Queue Move → State → Option Move → Option (Receipt Move)
  | 0, _, _, _, _ => none
  | fuel + 1, iteration, q, state, previous => do
    if q.pending.isEmpty then return ⟨[], q, .returned state.best, previous⟩
    if canSplit q.idx (q.idx + q.pending.length) depth cut (schedule.available iteration q) then
      return ⟨[], q, .handoff state, previous⟩
    let (selected, next) ← step (schedule.busy iteration q) q
    match selected with
    | none => run schedule depth cut search fuel (iteration + 1) next state previous
    | some move =>
      let score := -(search move false (-(state.alpha + 1)) (-state.alpha))
      let change := after state.alpha state.best score
      let chosen := if change.2.1 then some move else previous
      if change.2.2 then return ⟨[move], next, .returned score, chosen⟩
      let r ← run schedule depth cut search fuel (iteration + 1) next ⟨state.alpha, change.1⟩ chosen
      return { r with visited := move :: r.visited }

theorem run_exists (schedule : Schedule Move) (depth : Nat) (cut : Bool) (search : ChildSearch Move)
    (fuel iteration : Nat) (q : Queue Move) (state : State) (previous : Option Move) (enough : q.budget < fuel) :
    ∃ receipt, run schedule depth cut search fuel iteration q state previous = some receipt := by
  induction fuel generalizing iteration q state previous with
  | zero => omega
  | succ fuel ih =>
    rw [run]
    by_cases empty : q.pending.isEmpty
    · simp [empty]
    · by_cases split : canSplit q.idx (q.idx + q.pending.length) depth cut (schedule.available iteration q)
      · simp [empty, split]
      · cases advance : step (schedule.busy iteration q) q with
        | none =>
          have h := ((PvsPostpone.stepContract_proven Move).2.1 q _).mp advance
          simp [h] at empty
        | some pair =>
          obtain ⟨selected, next⟩ := pair
          have decrease := ((PvsPostpone.stepContract_proven Move).2.2 q _ _ _ advance).2.1
          cases selected with
          | none =>
            simpa [empty, split, advance] using ih (iteration + 1) next state previous (by omega)
          | some m =>
            let value := -(search m false (-(state.alpha + 1)) (-state.alpha))
            let change := after state.alpha state.best value
            let chosen := if change.2.1 then some m else previous
            simp only [empty, split, Bool.false_eq_true, ↓reduceIte]
            change ∃ r, (if change.2.2 then some ⟨[m], next, .returned value, chosen⟩ else
              (run schedule depth cut search fuel (iteration + 1) next ⟨state.alpha,change.1⟩ chosen).bind
                (fun r => some { r with visited := m :: r.visited })) = some r
            by_cases stop : change.2.2
            · simp [stop]
            · obtain ⟨r, returns⟩ := ih (iteration + 1) next ⟨state.alpha,change.1⟩ chosen (by omega)
              simp [stop, returns]

private theorem postponed_busy (q : Queue Move) (advance : step busy q = some (none, next)) : busy = true := by
  cases busy
  · obtain ⟨idx, fresh, late⟩ := q
    cases fresh <;> cases late <;> simp [step, shouldPostpone] at advance
  · rfl

theorem run_trace (schedule : Schedule Move) (depth : Nat) (cut : Bool) (done : ChildRelation Move)
    (search : ChildSearch Move) (beta : Int)
    (fuel iteration : Nat) (q : Queue Move) (state : State) (previous : Option Move) (receipt : Receipt Move)
    (result : run schedule depth cut search fuel iteration q state previous = some receipt)
    (calls : ∀ m ∈ receipt.visited, done m false (-(state.alpha + 1)) (-state.alpha)
      (search m false (-(state.alpha + 1)) (-state.alpha))) :
    Run done search false beta q state receipt.visited receipt.remaining receipt.outcome := by
  induction fuel generalizing iteration q state previous receipt with
  | zero => simp [run] at result
  | succ fuel ih =>
    rw [run] at result
    by_cases empty : q.pending.isEmpty
    · simp [empty] at result
      subst receipt
      exact .exhausted (List.isEmpty_iff.mp empty)
    · by_cases split : canSplit q.idx (q.idx + q.pending.length) depth cut (schedule.available iteration q)
      · simp [empty, split] at result
        subst receipt
        exact .handoff
      · cases advance : step (schedule.busy iteration q) q with
        | none => simp [empty, split, advance] at result
        | some pair =>
          obtain ⟨selected, next⟩ := pair
          cases selected with
          | none =>
            simp [empty, split, advance] at result
            exact .postpone (postponed_busy q advance ▸ advance) (ih _ _ _ _ _ result calls)
          | some m =>
            let value := -(search m false (-(state.alpha + 1)) (-state.alpha))
            let change := after state.alpha state.best value
            let chosen := if change.2.1 then some m else previous
            have same : moveScore false (q.idx == 0) search m state.alpha beta = value := by simp [moveScore, value]
            have update := physicalUpdate state value beta
            have numeric : (⟨state.alpha, change.1⟩ : State) = publish false beta state value := by
              obtain ⟨x,y⟩ := state
              simp [change, publish, Threading.SplitWindow.raisesAlpha, update.1]
            simp only [empty, split, advance, Bool.false_eq_true, ↓reduceIte] at result
            change (if change.2.2 then some ⟨[m], next, .returned value, chosen⟩ else
              (run schedule depth cut search fuel (iteration + 1) next ⟨state.alpha,change.1⟩ chosen).bind
                (fun r => some { r with visited := m :: r.visited })) = some receipt at result
            by_cases stop : change.2.2
            · have marks : marksCutoff false beta state value = true := update.2.2.1 ▸ stop
              simp [stop] at result
              subst receipt
              have invoked : MoveCalls done search false (q.idx == 0) m state.alpha beta := by
                simpa [MoveCalls] using calls m (by simp)
              change Run done search false beta q state [m] next (.returned value)
              rw [← same]
              exact .cutoff advance invoked (same ▸ marks)
            · have marks : marksCutoff false beta state value = false := by
                rw [← update.2.2.1]; exact Bool.eq_false_iff.mpr stop
              cases rec : run schedule depth cut search fuel (iteration + 1) next ⟨state.alpha,change.1⟩ chosen with
              | none => simp [stop, rec] at result
              | some later =>
                simp [stop, rec] at result
                subst receipt
                have invoked : MoveCalls done search false (q.idx == 0) m state.alpha beta := by
                  simpa [MoveCalls] using calls m (by simp)
                change Run done search false beta q state (m :: later.visited) later.remaining later.outcome
                refine .searched advance invoked (same ▸ marks) ?_
                rw [same, ← numeric]
                exact ih _ _ _ _ _ rec (fun x hx => calls x (List.mem_cons_of_mem m hx))

theorem physicalHeaderCursor (q : Queue Move) (count fresh depth : Nat) (cut available busy : Bool) (scout : Int)
    (total : count = q.idx + q.pending.length) (freshSlots : q.fresh.length = fresh - q.idx) :
    header q.idx count fresh depth cut available busy scout = queueHeader q depth cut available busy scout := by
  rw [← physicalHeader, total]
  cases left : q.fresh with
  | nil =>
    have notLater : ¬ q.idx + 1 < fresh := by simp [left] at freshSlots; omega
    have noFresh : ¬ q.idx + 1 < q.idx := by omega
    simp [header, notLater, noFresh]
  | cons m rest =>
    have same : fresh = q.idx + q.fresh.length := by
      rw [left] at freshSlots ⊢
      simp only [List.length_cons] at freshSlots ⊢
      omega
    rw [same, left]

open GeneralNonPvMir (order orderHint presearch Presearch)
open MoveList.Ordering (ttMoveValue)
open PvsMoveLoop (scoreInf)

theorem seeded {Move : Type} [DecidableEq Move] (priority : Move → Int) (done : ChildRelation Move)
    (search : ChildSearch Move) (first : Move) (rest : List Move) (hint : Option Move) (alpha beta : Int)
    (null : beta = alpha + 1) (legal : ∀ m, hint = some m → m ∈ first :: rest)
    (low : ∀ m sq, hint = some sq → m ∈ first :: rest → m ≠ sq → priority m < ttMoveValue)
    (calls : (first :: rest).length = 1 ∨ hint.isSome →
      done (hint.getD first) false (-(alpha + 1)) (-alpha) (search (hint.getD first) false (-(alpha + 1)) (-alpha)))
    (upper : (first :: rest).length = 1 ∨ hint.isSome →
      search (hint.getD first) false (-(alpha + 1)) (-alpha) < scoreInf)
    (start : presearch (first :: rest).length hint first alpha beta (fun m => search m false) = .owner idx best)
    (ordered : List Move) (same : ordered = order priority hint (first :: rest))
    {visited : List Move} {remaining : Queue Move} {outcome : Outcome}
    (suffix : Run done search false beta ⟨idx, ordered.drop idx, []⟩ ⟨alpha, best⟩ visited remaining outcome) :
    Run done search false beta ⟨0, ordered, []⟩ ⟨alpha, -scoreInf⟩
      (ordered.take idx ++ visited) remaining outcome := by
  by_cases used : (first :: rest).length = 1 ∨ hint.isSome
  · let m := hint.getD first
    have front : ordered.head? = some m := by
      rw [same]
      cases eq : hint with
      | some sq => simpa [m, eq] using (orderHint priority _ sq (legal _ eq) (fun x => low x sq eq)).2
      | none =>
        cases rest with
        | nil => simp [order, m, eq]
        | cons => simp [eq] at used
    cases ordered with
    | nil => cases front
    | cons actual tail =>
      cases Option.some.inj front
      let score := -(search m false (-(alpha + 1)) (-alpha))
      have numeric : moveScore false true search m alpha beta = score := by simp [moveScore, score]
      have invokes : MoveCalls done search false true m alpha beta := by simpa [MoveCalls, m] using calls used
      have sentinel : -scoreInf < moveScore false true search m alpha beta := by
        rw [numeric]
        have bound : search m false (-(alpha + 1)) (-alpha) < scoreInf := upper used
        dsimp [score]
        omega
      simp only [presearch, Bool.or_eq_true, decide_eq_true_eq, ite_eq_left used] at start
      change (if (decide (alpha < score) && decide (beta ≤ score)) then Presearch.storeLower m score
        else Presearch.owner 1 score) = _ at start
      by_cases cutoff : beta ≤ score
      · have above : alpha < score := by omega
        simp [cutoff, above] at start
      · simp [cutoff] at start
        obtain ⟨rfl, rfl⟩ := start
        simpa [numeric] using Run.presearchContinues m tail null invokes sentinel
          (numeric ▸ Int.lt_of_not_ge cutoff) (by simpa [numeric] using suffix)
  · simp only [presearch, Bool.or_eq_true, decide_eq_true_eq, ite_eq_right used, Presearch.owner.injEq] at start
    obtain ⟨rfl, rfl⟩ := start
    simpa [scoreInf] using suffix

open PvsBoardTree (GeneratedSquare build play legalSquares Position)
open PvsMoveLoop (Tree)
open TranspositionTable.ProbeRetry (RawAttempt)

def Completed (children : ChildResults) (child : Move → Tree) (alpha beta : Int)
    (earlier : List Move) (r : Receipt Move) (score : Int) : Prop :=
  match r.outcome with
  | .returned value => value = score
  | .handoff state =>
    let labels := (earlier ++ r.visited ++ r.remaining.pending).map child
    ∃ split : SplitExecution labels.length false alpha beta score,
      split.start.toNat = r.remaining.idx ∧ split.entry = state ∧ ReportCalls children labels split

theorem execution (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : Position)
    (first : GeneratedSquare mc b.val) (rest : List (GeneratedSquare mc b.val))
    (moves : legalSquares mc b.val = first :: rest) (priority : GeneratedSquare mc b.val → Int)
    (low : ∀ m ∈ legalSquares mc b.val, priority m < ttMoveValue)
    (attempts : List RawAttempt)
    (alpha beta : Int) (null : beta = alpha + 1) (search : ChildSearch (GeneratedSquare mc b.val))
    (schedule : Schedule (GeneratedSquare mc b.val)) (depth : Nat) (cut : Bool) (children : ChildResults)
    {receipt : Receipt (GeneratedSquare mc b.val)}
    (calls : ∀ m ∈ receipt.visited, children .serial (build mc fc (play mc fc b m).val)
      false (-(alpha + 1)) (-alpha) (search m false (-(alpha + 1)) (-alpha)))
    (firstCall : let hint := GeneralNonPvMir.checkedHint mc b (GeneralNonPvMir.probeHint b attempts)
      (first :: rest).length = 1 ∨ hint.isSome → let m := hint.getD first
        children .serial (build mc fc (play mc fc b m).val) false (-(alpha + 1)) (-alpha)
          (search m false (-(alpha + 1)) (-alpha)) ∧ search m false (-(alpha + 1)) (-alpha) < scoreInf)
    (start : let hint := GeneralNonPvMir.checkedHint mc b (GeneralNonPvMir.probeHint b attempts)
      presearch (first :: rest).length hint first alpha beta (fun m => search m false) = .owner idx best)
    (result : let hint := GeneralNonPvMir.checkedHint mc b (GeneralNonPvMir.probeHint b attempts)
      let ordered := order priority hint (first :: rest)
      run schedule depth cut search fuel 0 ⟨idx, ordered.drop idx, []⟩ ⟨alpha, best⟩ none = some receipt)
    (completed : let hint := GeneralNonPvMir.checkedHint mc b (GeneralNonPvMir.probeHint b attempts)
      let ordered := order priority hint (first :: rest)
      Completed children (fun m => build mc fc (play mc fc b m).val) alpha beta (ordered.take idx) receipt score) :
    ExecutionWithUses exit children (build mc fc b) false alpha beta score := by
  let hint := GeneralNonPvMir.checkedHint mc b (GeneralNonPvMir.probeHint b attempts)
  let ordered := order priority hint (first :: rest)
  let child := fun m => build mc fc (play mc fc b m).val
  have member : ∀ m : GeneratedSquare mc b.val, m ∈ legalSquares mc b.val := by
    intro m
    obtain ⟨v, present, eq⟩ := (PvsBoardTree.legalSquares_complete mc b.val m.val).mpr
      ((Board.getMoves_exact mc b.val _).mp m.property)
    rwa [show v = m from Subtype.ext eq] at present
  have legal := fun m (_ : hint = some m) => moves ▸ member m
  have lower := fun m sq (_ : hint = some sq) (h : m ∈ first :: rest) (_ : m ≠ sq) => low m (moves ▸ h)
  have perm := GeneralNonPvMir.orderPerm priority (first :: rest) hint legal lower
  have prefixRun := run_trace schedule depth cut (fun m => children .serial (child m)) search beta
    _ 0 _ _ _ _ result calls
  have trace := seeded priority (fun m => children .serial (child m)) search first rest hint alpha beta null legal lower
    (fun used => (firstCall used).1) (fun used => (firstCall used).2) start ordered rfl prefixRun
  rw [build.eq_def]
  split
  next head tail h =>
    obtain ⟨rfl, rfl⟩ := List.cons.inj (moves.symm.trans h)
    have all : ((⟨0, ordered, []⟩ : Queue (GeneratedSquare mc b.val)).pending.map child).Perm
        (child first :: rest.map child) := by simpa [Queue.pending, ordered] using perm.map child
    change Completed children child alpha beta (ordered.take idx) receipt score at completed
    cases outcome : receipt.outcome with
    | returned value =>
      simp [Completed, outcome] at completed
      rw [outcome, completed] at trace
      exact ExecutionWithUses.ofSerialRun child all rfl trace
    | handoff state =>
      simp only [Completed, outcome] at completed
      obtain ⟨split, cursor, entry, reports⟩ := completed
      rw [outcome, ← entry] at trace
      exact ExecutionWithUses.ofSerialHandoff child first all rfl split cursor trace reports
  next h => rw [moves] at h; cases h

abbrev SplitReturn (Move : Type) := List Move → Nat → Receipt Move → Option Int

def recordedOwner (schedule : Schedule Move) (depth : Nat) (cut : Bool) (search : ChildSearch Move)
    (reply : Move → Option Int) (split : SplitReturn Move) (alpha : Int)
    (ordered : List Move) (idx : Nat) (best : Int) : Option Int := do
  let q : Queue Move := ⟨idx, ordered.drop idx, []⟩
  let r ← run schedule depth cut search (q.budget + 1) 0 q ⟨alpha, best⟩ none
  if r.visited.all (fun m => (reply m).isSome) then
    match r.outcome with
    | .returned value => some value
    | .handoff _ => split ordered idx r
  else none

private theorem recordedOwner_returns (schedule : Schedule Move) (depth : Nat) (cut : Bool) (search : ChildSearch Move)
    (reply : Move → Option Int) (split : SplitReturn Move) (alpha : Int)
    (ordered : List Move) (idx : Nat) (best score : Int)
    (answer : recordedOwner schedule depth cut search reply split alpha ordered idx best = some score) :
    let q : Queue Move := ⟨idx, ordered.drop idx, []⟩
    ∃ r, run schedule depth cut search (q.budget + 1) 0 q ⟨alpha, best⟩ none = some r ∧
      (∀ m ∈ r.visited, (reply m).isSome) ∧
      (match r.outcome with
       | .returned v => v = score
       | .handoff _ => split ordered idx r = some score) := by
  intro q
  obtain ⟨r, normal⟩ := run_exists schedule depth cut search _ 0 q ⟨alpha,best⟩ none (Nat.lt_succ_self _)
  unfold recordedOwner at answer
  change (do
    let r ← run schedule depth cut search (q.budget + 1) 0 q ⟨alpha,best⟩ none
    if r.visited.all (fun m => (reply m).isSome) then
      match r.outcome with
      | .returned v => some v
      | .handoff _ => split ordered idx r
    else none) = some score at answer
  rw [normal] at answer
  simp at answer
  refine ⟨r, normal, answer.1, ?_⟩
  cases how : r.outcome <;> simpa [how] using answer.2

open GeneralNonPvMir (Child observe probeHint checkedHint)
open TranspositionTable (Depth)
open TableReads (ChildLookupFrom ProbeReadsFrom)

def scoutReply (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : Position) (depth : Depth)
    (cut : Bool) (child : Child) (alpha : Int) (m : GeneratedSquare mc b.val) : Option Int :=
  child (play mc fc b m).val (depth - 1) (!cut) (-(alpha+1)) (-alpha)

def scheduled (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : Position) (depth : Depth)
    (alpha beta : ScoreRefresh.Score) (paired checked cut : Bool) (mem : MoveList.Initialization.Memory Move.Move)
    (attempts : List RawAttempt) (reads : List (ChildLookupFrom mc fc b source))
    (priority : GeneratedSquare mc b.val → Int) (child : Child)
    (schedule : Schedule (GeneratedSquare mc b.val)) (split : SplitReturn (GeneratedSquare mc b.val)) : Option Int :=
  GeneralNonPvMir.frame mc fc b depth alpha beta paired checked cut mem attempts reads priority child
    (recordedOwner schedule depth.toNat cut (observe mc fc b depth cut child)
      (scoutReply mc fc b depth cut child alpha.raw.toInt) split alpha.raw.toInt)

theorem scheduledExecution (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : Position) (depth : Depth)
    (alpha beta : ScoreRefresh.Score) (paired checked cut : Bool) (mem : MoveList.Initialization.Memory Move.Move)
    (attempts : List RawAttempt) (reads : List (ChildLookupFrom mc fc b source))
    (priority : GeneratedSquare mc b.val → Int) (child : Child)
    (schedule : Schedule (GeneratedSquare mc b.val)) (split : SplitReturn (GeneratedSquare mc b.val))
    (full : ExactDepth.Full b depth) (null : beta.raw.toInt = alpha.raw.toInt + 1)
    (observed : ProbeReadsFrom source attempts)
    (low : ∀ m ∈ legalSquares mc b.val, priority m < ttMoveValue)
    (passed : legalSquares mc b.val = [] → Board.hasLegalMoves mc (PvsBoardTree.pass b).val = true → ∀ v,
      child (PvsBoardTree.pass b) depth (!cut) (-beta.raw.toInt) (-alpha.raw.toInt) = some v →
      children .serial (build mc fc (PvsBoardTree.pass b)) false (-beta.raw.toInt) (-alpha.raw.toInt) v)
    (scouted : ∀ m v, scoutReply mc fc b depth cut child alpha.raw.toInt m = some v →
      children .serial (build mc fc (play mc fc b m).val) false (-(alpha.raw.toInt+1)) (-alpha.raw.toInt) v ∧
        v < scoreInf)
    (reports : ∀ ordered idx r state v, r.outcome = .handoff state → split ordered idx r = some v →
      Completed children (fun m => build mc fc (play mc fc b m).val) alpha.raw.toInt beta.raw.toInt
        (ordered.take idx) r v)
    (answer : scheduled mc fc b depth alpha beta paired checked cut mem attempts reads priority child schedule split =
      some score) :
    ExecutionWithUses (CausalReplay.PrefixExitFrom mc fc source) children (build mc fc b)
      false alpha.raw.toInt beta.raw.toInt score := by
  apply GeneralNonPvMir.frameCases mc fc b depth alpha beta paired checked cut mem attempts reads priority child _
    full null observed low passed scouted _ answer
  intro first rest moves hint s a z seen pre idx best start result
  let ordered := order priority hint (first :: rest)
  let reply := scoutReply mc fc b depth cut child a
  obtain ⟨r, normal, ready, returned⟩ :=
    recordedOwner_returns schedule depth.toNat cut s reply split a ordered idx best score result
  apply execution mc fc b first rest moves priority low attempts a z null s schedule depth.toNat cut
    children (receipt := r) _ _ start normal _
  · intro m mem
    cases h : reply m with
    | none => have used := ready m mem; simp [h] at used
    | some v =>
      have witnessed := (scouted m v h).1
      change children .serial _ false _ _ ((reply m).getD 0)
      simpa [h] using witnessed
  · dsimp only
    intro used
    exact pre (seen ▸ used)
  · cases how : r.outcome with
    | returned v => simpa only [Completed, how] using (by simpa only [how] using returned : v = score)
    | handoff state => exact reports ordered idx r state score how (by simpa only [how] using returned)

end NeuralReversi.PvsParallel.GeneralOwnerMir
