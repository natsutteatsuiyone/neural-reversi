import NeuralReversi.PvsParallel.GeneralOwnerMir
import NeuralReversi.PvsParallel.Replay

namespace NeuralReversi.PvsParallel.SerialNonPv

open Aspiration (SoundResult Window)
open Bitboard.Moves (X86Config)
open Cutoffs (FlipConfig)
open PvsBoardTree (Position GeneratedSquare build play pass legalSquares minimax)
open PvsMoveLoop (ChildSearch scoreInf scoutWindow reflectedWindow)
open PvsPostpone (Queue step)
open GeneralOwnerMir (Schedule Receipt run recordedOwner scoutReply scheduled)
open GeneralNonPvMir (Action Child dispatch entryAction runPrefix frame checkedHint probeHint
  order observe presearch resumeOwner)
open TranspositionTable (Depth)
open TranspositionTable.ProbeRetry (RawAttempt)
open TableReads (ChildLookupFrom ProbeReadsFrom)

def schedule : Schedule Move := ⟨fun _ _ => false, fun _ _ => false⟩

def runFrame (mc : X86Config) (fc : FlipConfig) (b : Position) (depth : Depth)
    (alpha beta : ScoreRefresh.Score) (paired checked cut : Bool)
    (mem : MoveList.Initialization.Memory Move.Move) (attempts : List RawAttempt)
    (reads : List (ChildLookupFrom mc fc b source)) (priority : GeneratedSquare mc b.val → Int)
    (child : Child) : Option Int :=
  scheduled mc fc b depth alpha beta paired checked cut mem attempts reads priority child schedule
    (fun _ _ _ => none)

private theorem run_returned (search : ChildSearch Move) (depth fuel i : Nat) (cut : Bool)
    (q : Queue Move) (state : Threading.SplitWindow.State) (previous : Option Move) (r : Receipt Move)
    (result : run schedule depth cut search fuel i q state previous = some r) :
    ∃ value, r.outcome = .returned value := by
  induction fuel generalizing i q state previous r with
  | zero => simp [run] at result
  | succ fuel ih =>
    rw [run] at result
    by_cases empty : q.pending.isEmpty
    · simp [empty] at result
      subst r
      exact ⟨_, rfl⟩
    · simp only [empty, schedule, GeneralOwnerMir.canSplit, Bool.and_false, Bool.false_eq_true,
        ite_false] at result
      cases advance : step false q with
      | none => simp [advance] at result
      | some selected =>
        obtain ⟨move, next⟩ := selected
        cases move with
        | none =>
          exact ih _ _ _ _ _ (by simpa [schedule, advance] using result)
        | some m =>
          let value := -(search m false (-(state.alpha + 1)) (-state.alpha))
          let update := GeneralOwnerMir.after state.alpha state.best value
          let chosen := if update.2.1 then some m else previous
          simp only [advance, bind, Option.bind] at result
          change (if update.2.2 then some ⟨[m], next, .returned value, chosen⟩ else
            (run schedule depth cut search fuel (i + 1) next ⟨state.alpha, update.1⟩ chosen).bind
              (fun later => some { later with visited := m :: later.visited })) = some r at result
          by_cases stop : update.2.2
          · simp [stop] at result
            subst r
            exact ⟨_, rfl⟩
          · cases later : run schedule depth cut search fuel (i + 1) next
                ⟨state.alpha, update.1⟩ chosen with
            | none => simp [stop, later] at result
            | some tail =>
              simp [stop, later] at result
              subst r
              exact ih _ _ _ _ tail later

private theorem owner_returns (search : ChildSearch Move) (reply : Move → Option Int)
    (depth : Nat) (cut : Bool) (alpha : Int) (ordered : List Move) (idx : Nat) (best : Int)
    (available : ∀ m, ∃ score, reply m = some score) :
    ∃ score, recordedOwner schedule depth cut search reply (fun _ _ _ => none) alpha ordered idx best =
      some score := by
  let q : Queue Move := ⟨idx, ordered.drop idx, []⟩
  obtain ⟨r, result⟩ := GeneralOwnerMir.run_exists schedule depth cut search (q.budget + 1) 0 q
    ⟨alpha, best⟩ none (Nat.lt_succ_self _)
  obtain ⟨score, returned⟩ := run_returned search depth _ 0 cut q ⟨alpha, best⟩ none r result
  have seen : r.visited.all (fun m => (reply m).isSome) = true := by
    apply List.all_eq_true.mpr
    intro m _
    obtain ⟨v, h⟩ := available m
    simp [h]
  refine ⟨score, ?_⟩
  unfold recordedOwner
  change ((run schedule depth cut search (q.budget + 1) 0 q ⟨alpha, best⟩ none).bind _) = some score
  simp only [result, Option.bind_some, seen, ite_true, returned]

private def EntryFacts (depth : Nat) (stable mobile other parentCut childCut : Bool) : Action → Prop
  | .shallow => depth ≤ 12
  | .stability => stable = true
  | .pass => mobile = false ∧ other = true
  | .probe => parentCut = true
  | .enhanced => childCut = true
  | .rest => mobile = true
  | _ => True

private theorem dispatch_observed
    (selected : dispatch depth stable mobile other wipes parentCut childCut = action) :
    EntryFacts depth stable mobile other parentCut childCut action := by
  subst action
  by_cases small : depth ≤ 12
  · simp [dispatch, small, EntryFacts]
  by_cases stability : stable = true
  · simp [dispatch, small, stability, EntryFacts]
  cases mobile with
  | false => cases other <;> simp [dispatch, small, stability, EntryFacts]
  | true =>
    cases wipes with
    | true => simp [dispatch, small, stability, EntryFacts]
    | false =>
      cases parentCut with
      | true => simp [dispatch, small, stability, EntryFacts]
      | false => cases childCut <;> by_cases level : 14 ≤ depth <;>
          simp [dispatch, small, stability, level, EntryFacts]

theorem prefix_returns (mc : X86Config) (fc : FlipConfig) (b : Position) (depth : Depth)
    (alpha beta : ScoreRefresh.Score) (paired checked cut : Bool)
    (mem : MoveList.Initialization.Memory Move.Move) (attempts : List RawAttempt)
    (reads : List (ChildLookupFrom mc fc b source)) (child : Child)
    (full : ExactDepth.Full b depth) (deep : 12 < depth.toNat)
    (continued : entryAction mc fc b depth alpha beta paired checked mem attempts reads ≠ .rest)
    (passed : legalSquares mc b.val = [] → Board.hasLegalMoves mc (pass b).val = true →
      ∃ score, child (pass b) depth (!cut) (-beta.raw.toInt) (-alpha.raw.toInt) = some score) :
    ∃ score, runPrefix mc fc b depth alpha beta paired checked cut mem attempts reads child =
      some score := by
  unfold entryAction at continued
  unfold runPrefix
  dsimp only at continued ⊢
  split
  next _ selected =>
    obtain ⟨result, found⟩ := Option.isSome_iff_exists.mp (dispatch_observed selected)
    obtain ⟨lo, hi, _, _⟩ := (EndgameStability.cutoffContract_proven mc fc b depth.toNat
      ⟨(ScoreRefresh.toDiscDiff alpha).raw.toInt⟩).2 result found
    have signed : (BitVec.ofInt 32 result.value).toInt = result.value :=
      BitVec.toInt_ofInt_eq_self (by decide) (by omega) (by omega)
    let discs : ScoreRefresh.DiscScore := ⟨BitVec.ofInt 32 result.value⟩
    have within : ¬ (discs.raw.toInt < -127 ∨ discs.raw.toInt > 127) := by
      change ¬ ((BitVec.ofInt 32 result.value).toInt < -127 ∨ _)
      rw [signed]
      omega
    have converted : ScoreRefresh.fromDiscDiff checked discs = .ok ⟨discs.raw <<< 8⟩ := by
      simp only [ScoreRefresh.fromDiscDiff, within, decide_false, Bool.and_false, Bool.false_eq_true, ite_false]
    refine ⟨(discs.raw <<< 8).toInt, ?_⟩
    rw [found]
    change ((ScoreRefresh.fromDiscDiff checked discs).toOption.bind _) = _
    rw [converted]
    rfl
  next =>
    have result := PvEntryMir.terminal_solve b checked
    rw [← full] at result
    cases solved : ScoreRefresh.BoardBridge.solve checked b.val depth with
    | error e => simp [solved, Except.map] at result
    | ok score => exact ⟨score.raw.toInt, rfl⟩
  next _ selected =>
    have ⟨stuck, mobile⟩ := dispatch_observed selected
    have nil : legalSquares mc b.val = [] := by
      simpa [PvEntryMir.generated_count] using stuck
    obtain ⟨score, found⟩ := passed nil mobile
    exact ⟨-score, by simp [found]⟩
  next => exact ⟨_, rfl⟩
  next _ selected => exact Option.isSome_iff_exists.mp (dispatch_observed selected)
  next _ selected => exact Option.isSome_iff_exists.mp (dispatch_observed selected)
  next _ selected => exact (continued selected).elim
  next _ selected =>
    have small := dispatch_observed selected
    exact (Nat.not_le_of_gt deep small).elim

private theorem rest_returns (mc : X86Config) (fc : FlipConfig) (b : Position) (depth : Depth)
    (alpha beta : ScoreRefresh.Score) (paired checked cut : Bool)
    (mem : MoveList.Initialization.Memory Move.Move) (attempts : List RawAttempt)
    (reads : List (ChildLookupFrom mc fc b source)) (priority : GeneratedSquare mc b.val → Int)
    (child : Child) (entered : entryAction mc fc b depth alpha beta paired checked mem attempts reads = .rest)
    (hintLegal : (checkedHint mc b (probeHint b attempts)).isSome = (probeHint b attempts).isSome)
    (scouts : ∀ m, ∃ score, scoutReply mc fc b depth cut child alpha.raw.toInt m = some score) :
    ∃ score, runFrame mc fc b depth alpha beta paired checked cut mem attempts reads priority child =
      some score := by
  cases moves : legalSquares mc b.val with
  | nil =>
    have mobile := dispatch_observed entered
    simp [PvEntryMir.generated_count, moves, EntryFacts] at mobile
  | cons first rest =>
    let hint := checkedHint mc b (probeHint b attempts)
    let search := observe mc fc b depth cut child
    let a := alpha.raw.toInt
    let z := beta.raw.toInt
    let reply := scoutReply mc fc b depth cut child a
    let owner := recordedOwner schedule depth.toNat cut search reply (fun _ _ _ => none) a
      (order priority hint (first :: rest))
    let step := presearch (first :: rest).length hint first a z (fun m => search m false)
    have returned : ∃ score, resumeOwner owner step = some score := by
      cases step with
      | storeLower m v => exact ⟨v, rfl⟩
      | owner idx best => exact owner_returns search reply depth.toNat cut a _ idx best scouts
    obtain ⟨score, answer⟩ := returned
    refine ⟨score, ?_⟩
    unfold runFrame scheduled frame
    rw [entered, moves]
    simp only [guard, hintLegal, ite_true, pure, bind, Option.bind]
    change (do
      if (first :: rest).length = 1 || (probeHint b attempts).isSome then let _ ← reply (hint.getD first)
      resumeOwner owner step) = some score
    obtain ⟨v, seen⟩ := scouts (hint.getD first)
    by_cases used : ((decide ((first :: rest).length = 1)) || (probeHint b attempts).isSome) = true
    · simp only [used, ite_true]
      change (reply (hint.getD first)).bind (fun _ => resumeOwner owner step) = some score
      change reply (hint.getD first) = some v at seen
      rw [seen]
      exact answer
    · simp only [used, Bool.false_eq_true, ite_false]
      exact answer

/-- Each supplied result is for an actual placement or forced pass from this frame.
    The edge carries strict board rank decrease and preservation of exact depth. -/
abbrev ChildInduction (mc : X86Config) (fc : FlipConfig) (b : Position) (depth : Depth)
    (child : Child) (children : ChildResults) : Prop :=
  ∀ {next childDepth}, ExactDepth.RecursiveCall mc fc b depth next childDepth →
    ∀ childCut a z, Window a z → z = a + 1 →
      ∃ score, child next childDepth childCut a z = some score ∧
        children .serial (build mc fc next) false a z score ∧
        SoundResult (minimax mc fc next) a z score

theorem frame_returns (mc : X86Config) (fc : FlipConfig) (b : Position) (depth : Depth)
    (alpha beta : ScoreRefresh.Score) (paired checked cut : Bool)
    (mem : MoveList.Initialization.Memory Move.Move) (attempts : List RawAttempt)
    (reads : List (ChildLookupFrom mc fc b source)) (priority : GeneratedSquare mc b.val → Int)
    (child : Child) (children : ChildResults)
    (full : ExactDepth.Full b depth) (deep : 12 < depth.toNat)
    (window : Window alpha.raw.toInt beta.raw.toInt) (null : beta.raw.toInt = alpha.raw.toInt + 1)
    (observed : ProbeReadsFrom source attempts)
    (hintLegal : entryAction mc fc b depth alpha beta paired checked mem attempts reads = .rest →
      (checkedHint mc b (probeHint b attempts)).isSome = (probeHint b attempts).isSome)
    (low : ∀ m ∈ legalSquares mc b.val, priority m < MoveList.Ordering.ttMoveValue)
    (ih : ChildInduction mc fc b depth child children) :
    ∃ score, runFrame mc fc b depth alpha beta paired checked cut mem attempts reads priority child =
      some score ∧
      ExecutionWithUses (CausalReplay.PrefixExitFrom mc fc source) children (build mc fc b) false
        alpha.raw.toInt beta.raw.toInt score := by
  let a := alpha.raw.toInt
  let z := beta.raw.toInt
  have passes (stuck : legalSquares mc b.val = [])
      (mobile : Board.hasLegalMoves mc (pass b).val = true) :=
    ih (.forcedPass b depth ⟨PvsBoardTree.stuck_of_nil stuck, mobile⟩) (!cut) (-z) (-a)
      (reflectedWindow window) (by change z = a + 1 at null; omega)
  have scouts (m : GeneratedSquare mc b.val) :=
    ih (.ply b m depth) (!cut) (-(a + 1)) (-a) (scoutWindow window) (by omega)
  have result :
      ∃ score, runFrame mc fc b depth alpha beta paired checked cut mem attempts reads priority child =
        some score := by
    by_cases entered : entryAction mc fc b depth alpha beta paired checked mem attempts reads = .rest
    · exact rest_returns mc fc b depth alpha beta paired checked cut mem attempts reads priority child entered
        (hintLegal entered) (fun m => ⟨_, (scouts m).choose_spec.1⟩)
    · have prefixResult := prefix_returns mc fc b depth alpha beta paired checked cut mem attempts reads child full deep entered
        (fun stuck mobile => ⟨_, (passes stuck mobile).choose_spec.1⟩)
      unfold runFrame scheduled frame
      split
      next h => exact (entered h).elim
      next => exact prefixResult
  obtain ⟨score, answer⟩ := result
  refine ⟨score, answer, ?_⟩
  apply GeneralOwnerMir.scheduledExecution mc fc b depth alpha beta paired checked cut mem attempts reads
    priority child schedule (fun _ _ _ => none) full null observed low _ _ _ answer
  · intro stuck mobile v found
    obtain ⟨s, returned, proof, _⟩ := passes stuck mobile
    have same := Option.some.inj (returned.symm.trans found)
    exact same ▸ proof
  · intro m v found
    obtain ⟨s, returned, proof, sound⟩ := scouts m
    have same := Option.some.inj (returned.symm.trans found)
    exact same ▸ ⟨proof, sound.upper⟩
  · intro _ _ _ _ _ _ impossible
    contradiction

/-- A serial deep frame consumes recursive returns and reads from the supplied
    publication history. The returned score is bounded in its actual window. -/
theorem frame_returns_from_history (mc : X86Config) (fc : FlipConfig) (b : Position) (depth : Depth)
    (alpha beta : ScoreRefresh.Score) (paired checked cut : Bool)
    (mem : MoveList.Initialization.Memory Move.Move) (attempts : List RawAttempt)
    (reads : List (ChildLookupFrom mc fc b (·.publication history)))
    (priority : GeneratedSquare mc b.val → Int) (child : Child) (frames : Nat)
    (full : ExactDepth.Full b depth) (deep : 12 < depth.toNat)
    (window : Window alpha.raw.toInt beta.raw.toInt) (null : beta.raw.toInt = alpha.raw.toInt + 1)
    (observed : TableReads.ProbeReads history attempts)
    (published : ExactTableHistory.Publications mc fc history) (hints : TableReads.LegalHints history)
    (low : ∀ m ∈ legalSquares mc b.val, priority m < MoveList.Ordering.ttMoveValue)
    (ih : ∀ {next childDepth}, ExactDepth.RecursiveCall mc fc b depth next childDepth →
      ∀ childCut a z, Window a z → z = a + 1 →
        ∃ score, child next childDepth childCut a z = some score ∧
          boundedReplay frames (build mc fc next) false a z score) :
    ∃ score, runFrame mc fc b depth alpha beta paired checked cut mem attempts reads priority child =
      some score ∧
      boundedReplay (frames + 1) (build mc fc b) false alpha.raw.toInt beta.raw.toInt score ∧
      SoundResult (minimax mc fc b) alpha.raw.toInt beta.raw.toInt score := by
  have hint := GeneralNonPvMir.observedHint mc b observed hints
  have present : (checkedHint mc b (probeHint b attempts)).isSome = (probeHint b attempts).isSome := by
    simpa only [Option.isSome_map] using congrArg Option.isSome hint
  have children : ChildInduction mc fc b depth child (fun _ => boundedReplay frames) := by
    intro next nextDepth edge childCut a z window null
    obtain ⟨score, returned, replayed⟩ := ih edge childCut a z window null
    exact ⟨score, returned, replayed,
      replayContract_proven (build mc fc next) false a z score frames window (fun _ => null) replayed⟩
  obtain ⟨score, returned, frame⟩ :=
    frame_returns mc fc b depth alpha beta paired checked cut mem attempts reads priority child
      (fun _ => boundedReplay frames) full deep window null observed (fun _ => present) low
      (fun edge => children edge)
  have checked : boundedReplay (frames + 1) (build mc fc b) false alpha.raw.toInt beta.raw.toInt score :=
    frame.weaken (fun _ _ _ _ _ _ child => child) (fun _ _ _ _ _ _ _ exit => exit.checked published)
  exact ⟨score, returned, checked,
    replayContract_proven (build mc fc b) false alpha.raw.toInt beta.raw.toInt score (frames + 1)
      window (fun _ => null) checked⟩

end NeuralReversi.PvsParallel.SerialNonPv
