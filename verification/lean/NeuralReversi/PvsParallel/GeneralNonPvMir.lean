import NeuralReversi.PvsParallel.PvEntryMir
import NeuralReversi.PvsParallel.CausalReplay
import NeuralReversi.PvsParallel.GeneralNonPvMir.Program
import NeuralReversi.PvsParallel.TableReads
import NeuralReversi.MoveList.Ordering
import NeuralReversi.TranspositionTable.ReaderMir.ProbeProgram

namespace NeuralReversi.PvsParallel.GeneralNonPvMir

open PvsBoardTree (Position build pass legalSquares)
open Cutoffs (FlipConfig exactSelectivity key)
open Bitboard.Moves (X86Config)
open ScoreRefresh (Score fromDiscDiff toDiscDiff)
open TranspositionTable (Depth EntryData ProbeRetry.Result)
open TableReads (ChildLookupFrom ProbeReads ProbeReadsFrom)
open ExactTableHistory (Publications)

private def parent (b : Position) (depth : Depth) (beta : Score) (checked : Bool)
    (attempts : List TranspositionTable.ProbeRetry.RawAttempt) : Option Int := do
  let .hit _ entry := TranspositionTable.ProbeRetry.probe
    (attempts.map (TranspositionTable.ProbeRetry.observe (key b))) | none
  let cut ← (TranspositionTable.canCut checked entry ⟨beta.raw⟩ depth exactSelectivity true).toOption
  if cut then some entry.score.toInt else none

private theorem parentExecution (b : Position) (depth : Depth) (alpha beta : Score) (checked : Bool)
    (attempts : List TranspositionTable.ProbeRetry.RawAttempt)
    (reads : ProbeReadsFrom source attempts) (result : parent b depth beta checked attempts = some score) :
    CausalReplay.PrefixExitFrom mc fc source mc fc b false alpha.raw.toInt beta.raw.toInt score := by
  unfold parent at result
  cases found : TranspositionTable.ProbeRetry.probe
      (attempts.map (TranspositionTable.ProbeRetry.observe (key b))) with
  | miss slot => simp [found] at result
  | hit slot entry =>
    cases gate : TranspositionTable.canCut checked entry ⟨beta.raw⟩ depth exactSelectivity true with
    | error e => simp [found, gate, Except.toOption] at result
    | ok ok =>
      cases ok <;> simp [found, gate, Except.toOption] at result
      subst score
      exact .probe attempts slot entry ⟨alpha.raw⟩ ⟨beta.raw⟩ depth checked reads found gate

private def etc (q : PvsEnhancedCutoff.Query) (reads : List (ChildLookupFrom mc fc b source)) : Option Int :=
  (PvsEnhancedCutoff.scan q (reads.map ChildLookupFrom.observation)).map Prod.snd

def runPrefix (mc : X86Config) (fc : FlipConfig) (b : Position) (depth : Depth) (alpha beta : Score)
    (paired checked cut : Bool) (mem : MoveList.Initialization.Memory Move.Move)
    (attempts : List TranspositionTable.ProbeRetry.RawAttempt) (reads : List (ChildLookupFrom mc fc b source))
    (child : Position → Depth → Bool → Int → Int → Option Int) : Option Int := do
  let moves := PvEntryMir.generated mc fc b paired mem
  let stable := EndgameStability.tryCutoff b.val depth.toNat ⟨(toDiscDiff alpha).raw.toInt⟩
  let tt := parent b depth beta checked attempts
  let enhanced := etc ⟨depth, exactSelectivity, true, ⟨alpha.raw⟩⟩ reads
  match dispatch depth.toNat stable.isSome (moves.len.getD 0 != 0) (Board.hasLegalMoves mc (pass b).val)
      moves.wipeout.join.isSome tt.isSome enhanced.isSome with
  | .stability =>
    let discs ← stable
    return (← (fromDiscDiff checked ⟨BitVec.ofInt 32 discs.value⟩).toOption).raw.toInt
  | .terminal => return (← (ScoreRefresh.BoardBridge.solve checked b.val depth).toOption).raw.toInt
  | .pass => return -(← child (pass b) depth (!cut) (-beta.raw.toInt) (-alpha.raw.toInt))
  | .wipeout => some PvsMoveLoop.scoreMax
  | .probe => tt
  | .enhanced => enhanced
  | .rest | .shallow => none

theorem prefixExecution (mc : X86Config) (fc : FlipConfig) (b : Position) (depth : Depth) (alpha beta : Score)
    (paired checked cut : Bool) (mem : MoveList.Initialization.Memory Move.Move)
    (attempts : List TranspositionTable.ProbeRetry.RawAttempt) (reads : List (ChildLookupFrom mc fc b source))
    (child : Position → Depth → Bool → Int → Int → Option Int)
    (full : ExactDepth.Full b depth) (null : beta.raw.toInt = alpha.raw.toInt + 1)
    (observed : ProbeReadsFrom source attempts)
    (passCall : legalSquares mc b.val = [] → Board.hasLegalMoves mc (pass b).val = true → ∀ value,
      child (pass b) depth (!cut) (-beta.raw.toInt) (-alpha.raw.toInt) = some value →
      children .serial (build mc fc (pass b)) false (-beta.raw.toInt) (-alpha.raw.toInt) value)
    (answer : runPrefix mc fc b depth alpha beta paired checked cut mem attempts reads child = some score) :
    ExecutionWithUses (CausalReplay.PrefixExitFrom mc fc source) children (build mc fc b)
      false alpha.raw.toInt beta.raw.toInt score := by
  simp only [runPrefix, PvEntryMir.generated_count, Option.getD_some, dispatch] at answer
  by_cases small : depth.toNat ≤ 12
  · simp [small] at answer
  · simp only [ite_eq_right small] at answer
    cases stable : EndgameStability.tryCutoff b.val depth.toNat ⟨(toDiscDiff alpha).raw.toInt⟩ with
    | some discs =>
      cases converted : fromDiscDiff checked ⟨BitVec.ofInt 32 discs.value⟩ with
      | error reason => simp [stable, converted, Except.toOption] at answer
      | ok raw =>
        simp [stable, converted, Except.toOption] at answer
        subst score
        rw [null]
        exact .cutoff mc fc b (.stability alpha discs raw checked (full ▸ stable) converted)
    | none =>
      cases moves : legalSquares mc b.val with
      | nil =>
        cases other : Board.hasLegalMoves mc (pass b).val with
        | false =>
          cases result : ScoreRefresh.BoardBridge.solve checked b.val depth with
          | error reason => simp [stable, moves, other, result, Except.toOption] at answer
          | ok raw =>
            have value := PvEntryMir.terminal_solve b checked
            rw [← full, result] at value
            simp [Except.map] at value
            simp [stable, moves, other, result, Except.toOption] at answer
            rw [← answer, value, PvsBoardTree.terminal_tree mc fc b moves other]
            exact .leaf
        | true =>
          cases result : child (pass b) depth (!cut) (-beta.raw.toInt) (-alpha.raw.toInt) with
          | none => simp [stable, moves, other, result] at answer
          | some value =>
            simp [stable, moves, other, result] at answer
            have tree : build mc fc b = .pass (build mc fc (pass b)) := by
              rw [build.eq_def]
              split
              next eq => rw [moves] at eq; cases eq
              next => simp [other]
            rw [tree, ← answer]
            exact .pass (passCall moves other value result)
      | cons first rest =>
        cases cached : (PvEntryMir.generated mc fc b paired mem).wipeout.join with
        | some square =>
          obtain ⟨m, member, wipe⟩ := PvEntryMir.cached_wipe mc fc b paired mem (square := square) (by
            cases cache : (PvEntryMir.generated mc fc b paired mem).wipeout <;> simp_all)
          simp [stable, moves, cached] at answer
          rw [← answer]
          exact .cutoff mc fc b (.wipeout m member wipe)
        | none =>
          cases lookup : parent b depth beta checked attempts with
          | some tt =>
            simp [stable, moves, cached, lookup] at answer
            rw [← answer]
            exact .cutoff mc fc b (parentExecution b depth alpha beta checked attempts observed lookup)
          | none =>
            let q : PvsEnhancedCutoff.Query := ⟨depth, exactSelectivity, true, ⟨alpha.raw⟩⟩
            by_cases level : 14 ≤ depth.toNat
            · cases hit : PvsEnhancedCutoff.scan q (reads.map ChildLookupFrom.observation) with
              | none =>
                dsimp [q] at hit
                simp [stable, moves, cached, lookup, level, etc, hit] at answer
              | some found =>
                dsimp [q] at hit
                simp [stable, moves, cached, lookup, level, etc, hit] at answer
                rw [← answer, null]
                exact .cutoff mc fc b (.enhanced q reads found rfl rfl hit)
            · simp [stable, moves, cached, lookup, level] at answer

open PvsBoardTree (GeneratedSquare)
open MoveList.Ordering (ttMoveValue sortByKeyDesc swapList)
open TranspositionTable

def probeHint (b : Position) (attempts : List ProbeRetry.RawAttempt) : Option Bitboard.Square :=
  ReaderMir.bestMove (ProbeRetry.probe (attempts.map (ProbeRetry.observe (key b))))

def checkedHint (mc : X86Config) (b : Position) (sq : Option Bitboard.Square) : Option (GeneratedSquare mc b.val) := do
  let raw ← sq
  if valid : (Board.getMoves mc b.val).getLsbD raw.val = true then some ⟨raw, valid⟩ else none

theorem checkedHint_present (mc : X86Config) (b : Position) (hint : Option Bitboard.Square) :
    (checkedHint mc b hint).isSome = hint.isSome ↔
      ∀ sq, hint = some sq → (Board.getMoves mc b.val).getLsbD sq.val = true := by
  cases hint with
  | none => simp [checkedHint]
  | some sq =>
    cases legal : (Board.getMoves mc b.val).getLsbD sq.val <;> simp [checkedHint,legal]

theorem observedHint (mc : X86Config) (b : Position) (reads : ProbeReads history attempts)
    (hints : TableReads.LegalHints history) :
    (checkedHint mc b (probeHint b attempts)).map Subtype.val = probeHint b attempts := by
  cases probe : ProbeRetry.probe (attempts.map (ProbeRetry.observe (key b))) with
  | miss => simp [probeHint, probe, ReaderMir.bestMove, checkedHint]
  | hit slot entry =>
    cases decoded : Packing.decodeBestMove entry.bestMove with
    | none => simp [probeHint, probe, ReaderMir.bestMove, decoded, checkedHint]
    | some sq =>
      obtain ⟨move, _, rfl⟩ := (reads.legalHint hints probe).generated (mc := mc) decoded
      simp [probeHint, probe, ReaderMir.bestMove, decoded, checkedHint, move.property]

def order (key : Move → Int) [DecidableEq Move] (hint : Option Move) (moves : List Move) : List Move :=
  match moves, hint with
  | [a, b], some sq => if a = sq then [a, b] else swapList [a, b] 0 1
  | a :: b :: rest, _ =>
    sortByKeyDesc (fun m => if some m = hint then ttMoveValue else key m) (a :: b :: rest)
  | _, _ => moves

theorem orderHint {Move : Type} [DecidableEq Move] (key : Move → Int) (moves : List Move) (sq : Move)
    (member : sq ∈ moves) (low : ∀ m ∈ moves, m ≠ sq → key m < ttMoveValue) :
    (order key (some sq) moves).Perm moves ∧ (order key (some sq) moves).head? = some sq := by
  have sorted : MoveList.Ordering.TtPresearchOrder id key sq moves (order key (some sq) moves) := by
    cases moves with
    | nil => cases member
    | cons a rest =>
      cases rest with
      | nil => rfl
      | cons b rest =>
        cases rest with
        | nil => rfl
        | cons c rest =>
          simpa [order, MoveList.Ordering.TtPresearchOrder] using
            MoveList.Ordering.sortContract_proven Move (fun m => if m = sq then ttMoveValue else key m) (a :: b :: c :: rest)
  simpa using MoveList.Ordering.presearchedOrderContract_proven Move Move id moves (order key (some sq) moves)
    sq key ⟨sq, member, rfl⟩ low sorted

open PvsMoveLoop (ChildSearch MoveCalls moveScore scoreInf)
open Prefix.Serial (Run Outcome)
open PvsPostpone (Queue)

def PresearchSuffix (done : PvsMoveLoop.ChildRelation Move) (search : ChildSearch Move) (alpha beta : Int)
    (ordered : List Move) (outcome : Outcome) : Presearch Move → Prop
  | .storeLower _ score => outcome = .returned score
  | .owner idx best =>
    ∃ visited remaining, Run done search false beta ⟨idx, ordered.drop idx, []⟩ ⟨alpha, best⟩ visited remaining outcome

theorem presearchRun [DecidableEq Move] (key : Move → Int) (done : PvsMoveLoop.ChildRelation Move)
    (search : ChildSearch Move) (first : Move) (rest : List Move) (hint : Option Move) (alpha beta : Int)
    (null : beta = alpha + 1) (legal : ∀ m, hint = some m → m ∈ first :: rest)
    (low : ∀ m sq, hint = some sq → m ∈ first :: rest → m ≠ sq → key m < ttMoveValue)
    (calls : (first :: rest).length = 1 ∨ hint.isSome →
      done (hint.getD first) false (-(alpha + 1)) (-alpha) (search (hint.getD first) false (-(alpha + 1)) (-alpha)))
    (upper : (first :: rest).length = 1 ∨ hint.isSome → search (hint.getD first) false (-(alpha + 1)) (-alpha) < scoreInf)
    (outcome : Outcome)
    (suffix : PresearchSuffix done search alpha beta (order key hint (first :: rest)) outcome
      (presearch (first :: rest).length hint first alpha beta (fun m => search m false))) :
    ∃ visited remaining, Run done search false beta ⟨0, order key hint (first :: rest), []⟩
      ⟨alpha, -scoreInf⟩ visited remaining outcome := by
  by_cases used : (first :: rest).length = 1 ∨ hint.isSome
  · let m := hint.getD first
    have front : (order key hint (first :: rest)).head? = some m := by
      cases h : hint with
      | none =>
        cases rest with
        | nil => simp [m, order, h]
        | cons => simp [h] at used
      | some sq => simpa [m, h] using (orderHint key _ sq (legal _ h) (fun x => low x sq h)).2
    cases eq : order key hint (first :: rest) with
    | nil => simp [eq] at front
    | cons actual tail =>
      rw [eq] at front
      cases Option.some.inj front
      have invoked : MoveCalls done search false true m alpha beta := by simpa [MoveCalls, m] using calls used
      have sentinel : -scoreInf < moveScore false true search m alpha beta := by
        simp only [moveScore, Bool.false_and, Bool.false_eq_true, ↓reduceIte, false_and]
        have up : search m false (-(alpha + 1)) (-alpha) < scoreInf := upper used
        omega
      simp only [presearch, Bool.or_eq_true, decide_eq_true_eq, ite_eq_left used] at suffix
      change PresearchSuffix done search alpha beta _ outcome
        (if (decide (alpha < -search m false (-(alpha + 1)) (-alpha)) &&
          decide (beta ≤ -search m false (-(alpha + 1)) (-alpha))) then _ else _) at suffix
      by_cases cut : beta ≤ -(search m false (-(alpha + 1)) (-alpha))
      · have raised : alpha < -(search m false (-(alpha + 1)) (-alpha)) := by omega
        simp [cut, raised, PresearchSuffix] at suffix
        subst outcome
        exact ⟨_, _, Run.presearchCutoff m tail null invoked sentinel (by simpa [moveScore] using cut)⟩
      · simp [cut, PresearchSuffix, eq] at suffix
        obtain ⟨visited, remaining, suffix⟩ := suffix
        exact ⟨_, _, Run.presearchContinues m tail null invoked sentinel (by simpa [moveScore] using Int.lt_of_not_ge cut)
          (by simpa [moveScore] using suffix)⟩
  · simp only [presearch, Bool.or_eq_true, decide_eq_true_eq, ite_eq_right used] at suffix
    simpa [PresearchSuffix, scoreInf] using suffix

theorem physicalMoves (mc : X86Config) (fc : FlipConfig) (b : Position) (paired : Bool)
    (mem : MoveList.Initialization.Memory Move.Move) (i : Nat) (valid : i < (legalSquares mc b.val).length) :
    (PvEntryMir.generated mc fc b paired mem).slots i = some (MoveList.Generation.entry fc b.val
      (legalSquares mc b.val)[i].val) := by
  have values := PvsBoardTree.legalSquares_values mc b.val
  change (legalSquares mc b.val).map Subtype.val = MoveList.Generation.squares (Board.getMoves mc b.val) at values
  have same : MoveList.Generation.records fc b.val (Board.getMoves mc b.val) =
      (legalSquares mc b.val).map (fun m => MoveList.Generation.entry fc b.val m.val) := by
    simp only [MoveList.Generation.records, ← values, List.map_map, Function.comp_def]
  have slot := (MoveList.Generation.generationContract_proven mc fc b.val (Board.getMoves mc b.val)
    false paired mem b.property (fun _ h => h) (by simp)).2.1 i
  change (PvEntryMir.generated mc fc b paired mem).slots i = _ at slot
  simpa [same, valid] using slot

theorem orderPerm {Move : Type} [DecidableEq Move] (key : Move → Int) (moves : List Move) (hint : Option Move)
    (legal : ∀ m, hint = some m → m ∈ moves)
    (low : ∀ m sq, hint = some sq → m ∈ moves → m ≠ sq → key m < ttMoveValue) :
    (order key hint moves).Perm moves := by
  cases hint with
  | some sq => exact (orderHint key _ sq (legal _ rfl) (fun m => low m sq rfl)).1
  | none =>
    cases moves with
    | nil => exact .refl _
    | cons a rest =>
      cases rest with
      | nil => exact .refl _
      | cons b rest =>
        cases rest <;> simpa [order] using MoveList.Ordering.sortByKeyDesc_perm key _

theorem presearchExecution (mc : X86Config) (fc : FlipConfig) (b : Position) (first : GeneratedSquare mc b.val)
    (rest : List (GeneratedSquare mc b.val)) (moves : legalSquares mc b.val = first :: rest)
    (attempts : List ProbeRetry.RawAttempt)
    (seen : (checkedHint mc b (probeHint b attempts)).isSome = (probeHint b attempts).isSome)
    (priority : GeneratedSquare mc b.val → Int) (low : ∀ m ∈ first :: rest, priority m < ttMoveValue)
    (search : ChildSearch (GeneratedSquare mc b.val)) (alpha beta : Int) (null : beta = alpha + 1)
    (children : ChildResults)
    (calls : let hint := checkedHint mc b (probeHint b attempts)
      (first :: rest).length = 1 ∨ (probeHint b attempts).isSome →
        let m := hint.getD first
        let score := search m false (-(alpha + 1)) (-alpha)
        children .serial (build mc fc (PvsBoardTree.play mc fc b m).val) false (-(alpha + 1)) (-alpha) score ∧
          score < scoreInf)
    (suffix : let hint := checkedHint mc b (probeHint b attempts)
      PresearchSuffix (fun m => children .serial (build mc fc (PvsBoardTree.play mc fc b m).val)) search alpha beta
        (order priority hint (first :: rest)) (.returned score)
        (presearch (first :: rest).length hint first alpha beta (fun m => search m false))) :
    ExecutionWithUses exit children (build mc fc b) false alpha beta score := by
  let hint := checkedHint mc b (probeHint b attempts)
  have legal : ∀ m, hint = some m → m ∈ first :: rest := by
    intro m _
    obtain ⟨v, present, value⟩ := (PvsBoardTree.legalSquares_complete mc b.val m.val).mpr
      ((Board.getMoves_exact mc b.val _).mp m.property)
    have eq := Subtype.ext value
    rwa [eq, moves] at present
  have bounded := fun m (sq : GeneratedSquare mc b.val) (_ : hint = some sq)
    (mem : m ∈ first :: rest) (_ : m ≠ sq) => low m mem
  let done := fun m => children .serial (build mc fc (PvsBoardTree.play mc fc b m).val)
  obtain ⟨visited, remaining, run⟩ := presearchRun priority done search first rest hint alpha beta null legal bounded
    (fun used => (calls (seen ▸ used)).1) (fun used => (calls (seen ▸ used)).2) (.returned score) suffix
  have perm := orderPerm priority (first :: rest) hint legal bounded
  rw [build.eq_def]
  split
  next head tail h =>
    obtain ⟨rfl, rfl⟩ := List.cons.inj (moves.symm.trans h)
    apply ExecutionWithUses.ofSerialRun (fun m => build mc fc (PvsBoardTree.play mc fc b m).val) _ rfl run
    simpa [PvsPostpone.Queue.pending] using perm.map (fun m => build mc fc (PvsBoardTree.play mc fc b m).val)
  next h => rw [moves] at h; cases h

theorem publishedScore (published : Publications mc fc history) :
    ∀ b entry, (b, entry) ∈ history → Cutoffs.ExactEndgame entry → -32767 < entry.score.toInt := by
  induction published with
  | initial => simp
  | append prior write ih =>
    intro b entry member exact
    rcases List.mem_append.mp member with earlier | newest
    · exact ih _ _ earlier exact
    · cases List.mem_singleton.mp newest
      cases write with
      | other _ outside => contradiction
      | penalized priorEntry source _ => exact ih _ priorEntry (by assumption) exact
      | searched pv alpha beta score move depth generation ttPv previousKey previous written window result saved =>
        rw [(Cutoffs.storeContract_proven _ _ _ _ pv alpha beta score move depth generation ttPv
          previousKey previous _ prior.sound window result saved).1]
        exact result.lower

theorem parentGateReturns (b : Position) (depth : Depth) (beta : ScoreRefresh.Score) (checked : Bool)
    (published : Publications mc fc history) (reads : ProbeReads history attempts)
    (probe : ProbeRetry.probe (attempts.map (ProbeRetry.observe (key b))) = .hit slot entry) :
    ∃ cut, canCut checked entry ⟨beta.raw⟩ depth exactSelectivity true = .ok cut := by
  obtain ⟨attempt, tried, sampled, occupied⟩ := ProbeRetry.contract_proven.2 _ _ _ _ probe
  have previous := reads.published.matches slot entry attempt tried sampled occupied
  rw [canCut_contract]
  split
  next eligible =>
    have lower := publishedScore published _ _ previous (Cutoffs.exactEndgame_of_eligible eligible)
    simp [show entry.score.toInt ≠ -32768 by omega]
  next => exact ⟨false, rfl⟩

def entryAction (mc : X86Config) (fc : FlipConfig) (b : Position) (depth : Depth)
    (alpha beta : ScoreRefresh.Score) (paired checked : Bool) (mem : MoveList.Initialization.Memory Move.Move)
    (attempts : List ProbeRetry.RawAttempt) (reads : List (ChildLookupFrom mc fc b source)) : Action :=
  let moves := PvEntryMir.generated mc fc b paired mem
  dispatch depth.toNat (EndgameStability.tryCutoff b.val depth.toNat ⟨(toDiscDiff alpha).raw.toInt⟩).isSome
    (moves.len.getD 0 != 0) (Board.hasLegalMoves mc (pass b).val) moves.wipeout.join.isSome
    (parent b depth beta checked attempts).isSome (etc ⟨depth, exactSelectivity, true, ⟨alpha.raw⟩⟩ reads).isSome

abbrev Child := Position → Depth → Bool → Int → Int → Option Int

def observe (mc : X86Config) (fc : FlipConfig) (b : Position) (depth : Depth) (cut : Bool) (child : Child) :
    ChildSearch (GeneratedSquare mc b.val) :=
  fun m _ a z => (child (PvsBoardTree.play mc fc b m).val (depth - 1) (!cut) a z).getD 0

def resumeOwner (owner : Nat → Int → Option Int) : Presearch Move → Option Int
  | .storeLower _ value => some value
  | .owner idx best => owner idx best

def frame (mc : X86Config) (fc : FlipConfig) (b : Position) (depth : Depth) (alpha beta : ScoreRefresh.Score)
    (paired checked cut : Bool) (mem : MoveList.Initialization.Memory Move.Move)
    (attempts : List ProbeRetry.RawAttempt) (reads : List (ChildLookupFrom mc fc b source))
    (priority : GeneratedSquare mc b.val → Int) (child : Child)
    (owner : List (GeneratedSquare mc b.val) → Nat → Int → Option Int) : Option Int :=
  match entryAction mc fc b depth alpha beta paired checked mem attempts reads with
  | .rest => do
    let first :: rest := legalSquares mc b.val | none
    let hint := checkedHint mc b (probeHint b attempts)
    -- Rust plays a present square directly. A legal-tree replay requires that same square.
    guard (hint.isSome = (probeHint b attempts).isSome)
    let list := first :: rest
    let a := alpha.raw.toInt
    let z := beta.raw.toInt
    if list.length = 1 || (probeHint b attempts).isSome then
      let _ ← child (PvsBoardTree.play mc fc b (hint.getD first)).val (depth - 1) (!cut) (-(a + 1)) (-a)
    resumeOwner (owner (order priority hint list)) (presearch list.length hint first a z
      (fun m => observe mc fc b depth cut child m false))
  | _ => runPrefix mc fc b depth alpha beta paired checked cut mem attempts reads child

/-- On the direct-play branch, an illegal decoded square has no legal-tree replay.
    This is a condition on the verifier, not an extra check in the Rust search. -/
theorem frame_illegal_hint (mc : X86Config) (fc : FlipConfig) (b : Position) (depth : Depth)
    (alpha beta : ScoreRefresh.Score) (paired checked cut : Bool) (mem : MoveList.Initialization.Memory Move.Move)
    (attempts : List ProbeRetry.RawAttempt) (reads : List (ChildLookupFrom mc fc b source))
    (priority : GeneratedSquare mc b.val → Int) (child : Child)
    (owner : List (GeneratedSquare mc b.val) → Nat → Int → Option Int)
    (direct : entryAction mc fc b depth alpha beta paired checked mem attempts reads = .rest)
    (hit : probeHint b attempts = some sq) (illegal : (Board.getMoves mc b.val).getLsbD sq.val = false) :
    frame mc fc b depth alpha beta paired checked cut mem attempts reads priority child owner = none := by
  unfold frame
  rw [direct]
  have mismatch : ¬ (checkedHint mc b (probeHint b attempts)).isSome = (probeHint b attempts).isSome := by
    intro eq
    have valid := (checkedHint_present mc b _).mp eq sq hit
    rw [illegal] at valid
    contradiction
  cases legalSquares mc b.val <;> simp only [guard,ite_eq_right mismatch] <;> rfl

theorem frameCases (mc : X86Config) (fc : FlipConfig) (b : Position) (depth : Depth)
    (alpha beta : ScoreRefresh.Score) (paired checked cut : Bool) (mem : MoveList.Initialization.Memory Move.Move)
    (attempts : List ProbeRetry.RawAttempt) (reads : List (ChildLookupFrom mc fc b source))
    (priority : GeneratedSquare mc b.val → Int) (child : Child)
    (owner : List (GeneratedSquare mc b.val) → Nat → Int → Option Int)
    (full : ExactDepth.Full b depth) (null : beta.raw.toInt = alpha.raw.toInt + 1)
    (observed : ProbeReadsFrom source attempts)
    (low : ∀ m ∈ legalSquares mc b.val, priority m < ttMoveValue)
    (passCall : legalSquares mc b.val = [] → Board.hasLegalMoves mc (pass b).val = true → ∀ value,
      child (pass b) depth (!cut) (-beta.raw.toInt) (-alpha.raw.toInt) = some value →
      children .serial (build mc fc (pass b)) false (-beta.raw.toInt) (-alpha.raw.toInt) value)
    (call : ∀ m value, child (PvsBoardTree.play mc fc b m).val (depth - 1) (!cut)
      (-(alpha.raw.toInt + 1)) (-alpha.raw.toInt) = some value →
      children .serial (build mc fc (PvsBoardTree.play mc fc b m).val)
        false (-(alpha.raw.toInt + 1)) (-alpha.raw.toInt) value ∧ value < scoreInf)
    (ownerCase : ∀ first rest, legalSquares mc b.val = first :: rest →
      let hint := checkedHint mc b (probeHint b attempts)
      let s := observe mc fc b depth cut child
      let a := alpha.raw.toInt
      let z := beta.raw.toInt
      hint.isSome = (probeHint b attempts).isSome →
      ((first :: rest).length = 1 ∨ (probeHint b attempts).isSome → let m := hint.getD first
        children .serial (build mc fc (PvsBoardTree.play mc fc b m).val) false (-(a+1)) (-a)
          (s m false (-(a+1)) (-a)) ∧ s m false (-(a+1)) (-a) < scoreInf) →
      ∀ idx best, presearch (first :: rest).length hint first a z (fun m => s m false) = .owner idx best →
        owner (order priority hint (first :: rest)) idx best = some score →
        ExecutionWithUses (CausalReplay.PrefixExitFrom mc fc source) children (build mc fc b) false a z score)
    (answer : frame mc fc b depth alpha beta paired checked cut mem attempts reads priority child owner = some score) :
    ExecutionWithUses (CausalReplay.PrefixExitFrom mc fc source) children (build mc fc b)
      false alpha.raw.toInt beta.raw.toInt score := by
  unfold frame at answer
  split at answer
  next =>
    cases moves : legalSquares mc b.val with
    | nil => simp [moves] at answer
    | cons first rest =>
      let hint := checkedHint mc b (probeHint b attempts)
      let a := alpha.raw.toInt
      let z := beta.raw.toInt
      let s := observe mc fc b depth cut child
      let m := hint.getD first
      let looked := child (PvsBoardTree.play mc fc b m).val (depth - 1) (!cut) (-(a + 1)) (-a)
      let step := presearch (first :: rest).length hint first a z (fun m => s m false)
      let ordered := order priority hint (first :: rest)
      rw [moves] at answer
      by_cases seen : hint.isSome = (probeHint b attempts).isSome
      case neg => simp [hint] at seen; simp [guard,seen] at answer; cases answer
      simp only [guard,show (checkedHint mc b (probeHint b attempts)).isSome = _ from seen,
        ↓reduceIte,pure,bind,Option.bind] at answer
      change (do
        if (first :: rest).length = 1 || (probeHint b attempts).isSome then let _ ← looked
        resumeOwner (owner ordered) step) = some score at answer
      have consumed : (first :: rest).length = 1 ∨ (probeHint b attempts).isSome → ∃ v, looked = some v := by
        intro used
        cases eq : looked with
        | none => simp only [Bool.or_eq_true, decide_eq_true_eq, ite_eq_left used, eq] at answer; cases answer
        | some v => exact ⟨v, rfl⟩
      have returned : resumeOwner (owner ordered) step = some score := by
        by_cases used : (first :: rest).length = 1 ∨ (probeHint b attempts).isSome
        · obtain ⟨v, eq⟩ := consumed used
          simp only [Bool.or_eq_true, decide_eq_true_eq, ite_eq_left used, eq] at answer
          simpa using answer
        · simpa only [Bool.or_eq_true, decide_eq_true_eq, ite_eq_right used] using answer
      have firstCall : (first :: rest).length = 1 ∨ (probeHint b attempts).isSome →
          children .serial (build mc fc (PvsBoardTree.play mc fc b m).val) false (-(a+1)) (-a)
            (s m false (-(a+1)) (-a)) ∧ s m false (-(a+1)) (-a) < scoreInf := by
        intro used
        obtain ⟨v, eq⟩ := consumed used
        have seen := call m v eq
        show children .serial _ false _ _ (s m false (-(a + 1)) (-a)) ∧ s m false (-(a + 1)) (-a) < scoreInf
        have same : s m false (-(a + 1)) (-a) = v := by change looked.getD 0 = v; simp [eq]
        rwa [same]
      cases kind : step with
      | owner idx best =>
        exact ownerCase first rest moves seen firstCall idx best kind (by simpa only [kind, resumeOwner] using returned)
      | storeLower selected value =>
        apply presearchExecution mc fc b first rest moves attempts seen
          priority (moves ▸ low) s a z null children firstCall
        change PresearchSuffix _ s a z ordered (.returned score) step
        simp only [kind, resumeOwner, Option.some.injEq] at returned
        rw [kind, returned]
        rfl
  next =>
    exact prefixExecution mc fc b depth alpha beta paired checked cut mem attempts reads child full null
      observed passCall answer

theorem frameExecution (mc : X86Config) (fc : FlipConfig) (b : Position) (depth : Depth)
    (alpha beta : ScoreRefresh.Score) (paired checked cut : Bool) (mem : MoveList.Initialization.Memory Move.Move)
    (attempts : List ProbeRetry.RawAttempt) (reads : List (ChildLookupFrom mc fc b source))
    (priority : GeneratedSquare mc b.val → Int) (child : Child)
    (owner : List (GeneratedSquare mc b.val) → Nat → Int → Option Int)
    (full : ExactDepth.Full b depth) (null : beta.raw.toInt = alpha.raw.toInt + 1)
    (observed : ProbeReadsFrom source attempts)
    (low : ∀ m ∈ legalSquares mc b.val, priority m < ttMoveValue)
    (passCall : legalSquares mc b.val = [] → Board.hasLegalMoves mc (pass b).val = true → ∀ value,
      child (pass b) depth (!cut) (-beta.raw.toInt) (-alpha.raw.toInt) = some value →
      children .serial (build mc fc (pass b)) false (-beta.raw.toInt) (-alpha.raw.toInt) value)
    (call : ∀ m value, child (PvsBoardTree.play mc fc b m).val (depth - 1) (!cut)
      (-(alpha.raw.toInt + 1)) (-alpha.raw.toInt) = some value →
      children .serial (build mc fc (PvsBoardTree.play mc fc b m).val)
        false (-(alpha.raw.toInt + 1)) (-alpha.raw.toInt) value ∧ value < scoreInf)
    (suffix : ∀ ordered idx best value, owner ordered idx best = some value →
      ∃ visited remaining, Run (fun m => children .serial (build mc fc (PvsBoardTree.play mc fc b m).val))
        (observe mc fc b depth cut child) false beta.raw.toInt ⟨idx, ordered.drop idx, []⟩
        ⟨alpha.raw.toInt, best⟩ visited remaining (.returned value))
    (answer : frame mc fc b depth alpha beta paired checked cut mem attempts reads priority child owner = some score) :
    ExecutionWithUses (CausalReplay.PrefixExitFrom mc fc source) children (build mc fc b)
      false alpha.raw.toInt beta.raw.toInt score := by
  apply frameCases mc fc b depth alpha beta paired checked cut mem attempts reads priority child owner full null
    observed low passCall call _ answer
  intro first rest moves hint s a z seen pre idx best start result
  apply presearchExecution mc fc b first rest moves attempts seen
    priority (moves ▸ low) s a z null children pre
  change PresearchSuffix _ s a z _ (.returned score) (presearch _ hint first a z (fun m => s m false))
  rw [start]
  exact suffix _ _ _ _ result

end NeuralReversi.PvsParallel.GeneralNonPvMir
