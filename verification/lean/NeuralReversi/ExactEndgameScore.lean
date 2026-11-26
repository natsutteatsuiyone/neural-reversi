import NeuralReversi.PvsParallel.RootMultiPV
import NeuralReversi.SearchFallback

namespace NeuralReversi.ExactEndgameScore

open Bitboard (Square)
open Bitboard.Moves (X86Config)
open PvsBoardTree (Position minimax)
open PvsParallel
open SearchPublication (Row Stamp Snapshot capture)

abbrev FlipConfig := Flip.Dispatch.X86Config

theorem ranked_first (mc : X86Config) (fc : FlipConfig) (board : Position)
    (roots : List Square) (first : RootAspiration.Pick) (rest : List RootAspiration.Pick)
    (covers : Root.Covers mc board roots)
    (ranking : RootMultiPV.Ranked (Root.squareValue mc fc board) roots (first :: rest)) :
    first.square ∈ roots ∧ first.score = minimax mc fc board := by
  obtain ⟨found, chosen, maximal, _⟩ := ranking
  have positive : 0 < roots.length := by
    cases roots with
    | nil => simp at found
    | cons => simp
  have root := Root.value_zero (fc := fc) covers positive
  have lower : first.score ≤ Root.value mc fc board roots 0 := by
    rw [chosen]
    exact Root.square_value_le covers (by simpa using found)
  have upper : Root.value mc fc board roots 0 ≤ first.score := by
    unfold Root.value
    apply PvsMoveLoop.bestValue_le
    · rw [chosen]
      obtain ⟨move, _, boardSquare⟩ := List.mem_map.mp
        ((Root.choices_squares_mem (idx := 0) covers first.square).mpr (by simpa using found))
      rw [← boardSquare, Root.square_value_eq_child]
      have child := PvsMoveLoop.minimax_range
        (PvsBoardTree.build mc fc (PvsBoardTree.play mc fc board move).val)
      change -PvsMoveLoop.scoreInf ≤ -minimax mc fc (PvsBoardTree.play mc fc board move).val
      change -PvsMoveLoop.scoreMax ≤ minimax mc fc (PvsBoardTree.play mc fc board move).val ∧
        minimax mc fc (PvsBoardTree.play mc fc board move).val ≤ PvsMoveLoop.scoreMax at child
      unfold PvsMoveLoop.scoreInf PvsMoveLoop.scoreMax at *
      omega
    · intro move included
      rw [← Root.square_value_eq_child move]
      have member := (Root.choices_squares_mem covers move.val).mp (List.mem_map_of_mem included)
      exact maximal move.val (by simpa using member)
  exact ⟨found, (Int.le_antisymm lower upper).trans root⟩

def RankedScoreContract : Prop :=
  ∀ (mc : X86Config) (fc : FlipConfig) (board : Position) (roots : List Square)
    (count : Nat) (observe : RootMultiPV.Observations) (seed : RootMultiPV.Seed),
    Root.Covers mc board roots → 0 < count → count ≤ roots.length →
    RootMultiPV.Matched mc fc board observe seed →
    ∃ batch first, RootMultiPV.run observe seed count 0 roots = some batch ∧
      batch.picks.head? = some first ∧ first.square ∈ roots ∧ first.score = minimax mc fc board

theorem rankedScoreContract_proven : RankedScoreContract := by
  intro mc fc board roots count observe seed covers positive enough matched
  obtain ⟨batch, run, length, _, _, ranking, _⟩ :=
    RootMultiPV.contract_proven mc fc board roots count observe seed covers enough matched
  cases picks : batch.picks with
  | nil => simp [picks] at length; omega
  | cons first rest =>
    rw [picks] at ranking
    exact ⟨batch, first, run, by simp [picks], ranked_first mc fc board roots first rest covers ranking⟩

def SnapshotScoreContract : Prop :=
  ∀ (rows : List Row) (stamp : Stamp) (truth : Int) (nodes : Nat),
    (∀ row ∈ rows, row.score ≤ truth) →
    (∃ row ∈ rows, row.score = truth) →
    ∃ result, SearchFallback.fromSnapshot (capture rows stamp true) nodes = some result ∧
      result.best.score = truth ∧ result.endgame = true ∧ result.stamp = stamp

theorem snapshotScoreContract_proven : SnapshotScoreContract := by
  intro rows stamp truth nodes bounds existsBest
  obtain ⟨_, _, permutation, sorted, _⟩ := SearchPublication.captureContract_proven rows stamp true
  obtain ⟨winner, original, score⟩ := existsBest
  have winnerSaved := permutation.mem_iff.mpr original
  let saved := capture rows stamp true
  cases savedRows : saved.rows with
  | nil =>
    change winner ∈ saved.rows at winnerSaved
    simp [savedRows] at winnerSaved
  | cons first rest =>
    change saved.rows.Perm rows at permutation
    change saved.rows.Pairwise _ at sorted
    change winner ∈ saved.rows at winnerSaved
    rw [savedRows] at permutation sorted winnerSaved
    have firstBound := bounds first (permutation.mem_iff.mp (by simp))
    have winnerBound : winner.score ≤ first.score := by
      rcases List.mem_cons.mp winnerSaved with rfl | later
      · exact Int.le_refl _
      · exact (List.pairwise_cons.mp sorted).1 winner later
    have scoreExact : first.score = truth := by omega
    let report : SearchFallback.Report := ⟨first, first :: rest, stamp, true, nodes⟩
    refine ⟨report, ?_, scoreExact, rfl, rfl⟩
    change SearchFallback.fromSnapshot saved nodes = some report
    rw [SearchFallback.fromSnapshot, savedRows]
    rfl

def SingleResultContract : Prop :=
  ∀ (mc : X86Config) (fc : FlipConfig) (board : Position) (roots : List Square)
    (alpha beta : Int) (search : RootAspiration.Search) (pv : Square → List Square) (depth nodes : Nat),
    Root.Covers mc board roots → 0 < roots.length → Aspiration.Window alpha beta →
    RootAspiration.SoundSearch mc fc board roots 0 search →
    (∀ i a b, Aspiration.Window a b → (search i a b).aborted = false) →
    ∃ accepted result,
      Aspiration.runWith search 11 0 alpha beta 256 = some accepted ∧
      let rows := roots.map (fun square => (⟨square, accepted.trial.payload square, pv square⟩ : Row))
      SearchFallback.fromSnapshot (capture rows ⟨depth, 5⟩ true) nodes = some result ∧
        result.best.score = minimax mc fc board

private theorem singleResult_from_line {mc : X86Config} {fc : FlipConfig} {board : Position}
    (covers : Root.Covers mc board roots) (pv : Square → List Square) (depth nodes : Nat)
    (finished : RootAspiration.finish roots 0 accepted = some line)
    (scoreAt : accepted.trial.payload line.pick.square = line.pick.score)
    (valid : RootAspiration.LineSound (Root.squareValue mc fc board) roots 0 line) :
    let rows := roots.map (fun square => (⟨square, accepted.trial.payload square, pv square⟩ : Row))
    ∃ result, SearchFallback.fromSnapshot (capture rows ⟨depth, 5⟩ true) nodes = some result ∧
      result.best.score = minimax mc fc board := by
  have ranked : RootMultiPV.Ranked (Root.squareValue mc fc board) roots [line.pick] :=
    ⟨by simpa using valid.legal, valid.exactScore, by simpa using valid.maximal, trivial⟩
  have best := (ranked_first mc fc board roots line.pick [] covers ranked).2
  have order : line.order = RootOrder.sortFrom roots 0 accepted.trial.payload := by
    simp only [RootAspiration.finish, List.drop_zero] at finished
    split at finished
    · contradiction
    · cases Option.some.inj finished
      rfl
  have descending := (RootOrder.sortContract_proven roots 0 accepted.trial.payload (by omega)).2.2.2.1
  have head : line.order.head? = some line.pick.square := by simpa using valid.chosen
  have scores : ∀ square ∈ roots, accepted.trial.payload square ≤ minimax mc fc board := by
    intro square member
    have present := valid.order.mem_iff.mpr member
    rw [← best, ← scoreAt]
    have sorted : line.order.Pairwise (fun a b => accepted.trial.payload b ≤ accepted.trial.payload a) := by
      rw [order]
      simpa using descending
    cases ordering : line.order with
    | nil => simp [ordering] at head
    | cons front remaining =>
      have leader : front = line.pick.square := by simpa [ordering] using head
      rw [ordering] at present sorted
      rcases List.mem_cons.mp present with rfl | later
      · rw [leader]
        exact Int.le_refl _
      · rw [← leader]
        exact (List.pairwise_cons.mp sorted).1 square later
  obtain ⟨result, resultEq, scoreEq, _, _⟩ := snapshotScoreContract_proven
    (roots.map (fun square => ⟨square, accepted.trial.payload square, pv square⟩))
    ⟨depth, 5⟩ (minimax mc fc board) nodes
    (by
      intro row present
      obtain ⟨square, member, rfl⟩ := List.mem_map.mp present
      exact scores square member)
    ⟨⟨line.pick.square, accepted.trial.payload line.pick.square, pv line.pick.square⟩,
      List.mem_map_of_mem ranked.1, scoreAt.trans best⟩
  exact ⟨result, resultEq, scoreEq⟩

theorem singleResult_of_interior {mc : X86Config} {fc : FlipConfig} {board : Position}
    (covers : Root.Covers mc board roots) (inside : 0 < roots.length)
    {accepted : Aspiration.Accepted (Square → Int)}
    (interior : accepted.alpha < accepted.trial.score ∧ accepted.trial.score < accepted.beta)
    (attempt : RootOrder.SoundAttempt mc fc board roots 0 accepted.alpha accepted.beta
      accepted.trial.score accepted.trial.payload)
    (pv : Square → List Square) (depth nodes : Nat) :
    let rows := roots.map (fun square => (⟨square, accepted.trial.payload square, pv square⟩ : Row))
    ∃ result, SearchFallback.fromSnapshot (capture rows ⟨depth, 5⟩ true) nodes = some result ∧
      result.best.score = minimax mc fc board := by
  obtain ⟨line, finished, _, scoreAt, valid⟩ := RootAspiration.line_of_interior inside interior attempt
  exact singleResult_from_line covers pv depth nodes finished scoreAt valid

theorem singleResult_of_accepted {mc : X86Config} {fc : FlipConfig} {board : Position}
    (covers : Root.Covers mc board roots) (inside : 0 < roots.length)
    (witness : Aspiration.Witness search start fuel accepted)
    (attempt : RootOrder.SoundAttempt mc fc board roots 0 accepted.alpha accepted.beta
      accepted.trial.score accepted.trial.payload)
    (pv : Square → List Square) (depth nodes : Nat) :
    let rows := roots.map (fun square => (⟨square, accepted.trial.payload square, pv square⟩ : Row))
    ∃ result, SearchFallback.fromSnapshot (capture rows ⟨depth, 5⟩ true) nodes = some result ∧
      result.best.score = minimax mc fc board :=
  singleResult_of_interior covers inside witness.inside attempt pv depth nodes

theorem singleResultContract_proven : SingleResultContract := by
  intro mc fc board roots alpha beta search pv depth nodes covers positive window sound uninterrupted
  obtain ⟨line, accepted, found, _, lineFound, _, scoreAt, valid⟩ :=
    RootAspiration.lineContract_proven mc fc board roots 0 alpha beta search covers positive window
      sound uninterrupted
  have finished : RootAspiration.finish roots 0 accepted = some line := by
    simpa [RootAspiration.runLine, found] using lineFound
  obtain ⟨result, resultEq, scoreEq⟩ := singleResult_from_line covers pv depth nodes finished scoreAt valid
  exact ⟨accepted, result, found, resultEq, scoreEq⟩

def RootResultContract : Prop :=
  ∀ (mc : X86Config) (fc : FlipConfig) (board : Position) (roots : List Square)
    (count : Nat) (observe : RootMultiPV.Observations) (seed : RootMultiPV.Seed)
    (pv : Square → List Square) (depth nodes : Nat),
    Root.Covers mc board roots → 0 < count → count ≤ roots.length →
    RootMultiPV.Matched mc fc board observe seed →
    ∃ batch, RootMultiPV.run observe seed count 0 roots = some batch ∧
      ∀ remaining : List Row,
      (∀ row ∈ remaining, row.score ≤ minimax mc fc board) →
      let rows := batch.picks.map (fun pick => (⟨pick.square, pick.score, pv pick.square⟩ : Row)) ++ remaining
      ∃ result, SearchFallback.fromSnapshot (capture rows ⟨depth, 5⟩ true) nodes = some result ∧
        result.best.score = minimax mc fc board ∧ result.endgame = true ∧ result.stamp = ⟨depth, 5⟩

theorem rootResult_of_ranked {mc : X86Config} {fc : FlipConfig} {board : Position}
    (covers : Root.Covers mc board roots) (picks : List RootAspiration.Pick)
    (nonempty : picks ≠ []) (ranked : RootMultiPV.Ranked (Root.squareValue mc fc board) roots picks)
    (pv : Square → List Square) (depth nodes : Nat) (remaining : List Row)
    (bounded : ∀ row ∈ remaining, row.score ≤ minimax mc fc board) :
    let rows := picks.map (fun pick => (⟨pick.square, pick.score, pv pick.square⟩ : Row)) ++ remaining
    ∃ result, SearchFallback.fromSnapshot (capture rows ⟨depth, 5⟩ true) nodes = some result ∧
      result.best.score = minimax mc fc board ∧ result.endgame = true ∧ result.stamp = ⟨depth, 5⟩ := by
  cases picks with
  | nil => contradiction
  | cons first rest =>
    have order := ranked.ordered
    have exactScore := (ranked_first mc fc board roots first rest covers ranked).2
    apply snapshotScoreContract_proven
    · intro row member
      rcases List.mem_append.mp member with selected | extra
      · obtain ⟨pick, picksMember, rfl⟩ := List.mem_map.mp selected
        rcases List.mem_cons.mp picksMember with rfl | later
        · exact Int.le_of_eq exactScore
        · exact Int.le_trans ((List.pairwise_cons.mp order).1 pick later) (Int.le_of_eq exactScore)
      · exact bounded row extra
    · refine ⟨⟨first.square, first.score, pv first.square⟩, ?_, exactScore⟩
      exact List.mem_append_left _ (List.mem_map_of_mem (by simp))

theorem rootResultContract_proven : RootResultContract := by
  intro mc fc board roots count observe seed pv depth nodes covers positive enough matched
  obtain ⟨batch, run, length, _, _, ranked, _, _⟩ :=
    RootMultiPV.contract_proven mc fc board roots count observe seed covers enough matched
  refine ⟨batch, run, rootResult_of_ranked covers batch.picks ?_ ranked pv depth nodes⟩
  intro empty
  simp [empty] at length
  omega

def noMoveAwareScore (mc : X86Config) (engine : Position → Option Int) (b : Position) : Option Int :=
  if Board.hasLegalMoves mc b.val then engine b
  else if Board.hasLegalMoves mc (PvsBoardTree.pass b).val then
    (engine (PvsBoardTree.pass b)).map (fun score => -score)
  else some (PvsBoardTree.terminalScore b)

def ScoreDispatchContract : Prop :=
  ∀ (mc : X86Config) (fc : FlipConfig) (engine : Position → Option Int) (b : Position),
    (∀ q, Board.hasLegalMoves mc q.val = true → engine q = some (minimax mc fc q)) →
    noMoveAwareScore mc engine b = some (minimax mc fc b)

theorem scoreDispatchContract_proven : ScoreDispatchContract := by
  intro mc fc engine b exact
  unfold noMoveAwareScore
  cases moves : Board.hasLegalMoves mc b.val
  · have stuck := PvsBoardTree.nil_of_stuck moves
    cases other : Board.hasLegalMoves mc (PvsBoardTree.pass b).val
    · rw [PvsBoardTree.terminal_value mc fc b stuck other]
      simp [PvsBoardTree.terminalScore]
    · rw [PvsBoardTree.minimax_pass mc fc b stuck other]
      simp [exact (PvsBoardTree.pass b) other]
  · simp [exact b moves]

end NeuralReversi.ExactEndgameScore
