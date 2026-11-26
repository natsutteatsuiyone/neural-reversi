import NeuralReversi.PvsParallel.Interleaving.Turn
import NeuralReversi.PvsParallel.Interleaving.CausalReads
import NeuralReversi.EndgameSearch.CacheHistory

namespace NeuralReversi.PvsParallel.Interleaving.BoardFrame

open Aspiration (Window SoundResult)
open PvsBoardTree (Position GeneratedSquare legalSquares play pass minimax)
open PvsMoveLoop (Bound bestValue scoreInf scoreMax)
open TranspositionTable (canCut)
open EndgameSearch
open ScoreRefresh (Score)
open CausalTable (Values Lookups)

abbrev Children (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : Position) (prior : Values) :=
  fun m : GeneratedSquare mc b.val => prior (play mc fc b m).val

/-- A score-producing exact frame. The child relation and TT lookup relation are
    filled from earlier events in the same global execution. Worker caches are local. -/
inductive Frame (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (prior : Values) (reads : Lookups)
    (b : Position) : Bool → Int → Int → Int → Prop where
  | terminal (stuck : legalSquares mc b.val = []) (other : Board.hasLegalMoves mc (pass b).val = false) :
      Frame mc fc prior reads b pv alpha beta (PvsBoardTree.terminalScore b).val
  | pass (stuck : legalSquares mc b.val = []) (other : Board.hasLegalMoves mc (pass b).val = true)
      (child : prior (PvsBoardTree.pass b) pv (-beta) (-alpha) s) : Frame mc fc prior reads b pv alpha beta (-s)
  | wipe (m : GeneratedSquare mc b.val) (legal : m ∈ legalSquares mc b.val) (wiped : EndgameCutoffs.wipesOut fc b.val m.val = true) :
      Frame mc fc prior reads b pv alpha beta scoreMax
  | stable (alpha : Score) (result : Stability.Cutoff.DiscScore) (raw : Score) (checked : Bool)
      (cut : EndgameStability.tryCutoff b.val (Board.emptyCount b.val).toNat
        ⟨(ScoreRefresh.toDiscDiff alpha).raw.toInt⟩ = some result)
      (word : ScoreRefresh.fromDiscDiff checked ⟨BitVec.ofInt 32 result.value⟩ = .ok raw) :
      Frame mc fc prior reads b false alpha.raw.toInt (alpha.raw.toInt+1) raw.raw.toInt
  | table (beta : TranspositionTable.Score) (seen : reads b e)
      (hit : canCut checked e beta depth Cutoffs.exactSelectivity true = .ok true) :
      Frame mc fc prior reads b false alpha beta.raw.toInt e.score.toInt
  | enhanced (q : PvsEnhancedCutoff.Query) (probes : List PvsEnhancedCutoff.Probe) (hit : PvsEnhancedCutoff.Hit)
      (exact : q.selectivity = Cutoffs.exactSelectivity) (endgame : q.endgame = true)
      (seen : ∀ probe ∈ probes, ∀ e, probe.2 = some e → ∃ m ∈ legalSquares mc b.val,
        m.val = probe.1 ∧ reads (play mc fc b m).val e)
      (cut : PvsEnhancedCutoff.scan q probes = some hit) :
      Frame mc fc prior reads b false q.alpha.raw.toInt (q.alpha.raw.toInt+1) hit.2
  | fixed (result : ScoreRefresh.DiscScore) (alpha beta raw : Score)
      (solved : EndgameSearch.Direct.FixedDisc target bmi2 b (ScoreRefresh.toDiscDiff alpha).raw result.raw)
      (converted : ScoreRefresh.fromDiscDiff checked result = .ok raw) :
      Frame mc fc prior reads b false alpha.raw.toInt beta.raw.toInt raw.raw.toInt
  | shallow (p : CacheHistory.Pool) (timeline : CacheHistory.Timeline p) (worker atStep : Nat)
      (alpha beta raw : Score) (lo : 5 ≤ (Board.emptyCount b.val).toNat) (hi : (Board.emptyCount b.val).toNat ≤ 6)
      (solved : let h := (timeline.readAt mc fc worker atStep).history
        let result := EndgameSearch.run mc fc (EndgameSearch.orderedOptions target bmi2 mc fc p.address) .shallow b
          (ScoreRefresh.toDiscDiff alpha).raw.toInt h
        ScoreRefresh.fromDiscDiff checked ⟨BitVec.ofInt 32 result.score⟩ = .ok raw) :
      Frame mc fc prior reads b false alpha.raw.toInt beta.raw.toInt raw.raw.toInt
  | cached (p : CacheHistory.Pool) (timeline : CacheHistory.Timeline p) (worker atStep : Nat)
      (alpha beta raw : Score) (s : EndgameSearch.Cancellation.State)
      (localState : s.history = (timeline.readAt mc fc worker atStep).history) (aligned : s.empties.matches b)
      (solved : let opts := EndgameSearch.orderedOptions target bmi2 mc fc p.address
        let out := EndgameSearch.Cancellation.run mc fc opts b (ScoreRefresh.toDiscDiff alpha).raw.toInt s
        ScoreRefresh.fromDiscDiff checked ⟨BitVec.ofInt 32 (out.score.getD 0)⟩ = .ok raw ∧
          EndgameSearch.Cancellation.publish out aborted raw = some raw) :
      Frame mc fc prior reads b false alpha.raw.toInt beta.raw.toInt raw.raw.toInt
  | moves (first : GeneratedSquare mc b.val) (rest order : List (GeneratedSquare mc b.val))
      (generated : legalSquares mc b.val = first :: rest) (perm : order.Perm (first :: rest))
      (ran : Turn (Children mc fc b prior) order pv alpha beta s) : Frame mc fc prior reads b pv alpha beta s

theorem preserves : CausalTable.Preserves mc fc (Frame mc fc) := by
  intro children reads valid usable b pv alpha beta v window null frame
  cases frame with
  | terminal stuck other =>
    apply bound_sound _ window
    have tree : PvsBoardTree.build mc fc b = .leaf (PvsBoardTree.terminalScore b) := by
      rw [PvsBoardTree.build.eq_def]
      split
      next eq => rw [stuck] at eq; contradiction
      next => simp [other]
    rw [tree,PvsMoveLoop.minimax.eq_def]
    exact ⟨fun _ => Int.le_refl _,fun _ => Int.le_refl _⟩
  | pass stuck other issued =>
    apply bound_sound _ window
    change Bound (minimax mc fc b) _ _ _
    rw [PvsBoardTree.minimax_pass mc fc b stuck other]
    exact (Bound.ofSoundResult (valid _ _ _ _ _ (PvsMoveLoop.reflectedWindow window)
      (fun np => by have := null np; omega) issued)).negate
  | wipe m legal hit =>
    exact bound_sound _ window (Cutoffs.contract_proven mc fc b _ _ _ _ null (.wipeout m legal hit))
  | stable a res raw checked cut word =>
    exact bound_sound _ window (Cutoffs.contract_proven mc fc b _ _ _ _ null (.stability a res raw checked cut word))
  | table be seen hit => exact (usable b _ seen).cut _ be _ (null rfl) hit
  | enhanced q probes hit exactLevel phase seen cut =>
    apply bound_sound _ window
    apply Cutoffs.etcBoundFrom mc fc b q probes hit _ cut
    intro probe member e stored eligible
    obtain ⟨m,legal,_,found⟩ := seen probe member e stored
    exact ⟨m,legal,((usable _ _ found).1 (Cutoffs.exactEndgame_of_eligible (by simpa [exactLevel,phase] using eligible))).1⟩
  | fixed result a be raw solved converted =>
    exact bound_sound _ window ((EndgameSearch.Direct.Returned.fixed result solved converted).bound be (null rfl))
  | shallow p timeline worker pos a be raw lo hi solved =>
    exact bound_sound _ window ((EndgameSearch.Direct.Returned.shallow p.address _ lo hi
      (timeline.reads mc fc worker pos).1 solved).bound be (null rfl))
  | cached p timeline worker pos a be raw state localState aligned solved =>
    have cache := localState ▸ (timeline.reads mc fc worker pos).1
    obtain ⟨_,_,_,_,converted,_,ok⟩ := EndgameSearch.Cancellation.boundaryContract_proven _ _ mc fc p.address b a be state
      _ _ aligned cache (null rfl)
    cases Except.ok.inj (solved.1.symm.trans converted)
    exact ok solved.2
  | moves first rest order generated perm ran =>
    apply bound_sound _ window
    change Bound (minimax mc fc b) _ _ _
    rw [PvsBoardTree.minimax_moves mc fc b first rest generated]
    let value := fun m : GeneratedSquare mc b.val => -(minimax mc fc (play mc fc b m).val)
    have identical := PvsMoveLoop.bestValue_eq_of_same_moves value (-scoreInf) (fun _ => perm.mem_iff)
    change Bound (bestValue value (first :: rest) (-scoreInf)) _ _ _
    rw [← identical]
    apply ran.bound (fun empty => by simp [empty] at perm) window null
    · intro m _
      dsimp [value,minimax]
      have := (PvsMoveLoop.minimax_range (PvsBoardTree.build mc fc (play mc fc b m).val)).1
      omega
    · intro m _ cp lo hi score win narrow child
      simpa [value] using valid _ _ _ _ _ win narrow child

end NeuralReversi.PvsParallel.Interleaving.BoardFrame
