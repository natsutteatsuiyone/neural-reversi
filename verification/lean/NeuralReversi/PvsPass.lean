module
public import NeuralReversi.Aspiration
public import NeuralReversi.Board.MovesProperties
public import NeuralReversi.Solve2.Control

@[expose] public section

namespace NeuralReversi.PvsPass

open Board (Board)
open ScoreRefresh (Score)
open Bitboard.Moves (X86Config LegalMove)

abbrev Search := Board → BitVec 32 → Score → Score → Bool → Bool → Score

def pass (mc : X86Config) (checked : Bool) (b : Board) (depth : BitVec 32)
    (alpha beta : Score) (pv cut : Bool) (search : Search) : Option (Except String Score) :=
  if Board.getMoves mc b = 0 ∧ Board.hasLegalMoves mc (Board.switchPlayers b) = true then some do
    let low ← ScoreRefresh.negate checked beta
    let high ← ScoreRefresh.negate checked alpha
    ScoreRefresh.negate checked (search (Board.switchPlayers b) depth low high pv (!pv && !cut))
  else none

private theorem neg_ok (checked : Bool) (s : Score) (lo : -32767 ≤ s.raw.toInt)
    (hi : s.raw.toInt ≤ 32767) :
    ScoreRefresh.negate checked s = .ok ⟨-s.raw⟩ ∧ (-s.raw).toInt = -s.raw.toInt := by
  have safe : ¬ ScaledScore.outsideI32 (-s.raw.toInt) := by unfold ScaledScore.outsideI32; omega
  exact ⟨by simp [ScoreRefresh.negate, ScaledScore.neg_overflow_spec, safe], Solve2.neg_safe _ (by omega)⟩

private theorem reflect {v a b r : Int} (s : Aspiration.SoundResult v (-b) (-a) r) :
    Aspiration.SoundResult (-v) a b (-r) := by
  obtain ⟨lo, hi, low, high, exactScore⟩ := s
  refine ⟨by omega, by omega, ?_, ?_, ?_⟩
  · intro h
    have := high (by omega)
    omega
  · intro h
    have := low (by omega)
    omega
  · intro h₁ h₂
    have := exactScore (by omega) (by omega)
    omega

def Contract : Prop :=
  ∀ mc checked b depth alpha beta pv cut search,
    let result := pass mc checked b depth alpha beta pv cut search
    let forced := (∀ i, ¬ LegalMove b.player b.opponent i) ∧ ∃ i, LegalMove b.opponent b.player i
    (result.isSome = true ↔ forced) ∧
    (forced → Aspiration.Window alpha.raw.toInt beta.raw.toInt →
      let childAlpha : Score := ⟨-beta.raw⟩
      let childBeta : Score := ⟨-alpha.raw⟩
      let child := search (Board.switchPlayers b) depth childAlpha childBeta pv (!pv && !cut)
      childAlpha.raw.toInt = -beta.raw.toInt ∧ childBeta.raw.toInt = -alpha.raw.toInt ∧
      Aspiration.Window childAlpha.raw.toInt childBeta.raw.toInt ∧
      ∀ childValue, Aspiration.SoundResult childValue childAlpha.raw.toInt childBeta.raw.toInt child.raw.toInt →
        result = some (.ok ⟨-child.raw⟩) ∧
        Aspiration.SoundResult (-childValue) alpha.raw.toInt beta.raw.toInt (-child.raw).toInt)

theorem contract_proven : Contract := by
  intro mc checked b depth alpha beta pv cut search
  dsimp only
  have guard : (Board.getMoves mc b = 0 ∧ Board.hasLegalMoves mc (Board.switchPlayers b) = true) ↔
      (∀ i, ¬LegalMove b.player b.opponent i) ∧ ∃ i, LegalMove b.opponent b.player i :=
    and_congr (Board.moves_zero_iff mc b) (Board.hasLegalMoves_exact mc (Board.switchPlayers b))
  refine ⟨?_, ?_⟩
  · unfold pass
    split <;> simp_all
  · intro forced w
    have ⟨lo, ordered, hi⟩ := w
    obtain ⟨negA, valA⟩ := neg_ok checked alpha lo (by omega)
    obtain ⟨negB, valB⟩ := neg_ok checked beta (by omega) hi
    refine ⟨valB, valA, ?_, ?_⟩
    · rw [valB, valA]
      constructor <;> omega
    · intro value sound
      let child := search (Board.switchPlayers b) depth ⟨-beta.raw⟩ ⟨-alpha.raw⟩ pv (!pv && !cut)
      change Aspiration.SoundResult value (-beta.raw).toInt (-alpha.raw).toInt child.raw.toInt at sound
      rw [valB, valA] at sound
      obtain ⟨negChild, valChild⟩ := neg_ok checked child (Int.le_of_lt sound.lower) (Int.le_of_lt sound.upper)
      refine ⟨?_, ?_⟩
      · simp [pass, guard.mpr forced, negA, negB, child] at negChild ⊢
        exact negChild
      · change Aspiration.SoundResult (-value) alpha.raw.toInt beta.raw.toInt (-child.raw).toInt
        rw [valChild]
        exact reflect sound

end NeuralReversi.PvsPass
