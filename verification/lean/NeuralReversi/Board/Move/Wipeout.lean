module
public import NeuralReversi.Board.Move.Documentation
public import NeuralReversi.ScoreRefresh.BoardBridge

@[expose] public section

namespace NeuralReversi.Board

private theorem legalMove_ne_zero {p o : Bitboard.Bits} {i : Nat}
    (legal : Bitboard.Moves.LegalMove p o i) : p ≠ 0 ∧ o ≠ 0 := by
  have rayNeedsDiscs {dr dc : Int} (ray : Bitboard.Moves.Ray p o i dr dc) :
      p ≠ 0 ∧ o ≠ 0 := by
    obtain ⟨anchor, d, _, depth, _, friendly, _, chain⟩ := ray
    obtain ⟨_, _, _, enemy⟩ := chain 1 (by decide) (by omega)
    constructor
    · rintro rfl
      simp at friendly
    · rintro rfl
      simp at enemy
  rcases legal.2.2.2 with h | h | h | h | h | h | h | h <;> exact rayNeedsDiscs h

def WipeoutScoreContract : Prop :=
  ∀ movesConfig flipConfig checked overflowChecks (b : Board) (sq : Bitboard.Square),
    wellFormed b → (getMoves movesConfig b).getLsbD sq.val = true →
    Flip.Dispatch.resultValue flipConfig sq b.player b.opponent = b.opponent →
    ∃ child, makeMove flipConfig checked overflowChecks b sq = .ok child ∧
      isGameOver movesConfig child = true ∧
      ScoreRefresh.BoardBridge.solve checked child (emptyCount child) = .ok ScoreRefresh.scoreMin

theorem wipeoutScoreContract_proven : WipeoutScoreContract := by
  intro mc fc checked overflowChecks b sq wf generated wipeout
  let child := place b b.opponent sq
  have played := (precomputedMoveContract_proven mc fc checked overflowChecks b sq wf generated).2
  rw [wipeout] at played
  have noDiscs : child.player = 0 := by simp [child, place]
  have discs := (legalMove_ne_zero ((getMoves_exact mc b sq.val).mp generated)).2
  have survivor : child.opponent ≠ 0 := by
    intro empty
    exact discs ((BitVec.or_eq_zero_iff.mp empty).2)
  refine ⟨child, played, (isGameOver_exact mc child).mpr ⟨?_, ?_⟩, ?_⟩
  · exact fun _ legal => (legalMove_ne_zero legal).1 noDiscs
  · exact fun _ legal => (legalMove_ne_zero legal).2 noDiscs
  · have disjoint := place_wellFormed b b.opponent sq wf
    have counts := countConservation_proven child disjoint
    have zero : playerCount child = 0 := by simp [playerCount, noDiscs, Bitboard.count]
    have someDiscs : opponentCount child ≠ 0 := fun h =>
      survivor ((Bitboard.populationCountContract_proven _).2.2.1.mp h)
    have positive : 0 < (opponentCount child).toNat := by
      apply Nat.pos_of_ne_zero
      exact fun h => someDiscs (BitVec.eq_of_toNat_eq h)
    have zeroNat : (playerCount child).toNat = 0 := by rw [zero]; rfl
    have loss : AwardedScore child = -64 := by
      rw [zeroNat] at counts
      simp [AwardedScore, zeroNat, (Nat.ne_of_gt positive).symm]
      omega
    obtain ⟨score, release, debug, awarded, _⟩ :=
      ScoreRefresh.BoardBridge.solveDiscDiff_awarded child disjoint
    have raw : score.raw = -64 := BitVec.eq_of_toInt_eq (awarded.trans loss)
    cases checked <;> simp only [ScoreRefresh.BoardBridge.solve, release, debug,
      ScoreRefresh.fromDiscDiff, raw] <;> rfl

end NeuralReversi.Board
