import NeuralReversi.Board.Move.Properties
import NeuralReversi.Board.MovesProperties

namespace NeuralReversi.Board

def CheckedRejectsAllIllegalMoves : Prop :=
  ∀ (config : Flip.Dispatch.X86Config) (b : Board) (sq : Bitboard.Square),
    wellFormed b → ¬ Bitboard.Moves.LegalMove b.player b.opponent sq.val →
    ∃ message, makeMove config true true b sq = .error message

private theorem directionalBit_to_ray {sq : Flip.Square} {d : Fin 8} {player opponent : Bitboard.Bits} {i : Nat}
    (h : Flip.DirectionalBit sq d player opponent i) :
    Bitboard.Moves.Ray player opponent sq.val (-(Flip.delta d).1) (-(Flip.delta d).2) := by
  obtain ⟨_, _, anchor, distance, hanc, hlo, hhi, hpl, _, hray, hchain, _⟩ := h
  refine ⟨anchor, distance, hanc, hlo, hhi, hpl, ?_, ?_⟩
  · unfold Bitboard.Moves.OnRay at hray ⊢
    rw [Int.mul_neg, Int.mul_neg]
    omega
  · intro k hk hkd
    obtain ⟨idx, hidx, hmid, hopp⟩ := hchain (distance - k) (by omega) (by omega)
    refine ⟨idx, hidx, ?_, hopp⟩
    unfold Bitboard.Moves.OnRay at hray hmid ⊢
    have hsub : ((distance - k : Nat) : Int) = (distance : Int) - (k : Int) := by omega
    rw [hsub] at hmid
    have hmul1 : ((distance : Int) - (k : Int)) * (Flip.delta d).1 =
        (distance : Int) * (Flip.delta d).1 - (k : Int) * (Flip.delta d).1 := Int.sub_mul _ _ _
    have hmul2 : ((distance : Int) - (k : Int)) * (Flip.delta d).2 =
        (distance : Int) * (Flip.delta d).2 - (k : Int) * (Flip.delta d).2 := Int.sub_mul _ _ _
    have hneg1 : (k : Int) * (-(Flip.delta d).1) = -((k : Int) * (Flip.delta d).1) := Int.mul_neg _ _
    have hneg2 : (k : Int) * (-(Flip.delta d).2) = -((k : Int) * (Flip.delta d).2) := Int.mul_neg _ _
    have hnegd1 : (distance : Int) * (-(Flip.delta d).1) = -((distance : Int) * (Flip.delta d).1) := Int.mul_neg _ _
    have hnegd2 : (distance : Int) * (-(Flip.delta d).2) = -((distance : Int) * (Flip.delta d).2) := Int.mul_neg _ _
    omega

theorem coordinateBit_bracketed {sq : Flip.Square} {player opponent : Bitboard.Bits} {i : Nat}
    (h : Flip.CoordinateBit sq player opponent i) :
    Bitboard.Moves.Bracketed player opponent sq.val := by
  obtain ⟨d, hd⟩ := h
  have hr := directionalBit_to_ray hd
  rcases d with ⟨v, hv⟩
  match v with
  | 0 => exact Or.inr (Or.inl (by simpa [Flip.delta] using hr))
  | 1 => exact Or.inr (Or.inr (Or.inr (Or.inl (by simpa [Flip.delta] using hr))))
  | 2 => exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (by simpa [Flip.delta] using hr)))))))
  | 3 => exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl (by simpa [Flip.delta] using hr)))))))
  | 4 => exact Or.inl (by simpa [Flip.delta] using hr)
  | 5 => exact Or.inr (Or.inr (Or.inl (by simpa [Flip.delta] using hr)))
  | 6 => exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inl (by simpa [Flip.delta] using hr)))))
  | 7 => exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl (by simpa [Flip.delta] using hr))))))
  | n + 8 => omega

theorem checkedRejectsAllIllegalMoves_proven : CheckedRejectsAllIllegalMoves := by
  intro config b sq wf illegal
  rw [makeMove_resolved]
  by_cases occ : b.player.getLsbD sq.val = true ∨ b.opponent.getLsbD sq.val = true
  · refine ⟨"sq must be an empty square", ?_⟩
    simp only [occ, decide_true, Bool.true_and, ↓reduceIte]
  · have hpf : b.player.getLsbD sq.val = false := by
      cases hp : b.player.getLsbD sq.val
      · rfl
      · exact False.elim (occ (.inl hp))
    have hof : b.opponent.getLsbD sq.val = false := by
      cases ho : b.opponent.getLsbD sq.val
      · rfl
      · exact False.elim (occ (.inr ho))
    have hzero : Flip.Dispatch.resultValue config sq b.player b.opponent = 0 := by
      by_cases hne : Flip.Dispatch.resultValue config sq b.player b.opponent = 0
      · exact hne
      · rw [Flip.Dispatch.value_disjoint_scalar config sq b.player b.opponent wf] at hne
        have hctz := BitVec.getLsbD_true_ctz_of_ne_zero hne
        have hcoord := (Flip.value_coordinate sq b.player b.opponent _).mp hctz
        exact False.elim (illegal ⟨sq.isLt, hpf, hof, coordinateBit_bracketed hcoord⟩)
    refine ⟨"assertion failed: !flipped.is_empty()", ?_⟩
    simp only [occ, hzero, decide_false, decide_true, Bool.true_and, Bool.false_eq_true, ↓reduceIte]

theorem bracketed_has_flip (b : Board) (sq : Bitboard.Square) (wf : wellFormed b)
    (bracket : Bitboard.Moves.Bracketed b.player b.opponent sq.val) :
    ∃ i, Flip.CoordinateBit sq b.player b.opponent i := by
  have direction : ∃ d : Fin 8, Bitboard.Moves.Ray b.player b.opponent sq.val
      (-(Flip.delta d).1) (-(Flip.delta d).2) := by
    rcases bracket with h | h | h | h | h | h | h | h
    · exact ⟨4, h⟩
    · exact ⟨0, h⟩
    · exact ⟨5, h⟩
    · exact ⟨1, h⟩
    · exact ⟨6, h⟩
    · exact ⟨7, h⟩
    · exact ⟨3, h⟩
    · exact ⟨2, h⟩
  obtain ⟨d, anchor, distance, ha, hlo, hhi, friendly, last, chain⟩ := direction
  have exclusive : b.opponent.getLsbD anchor = false := by
    have disjoint := congrArg (fun bits : Bitboard.Bits => bits.getLsbD anchor) wf
    simpa [friendly] using disjoint
  have reverseChain : ∀ k, 0 < k → k < distance → ∃ i, i < 64 ∧
      Bitboard.Moves.OnRay sq.val i k (Flip.delta d).1 (Flip.delta d).2 ∧
      b.opponent.getLsbD i = true := by
    intro k positive interior
    obtain ⟨i, hi, middle, opponent⟩ := chain (distance - k) (by omega) (by omega)
    refine ⟨i, hi, ?_, opponent⟩
    have sub : ((distance - k : Nat) : Int) = (distance : Int) - k := by omega
    simp only [Bitboard.Moves.OnRay, Int.mul_neg, sub, Int.sub_mul] at last middle ⊢
    omega
  obtain ⟨i, hi, neighbor, _⟩ := reverseChain 1 (by decide) (by omega)
  refine ⟨i, d, sq.isLt, hi, anchor, distance, ha, hlo, hhi, friendly, exclusive,
    ?_, reverseChain, 1, by decide, by omega, neighbor⟩
  simp only [Bitboard.Moves.OnRay, Int.mul_neg] at last ⊢
  omega

theorem legal_iff_empty_flip (b : Board) (sq : Bitboard.Square) (wf : wellFormed b) :
    Bitboard.Moves.LegalMove b.player b.opponent sq.val ↔
      (b.player.getLsbD sq.val = false ∧ b.opponent.getLsbD sq.val = false) ∧
      Flip.value sq b.player b.opponent ≠ 0 := by
  constructor
  · rintro ⟨_, p, o, bracket⟩
    refine ⟨⟨p, o⟩, ?_⟩
    obtain ⟨i, witness⟩ := bracketed_has_flip b sq wf bracket
    have bit := (Flip.value_coordinate sq b.player b.opponent i).mpr witness
    intro zero
    simp [zero] at bit
  · rintro ⟨⟨p, o⟩, nonzero⟩
    have bit := BitVec.getLsbD_true_ctz_of_ne_zero nonzero
    exact ⟨sq.isLt, p, o, coordinateBit_bracketed ((Flip.value_coordinate sq b.player b.opponent _).mp bit)⟩

def CheckedAcceptsGeneratedMoves : Prop :=
  ∀ movesConfig flipConfig (b : Board) (sq : Bitboard.Square), wellFormed b →
    (getMoves movesConfig b).getLsbD sq.val = true →
    let flipped := Flip.Dispatch.resultValue flipConfig sq b.player b.opponent
    flipped ≠ 0 ∧ makeMove flipConfig true true b sq = .ok (place b flipped sq)

theorem checkedAcceptsGeneratedMoves_proven : CheckedAcceptsGeneratedMoves := by
  intro movesConfig flipConfig b sq wf generated
  obtain ⟨_, emptyPlayer, emptyOpponent, bracket⟩ := (getMoves_exact movesConfig b sq.val).mp generated
  obtain ⟨i, captured⟩ := bracketed_has_flip b sq wf bracket
  have nonempty : Flip.Dispatch.resultValue flipConfig sq b.player b.opponent ≠ 0 := by
    rw [Flip.Dispatch.value_disjoint_scalar _ _ _ _ wf]
    intro zero
    have bit := (Flip.value_coordinate sq b.player b.opponent i).mpr captured
    simp [zero] at bit
  refine ⟨nonempty, ?_⟩
  rw [makeMove_resolved]
  simp only [emptyPlayer, emptyOpponent, Bool.false_eq_true, or_self, nonempty,
    decide_false, Bool.and_false, ↓reduceIte, placeAny_valid_eq]

def PrecomputedMoveContract : Prop :=
  ∀ movesConfig flipConfig debugAssertions overflowChecks (b : Board) (sq : Bitboard.Square),
    wellFormed b → (getMoves movesConfig b).getLsbD sq.val = true →
    let flipped := Flip.Dispatch.resultValue flipConfig sq b.player b.opponent
    let child := place b flipped sq
    makeMoveWithFlippedAny debugAssertions overflowChecks b flipped sq = .ok child ∧
      makeMove flipConfig debugAssertions overflowChecks b sq = .ok child

theorem precomputedMoveContract_proven : PrecomputedMoveContract := by
  intro movesConfig flipConfig debugAssertions overflowChecks b sq wf generated
  obtain ⟨_, emptyPlayer, emptyOpponent, _⟩ := (getMoves_exact movesConfig b sq.val).mp generated
  have nonempty := (checkedAcceptsGeneratedMoves_proven movesConfig flipConfig b sq wf generated).1
  have empty : emptySquares b &&& Bitboard.squareMask sq ≠ 0#64 := by
    intro zero
    have missing := (Bitboard.and_squareMask_zero _ _).mp zero
    simp [emptySquares, sq.isLt, emptyPlayer, emptyOpponent, -BitVec.getLsbD_eq_getElem] at missing
  have subset : Flip.Dispatch.resultValue flipConfig sq b.player b.opponent &&& ~~~b.opponent = 0#64 := by
    rw [Flip.Dispatch.value_disjoint_scalar _ _ _ _ wf]
    apply BitVec.eq_of_getLsbD_eq
    intro i hi
    cases captured : (Flip.value sq b.player b.opponent).getLsbD i with
    | false => simp only [BitVec.getLsbD_and, captured, Bool.false_and, BitVec.getLsbD_zero]
    | true =>
      simp only [BitVec.getLsbD_and, BitVec.getLsbD_not, Flip.flipped_is_opponent _ _ _ _ captured,
        Bool.not_true, Bool.and_false, BitVec.getLsbD_zero]
  constructor
  · rw [makeMoveWithFlippedAny_valid]
    have guarded := playableGuardedMoveContract_proven b
      (Flip.Dispatch.resultValue flipConfig sq b.player b.opponent) sq
    cases debugAssertions
    · exact guarded.1
    · exact guarded.2.1.mpr ⟨nonempty, empty, subset⟩
  · rw [makeMove_resolved]
    simp only [emptyPlayer, emptyOpponent, Bool.false_eq_true, or_self, nonempty,
      decide_false, Bool.and_false, ↓reduceIte, placeAny_valid_eq]

end NeuralReversi.Board
