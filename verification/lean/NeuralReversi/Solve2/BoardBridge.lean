module
public import NeuralReversi.Board.SetPlacement
public import NeuralReversi.Bitboard.SquareProperties
public import NeuralReversi.Flip.Properties

@[expose] public section

namespace NeuralReversi.Solve2
abbrev Word := BitVec 64
abbrev Square := Fin 64

structure TwoEmpty where
  player : Word
  opponent : Word
  first : Square
  second : Square
  distinct : first ≠ second
  disjoint : player &&& opponent = 0
  empty_exact : ~~~(player ||| opponent) =
    Bitboard.squareMask first ||| Bitboard.squareMask second

def TwoEmpty.pass (b : TwoEmpty) : TwoEmpty where
  player := b.opponent
  opponent := b.player
  first := b.first
  second := b.second
  distinct := b.distinct
  disjoint := by rw [BitVec.and_comm]; exact b.disjoint
  empty_exact := by rw [BitVec.or_comm]; exact b.empty_exact

def TwoEmpty.exchange (b : TwoEmpty) : TwoEmpty where
  player := b.player
  opponent := b.opponent
  first := b.second
  second := b.first
  distinct := Ne.symm b.distinct
  disjoint := b.disjoint
  empty_exact := b.empty_exact.trans (BitVec.or_comm _ _)

def FlipSubset (flipped opponent : Word) : Prop :=
  ∀ i : Nat, flipped.getLsbD i = true → opponent.getLsbD i = true

theorem scalar_flip_subset (sq : Square) (p o : Word) :
    FlipSubset (Flip.value ⟨sq.val, by omega⟩ p o) o := by
  intro i hi
  exact Flip.flipped_is_opponent _ _ _ _ hi

def TwoEmpty.asBoard (b : TwoEmpty) : Board.Board := ⟨b.player, b.opponent⟩

def childBoard (b : TwoEmpty) (flipped : Word) : Board.Board :=
  ⟨b.opponent ^^^ flipped, b.player ^^^ flipped ^^^ Bitboard.squareMask b.first⟩

theorem two_empty_bits (b : TwoEmpty) (i : Fin 64) :
    (b.player.getLsbD i.val = false ∧ b.opponent.getLsbD i.val = false) ↔
      (i = b.first ∨ i = b.second) := by
  have h := congrArg (fun v : Word => v.getLsbD i.val) b.empty_exact
  simp only [BitVec.getLsbD_not, BitVec.getLsbD_or, Bitboard.squareMask_bit] at h
  have hi := i.isLt
  cases hp : b.player.getLsbD i.val <;> cases ho : b.opponent.getLsbD i.val <;>
    simp_all [Fin.ext_iff, -BitVec.getLsbD_eq_getElem]

theorem first_empty (b : TwoEmpty) :
    b.player.getLsbD b.first.val = false ∧ b.opponent.getLsbD b.first.val = false :=
  (two_empty_bits b b.first).mpr (Or.inl rfl)

theorem second_empty (b : TwoEmpty) :
    b.player.getLsbD b.second.val = false ∧ b.opponent.getLsbD b.second.val = false :=
  (two_empty_bits b b.second).mpr (Or.inr rfl)

theorem flip_at_empty_false (b : TwoEmpty) (flipped : Word) (subset : FlipSubset flipped b.opponent)
    (sq : Square) (empty : b.opponent.getLsbD sq.val = false) :
    flipped.getLsbD sq.val = false := by
  cases h : flipped.getLsbD sq.val with
  | false => rfl
  | true => have h' := subset sq.val h; simp only [empty, Bool.false_eq_true] at h'

theorem child_remaining_empty (b : TwoEmpty) (flipped : Word) (subset : FlipSubset flipped b.opponent) :
    (childBoard b flipped).player.getLsbD b.second.val = false := by
  have ho := (second_empty b).2
  have hf := flip_at_empty_false b flipped subset b.second ho
  simp [childBoard, ho, hf, -BitVec.getLsbD_eq_getElem]

theorem child_opponent_reconstructed (b : TwoEmpty) (flipped : Word)
    (subset : FlipSubset flipped b.opponent) :
    (childBoard b flipped).opponent =
      ~~~((childBoard b flipped).player ||| Bitboard.squareMask b.second) := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  have hd := congrArg (fun v : Word => v.getLsbD i) b.disjoint
  have he := two_empty_bits b ⟨i,hi⟩
  have hf := subset i
  have hn : b.first.val ≠ b.second.val := fun h => b.distinct (Fin.ext h)
  simp only [BitVec.getLsbD_and] at hd
  simp only [childBoard, BitVec.getLsbD_xor, BitVec.getLsbD_not,
    BitVec.getLsbD_or, Bitboard.squareMask_bit]
  by_cases ha : i = b.first.val <;> by_cases hb : i = b.second.val <;>
    cases hp : b.player.getLsbD i <;> cases ho : b.opponent.getLsbD i <;>
    cases hflip : flipped.getLsbD i <;>
    simp_all [Fin.ext_iff, -BitVec.getLsbD_eq_getElem]

theorem child_exactly_one_empty (b : TwoEmpty) (flipped : Word)
    (subset : FlipSubset flipped b.opponent) :
    Board.emptySquares (childBoard b flipped) = Bitboard.squareMask b.second := by
  unfold Board.emptySquares
  rw [child_opponent_reconstructed b flipped subset]
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  have hr := child_remaining_empty b flipped subset
  by_cases same : i = b.second.val
  · simp [same, hr, Bitboard.squareMask_bit, -BitVec.getLsbD_eq_getElem]
  · cases h : (childBoard b flipped).player.getLsbD i <;>
      simp [same, h, hi, Bitboard.squareMask_bit, -BitVec.getLsbD_eq_getElem]

theorem child_eq_setPlacement (b : TwoEmpty) (flipped : Word)
    (subset : FlipSubset flipped b.opponent) :
    childBoard b flipped = Board.SetPlacement.placeMask b.asBoard flipped
      (Bitboard.squareMask b.first) := by
  symm
  apply Board.SetPlacement.equals_xor_of_valid
  · exact b.disjoint
  · apply BitVec.eq_of_getLsbD_eq
    intro i hi
    have h := subset i
    cases hf : flipped.getLsbD i <;> cases ho : b.opponent.getLsbD i <;>
      simp_all [TwoEmpty.asBoard, -BitVec.getLsbD_eq_getElem]
  · have h := first_empty b
    apply BitVec.eq_of_getLsbD_eq
    intro i hi
    by_cases he : i = b.first.val
    · simp [TwoEmpty.asBoard, Bitboard.squareMask_bit, he, h.1, h.2,
        -BitVec.getLsbD_eq_getElem]
    · simp [TwoEmpty.asBoard, Bitboard.squareMask_bit, he,
        -BitVec.getLsbD_eq_getElem]

end NeuralReversi.Solve2
