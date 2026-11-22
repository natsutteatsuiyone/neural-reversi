import NeuralReversi.Solve2.BoardBridge
import NeuralReversi.Board.Counting

namespace NeuralReversi.Solve34
abbrev Word := BitVec 64
abbrev Square := Fin 64

def emptyMask : List Square → Word
  | [] => 0
  | s :: ss => Bitboard.squareMask s ||| emptyMask ss

theorem emptyMask_bit (ss : List Square) (i : Fin 64) :
    (emptyMask ss).getLsbD i.val = true ↔ i ∈ ss := by
  induction ss with
  | nil => simp [emptyMask]
  | cons s ss ih =>
    simp only [emptyMask, BitVec.getLsbD_or, Bool.or_eq_true,
      Bitboard.squareMask_bit, decide_eq_true_eq, ih, List.mem_cons]
    simp [Fin.ext_iff]

structure EmptyBoard (n : Nat) where
  player : Word
  opponent : Word
  squares : List Square
  length_eq : squares.length = n
  nodup : squares.Nodup
  disjoint : player &&& opponent = 0
  empty_exact : ~~~(player ||| opponent) = emptyMask squares

def EmptyBoard.asBoard (b : EmptyBoard n) : Board.Board := ⟨b.player, b.opponent⟩

def EmptyBoard.ofBoard (b : Board.Board) (wf : Board.wellFormed b) :
    EmptyBoard (Board.emptyCount b).toNat where
  player := b.player
  opponent := b.opponent
  squares := (List.finRange 64).filter fun sq => (Board.emptySquares b).getLsbD sq.val
  length_eq := by
    have indices : (List.finRange 64).map Fin.val = List.range 64 := by decide
    change _ = (Bitboard.count (Board.emptySquares b)).toNat
    rw [← List.countP_eq_length_filter, (Bitboard.populationCountContract_proven (Board.emptySquares b)).1,
      ← indices, List.countP_map]
    rfl
  nodup := (List.nodup_finRange 64).filter _
  disjoint := wf
  empty_exact := by
    apply BitVec.eq_of_getLsbD_eq
    intro i hi
    apply Bool.eq_iff_iff.mpr
    rw [emptyMask_bit _ ⟨i, hi⟩]
    exact ⟨fun h => List.mem_filter.mpr ⟨by simp, h⟩, fun h => (List.mem_filter.mp h).2⟩

def EmptyBoard.pass (b : EmptyBoard n) : EmptyBoard n where
  player := b.opponent
  opponent := b.player
  squares := b.squares
  length_eq := b.length_eq
  nodup := b.nodup
  disjoint := by rw [BitVec.and_comm]; exact b.disjoint
  empty_exact := by rw [BitVec.or_comm]; exact b.empty_exact

theorem empty_bits (b : EmptyBoard n) (i : Fin 64) :
    (b.player.getLsbD i.val = false ∧ b.opponent.getLsbD i.val = false) ↔ i ∈ b.squares := by
  rw [← emptyMask_bit, ← b.empty_exact]
  simp [i.isLt, -BitVec.getLsbD_eq_getElem]

def childBoard (b : EmptyBoard n) (sq : Square) (flipped : Word) : Board.Board :=
  ⟨b.opponent ^^^ flipped, b.player ^^^ flipped ^^^ Bitboard.squareMask sq⟩

theorem child_eq_setPlacement (b : EmptyBoard n) (sq : Square) (mem : sq ∈ b.squares)
    (f : Word) (subset : Solve2.FlipSubset f b.opponent) :
    childBoard b sq f = Board.SetPlacement.placeMask b.asBoard f (Bitboard.squareMask sq) := by
  symm
  apply Board.SetPlacement.equals_xor_of_valid
  · exact b.disjoint
  · apply BitVec.eq_of_getLsbD_eq
    intro i hi
    have h := subset i
    cases hf : f.getLsbD i <;> cases ho : b.opponent.getLsbD i <;>
      simp_all [EmptyBoard.asBoard, -BitVec.getLsbD_eq_getElem]
  · have h := (empty_bits b sq).mpr mem
    apply BitVec.eq_of_getLsbD_eq
    intro i hi
    by_cases he : i = sq.val
    · simp [EmptyBoard.asBoard, Bitboard.squareMask_bit, he, h.1, h.2,
        -BitVec.getLsbD_eq_getElem]
    · simp [EmptyBoard.asBoard, Bitboard.squareMask_bit, he,
        -BitVec.getLsbD_eq_getElem]

theorem child_disjoint (b : EmptyBoard n) (sq : Square) (mem : sq ∈ b.squares)
    (f : Word) (subset : Solve2.FlipSubset f b.opponent) :
    (childBoard b sq f).player &&& (childBoard b sq f).opponent = 0 := by
  rw [child_eq_setPlacement b sq mem f subset]
  exact Board.SetPlacement.wellFormed_preserved _ _ _ b.disjoint

theorem child_empty_exact (b : EmptyBoard n) (sq : Square) (mem : sq ∈ b.squares)
    (f : Word) (subset : Solve2.FlipSubset f b.opponent) :
    Board.emptySquares (childBoard b sq f) = emptyMask (b.squares.erase sq) := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  have hd := congrArg (fun w : Word => w.getLsbD i) b.disjoint
  have he := empty_bits b ⟨i, hi⟩
  have hm := emptyMask_bit (b.squares.erase sq) ⟨i, hi⟩
  have hmem := b.nodup.mem_erase_iff (a := (⟨i,hi⟩ : Square)) (b := sq)
  have hf := subset i
  simp only [BitVec.getLsbD_and] at hd
  simp only [Board.emptySquares, childBoard, BitVec.getLsbD_not, BitVec.getLsbD_or,
    BitVec.getLsbD_xor, Bitboard.squareMask_bit]
  by_cases hs : i = sq.val <;>
    cases hp : b.player.getLsbD i <;> cases ho : b.opponent.getLsbD i <;>
    cases hflip : f.getLsbD i <;> cases hmval : (emptyMask (b.squares.erase sq)).getLsbD i <;>
    simp_all [Fin.ext_iff, -BitVec.getLsbD_eq_getElem]

def EmptyBoard.play (b : EmptyBoard n) (sq : Square) (mem : sq ∈ b.squares)
    (f : Word) (subset : Solve2.FlipSubset f b.opponent) : EmptyBoard (n - 1) where
  player := (childBoard b sq f).player
  opponent := (childBoard b sq f).opponent
  squares := b.squares.erase sq
  length_eq := by rw [List.length_erase_of_mem mem, b.length_eq]
  nodup := b.nodup.erase sq
  disjoint := child_disjoint b sq mem f subset
  empty_exact := child_empty_exact b sq mem f subset

private theorem list_two (ss : List Square) (h : ss.length = 2) :
    ss = [ss[0]'(by omega), ss[1]'(by omega)] := by
  match ss with
  | [] => simp at h
  | a :: [] => simp at h
  | a :: c :: [] => rfl
  | a :: c :: d :: ds => simp at h

theorem EmptyBoard.two_squares (b : EmptyBoard 2) :
    b.squares = [b.squares[0]'(by have := b.length_eq; omega),
                 b.squares[1]'(by have := b.length_eq; omega)] :=
  list_two b.squares b.length_eq

def EmptyBoard.toTwoEmpty (b : EmptyBoard 2) : Solve2.TwoEmpty where
  player := b.player
  opponent := b.opponent
  first := b.squares[0]'(by have := b.length_eq; omega)
  second := b.squares[1]'(by have := b.length_eq; omega)
  distinct := by
    have h := b.nodup
    rw [b.two_squares] at h
    simpa using h
  disjoint := b.disjoint
  empty_exact := by
    rw [b.empty_exact]
    exact (congrArg emptyMask b.two_squares).trans (by simp [emptyMask])

theorem play_order (b : EmptyBoard n) (sq : Square) (mem : sq ∈ b.squares)
    (f : Word) (subset : Solve2.FlipSubset f b.opponent) :
    (b.play sq mem f subset).squares = b.squares.erase sq := rfl

theorem erase_three (a b c : Square) (h : [a,b,c].Nodup) :
    [a,b,c].erase a = [b,c] ∧ [a,b,c].erase b = [a,c] ∧
    [a,b,c].erase c = [a,b] := by
  simp at h
  rcases h with ⟨⟨hab, hac⟩, hbc⟩
  simp [hab, hac, hbc]

theorem erase_four (a b c d : Square) (h : [a,b,c,d].Nodup) :
    [a,b,c,d].erase a = [b,c,d] ∧ [a,b,c,d].erase b = [a,c,d] ∧
    [a,b,c,d].erase c = [a,b,d] ∧ [a,b,c,d].erase d = [a,b,c] := by
  simp at h
  rcases h with ⟨⟨hab, hac, had⟩, ⟨hbc, hbd⟩, hcd⟩
  simp [hab, hac, had, hbc, hbd, hcd]

end NeuralReversi.Solve34
