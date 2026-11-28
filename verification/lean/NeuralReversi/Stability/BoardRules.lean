module
public import NeuralReversi.Stability.EdgeEncoding
public import NeuralReversi.Board.Move.Documentation

@[expose] public section

namespace NeuralReversi.Stability.Semantics

open NeuralReversi.Bitboard (Bits)
open NeuralReversi.Bitboard.Moves (OnRay)
abbrev BitBoard := NeuralReversi.Board.Board

private def direction (d : Fin 8) : Direction :=
  match d.val with
  | 0 => .e
  | 1 => .s
  | 2 => .se
  | 3 => .sw
  | 4 => .w
  | 5 => .n
  | 6 => .nw
  | _ => .ne

private theorem direction_delta : ∀ d, (direction d).delta = Flip.delta d := by decide

private theorem direction_covers (d : Direction) : ∃ f : Fin 8, direction f = d := by
  cases d <;> decide

private theorem along_iff (a x : Square) (d : Fin 8) (k : Nat) :
    Along a x (direction d) k ↔
      OnRay (boardIndex a).val (boardIndex x).val k (Flip.delta d).1 (Flip.delta d).2 := by
  change Along a x (direction d) k ↔
    (((boardIndex x).val / 8 : Nat) : Int) =
      ((boardIndex a).val / 8 : Nat) + (k : Int) * (Flip.delta d).1 ∧
    (((boardIndex x).val % 8 : Nat) : Int) =
      ((boardIndex a).val % 8 : Nat) + (k : Int) * (Flip.delta d).2
  simp only [Along, direction_delta, boardIndex_row, boardIndex_col]

private theorem opposite_twice (c : Color) : Opposite (Opposite c) = c := by cases c <;> rfl

def stateAt (b : BitBoard) (c : Color) : State :=
  ⟨decodeBoard b.player b.opponent c, c⟩

theorem stateAt_represents (b : BitBoard) (c : Color) (wf : NeuralReversi.Board.wellFormed b) :
    Represents (stateAt b c) c b.player b.opponent :=
  decodeBoard_represents b.player b.opponent c wf

private theorem not_both {b : BitBoard} (wf : NeuralReversi.Board.wellFormed b) (i : Nat) :
    (b.player.getLsbD i && b.opponent.getLsbD i) = false := by
  have eq := congrArg (fun w : Bits => w.getLsbD i) wf
  simpa using eq

private theorem player_excludes_opponent {b : BitBoard} (wf : NeuralReversi.Board.wellFormed b)
    {i : Nat} (p : b.player.getLsbD i = true) : b.opponent.getLsbD i = false := by
  simpa only [p, Bool.true_and] using not_both wf i

private theorem empty_iff (b : BitBoard) (c : Color) (sq : Square) :
    (stateAt b c).board sq = none ↔
      b.player.getLsbD (boardIndex sq).val = false ∧ b.opponent.getLsbD (boardIndex sq).val = false := by
  dsimp only [stateAt, decodeBoard, decodeColor]
  cases b.player.getLsbD (boardIndex sq).val <;>
    cases b.opponent.getLsbD (boardIndex sq).val <;> simp

theorem capture_iff_coordinate (b : BitBoard) (c : Color) (a x : Square)
    (wf : NeuralReversi.Board.wellFormed b) :
    Captured (stateAt b c).board c a x ↔
      Flip.CoordinateBit (boardIndex a) b.player b.opponent (boardIndex x).val := by
  have repr := stateAt_represents b c wf
  constructor
  · rintro ⟨dir, n, lo, hi, ⟨stop, hr, owner⟩, chain, k, hk, hn, captured⟩
    obtain ⟨d, rfl⟩ := direction_covers dir
    have friendly := (repr.1 stop).mpr owner
    refine ⟨d, (boardIndex a).isLt, (boardIndex x).isLt, (boardIndex stop).val, n + 1,
      (boardIndex stop).isLt, by omega, by omega, friendly, player_excludes_opponent wf friendly,
      (along_iff a stop d _).mp hr, ?_, k, by omega, by omega, (along_iff a x d _).mp captured⟩
    intro j lower upper
    obtain ⟨y, path, color⟩ := chain j (by omega) (by omega)
    exact ⟨(boardIndex y).val, (boardIndex y).isLt, (along_iff a y d j).mp path, (repr.2 y).mpr color⟩
  · rintro ⟨d, _, _, stop, dist, stopLt, lower, upper, friendly, _, hr, chain, k, hk, hd, captured⟩
    have stopEq : dist - 1 + 1 = dist := by omega
    refine ⟨direction d, dist - 1, by omega, by omega,
      ⟨boardSquare ⟨stop, stopLt⟩, ?_, ?_⟩, ?_, k, by omega, by omega,
      (along_iff a x d k).mpr captured⟩
    · apply (along_iff _ _ _ _).mpr
      simpa only [boardIndex_square, stopEq] using hr
    · apply (repr.1 _).mp
      simpa only [Marks, boardIndex_square] using friendly
    · intro j lo hi
      obtain ⟨i, ib, path, opponent⟩ := chain j (by omega) (by omega)
      refine ⟨boardSquare ⟨i, ib⟩, (along_iff _ _ _ _).mpr ?_, (repr.2 _).mp ?_⟩
      · simpa only [boardIndex_square] using path
      · simpa only [Marks, boardIndex_square] using opponent

theorem legal_iff_relative (b : BitBoard) (c : Color) (sq : Bitboard.Square)
    (wf : NeuralReversi.Board.wellFormed b) :
    LegalMove (stateAt b c).board c (boardSquare sq) ↔
      Bitboard.Moves.LegalMove b.player b.opponent sq.val := by
  have empty := empty_iff b c (boardSquare sq)
  simp only [boardIndex_square] at empty
  constructor
  · rintro ⟨vacant, x, captured⟩
    have bit := (capture_iff_coordinate b c (boardSquare sq) x wf).mp captured
    rw [boardIndex_square] at bit
    exact ⟨sq.isLt, (empty.mp vacant).1, (empty.mp vacant).2,
      NeuralReversi.Board.coordinateBit_bracketed bit⟩
  · rintro ⟨_, p, o, bracket⟩
    obtain ⟨i, bit⟩ := NeuralReversi.Board.bracketed_has_flip b sq wf bracket
    obtain ⟨d, hsquare, hi, payload⟩ := bit
    refine ⟨empty.mpr ⟨p, o⟩, boardSquare ⟨i, hi⟩, ?_⟩
    apply (capture_iff_coordinate _ _ _ _ wf).mpr
    simp only [boardIndex_square]
    exact ⟨d, hsquare, hi, payload⟩

theorem stateAt_switch (b : BitBoard) (c : Color) (wf : NeuralReversi.Board.wellFormed b) :
    stateAt (NeuralReversi.Board.switchPlayers b) (Opposite c) =
      ⟨(stateAt b c).board, Opposite c⟩ := by
  suffices (stateAt (NeuralReversi.Board.switchPlayers b) (Opposite c)).board =
      (stateAt b c).board by
    cases b
    simp only [stateAt, NeuralReversi.Board.switchPlayers] at this ⊢
    rw [this]
  funext x
  have disjoint := not_both wf (boardIndex x).val
  unfold stateAt decodeBoard decodeColor NeuralReversi.Board.switchPlayers
  cases hp : b.player.getLsbD (boardIndex x).val <;>
    cases ho : b.opponent.getLsbD (boardIndex x).val <;>
    simp_all [opposite_twice]

private theorem place_cell (c : Color) (p o flip : Bool) (target : Prop) [Decidable target]
    (apart : (p && o) = false) (flipOpponent : flip = true → o = true)
    (empty : target → o = false) :
    (if o && !flip then some (Opposite c)
     else if (p || (decide target && !o)) || flip then some c else none) =
    (if target ∨ flip = true then some c else if p then some c else if o then some (Opposite c) else none) := by
  cases p <;> cases o <;> cases flip <;> by_cases target <;> simp_all

theorem stateAt_place (b : BitBoard) (c : Color) (sq : Bitboard.Square)
    (wf : NeuralReversi.Board.wellFormed b)
    (legal : Bitboard.Moves.LegalMove b.player b.opponent sq.val) :
    stateAt (NeuralReversi.Board.place b (Flip.value sq b.player b.opponent) sq) (Opposite c) =
      ⟨place (stateAt b c).board c (boardSquare sq), Opposite c⟩ := by
  classical
  suffices (stateAt (NeuralReversi.Board.place b (Flip.value sq b.player b.opponent) sq)
      (Opposite c)).board = place (stateAt b c).board c (boardSquare sq) by
    simp only [stateAt] at this ⊢
    rw [this]
  funext x
  let i := (boardIndex x).val
  have squareEq : x = boardSquare sq ↔ i = sq.val := by
    constructor
    · rintro rfl
      simp only [i, boardIndex_square]
    · intro same
      have := congrArg boardSquare (Fin.ext same : boardIndex x = sq)
      simpa only [boardSquare_index] using this
  have captureEq : Captured (stateAt b c).board c (boardSquare sq) x ↔
      (Flip.value sq b.player b.opponent).getLsbD i = true := by
    rw [capture_iff_coordinate _ _ _ _ wf, boardIndex_square, Flip.value_coordinate]
  have apart := not_both wf i
  have flipOpponent := Flip.flipped_is_opponent sq b.player b.opponent i
  unfold place
  simp only [squareEq, captureEq]
  simp only [stateAt, decodeBoard, decodeColor, NeuralReversi.Board.place,
    BitVec.getLsbD_and, BitVec.getLsbD_or, BitVec.getLsbD_not, Bitboard.squareMask_bit, opposite_twice]
  change (if b.opponent.getLsbD i && (decide (i < 64) &&
      !(Flip.value sq b.player b.opponent).getLsbD i) then some (Opposite c)
    else if (b.player.getLsbD i || (decide (i = sq.val) &&
        (decide (i < 64) && !b.opponent.getLsbD i))) ||
        (Flip.value sq b.player b.opponent).getLsbD i then some c else none) = _
  have ib : i < 64 := (boardIndex x).isLt
  simp only [ib, decide_true, Bool.true_and]
  exact place_cell c _ _ _ (i = sq.val) apart flipOpponent (fun same => same ▸ legal.2.2.1)

end NeuralReversi.Stability.Semantics
