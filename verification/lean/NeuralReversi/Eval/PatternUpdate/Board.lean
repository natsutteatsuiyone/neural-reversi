import NeuralReversi.Eval.PatternUpdate.Tables
import NeuralReversi.Board.Model
import NeuralReversi.Bitboard.SquareProperties

namespace NeuralReversi.Eval.PatternUpdate

open Bitboard (Bits)
open NeuralReversi.Board (Board switchPlayers place wellFormed)

def digit (b : Board) (i : Nat) : Int :=
  if b.player.getLsbD i then 0 else if b.opponent.getLsbD i then 1 else 2

def pairedCode (squares : List Nat) (b : Board) : Int × Int :=
  (code (digit b) squares, code (digit (switchPlayers b)) squares)

def placeCompatible (b : Board) (sq : Square.Square) (flipped : Bits) : Prop :=
  wellFormed b ∧ b.player.getLsbD sq.val = false ∧ b.opponent.getLsbD sq.val = false ∧
    ∀ i, flipped.getLsbD i = true → b.opponent.getLsbD i = true

def DigitUpdateContract : Prop :=
  ∀ b sq flipped, placeCompatible b sq flipped → ∀ i, i < 64 →
    let placed := squareIndicator sq.val i
    let flips := flipDigits flipped i
    digit (switchPlayers (place b flipped sq)) i = digit b i - (2 * placed + flips) ∧
    digit (place b flipped sq) i = digit (switchPlayers b) i - placed + flips

theorem digitUpdateContract_proven : DigitUpdateContract := by
  intro b sq flipped placement i iOk
  obtain ⟨disjoint, pEmpty, oEmpty, flipFromO⟩ := placement
  have onlyOne := congrArg (fun x : Bits => x.getLsbD i) disjoint
  simp only [BitVec.getLsbD_and] at onlyOne
  unfold digit place switchPlayers
  dsimp only
  simp only [BitVec.getLsbD_and, BitVec.getLsbD_or, BitVec.getLsbD_not, Bitboard.squareMask_bit,
    iOk, decide_true, Bool.true_and]
  by_cases put : i = sq.val
  · subst i
    have noFlip : flipped.getLsbD sq.val = false := by
      cases f : flipped.getLsbD sq.val
      · rfl
      · exact False.elim (Bool.false_ne_true (oEmpty.symm.trans (flipFromO _ f)))
    simp only [squareIndicator, flipDigits, bit, pEmpty, oEmpty, noFlip, decide_true]
    decide
  · have notPut : sq.val ≠ i := Ne.symm put
    have flipOwner := flipFromO i
    cases p : b.player.getLsbD i <;> cases o : b.opponent.getLsbD i <;>
      cases f : flipped.getLsbD i <;>
      simp_all [squareIndicator, flipDigits, bit]

def moverBoard (opponentMoves : Bool) (root : Board) : Board :=
  if opponentMoves then switchPlayers root else root

def rootAfter (opponentMoves : Bool) (root : Board) (sq : Square.Square) (flipped : Bits) : Board :=
  let next := place (moverBoard opponentMoves root) flipped sq
  if opponentMoves then next else switchPlayers next

def updatePair (opponentMoves : Bool) (p o f s : Int) : Int × Int :=
  if opponentMoves then (p - f + s, o - (2 * f + s))
  else (p - (2 * f + s), o - f + s)

theorem code_mover (squares : List Nat) (b : Board) (sq : Square.Square) (flipped : Bits)
    (valid : placeCompatible b sq flipped) (coords : ∀ i ∈ squares, i < 64)
    (unique : squares.Nodup) :
    pairedCode squares (switchPlayers (place b flipped sq)) =
      updatePair false (pairedCode squares b).1 (pairedCode squares b).2
        (lookupWeight sq.val squares) (flipCode squares flipped) := by
  unfold pairedCode updatePair flipCode
  simp only [Bool.false_eq_true, ite_false]
  have swap : switchPlayers (switchPlayers (place b flipped sq)) = place b flipped sq := rfl
  rw [swap, lookupWeightContract_proven squares _ unique]
  congr 1
  · rw [← code_scale, ← code_add, ← code_sub]
    apply code_congr
    intro i hi
    exact (digitUpdateContract_proven b sq flipped valid i (coords i hi)).1
  · rw [← code_sub, ← code_add]
    apply code_congr
    intro i hi
    exact (digitUpdateContract_proven b sq flipped valid i (coords i hi)).2

def BoardUpdateContract : Prop :=
  ∀ opponentMoves root sq flipped (lane : Fin 32),
    placeCompatible (moverBoard opponentMoves root) sq flipped →
    let squares := pattern lane
    let before := pairedCode squares root
    pairedCode squares (rootAfter opponentMoves root sq flipped) =
      updatePair opponentMoves before.1 before.2 (lookupWeight sq.val squares) (flipCode squares flipped)

theorem boardUpdateContract_proven : BoardUpdateContract := by
  intro opponentMoves root sq flipped lane compatible
  obtain ⟨unique, _, bounded⟩ := patternGeometryContract_proven lane
  have coords : ∀ i ∈ pattern lane, i < 64 := by
    intro i h
    exact of_decide_eq_true (List.all_eq_true.mp bounded i h)
  cases opponentMoves
  · exact code_mover (pattern lane) root sq flipped compatible coords unique
  · have flippedRoot := code_mover (pattern lane) (switchPlayers root) sq flipped compatible coords unique
    have exchange := congrArg (fun x : Int × Int => (x.2, x.1)) flippedRoot
    exact exchange

def FeatureBoundsContract : Prop :=
  ∀ b (lane : Fin 32),
    let bounds := fun (v : Int) => 0 ≤ v ∧ v < 3 ^ (pattern lane).length ∧ v < 65536
    bounds (pairedCode (pattern lane) b).1 ∧ bounds (pairedCode (pattern lane) b).2

theorem featureBoundsContract_proven : FeatureBoundsContract := by
  intro b lane
  have length := (patternGeometryContract_proven lane).2.1
  have naturalBound : (3 : Nat) ^ (pattern lane).length ≤ 19683 :=
    Nat.pow_le_pow_right (by decide) length
  have powBound : (3 : Int) ^ (pattern lane).length ≤ 19683 := by
    have cast := Int.natCast_pow 3 (pattern lane).length
    change Int.ofNat (3 ^ (pattern lane).length) = (3 : Int) ^ (pattern lane).length at cast
    rw [← cast]
    exact Int.ofNat_le.mpr naturalBound
  have bounds (board : Board) :
      0 ≤ code (digit board) (pattern lane) ∧
      code (digit board) (pattern lane) < 3 ^ (pattern lane).length ∧
      code (digit board) (pattern lane) < 65536 := by
    have h := code_bound (digit board) (pattern lane) (fun i _ => by
      unfold digit
      split <;> (try split) <;> decide)
    exact ⟨h.1, h.2, by omega⟩
  exact ⟨bounds b, bounds (switchPlayers b)⟩

end NeuralReversi.Eval.PatternUpdate
