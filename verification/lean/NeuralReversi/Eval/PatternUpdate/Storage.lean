module
public import NeuralReversi.Eval.PatternUpdate.Board

@[expose] public section

namespace NeuralReversi.Eval.PatternUpdate

open Bitboard (Bits)

def updateAllowed (ply : Nat) (flips : Bits) : Bool :=
  decide (ply < 60) && !(flips == 0)

def UpdateGuardContract : Prop :=
  ∀ ply flips, updateAllowed ply flips = true ↔
    (ply < 61 ∧ ply + 1 < 61 ∧ flips ≠ 0 ∧ flips.ctz.toNat < 64)

theorem updateGuardContract_proven : UpdateGuardContract := by
  intro ply flips
  have bound : flips ≠ 0 → flips.ctz.toNat < 64 := by
    intro nonempty
    exact (BitVec.ctz_lt_iff_ne_zero.mpr nonempty)
  simp only [updateAllowed, Bool.and_eq_true, decide_eq_true_eq,
    Bool.not_eq_true', beq_eq_false_iff_ne]
  constructor
  · intro h
    exact ⟨by omega, by omega, h.2, bound h.2⟩
  · intro h
    exact ⟨by omega, h.2.2.1⟩

abbrev Word := BitVec 16

def sideMask (opponentMoves : Bool) : Word := 0 - (BitVec.ofBool opponentMoves).setWidth 16

def SideMaskContract : Prop :=
  ∀ opponentMoves (playerDelta opponentDelta : Word),
    let mask := sideMask opponentMoves
    (mask = if opponentMoves then 65535#16 else 0#16) ∧
    ((playerDelta &&& ~~~mask) ||| (opponentDelta &&& mask)) =
      if opponentMoves then opponentDelta else playerDelta

theorem sideMaskContract_proven : SideMaskContract := by
  intro side p o
  cases side
  · change 0#16 = 0#16 ∧ (p &&& ~~~0#16 ||| o &&& 0#16) = p
    simp
    exact BitVec.and_allOnes
  · change 65535#16 = 65535#16 ∧ (p &&& ~~~65535#16 ||| o &&& 65535#16) = o
    change 65535#16 = 65535#16 ∧
      (p &&& ~~~(BitVec.allOnes 16) ||| o &&& BitVec.allOnes 16) = o
    simp
    exact BitVec.and_allOnes

def scalarWords (opponentMoves : Bool) (p o f s : Word) : Word × Word :=
  if opponentMoves then (p - f + s, o - (f <<< 1) - s)
  else (p - (f <<< 1) - s, o - f + s)

def laneWords (opponentMoves : Bool) (p o f s : Word) : Word × Word :=
  let capture := (f <<< 1) + s
  let other := f - s
  if opponentMoves then (p - other, o - capture) else (p - capture, o - other)

def neonWords (opponentMoves : Bool) (p o f s : Word) : Word × Word :=
  if opponentMoves then (p - f + s, (o - f) - (f + s))
  else ((p - f) - (f + s), o - f + s)

def wordPair (values : Int × Int) : Word × Word :=
  (BitVec.ofInt 16 values.1, BitVec.ofInt 16 values.2)

private theorem subWord (x y : Int) :
    BitVec.ofInt 16 x - BitVec.ofInt 16 y = BitVec.ofInt 16 (x - y) := by
  rw [BitVec.sub_eq_add_neg, ← BitVec.ofInt_neg, ← BitVec.ofInt_add, Int.sub_eq_add_neg]

private theorem shiftWord (x : Int) :
    BitVec.ofInt 16 x <<< 1 = BitVec.ofInt 16 (x + x) := by
  rw [BitVec.shiftLeft_eq_mul_twoPow]
  change BitVec.ofInt 16 x * 2#16 = _
  rw [BitVec.mul_two, ← BitVec.ofInt_add]

def WordUpdateContract : Prop :=
  ∀ opponentMoves p o f s,
    let args := wordPair (p, o)
    let place := BitVec.ofInt 16 f
    let flips := BitVec.ofInt 16 s
    let want := wordPair (updatePair opponentMoves p o f s)
    scalarWords opponentMoves args.1 args.2 place flips = want ∧
    laneWords opponentMoves args.1 args.2 place flips = want ∧
    neonWords opponentMoves args.1 args.2 place flips = want

theorem wordUpdateContract_proven : WordUpdateContract := by
  intro side p o f s
  cases side <;> simp only [wordPair, scalarWords, laneWords, neonWords, updatePair,
    Bool.false_eq_true, ite_false, ite_true, shiftWord, subWord, ← BitVec.ofInt_add] <;>
    refine ⟨?_, ?_, ?_⟩ <;>
    congr 1 <;> apply congrArg (BitVec.ofInt 16) <;> omega

def KernelRebuildContract : Prop :=
  ∀ side b sq flipped (lane : Fin 32),
    placeCompatible (moverBoard side b) sq flipped →
    let squares := pattern lane
    let before := wordPair (pairedCode squares b)
    let f := BitVec.ofInt 16 (lookupWeight sq.val squares)
    let s := BitVec.ofInt 16 (flipCode squares flipped)
    let rebuilt := wordPair (pairedCode squares (rootAfter side b sq flipped))
    scalarWords side before.1 before.2 f s = rebuilt ∧
    laneWords side before.1 before.2 f s = rebuilt ∧
    neonWords side before.1 before.2 f s = rebuilt

theorem kernelRebuildContract_proven : KernelRebuildContract := by
  intro side b sq flipped lane compatible
  dsimp only
  rw [boardUpdateContract_proven side b sq flipped lane compatible]
  exact wordUpdateContract_proven side _ _ _ _

def storeSuccessor {α : Type} (slots : Nat → α) (ply : Nat) (next : α) (i : Nat) : α :=
  if i = ply + 1 then next else slots i

def SlotWriteContract : Prop :=
  ∀ (α : Type) (slots : Nat → α) ply (first second : α),
    storeSuccessor slots ply first (ply + 1) = first ∧
    (∀ i, i ≠ ply + 1 → storeSuccessor slots ply first i = slots i) ∧
    storeSuccessor slots ply first ply = slots ply ∧
    storeSuccessor (storeSuccessor slots ply first) ply second = storeSuccessor slots ply second

theorem slotWriteContract_proven : SlotWriteContract := by
  intro α slots ply first second
  refine ⟨by simp [storeSuccessor], fun i h => by simp [storeSuccessor, h],
    by simp [storeSuccessor], ?_⟩
  funext i
  by_cases h : i = ply + 1 <;> simp [storeSuccessor, h]

def currentCode (opponentTurn : Bool) (pair : Int × Int) : Int :=
  if opponentTurn then pair.2 else pair.1

def PerspectiveContract : Prop :=
  ∀ squares side b sq flipped,
    currentCode (!side) (pairedCode squares (rootAfter side b sq flipped)) =
      code (digit (NeuralReversi.Board.place (moverBoard side b) flipped sq)) squares ∧
    currentCode (!side) (pairedCode squares b) =
      code (digit (NeuralReversi.Board.switchPlayers (moverBoard side b))) squares ∧
    currentCode (!(!side)) (pairedCode squares b) = currentCode side (pairedCode squares b)

theorem perspectiveContract_proven : PerspectiveContract := by
  intro squares side b sq flipped
  cases side <;> exact ⟨rfl, rfl, rfl⟩

end NeuralReversi.Eval.PatternUpdate
