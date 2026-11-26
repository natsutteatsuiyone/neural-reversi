import NeuralReversi.Solve2.Control
import NeuralReversi.Bitboard.CornerProperties
import NeuralReversi.Board.Move.Model
import NeuralReversi.Board.MovesModel
import NeuralReversi.MidgameAspiration.Model

namespace NeuralReversi.LateMoveReduction

structure Node where
  depth : Nat
  moveCount : Nat
  moves : Nat
  endgame : Bool
  pv : Bool
  selective : Bool

def disabled (n : Node) : Bool :=
  n.endgame || n.pv || !n.selective || n.depth < 5 || n.moveCount ≤ 2 || n.moves < 4

def maxReduction (depth : Nat) : Nat :=
  min (max (depth / 3) 1) (depth - 2)

def reduction (n : Node) (proposed : Int) : Nat :=
  if disabled n then 0 else
    min proposed.toNat (maxReduction n.depth)

abbrev Search := Nat → Int → Int → Bool → Int

def scout (search : Search) (depth r : Nat) (alpha : Int) (cut : Bool) : Int :=
  -search (depth - 1 - r) (-(alpha + 1)) (-alpha) (decide (0 < r) || !cut)

def searchMove (search : Search) (depth r : Nat) (alpha : Int) (cut : Bool) : Int :=
  let first := scout search depth r alpha cut
  if 0 < r ∧ alpha < first then
    -search (depth - 1) (-(alpha + 1)) (-alpha) (!cut)
  else first

def Contract : Prop :=
  ∀ (n : Node) (proposed : Int), 0 < n.depth →
    let r := reduction n proposed
    let child := n.depth - 1 - r
    (n.depth < 2^32 → maxReduction n.depth < 2^31) ∧
    (disabled n = true → r = 0) ∧
    r ≤ n.depth / 3 ∧ child + r + 1 = n.depth ∧ child < n.depth ∧ (0 < r → 0 < child) ∧
    ∀ (search : Search) (alpha : Int) (cut : Bool),
      let first := scout search n.depth r alpha cut
      let fullChild := search (n.depth - 1) (-(alpha + 1)) (-alpha) (!cut)
      let result := searchMove search n.depth r alpha cut
      (first ≤ alpha → result = first) ∧
      (alpha < first → result = -fullChild) ∧
      (alpha < result → result = -fullChild ∧
        ∀ value, Solve2.NWS (-(alpha + 1)) value fullChild → result ≤ -value)

theorem contract_proven : Contract := by
  intro n proposed positive
  dsimp only
  let r := reduction n proposed
  have bounds : r ≤ n.depth / 3 ∧ n.depth - 1 - r + r + 1 = n.depth ∧
      n.depth - 1 - r < n.depth ∧ (0 < r → 0 < n.depth - 1 - r) := by
    dsimp [r, reduction, maxReduction]
    split
    · omega
    · rename_i enabled
      have minDepth : 5 ≤ n.depth := by
        simp [disabled] at enabled
        omega
      omega
  refine ⟨by simp only [maxReduction]; omega, fun h => by simp [reduction, h], bounds.1, bounds.2.1, bounds.2.2.1,
    bounds.2.2.2, ?_⟩
  intro search alpha cut
  generalize reduction n proposed = k
  by_cases zero : k = 0
  · simp [zero, searchMove, scout]
    intro high value sound
    have upper := sound.1 (by omega)
    omega
  · have pos : 0 < k := Nat.pos_of_ne_zero zero
    by_cases high : alpha < scout search n.depth k alpha cut
    · simp [searchMove, pos, high, Int.not_le.mpr high]
      intro above value sound
      have upper := sound.1 (by omega)
      omega
    · simp [searchMove, pos, high]

def logTable : Array Nat :=
  #[   0,    0,  177,  281,  355,  412,  459,  498,  532,  562,
     589,  614,  636,  657,  676,  693,  710,  725,  740,  754,
     767,  779,  791,  803,  814,  824,  834,  844,  853,  862,
     871,  879,  887,  895,  903,  910,  917,  924,  931,  938,
     944,  951,  957,  963,  969,  975,  980,  986,  991,  996,
    1001, 1007, 1012, 1016, 1021, 1026, 1030, 1035, 1039, 1044,
    1048]

def logAt (n : Nat) : Nat :=
  logTable[min n 60]'(by simp [logTable]; omega)

private theorem logAt_bound (n : Nat) : logAt n ≤ 1048 := by
  have entries : ∀ i : Fin 61, logTable[i.val]'i.isLt ≤ 1048 := by
    decide
  exact entries ⟨min n 60, by omega⟩

def curve (depth moveCount : Nat) : Nat := 128 + logAt depth * logAt moveCount / 512

def CurveContract : Prop :=
  (∀ n, 60 ≤ n → logAt n = 1048) ∧
  ∀ depth moveCount,
    logAt depth * logAt moveCount ≤ 1098304 ∧
    128 ≤ curve depth moveCount ∧ curve depth moveCount ≤ 2273

theorem curveContract_proven : CurveContract := by
  refine ⟨?_, ?_⟩
  · intro n high
    simp [logAt, Nat.min_eq_right high, logTable]
  · intro depth moveCount
    have product := Nat.mul_le_mul (logAt_bound depth) (logAt_bound moveCount)
    exact ⟨product, by unfold curve; omega, by unfold curve; omega⟩

open MidgameAspiration (FiniteScore)

def ttMoveValue : Int := 2 ^ 30
def midgameOrderingValue (estimate : Int) (mobility potential : BitVec 64) : Int :=
  estimate - (Bitboard.cornerWeightedCount mobility).toNat * 64 -
    (Bitboard.cornerWeightedCount potential).toNat * 32

def OrderingContract : Prop :=
  ∀ estimate mobility potential, FiniteScore estimate →
    let first := estimate - (Bitboard.cornerWeightedCount mobility).toNat * (64 : Int)
    let value := midgameOrderingValue estimate mobility potential
    (-39294 : Int) ≤ value ∧ value ≤ first ∧ first ≤ estimate ∧ estimate < ttMoveValue

theorem orderingContract_proven : OrderingContract := by
  intro estimate mobility potential finite first value
  have mobilityBound := (Bitboard.cornerWeightedCountContract_proven mobility).2.1
  have potentialBound := (Bitboard.cornerWeightedCountContract_proven potential).2.1
  unfold FiniteScore at finite
  dsimp [value, first, midgameOrderingValue, ttMoveValue]
  omega

def margin (shortfall : Int) : Int := min (max (shortfall / 2) (-256)) 256
def pvCredit (ttPv : Bool) : Int := if ttPv then 256 else 0

def proposedReduction (n : Node) (cut ttPv : Bool) (alpha moveValue : Int) : Int :=
  let plies := ((curve n.depth n.moveCount : Int) + margin (alpha - moveValue) - pvCredit ttPv) / 256
  if !cut && decide (2 < plies) then plies - 1 else plies

def scheduled (n : Node) (cut ttPv : Bool) (alpha moveValue : Int) : Nat :=
  reduction n (proposedReduction n cut ttPv alpha moveValue)

def ArithmeticContract : Prop :=
  ∀ n cut ttPv alpha moveValue,
    (-32767 : Int) ≤ alpha → alpha ≤ 32767 → -39294 ≤ moveValue → moveValue ≤ ttMoveValue →
    let shortfall : Int := alpha - moveValue
    let fixed : Int := (curve n.depth n.moveCount : Int) + margin shortfall - pvCredit ttPv
    (-2147483648 : Int) ≤ shortfall ∧ shortfall ≤ 2147483647 ∧
    -384 ≤ fixed ∧ fixed ≤ 2529 ∧
    -2 ≤ proposedReduction n cut ttPv alpha moveValue ∧
    scheduled n cut ttPv alpha moveValue ≤ 9 ∧
    (disabled n = true → scheduled n cut ttPv alpha moveValue = 0)

theorem arithmeticContract_proven : ArithmeticContract := by
  intro n cut ttPv alpha moveValue alphaLo alphaHi lo hi shortfall fixed
  have curveBound := (curveContract_proven.2 n.depth n.moveCount).2
  have marginBound : -256 ≤ margin shortfall ∧ margin shortfall ≤ 256 := by
    unfold margin
    omega
  have creditBound : 0 ≤ pvCredit ttPv ∧ pvCredit ttPv ≤ 256 := by
    cases ttPv <;> decide
  have shortfallBound : -2147483648 ≤ shortfall ∧ shortfall ≤ 2147483647 := by
    dsimp [shortfall]
    unfold ttMoveValue at hi
    omega
  have fixedBound : -384 ≤ fixed ∧ fixed ≤ 2529 := by
    dsimp [fixed]
    omega
  have pliesBound : -2 ≤ fixed / 256 ∧ fixed / 256 ≤ 9 := by omega
  have proposal : -2 ≤ proposedReduction n cut ttPv alpha moveValue ∧
      proposedReduction n cut ttPv alpha moveValue ≤ 9 := by
    unfold proposedReduction
    change (if !cut && decide (2 < fixed / 256) then fixed / 256 - 1 else fixed / 256) ≥ -2 ∧
      (if !cut && decide (2 < fixed / 256) then fixed / 256 - 1 else fixed / 256) ≤ 9
    cases cut <;>
      by_cases reduces : 2 < fixed / 256 <;> simp [reduces] <;> omega
  refine ⟨shortfallBound.1, shortfallBound.2, fixedBound.1, fixedBound.2, proposal.1, ?_, ?_⟩
  · unfold scheduled reduction
    split <;> omega
  · intro disabled
    simp [scheduled, reduction, disabled]

private theorem fits_i32 (value : Int) (lo : -2147483648 ≤ value) (hi : value < 2147483648) :
    (BitVec.ofInt 32 value).toInt = value :=
  BitVec.toInt_ofInt_eq_self (by decide) lo hi

private theorem bmod_i32 (value : Int) (lo : -2147483648 ≤ value) (hi : value < 2147483648) :
    value.bmod (2 ^ 32) = value := by
  rw [← BitVec.toInt_ofInt (n := 32)]
  exact fits_i32 value lo hi

def WordContract : Prop :=
  (∀ alpha moveValue : Int,
    (-32767 : Int) ≤ alpha → alpha ≤ 32767 → -39294 ≤ moveValue → moveValue ≤ ttMoveValue →
    ((BitVec.ofInt 32 alpha - BitVec.ofInt 32 moveValue).sshiftRight 1).toInt =
      (alpha - moveValue) / 2) ∧
  (∀ (depth moveCount : Nat) (shortfall : Int) (ttPv : Bool),
    let base : Int := curve depth moveCount
    let credit := pvCredit ttPv
    let fixed := base + margin shortfall - credit
    ((BitVec.ofInt 32 base + BitVec.ofInt 32 (margin shortfall) -
      BitVec.ofInt 32 credit).sshiftRight 8).toInt = fixed / 256)

theorem wordContract_proven : WordContract := by
  constructor
  · intro alpha value alphaLo alphaHi valueLo valueHi
    unfold ttMoveValue at valueHi
    rw [BitVec.toInt_sshiftRight, BitVec.toInt_sub,
      fits_i32 alpha (by omega) (by omega), fits_i32 value (by omega) (by omega)]
    rw [bmod_i32 (alpha - value) (by omega) (by omega), Int.shiftRight_eq_div_pow]
    rfl
  · intro depth moveCount shortfall ttPv base credit fixed
    have baseBound : 128 ≤ base ∧ base ≤ 2273 := by
      have := (curveContract_proven.2 depth moveCount).2
      dsimp [base]
      omega
    have marginBound : -256 ≤ margin shortfall ∧ margin shortfall ≤ 256 := by
      unfold margin
      omega
    have creditBound : 0 ≤ credit ∧ credit ≤ 256 := by cases ttPv <;> decide
    rw [BitVec.toInt_sshiftRight, BitVec.toInt_sub, BitVec.toInt_add,
      fits_i32 base (by omega) (by omega), fits_i32 (margin shortfall) (by omega) (by omega),
      fits_i32 credit (by omega) (by omega)]
    rw [bmod_i32 (base + margin shortfall) (by omega) (by omega)]
    change fixed.bmod (2 ^ 32) >>> 8 = fixed / 256
    rw [bmod_i32 fixed (by dsimp [fixed]; omega) (by dsimp [fixed]; omega), Int.shiftRight_eq_div_pow]
    rfl

def OrderingChildPlyContract : Prop :=
  ∀ ply empties remaining : Nat, ply + empties = 60 → 1 < remaining → remaining ≤ empties →
    ply + 1 < 60

theorem orderingChildPlyContract_proven : OrderingChildPlyContract := by
  intro ply empties remaining numbering plural available
  omega

def OrderingRouteContract : Prop :=
  ∀ (n : Node) presearched, disabled n = false → presearched ≤ 1 →
    n.endgame = false ∧ 1 < n.moves - presearched ∧ 5 ≤ n.depth

theorem orderingRouteContract_proven : OrderingRouteContract := by
  intro n presearched enabled presearchBound
  have midgame : n.endgame = false := by
    cases phase : n.endgame
    · rfl
    · simp [disabled, phase] at enabled
  simp [disabled] at enabled
  exact ⟨midgame, by omega, by omega⟩

theorem terminal_child_with_empty_squares :
    let board : Board.Board := ⟨132096, 9223372036854776320⟩
    let next : Board.Board := ⟨9223372036854775808, 132864⟩
    Board.getMoves ⟨false, false⟩ board = 258 ∧
    Board.makeMove ⟨false, false, false⟩ true true board ⟨8, by decide⟩ = .ok next ∧
    Board.emptyCount next = 59 ∧ Board.isGameOver ⟨false, false⟩ next = true := by
  dsimp only
  exact ⟨by decide, rfl, by decide, by decide⟩

end NeuralReversi.LateMoveReduction
