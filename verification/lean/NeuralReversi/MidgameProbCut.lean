import NeuralReversi.Binary64Arithmetic
import NeuralReversi.EndgameProbCut

namespace NeuralReversi.MidgameProbCut

open Aspiration (SoundResult Window)
open EndgameProbCut (clamp)
open PvsMoveLoop (scoreInf scoreMax)

def shallowDepth (depth : Nat) : Nat :=
  2 * (depth / 6)

def DepthContract : Prop :=
  (∀ depth < 2 ^ 32, shallowDepth depth < 2 ^ 32) ∧
  (∀ depth, 0 < depth → shallowDepth depth < depth) ∧
  (∀ depth, shallowDepth depth = 0 ↔ depth < 6) ∧
  (∀ depth < 60, shallowDepth depth % 2 = 0 ∧ shallowDepth depth ≤ 18)

theorem depthContract_proven : DepthContract := by
  unfold DepthContract shallowDepth
  refine ⟨?_, ?_, ?_, ?_⟩ <;> (intro; omega)

inductive Reached : Bool → Nat → Nat → Prop where
  | root : ply + depth ≤ 60 → Reached true ply depth
  | pass : Reached root ply depth → Reached root ply depth
  | reduce : Reached root ply depth → next ≤ depth → Reached root ply next
  | place : Reached root ply depth → next < depth → Reached false (ply + 1) next
  | verify : Reached false ply depth → 0 < shallowDepth depth → Reached false ply (shallowDepth depth)

def ReachContract : Prop :=
  ∀ root ply depth, Reached root ply depth →
    ply + depth ≤ 60 ∧ (root = false → 0 < ply) ∧
    (root = false → 4 ≤ depth → ply < 60 ∧ depth < 60)

theorem reachContract_proven : ReachContract := by
  intro root ply depth reached
  have bounds : ply + depth ≤ 60 ∧ (root = false → 0 < ply) := by
    induction reached with
    | root h => exact ⟨h, by simp⟩
    | pass _ ih => exact ih
    | reduce _ smaller ih => exact ⟨by have := ih.1; omega, ih.2⟩
    | place _ smaller ih => exact ⟨by have := ih.1; omega, by omega⟩
    | @verify ply depth _ smaller ih =>
      have reduction := depthContract_proven.2.1 depth (by unfold shallowDepth at smaller; omega)
      exact ⟨by have := ih.1; omega, ih.2⟩
  exact ⟨bounds.1, bounds.2, fun nonroot gate => by have := bounds.2 nonroot; have := bounds.1; omega⟩

def boardPly (board : PvsBoardTree.Position) : Nat :=
  60 - (Board.emptyCount board.val).toNat

def BoardPlyContract : Prop :=
  ∀ (mc : Bitboard.Moves.X86Config) (fc : Flip.Dispatch.X86Config)
    (board : PvsBoardTree.Position) (move : PvsBoardTree.GeneratedSquare mc board.val),
    (Board.emptyCount board.val).toNat ≤ 60 →
    boardPly (PvsBoardTree.pass board) = boardPly board ∧
    boardPly (PvsBoardTree.play mc fc board move).val = boardPly board + 1 ∧
    ∀ requested, Reached true (boardPly board) (min (max 1 requested) (Board.emptyCount board.val).toNat)

theorem boardPlyContract_proven : BoardPlyContract := by
  intro mc fc board move fits
  have accepted := (Board.precomputedMoveContract_proven mc fc true true
    board.val move.val board.property move.property).1
  have consumes := (Board.checkedPlacementProgress_proven true _ _ _ _ board.property accepted).2
  change (Board.emptyCount (PvsBoardTree.play mc fc board move).val.val).toNat + 1 =
    (Board.emptyCount board.val).toNat at consumes
  refine ⟨?_, ?_, fun requested => .root ?_⟩
  · change 60 - (Board.emptyCount (Board.switchPlayers board.val)).toNat = boardPly board
    rw [Board.switchPlayers_emptyCount]
    rfl
  · unfold boardPly
    omega
  · unfold boardPly
    omega

def table (cell : Fin 60 → Fin 60 → α) : Array (Array α) :=
  Array.ofFn fun row => Array.ofFn (cell row)

def lookup (entries : Array (Array α)) (ply depth : Nat) : Option α := do
  let row ← entries[ply]?
  row[depth]?

def TableContract : Prop :=
  ∀ (Value : Type) (cell : Fin 60 → Fin 60 → Value) (ply depth : Nat),
    lookup (table cell) ply depth =
      if p : ply < 60 then if d : depth < 60 then some (cell ⟨ply, p⟩ ⟨depth, d⟩) else none
      else none

theorem tableContract_proven : TableContract := by
  intro Value cell ply depth
  by_cases row : ply < 60 <;> by_cases column : depth < 60 <;> simp [lookup, table, row, column]

def publicThreshold (threshold : Int) : Int :=
  clamp (-scoreInf) scoreInf threshold

def verificationBeta (threshold : Int) : Int :=
  max (-scoreInf + 1) (publicThreshold threshold)

def ThresholdContract : Prop :=
  (∀ threshold, -scoreInf ≤ publicThreshold threshold ∧ publicThreshold threshold ≤ scoreInf) ∧
  (∀ threshold, scoreMax ≤ verificationBeta threshold ↔ scoreMax ≤ threshold) ∧
  (∀ threshold, verificationBeta threshold < scoreMax →
    Window (verificationBeta threshold - 1) (verificationBeta threshold)) ∧
  (∀ threshold score, -scoreInf < score → score < scoreInf →
    (publicThreshold threshold ≤ score ↔ threshold ≤ score) ∧
    (verificationBeta threshold ≤ score ↔ threshold ≤ score))

theorem thresholdContract_proven : ThresholdContract := by
  unfold ThresholdContract
  simp only [publicThreshold, verificationBeta, clamp, scoreInf, scoreMax]
  refine ⟨fun _ => ⟨by omega, by omega⟩, fun _ => by omega, fun _ _ => ⟨by omega, by omega, by omega⟩,
    fun _ _ _ _ => ⟨by omega, by omega⟩⟩

structure Coefficients where
  mean : Int
  sigma : Int
  mean0 : Int
  sigma0 : Int

def thresholdsFromUnits (beta t : Int) (coefficients : Coefficients) (cutNode : Bool) :
    Option (Int × Int) := do
  let quantum : Int := EndgameTiming.quantum
  let raw := beta * quantum
  let half := quantum / 2
  let verificationDeviation ← Binary64Arithmetic.mul t coefficients.sigma
  let verificationOffset ← Binary64Arithmetic.add raw verificationDeviation
  let verification ← Binary64Arithmetic.sub verificationOffset coefficients.mean
  let halfMean0 ← Binary64Arithmetic.mul half coefficients.mean0
  let evaluationMean ← Binary64Arithmetic.add halfMean0 coefficients.mean
  let halfT ← Binary64Arithmetic.mul t half
  let evaluationDeviation0 ← Binary64Arithmetic.mul halfT coefficients.sigma0
  let evaluationDeviation ← Binary64Arithmetic.add evaluationDeviation0 coefficients.sigma
  let allMargin ← if cutNode then some 0 else Binary64Arithmetic.mul coefficients.sigma0 (3 * half)
  let preliminary ← Binary64Arithmetic.sub raw evaluationDeviation
  let corrected ← Binary64Arithmetic.sub preliminary evaluationMean
  let screen ← Binary64Arithmetic.add corrected allMargin
  return (EndgameProbCut.roundedUnits true verification, EndgameProbCut.roundedUnits false screen)

structure Call where
  depth : Nat
  alpha : Int
  beta : Int
  selectivity : Nat
  cutNode : Bool
  deriving DecidableEq

inductive Plan where
  | skip
  | useStatic (beta : Int)
  | verify (call : Call)
  deriving DecidableEq

def plan (pv : Bool) (depth selectivity : Nat) (evaluation rawVerification rawScreen : Int) : Plan :=
  let beta := verificationBeta rawVerification
  let pcDepth := shallowDepth depth
  if pv = true ∨ depth < 4 ∨ selectivity = 5 then .skip
  else if scoreMax ≤ beta ∨ evaluation < publicThreshold rawScreen then .skip
  else if pcDepth = 0 then .useStatic beta
  else .verify ⟨pcDepth, beta - 1, beta, 5, true⟩

structure Outcome where
  call : Option Call
  selectivity : Nat
  cutoff : Option Int
  deriving DecidableEq

def run (selectivity : Nat) (beta evaluation : Int) (verify : Call → Int) : Plan → Outcome
  | .skip => ⟨none, selectivity, none⟩
  | .useStatic requested => ⟨none, selectivity, if requested ≤ evaluation then some beta else none⟩
  | .verify call => ⟨some call, selectivity, if call.beta ≤ verify call then some beta else none⟩

def CallContract : Prop :=
  ∀ (pv : Bool) (depth selectivity : Nat) (beta evaluation rawVerification rawScreen : Int) (verify : Call → Int),
    let expected := plan pv depth selectivity evaluation rawVerification rawScreen
    let result := run selectivity beta evaluation verify expected
    result.selectivity = selectivity ∧
    (∀ call, result.call = some call →
      0 < call.depth ∧ call.depth < depth ∧ call.beta = call.alpha + 1 ∧ Window call.alpha call.beta ∧
      call.selectivity = 5 ∧ call.cutNode = true ∧
      ∀ d, plan false d call.selectivity evaluation rawVerification rawScreen = .skip) ∧
    (shallowDepth depth = 0 → result.call = none)

theorem callContract_proven : CallContract := by
  intro pv depth selectivity beta evaluation rawVerification rawScreen verify expected result
  dsimp only [result, expected, plan]
  split
  · simp [run]
  · split
    · simp [run]
    · rename_i eligible fits
      split
      · simp [run]
      · rename_i searched
        refine ⟨rfl, ?_, fun premise => False.elim (searched premise)⟩
        intro call actual
        simp only [run, Option.some.injEq] at actual
        subst call
        have lessDepth := depthContract_proven.2.1 depth (by unfold shallowDepth at searched; omega)
        have window := thresholdContract_proven.2.2.1 rawVerification (by omega)
        refine ⟨Nat.pos_of_ne_zero searched, lessDepth, ?_, window, rfl, rfl, ?_⟩
        · change verificationBeta rawVerification = verificationBeta rawVerification - 1 + 1
          omega
        · intro d
          simp

def CutContract : Prop :=
  ∀ (pv : Bool) (depth selectivity : Nat) (beta evaluation rawVerification rawScreen returned : Int)
    (verify : Call → Int) (shallow : Int),
    -scoreInf < evaluation → evaluation < scoreInf →
    -scoreInf < shallow → shallow < scoreInf →
    (∀ call, Window call.alpha call.beta → SoundResult shallow call.alpha call.beta (verify call)) →
    let expected := plan pv depth selectivity evaluation rawVerification rawScreen
    let result := run selectivity beta evaluation verify expected
    (result.cutoff = some returned ↔
      pv = false ∧ 4 ≤ depth ∧ selectivity ≠ 5 ∧ rawVerification < scoreMax ∧ rawScreen ≤ evaluation ∧
      rawVerification ≤ (if shallowDepth depth = 0 then evaluation else shallow) ∧ returned = beta) ∧
    (∀ full, result.cutoff = some returned →
      (if shallowDepth depth = 0 then evaluation else shallow) - (rawVerification - beta) ≤ full →
      returned ≤ full)

theorem cutContract_proven : CutContract := by
  intro pv depth selectivity beta evaluation rawVerification rawScreen returned verify shallow lower upper shallowLow shallowHigh
    sound expected result
  have cap : scoreMax ≤ verificationBeta rawVerification ↔ scoreMax ≤ rawVerification :=
    thresholdContract_proven.2.1 rawVerification
  have screen : publicThreshold rawScreen ≤ evaluation ↔ rawScreen ≤ evaluation :=
    (thresholdContract_proven.2.2.2 rawScreen evaluation lower upper).1
  have static : verificationBeta rawVerification ≤ evaluation ↔ rawVerification ≤ evaluation :=
    (thresholdContract_proven.2.2.2 rawVerification evaluation lower upper).2
  have decision : result.cutoff = some returned ↔
      pv = false ∧ 4 ≤ depth ∧ selectivity ≠ 5 ∧ rawVerification < scoreMax ∧ rawScreen ≤ evaluation ∧
      rawVerification ≤ (if shallowDepth depth = 0 then evaluation else shallow) ∧ returned = beta := by
    have searched : verificationBeta rawVerification < scoreMax →
        let call : Call := ⟨shallowDepth depth, verificationBeta rawVerification - 1,
          verificationBeta rawVerification, 5, true⟩
        verificationBeta rawVerification ≤ verify call ↔ rawVerification ≤ shallow := by
      intro fit call
      have obs := sound call (thresholdContract_proven.2.2.1 rawVerification fit)
      have shifted := ShallowSearch.reachesThreshold _ _ _ obs
      exact shifted.trans ((thresholdContract_proven.2.2.2 rawVerification shallow shallowLow shallowHigh).2)
    by_cases eligible : pv = true ∨ depth < 4 ∨ selectivity = 5
    · have excluded : ¬ (pv = false ∧ 4 ≤ depth ∧ selectivity ≠ 5) := by
        rcases eligible with isPV | short | exactLevel <;> simp_all <;> omega
      simp only [result, expected, plan, ite_eq_left eligible, run, reduceCtorEq]
      exact (iff_false_intro fun h => excluded ⟨h.1, h.2.1, h.2.2.1⟩).symm
    · have groups : pv = false ∧ 4 ≤ depth ∧ selectivity ≠ 5 := by
        cases pv <;> simp_all <;> omega
      dsimp only [result, expected, plan]
      rw [ite_eq_right eligible]
      by_cases deny : scoreMax ≤ verificationBeta rawVerification ∨ evaluation < publicThreshold rawScreen
      · simp only [ite_eq_left deny, run, reduceCtorEq]
        apply (iff_false_intro ?_).symm
        rintro ⟨_, _, _, uncapped, screened, _, _⟩
        exact deny.elim (fun exceeds => Int.not_le_of_gt uncapped (cap.mp exceeds))
          (fun misses => Int.not_lt_of_ge (screen.mpr screened) misses)
      · have covers : verificationBeta rawVerification < scoreMax := by omega
        have room : rawVerification < scoreMax := by omega
        have passesScreen : rawScreen ≤ evaluation := screen.mp (by omega)
        rw [ite_eq_right deny]
        by_cases zero : shallowDepth depth = 0
        · simp [zero, run, static, groups.1, groups.2.1, groups.2.2, room, passesScreen, eq_comm]
        · simp [zero, run, searched covers, groups.1, groups.2.1, groups.2.2, room, passesScreen, eq_comm]
  exact ⟨decision, fun full pass margin => by
    obtain ⟨_, _, _, _, _, reached, equal⟩ := decision.mp pass
    omega⟩

example :
    lookup (table fun (p d : Fin 60) => (p.val, d.val)) 1 2 = some (1, 2) ∧
    lookup (table fun (p d : Fin 60) => (p.val, d.val)) 0 60 = none ∧
    plan false 5 0 0 (-32768) (-32768) = .useStatic (-32766) ∧
    plan false 6 0 0 (-32768) (-32768) = .verify ⟨2, -32767, -32766, 5, true⟩ ∧
    plan false 6 5 0 (-32768) (-32768) = .skip ∧
    plan false 6 0 0 16384 (-32768) = .skip := by decide

end NeuralReversi.MidgameProbCut
