import NeuralReversi.EndgameSearch.FourOrder
import NeuralReversi.PvsParallel.ExactDepth
import NeuralReversi.Solve34.Target

namespace NeuralReversi.EndgameSearch.Direct

open Bitboard.Moves (X86Config)
open PvsBoardTree (Position minimax)
open ScoreRefresh (Score DiscScore toDiscDiff fromDiscDiff)
open EndgameTreeCache (Address Store discValue soundHistory)
open Flip.Dispatch (TargetConfig)

private theorem one_bounds (target : TargetConfig) (bmi2 : Bool) (mc : X86Config)
    (fc : FlipConfig) (b : Solve1.Bmi2.Board) (alpha : BitVec 32) :
    let result := (Solve2.solve1On target bmi2 b alpha).toInt
    EndgameCache.SearchRange result ∧ Solve2.NWS alpha.toInt (discValue mc fc b.asBoard) result := by
  rw [Solve2.solve1On_eq]
  have nws := (Solve1.treeContract_proven mc fc b alpha).1
  have hp := Solve1.Bmi2.popCount_bound b.player
  have hf := Solve1.Bmi2.reference_bound b.player b.square
  have ho := Solve1.Bmi2.reference_bound (~~~b.player) b.square
  constructor
  · unfold EndgameCache.SearchRange Solve1.Bmi2.solve
    rw [Solve1.Bmi2.countWide_toNat, Solve1.Bmi2.countOppWide_toNat,
      Solve1.bmi2_bridge _ _ _ alpha hp (by omega) (by omega)]
    have bounds := Solve34.leaf_count_bounds _ _ _ alpha.toInt hp
      (show 2 * Solve1.Bmi2.reference b.player b.square.val ≤ 48 by omega)
      (show 2 * Solve1.Bmi2.reference (~~~b.player) b.square.val ≤ 48 by omega)
    omega
  · exact ⟨fun _ => nws.1, fun above => Int.le_of_eq (nws.2.1 above)⟩

inductive FixedDisc (target : TargetConfig) (bmi2 : Bool) (b : Position) (alpha : BitVec 32) :
    BitVec 32 → Prop
  | one (board : Solve1.Bmi2.Board) : board.asBoard = b.val →
      FixedDisc target bmi2 b alpha (Solve2.solve1On target bmi2 board alpha)
  | two (board : Solve34.EmptyBoard 2) : board.asBoard = b.val →
      FixedDisc target bmi2 b alpha (Solve2.targetRun target bmi2 board.toTwoEmpty alpha).score
  | three (board : Solve34.EmptyBoard 3) : board.asBoard = b.val →
      FixedDisc target bmi2 b alpha (Solve34.target3Run target bmi2 board alpha).score
  | four (board : Solve34.EmptyBoard 4) : board.asBoard = b.val →
      FixedDisc target bmi2 b alpha (Solve34.target4Run target bmi2 board alpha).score

theorem FixedDisc.bounds (mc : X86Config) (fc : FlipConfig)
    (call : FixedDisc target bmi2 board alpha result) (safe : Solve34.WindowSafe alpha) :
    EndgameCache.SearchRange result.toInt ∧ Solve2.NWS alpha.toInt (discValue mc fc board.val) result.toInt := by
  obtain ⟨two, three, four⟩ := Solve34.targetTreeContract_proven target bmi2 mc fc alpha safe
  cases call with
  | one b same => exact same ▸ one_bounds target bmi2 mc fc b alpha
  | two b same => exact same ▸ two b
  | three b same => exact same ▸ three b
  | four b same => exact same ▸ four b

inductive Returned (target : TargetConfig) (bmi2 : Bool) (mc : X86Config) (fc : FlipConfig)
    (board : Position) (alpha : Score) (checked : Bool) (raw : Score) : Prop
  | zero : PvsParallel.ExactDepth.Full board 0 →
      ScoreRefresh.BoardBridge.finalScore checked board.val = .ok raw →
      Returned target bmi2 mc fc board alpha checked raw
  | fixed (result : DiscScore) :
      FixedDisc target bmi2 board (toDiscDiff alpha).raw result.raw →
      fromDiscDiff checked result = .ok raw → Returned target bmi2 mc fc board alpha checked raw
  | shallow (address : Address) (history : List Store) :
      5 ≤ (Board.emptyCount board.val).toNat → (Board.emptyCount board.val).toNat ≤ 6 →
      soundHistory mc fc history →
      (let result := run mc fc (orderedOptions target bmi2 mc fc address) .shallow board
         (toDiscDiff alpha).raw.toInt history
       fromDiscDiff checked ⟨BitVec.ofInt 32 result.score⟩ = .ok raw) →
      Returned target bmi2 mc fc board alpha checked raw

theorem Returned.bound (call : Returned target bmi2 mc fc board alpha checked raw)
    (beta : Score) (null : beta.raw.toInt = alpha.raw.toInt + 1) :
    PvsMoveLoop.Bound (minimax mc fc board) alpha.raw.toInt beta.raw.toInt raw.raw.toInt := by
  cases call with
  | zero full evaluated =>
    have h := (PvsParallel.ExactDepth.zeroDepthContract_proven mc fc board checked full).2.2
    rw [evaluated] at h
    have exactScore := Except.ok.inj h
    exact ⟨fun _ => Int.le_of_eq exactScore.symm, fun _ => Int.le_of_eq exactScore⟩
  | fixed result fixed converted =>
    have floor := ScoreRefresh.disc_floor alpha
    have low := BitVec.le_toInt alpha.raw
    have high := BitVec.toInt_lt (x := alpha.raw)
    have safe : Solve34.WindowSafe (toDiscDiff alpha).raw := by
      unfold Solve34.WindowSafe
      rw [floor]
      omega
    obtain ⟨range, nws⟩ := fixed.bounds mc fc safe
    have bound := EndgameTreeCache.rawWindow_sound mc fc board alpha.raw.toInt result.raw.toInt range
      (floor ▸ nws)
    obtain ⟨score, cast, _, scaled, _⟩ := ScoreRefresh.bounded_roundtrip result range.1 range.2
    have within : ¬ (result.raw.toInt < -127 ∨ result.raw.toInt > 127) := by
      obtain ⟨_, _⟩ := range
      omega
    have eq : fromDiscDiff checked result = .ok score := by
      simpa [fromDiscDiff, within] using cast
    cases Except.ok.inj (converted.symm.trans eq)
    rw [← null, Int.mul_comm, ← scaled] at bound
    exact PvsMoveLoop.Bound.ofSoundResult bound
  | shallow address history _ _ sound converted =>
    obtain ⟨score, cast, valid, _, _⟩ :=
      orderedContract_proven target bmi2 mc fc address .shallow board alpha beta history checked sound null
    cases Except.ok.inj (converted.symm.trans cast)
    exact PvsMoveLoop.Bound.ofSoundResult valid

end NeuralReversi.EndgameSearch.Direct
