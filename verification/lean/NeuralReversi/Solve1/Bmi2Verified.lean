module
public import NeuralReversi.Solve1.Bmi2
public import NeuralReversi.Solve1.Policy

@[expose] public section

namespace NeuralReversi.Solve1.Bmi2

structure Board where
  player : Word
  square : Fin 64
  empty : player.getLsbD square.val = false

def popCount (p : Word) : Nat := p.cpop.toNat

def solve (b : Board) (alpha : ScoreWord) : ScoreWord :=
  bmi2Word (popCount b.player) (countWide b.player b.square.val).toNat
    (countOppWide b.player b.square.val).toNat alpha

def exactScore (b : Board) : Int :=
  exactCounts (popCount b.player) (2 * reference b.player b.square.val)
    (2 * reference (~~~b.player) b.square.val)

theorem popCount_bound (p : Word) : popCount p ≤ 64 := BitVec.toNat_cpop_le p

theorem solve_nws (b : Board) (alpha : ScoreWord) :
    NWSResult alpha.toInt (exactScore b) (solve b alpha).toInt := by
  have hf := reference_bound b.player b.square
  have ho := reference_bound (~~~b.player) b.square
  unfold solve exactScore
  rw [countWide_toNat, countOppWide_toNat]
  exact (backend_nws (popCount b.player) (2 * reference b.player b.square.val)
    (2 * reference (~~~b.player) b.square.val) alpha (popCount_bound b.player)
    (by omega) (by omega)).1

theorem exactScore_terminal (b : Board) :
    let n : Int := popCount b.player
    let f : Int := reference b.player b.square.val
    let fo : Int := reference (~~~b.player) b.square.val
    exactScore b =
      (if f > 0 then (n + 1 + f) - (63 - n - f)
       else if fo > 0 then (n - fo) - (64 - n + fo)
       else if n > 63 - n then (n + 1) - (63 - n) else n - (64 - n)) := by
  exact exactCounts_terminal (popCount b.player) (reference b.player b.square.val)
    (reference (~~~b.player) b.square.val)

end NeuralReversi.Solve1.Bmi2
