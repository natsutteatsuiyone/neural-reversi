import NeuralReversi.Solve1.ScalarParams
import NeuralReversi.Solve1.Projection
import NeuralReversi.Solve1.Geometry
namespace NeuralReversi.Solve1.Scalar

def laneMask (s : Nat) (lane : Fin 4) : Word :=
  let pp := params s
  match lane.val with
  | 0 => pp.maskV
  | 1 => 255 <<< pp.rowShift
  | 2 => pp.maskD7
  | _ => pp.maskD9

def laneIndex (s : Nat) (lane : Fin 4) (p : Word) : Byte :=
  let pp := params s
  match lane.val with
  | 0 => ((p &&& pp.maskV) * pp.multV).extractLsb' 56 8
  | 1 => (p >>> pp.rowShift).setWidth 8
  | 2 => ((((p &&& pp.maskD7) + pp.addend7) &&& pp.postMask7) * pp.multD7).extractLsb' 56 8
  | _ => ((p &&& pp.maskD9) * 0x0101010101010101).extractLsb' 56 8

def lanePos (s : Nat) (lane : Fin 4) : Fin 8 :=
  let pp := params s
  let n := match lane.val with | 0 => pp.t0 | 1 => pp.t1 | 2 => pp.t2 | _ => pp.t3
  ⟨n % 8, Nat.mod_lt _ (by decide)⟩

def laneReference (s : Nat) (lane : Fin 4) (p : Word) : Nat :=
  let r := fun dr dc => Geometry.ray p 0 (Geometry.raySquares s dr dc)
  match lane.val with
  | 0 => r 1 0 + r (-1) 0
  | 1 => r 0 1 + r 0 (-1)
  | 2 => if (params s).maskD9 = 0 then r 1 1 + r (-1) (-1) + r 1 (-1) + r (-1) 1
         else r 1 (-1) + r (-1) 1
  | _ => if (params s).maskD9 = 0 then 0 else r 1 1 + r (-1) (-1)

theorem and255 (x : Byte) : x &&& (255#8) = x := by
  change x &&& BitVec.allOnes 8 = x
  exact BitVec.and_allOnes
end NeuralReversi.Solve1.Scalar
