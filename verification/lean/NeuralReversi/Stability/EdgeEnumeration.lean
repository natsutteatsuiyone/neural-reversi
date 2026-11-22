import NeuralReversi.Stability.EdgeModel

namespace NeuralReversi.Stability.Edge

def sourceEmptyOrder : Nat → Byte → List Nat
  | 0, _ => []
  | fuel + 1, bits => if bits = 0 then [] else
      bits.ctz.toNat :: sourceEmptyOrder fuel (bits &&& (bits - 1))

def emptyLoopStops : Nat → Byte → Bool
  | 0, bits => bits == 0
  | fuel + 1, bits => if bits == 0 then true else emptyLoopStops fuel (bits &&& (bits - 1))

def sourceSubsets (m p : Byte) : Nat → List Byte
  | 0 => []
  | fuel + 1 =>
      let next := (p - m) &&& m
      p :: if next == 0 then [] else sourceSubsets m next fuel

def subsetLoopStops (m p : Byte) : Nat → Bool
  | 0 => false
  | fuel + 1 =>
      let next := (p - m) &&& m
      if next == 0 then true else subsetLoopStops m next fuel

set_option maxRecDepth 100000
set_option maxHeartbeats 0

private theorem empty_order_finite : ∀ n : Fin 256,
    sourceEmptyOrder 8 (BitVec.ofNat 8 n.val) =
      (List.range 8).filter (fun x => (BitVec.ofNat 8 n.val).getLsbD x) ∧
    emptyLoopStops 8 (BitVec.ofNat 8 n.val) = true := by decide +kernel

theorem source_empty_order (bits : Byte) :
    sourceEmptyOrder 8 bits = (List.range 8).filter bits.getLsbD ∧
    emptyLoopStops 8 bits = true := by
  simpa only [BitVec.ofNat_toNat, BitVec.setWidth_eq] using
    empty_order_finite ⟨bits.toNat, bits.isLt⟩

private theorem subset_stops_finite : ∀ n : Fin 256,
    subsetLoopStops (BitVec.ofNat 8 n.val) 0 256 = true := by decide +kernel

theorem source_subset_loop_stops (mask : Byte) : subsetLoopStops mask 0 256 = true := by
  simpa only [BitVec.ofNat_toNat, BitVec.setWidth_eq] using
    subset_stops_finite ⟨mask.toNat, mask.isLt⟩

private theorem subsets_finite : ∀ n : Fin 256,
    sourceSubsets (BitVec.ofNat 8 n.val) 0 256 =
      ((List.range 256).filter (fun p =>
        BitVec.ofNat 8 p &&& BitVec.ofNat 8 n.val == BitVec.ofNat 8 p)).map (BitVec.ofNat 8) := by decide +kernel

theorem source_subsets_exact (mask : Byte) :
    sourceSubsets mask 0 256 =
      ((List.range 256).filter (fun p =>
        BitVec.ofNat 8 p &&& mask == BitVec.ofNat 8 p)).map (BitVec.ofNat 8) := by
  simpa only [BitVec.ofNat_toNat, BitVec.setWidth_eq] using
    subsets_finite ⟨mask.toNat, mask.isLt⟩

end NeuralReversi.Stability.Edge
