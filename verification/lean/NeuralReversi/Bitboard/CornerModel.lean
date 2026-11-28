module
public import NeuralReversi.Bitboard.Model

@[expose] public section

namespace NeuralReversi.Bitboard

def cornerMask : Bits := 0x0000000000000001 ||| 0x0000000000000080 |||
  0x0100000000000000 ||| 0x8000000000000000

def corners (bits : Bits) : Bits := bits &&& cornerMask
def nonCorners (bits : Bits) : Bits := bits &&& ~~~cornerMask

def cornerWeightedCount (bits : Bits) : BitVec 32 := count bits + count (corners bits)

def cornerStableMask (bits : Bits) : Bits :=
  let aFileCorners : Bits := 0x0000000000000001 ||| 0x0100000000000000
  let hFileCorners : Bits := 0x0000000000000080 ||| 0x8000000000000000
  let rank1Corners : Bits := 0x0000000000000001 ||| 0x0000000000000080
  let rank8Corners : Bits := 0x0100000000000000 ||| 0x8000000000000000
  ((((aFileCorners &&& bits) <<< 1) ||| ((hFileCorners &&& bits) >>> 1) |||
    ((rank1Corners &&& bits) <<< 8) ||| ((rank8Corners &&& bits) >>> 8) ||| cornerMask) &&& bits)

def cornerStability (bits : Bits) : BitVec 32 := count (cornerStableMask bits)

def cornerCoordinate (i : Nat) : Bool :=
  decide ((i / 8 = 0 ∨ i / 8 = 7) ∧ (i % 8 = 0 ∨ i % 8 = 7))

def cornerStencil (i : Nat) : Bool :=
  cornerCoordinate i ||
  decide ((i / 8 = 0 ∨ i / 8 = 7) ∧ (i % 8 = 1 ∨ i % 8 = 6)) ||
  decide ((i % 8 = 0 ∨ i % 8 = 7) ∧ (i / 8 = 1 ∨ i / 8 = 6))

def cornerSupportedCell (bits : Bits) (i : Nat) : Bool :=
  bits.getLsbD i &&
    (cornerCoordinate i ||
      (decide (i / 8 = 0 ∨ i / 8 = 7) &&
        ((decide (i % 8 = 1) && bits.getLsbD (i - 1)) ||
         (decide (i % 8 = 6) && bits.getLsbD (i + 1)))) ||
      (decide (i % 8 = 0 ∨ i % 8 = 7) &&
        ((decide (i / 8 = 1) && bits.getLsbD (i - 8)) ||
         (decide (i / 8 = 6) && bits.getLsbD (i + 8)))))

end NeuralReversi.Bitboard
