module
public import NeuralReversi.Solve1.LineTable
public import NeuralReversi.Solve1.Geometry

@[expose] public section

namespace NeuralReversi.Solve1.Bmi2
abbrev Word := BitVec 64
abbrev Byte := BitVec 8
abbrev Index := BitVec 16
structure UnionParams where
  fileCells : List Nat
  unionCells : List Nat
  d0Map : List (Nat × Nat)
  d1Map : List (Nat × Nat)
  d0Pos : Nat
  d1Pos : Nat
  rowPos : Nat
  filePos : Nat
  rankShift : Nat
  offset : Nat
  size : Nat
  deriving Repr, Inhabited

def up0 : UnionParams := ⟨[0,8,16,24,32,40,48,56], [9,18,27,36,45,54,63], [], [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7)], 0, 0, 0, 0, 0, 0, 128⟩
def up1 : UnionParams := ⟨[1,9,17,25,33,41,49,57], [8,10,19,28,37,46,55], [(0,1)], [(1,1),(2,2),(3,3),(4,4),(5,5),(6,6)], 0, 0, 1, 0, 0, 128, 128⟩
def up2 : UnionParams := ⟨[2,10,18,26,34,42,50,58], [9,11,16,20,29,38,47], [(0,1),(2,2)], [(1,1),(3,2),(4,3),(5,4),(6,5)], 0, 0, 2, 0, 0, 256, 128⟩
def up3 : UnionParams := ⟨[3,11,19,27,35,43,51,59], [10,12,17,21,24,30,39], [(0,1),(2,2),(4,3)], [(1,1),(3,2),(5,3),(6,4)], 0, 0, 3, 0, 0, 384, 128⟩
def up4 : UnionParams := ⟨[4,12,20,28,36,44,52,60], [11,13,18,22,25,31,32], [(0,1),(2,2),(4,3),(6,4)], [(1,1),(3,2),(5,3)], 0, 0, 4, 0, 0, 512, 128⟩
def up5 : UnionParams := ⟨[5,13,21,29,37,45,53,61], [12,14,19,23,26,33,40], [(0,1),(2,2),(4,3),(5,4),(6,5)], [(1,1),(3,2)], 0, 0, 5, 0, 0, 640, 128⟩
def up6 : UnionParams := ⟨[6,14,22,30,38,46,54,62], [13,15,20,27,34,41,48], [(0,1),(2,2),(3,3),(4,4),(5,5),(6,6)], [(1,1)], 0, 0, 6, 0, 0, 768, 128⟩
def up7 : UnionParams := ⟨[7,15,23,31,39,47,55,63], [14,21,28,35,42,49,56], [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7)], [], 0, 0, 7, 0, 0, 896, 128⟩
def up8 : UnionParams := ⟨[0,8,16,24,32,40,48,56], [1,17,26,35,44,53,62], [(0,0)], [(1,1),(2,2),(3,3),(4,4),(5,5),(6,6)], 1, 0, 0, 1, 8, 1024, 128⟩
def up9 : UnionParams := ⟨[1,9,17,25,33,41,49,57], [0,2,16,18,27,36,45,54,63], [(1,0),(2,2)], [(0,0),(3,2),(4,3),(5,4),(6,5),(7,6),(8,7)], 1, 1, 1, 1, 8, 1152, 512⟩
def up10 : UnionParams := ⟨[2,10,18,26,34,42,50,58], [1,3,17,19,24,28,37,46,55], [(1,0),(2,2),(4,3)], [(0,0),(3,2),(5,3),(6,4),(7,5),(8,6)], 1, 1, 2, 1, 8, 1664, 512⟩
def up11 : UnionParams := ⟨[3,11,19,27,35,43,51,59], [2,4,18,20,25,29,32,38,47], [(1,0),(2,2),(4,3),(6,4)], [(0,0),(3,2),(5,3),(7,4),(8,5)], 1, 1, 3, 1, 8, 2176, 512⟩
def up12 : UnionParams := ⟨[4,12,20,28,36,44,52,60], [3,5,19,21,26,30,33,39,40], [(1,0),(2,2),(4,3),(6,4),(8,5)], [(0,0),(3,2),(5,3),(7,4)], 1, 1, 4, 1, 8, 2688, 512⟩
def up13 : UnionParams := ⟨[5,13,21,29,37,45,53,61], [4,6,20,22,27,31,34,41,48], [(1,0),(2,2),(4,3),(6,4),(7,5),(8,6)], [(0,0),(3,2),(5,3)], 1, 1, 5, 1, 8, 3200, 512⟩
def up14 : UnionParams := ⟨[6,14,22,30,38,46,54,62], [5,7,21,23,28,35,42,49,56], [(1,0),(2,2),(4,3),(5,4),(6,5),(7,6),(8,7)], [(0,0),(3,2)], 1, 1, 6, 1, 8, 3712, 512⟩
def up15 : UnionParams := ⟨[7,15,23,31,39,47,55,63], [6,22,29,36,43,50,57], [(1,1),(2,2),(3,3),(4,4),(5,5),(6,6)], [(0,0)], 0, 1, 7, 1, 8, 4224, 128⟩
def up16 : UnionParams := ⟨[0,8,16,24,32,40,48,56], [2,9,25,34,43,52,61], [(0,0),(1,1)], [(2,1),(3,2),(4,3),(5,4),(6,5)], 2, 0, 0, 2, 16, 4352, 128⟩
def up17 : UnionParams := ⟨[1,9,17,25,33,41,49,57], [3,8,10,24,26,35,44,53,62], [(0,0),(2,1),(3,3)], [(1,0),(4,2),(5,3),(6,4),(7,5),(8,6)], 2, 1, 1, 2, 16, 4480, 512⟩
def up18 : UnionParams := ⟨[2,10,18,26,34,42,50,58], [0,4,9,11,25,27,32,36,45,54,63], [(1,0),(3,1),(4,3),(6,4)], [(0,0),(2,1),(5,3),(7,4),(8,5),(9,6),(10,7)], 2, 2, 2, 2, 16, 4992, 2048⟩
def up19 : UnionParams := ⟨[3,11,19,27,35,43,51,59], [1,5,10,12,26,28,33,37,40,46,55], [(1,0),(3,1),(4,3),(6,4),(8,5)], [(0,0),(2,1),(5,3),(7,4),(9,5),(10,6)], 2, 2, 3, 2, 16, 7040, 2048⟩
def up20 : UnionParams := ⟨[4,12,20,28,36,44,52,60], [2,6,11,13,27,29,34,38,41,47,48], [(1,0),(3,1),(4,3),(6,4),(8,5),(10,6)], [(0,0),(2,1),(5,3),(7,4),(9,5)], 2, 2, 4, 2, 16, 9088, 2048⟩
def up21 : UnionParams := ⟨[5,13,21,29,37,45,53,61], [3,7,12,14,28,30,35,39,42,49,56], [(1,0),(3,1),(4,3),(6,4),(8,5),(9,6),(10,7)], [(0,0),(2,1),(5,3),(7,4)], 2, 2, 5, 2, 16, 11136, 2048⟩
def up22 : UnionParams := ⟨[6,14,22,30,38,46,54,62], [4,13,15,29,31,36,43,50,57], [(2,0),(3,2),(5,3),(6,4),(7,5),(8,6)], [(0,0),(1,1),(4,3)], 1, 2, 6, 2, 16, 13184, 512⟩
def up23 : UnionParams := ⟨[7,15,23,31,39,47,55,63], [5,14,30,37,44,51,58], [(2,1),(3,2),(4,3),(5,4),(6,5)], [(0,0),(1,1)], 0, 2, 7, 2, 16, 13696, 128⟩
def up24 : UnionParams := ⟨[0,8,16,24,32,40,48,56], [3,10,17,33,42,51,60], [(0,0),(1,1),(2,2)], [(3,1),(4,2),(5,3),(6,4)], 3, 0, 0, 3, 24, 13824, 128⟩
def up25 : UnionParams := ⟨[1,9,17,25,33,41,49,57], [4,11,16,18,32,34,43,52,61], [(0,0),(1,1),(3,2),(4,4)], [(2,0),(5,2),(6,3),(7,4),(8,5)], 3, 1, 1, 3, 24, 13952, 512⟩
def up26 : UnionParams := ⟨[2,10,18,26,34,42,50,58], [5,8,12,17,19,33,35,40,44,53,62], [(0,0),(2,1),(4,2),(5,4),(7,5)], [(1,0),(3,1),(6,3),(8,4),(9,5),(10,6)], 3, 2, 2, 3, 24, 14464, 2048⟩
def up27 : UnionParams := ⟨[3,11,19,27,35,43,51,59], [0,6,9,13,18,20,34,36,41,45,48,54,63], [(1,0),(3,1),(5,2),(6,4),(8,5),(10,6)], [(0,0),(2,1),(4,2),(7,4),(9,5),(11,6),(12,7)], 3, 3, 3, 3, 24, 16512, 8192⟩
def up28 : UnionParams := ⟨[4,12,20,28,36,44,52,60], [1,7,10,14,19,21,35,37,42,46,49,55,56], [(1,0),(3,1),(5,2),(6,4),(8,5),(10,6),(12,7)], [(0,0),(2,1),(4,2),(7,4),(9,5),(11,6)], 3, 3, 4, 3, 24, 24704, 8192⟩
def up29 : UnionParams := ⟨[5,13,21,29,37,45,53,61], [2,11,15,20,22,36,38,43,47,50,57], [(2,0),(4,1),(5,3),(7,4),(9,5),(10,6)], [(0,0),(1,1),(3,2),(6,4),(8,5)], 2, 3, 5, 3, 24, 32896, 2048⟩
def up30 : UnionParams := ⟨[6,14,22,30,38,46,54,62], [3,12,21,23,37,39,44,51,58], [(3,0),(4,2),(6,3),(7,4),(8,5)], [(0,0),(1,1),(2,2),(5,4)], 1, 3, 6, 3, 24, 34944, 512⟩
def up31 : UnionParams := ⟨[7,15,23,31,39,47,55,63], [4,13,22,38,45,52,59], [(3,1),(4,2),(5,3),(6,4)], [(0,0),(1,1),(2,2)], 0, 3, 7, 3, 24, 35456, 128⟩
def up32 : UnionParams := ⟨[0,8,16,24,32,40,48,56], [4,11,18,25,41,50,59], [(0,0),(1,1),(2,2),(3,3)], [(4,1),(5,2),(6,3)], 4, 0, 0, 4, 32, 35584, 128⟩
def up33 : UnionParams := ⟨[1,9,17,25,33,41,49,57], [5,12,19,24,26,40,42,51,60], [(0,0),(1,1),(2,2),(4,3),(5,5)], [(3,0),(6,2),(7,3),(8,4)], 4, 1, 1, 4, 32, 35712, 512⟩
def up34 : UnionParams := ⟨[2,10,18,26,34,42,50,58], [6,13,16,20,25,27,41,43,48,52,61], [(0,0),(1,1),(3,2),(5,3),(6,5),(8,6)], [(2,0),(4,1),(7,3),(9,4),(10,5)], 4, 2, 2, 4, 32, 36224, 2048⟩
def up35 : UnionParams := ⟨[3,11,19,27,35,43,51,59], [7,8,14,17,21,26,28,42,44,49,53,56,62], [(0,0),(2,1),(4,2),(6,3),(7,5),(9,6),(11,7)], [(1,0),(3,1),(5,2),(8,4),(10,5),(12,6)], 4, 3, 3, 4, 32, 38272, 8192⟩
def up36 : UnionParams := ⟨[4,12,20,28,36,44,52,60], [0,9,15,18,22,27,29,43,45,50,54,57,63], [(2,0),(4,1),(6,2),(7,4),(9,5),(11,6)], [(0,0),(1,1),(3,2),(5,3),(8,5),(10,6),(12,7)], 3, 4, 4, 4, 32, 46464, 8192⟩
def up37 : UnionParams := ⟨[5,13,21,29,37,45,53,61], [1,10,19,23,28,30,44,46,51,55,58], [(3,0),(5,1),(6,3),(8,4),(10,5)], [(0,0),(1,1),(2,2),(4,3),(7,5),(9,6)], 2, 4, 5, 4, 32, 54656, 2048⟩
def up38 : UnionParams := ⟨[6,14,22,30,38,46,54,62], [2,11,20,29,31,45,47,52,59], [(4,0),(5,2),(7,3),(8,4)], [(0,0),(1,1),(2,2),(3,3),(6,5)], 1, 4, 6, 4, 32, 56704, 512⟩
def up39 : UnionParams := ⟨[7,15,23,31,39,47,55,63], [3,12,21,30,46,53,60], [(4,1),(5,2),(6,3)], [(0,0),(1,1),(2,2),(3,3)], 0, 4, 7, 4, 32, 57216, 128⟩
def up40 : UnionParams := ⟨[0,8,16,24,32,40,48,56], [5,12,19,26,33,49,58], [(0,0),(1,1),(2,2),(3,3),(4,4)], [(5,1),(6,2)], 5, 0, 0, 5, 40, 57344, 128⟩
def up41 : UnionParams := ⟨[1,9,17,25,33,41,49,57], [6,13,20,27,32,34,48,50,59], [(0,0),(1,1),(2,2),(3,3),(5,4),(6,6)], [(4,0),(7,2),(8,3)], 5, 1, 1, 5, 40, 57472, 512⟩
def up42 : UnionParams := ⟨[2,10,18,26,34,42,50,58], [7,14,21,24,28,33,35,49,51,56,60], [(0,0),(1,1),(2,2),(4,3),(6,4),(7,6),(9,7)], [(3,0),(5,1),(8,3),(10,4)], 5, 2, 2, 5, 40, 57984, 2048⟩
def up43 : UnionParams := ⟨[3,11,19,27,35,43,51,59], [15,16,22,25,29,34,36,50,52,57,61], [(0,0),(2,1),(4,2),(6,3),(7,5),(9,6)], [(1,0),(3,1),(5,2),(8,4),(10,5)], 4, 3, 3, 5, 40, 60032, 2048⟩
def up44 : UnionParams := ⟨[4,12,20,28,36,44,52,60], [8,17,23,26,30,35,37,51,53,58,62], [(2,0),(4,1),(6,2),(7,4),(9,5)], [(0,0),(1,1),(3,2),(5,3),(8,5),(10,6)], 3, 4, 4, 5, 40, 62080, 2048⟩
def up45 : UnionParams := ⟨[5,13,21,29,37,45,53,61], [0,9,18,27,31,36,38,52,54,59,63], [(4,0),(6,1),(7,3),(9,4)], [(0,0),(1,1),(2,2),(3,3),(5,4),(8,6),(10,7)], 2, 5, 5, 5, 40, 64128, 2048⟩
def up46 : UnionParams := ⟨[6,14,22,30,38,46,54,62], [1,10,19,28,37,39,53,55,60], [(5,0),(6,2),(8,3)], [(0,0),(1,1),(2,2),(3,3),(4,4),(7,6)], 1, 5, 6, 5, 40, 66176, 512⟩
def up47 : UnionParams := ⟨[7,15,23,31,39,47,55,63], [2,11,20,29,38,54,61], [(5,1),(6,2)], [(0,0),(1,1),(2,2),(3,3),(4,4)], 0, 5, 7, 5, 40, 66688, 128⟩
def up48 : UnionParams := ⟨[0,8,16,24,32,40,48,56], [6,13,20,27,34,41,57], [(0,0),(1,1),(2,2),(3,3),(4,4),(5,5)], [(6,1)], 6, 0, 0, 6, 48, 66816, 128⟩
def up49 : UnionParams := ⟨[1,9,17,25,33,41,49,57], [7,14,21,28,35,40,42,56,58], [(0,0),(1,1),(2,2),(3,3),(4,4),(6,5),(7,7)], [(5,0),(8,2)], 6, 1, 1, 6, 48, 66944, 512⟩
def up50 : UnionParams := ⟨[2,10,18,26,34,42,50,58], [15,22,29,32,36,41,43,57,59], [(0,0),(1,1),(2,2),(4,3),(6,4),(7,6)], [(3,0),(5,1),(8,3)], 5, 2, 2, 6, 48, 67456, 512⟩
def up51 : UnionParams := ⟨[3,11,19,27,35,43,51,59], [23,24,30,33,37,42,44,58,60], [(0,0),(2,1),(4,2),(6,3),(7,5)], [(1,0),(3,1),(5,2),(8,4)], 4, 3, 3, 6, 48, 67968, 512⟩
def up52 : UnionParams := ⟨[4,12,20,28,36,44,52,60], [16,25,31,34,38,43,45,59,61], [(2,0),(4,1),(6,2),(7,4)], [(0,0),(1,1),(3,2),(5,3),(8,5)], 3, 4, 4, 6, 48, 68480, 512⟩
def up53 : UnionParams := ⟨[5,13,21,29,37,45,53,61], [8,17,26,35,39,44,46,60,62], [(4,0),(6,1),(7,3)], [(0,0),(1,1),(2,2),(3,3),(5,4),(8,6)], 2, 5, 5, 6, 48, 68992, 512⟩
def up54 : UnionParams := ⟨[6,14,22,30,38,46,54,62], [0,9,18,27,36,45,47,61,63], [(6,0),(7,2)], [(0,0),(1,1),(2,2),(3,3),(4,4),(5,5),(8,7)], 1, 6, 6, 6, 48, 69504, 512⟩
def up55 : UnionParams := ⟨[7,15,23,31,39,47,55,63], [1,10,19,28,37,46,62], [(6,1)], [(0,0),(1,1),(2,2),(3,3),(4,4),(5,5)], 0, 6, 7, 6, 48, 70016, 128⟩
def up56 : UnionParams := ⟨[0,8,16,24,32,40,48,56], [7,14,21,28,35,42,49], [(0,0),(1,1),(2,2),(3,3),(4,4),(5,5),(6,6)], [], 7, 0, 0, 7, 56, 70144, 128⟩
def up57 : UnionParams := ⟨[1,9,17,25,33,41,49,57], [15,22,29,36,43,48,50], [(0,0),(1,1),(2,2),(3,3),(4,4),(6,5)], [(5,0)], 6, 1, 1, 7, 56, 70272, 128⟩
def up58 : UnionParams := ⟨[2,10,18,26,34,42,50,58], [23,30,37,40,44,49,51], [(0,0),(1,1),(2,2),(4,3),(6,4)], [(3,0),(5,1)], 5, 2, 2, 7, 56, 70400, 128⟩
def up59 : UnionParams := ⟨[3,11,19,27,35,43,51,59], [31,32,38,41,45,50,52], [(0,0),(2,1),(4,2),(6,3)], [(1,0),(3,1),(5,2)], 4, 3, 3, 7, 56, 70528, 128⟩
def up60 : UnionParams := ⟨[4,12,20,28,36,44,52,60], [24,33,39,42,46,51,53], [(2,0),(4,1),(6,2)], [(0,0),(1,1),(3,2),(5,3)], 3, 4, 4, 7, 56, 70656, 128⟩
def up61 : UnionParams := ⟨[5,13,21,29,37,45,53,61], [16,25,34,43,47,52,54], [(4,0),(6,1)], [(0,0),(1,1),(2,2),(3,3),(5,4)], 2, 5, 5, 7, 56, 70784, 128⟩
def up62 : UnionParams := ⟨[6,14,22,30,38,46,54,62], [8,17,26,35,44,53,55], [(6,0)], [(0,0),(1,1),(2,2),(3,3),(4,4),(5,5)], 1, 6, 6, 7, 56, 70912, 128⟩
def up63 : UnionParams := ⟨[7,15,23,31,39,47,55,63], [0,9,18,27,36,45,54], [], [(0,0),(1,1),(2,2),(3,3),(4,4),(5,5),(6,6)], 0, 7, 7, 7, 56, 71040, 128⟩

def unionParamsTable : Array UnionParams := #[up0, up1, up2, up3, up4, up5, up6, up7, up8, up9, up10, up11, up12, up13, up14, up15, up16, up17, up18, up19, up20, up21, up22, up23, up24, up25, up26, up27, up28, up29, up30, up31, up32, up33, up34, up35, up36, up37, up38, up39, up40, up41, up42, up43, up44, up45, up46, up47, up48, up49, up50, up51, up52, up53, up54, up55, up56, up57, up58, up59, up60, up61, up62, up63]

def unionParams (s : Nat) : UnionParams :=
  unionParamsTable.getD s default

theorem unionParams_0 : unionParams 0 = up0 := rfl
theorem unionParams_1 : unionParams 1 = up1 := rfl
theorem unionParams_2 : unionParams 2 = up2 := rfl
theorem unionParams_3 : unionParams 3 = up3 := rfl
theorem unionParams_4 : unionParams 4 = up4 := rfl
theorem unionParams_5 : unionParams 5 = up5 := rfl
theorem unionParams_6 : unionParams 6 = up6 := rfl
theorem unionParams_7 : unionParams 7 = up7 := rfl
theorem unionParams_8 : unionParams 8 = up8 := rfl
theorem unionParams_9 : unionParams 9 = up9 := rfl
theorem unionParams_10 : unionParams 10 = up10 := rfl
theorem unionParams_11 : unionParams 11 = up11 := rfl
theorem unionParams_12 : unionParams 12 = up12 := rfl
theorem unionParams_13 : unionParams 13 = up13 := rfl
theorem unionParams_14 : unionParams 14 = up14 := rfl
theorem unionParams_15 : unionParams 15 = up15 := rfl
theorem unionParams_16 : unionParams 16 = up16 := rfl
theorem unionParams_17 : unionParams 17 = up17 := rfl
theorem unionParams_18 : unionParams 18 = up18 := rfl
theorem unionParams_19 : unionParams 19 = up19 := rfl
theorem unionParams_20 : unionParams 20 = up20 := rfl
theorem unionParams_21 : unionParams 21 = up21 := rfl
theorem unionParams_22 : unionParams 22 = up22 := rfl
theorem unionParams_23 : unionParams 23 = up23 := rfl
theorem unionParams_24 : unionParams 24 = up24 := rfl
theorem unionParams_25 : unionParams 25 = up25 := rfl
theorem unionParams_26 : unionParams 26 = up26 := rfl
theorem unionParams_27 : unionParams 27 = up27 := rfl
theorem unionParams_28 : unionParams 28 = up28 := rfl
theorem unionParams_29 : unionParams 29 = up29 := rfl
theorem unionParams_30 : unionParams 30 = up30 := rfl
theorem unionParams_31 : unionParams 31 = up31 := rfl
theorem unionParams_32 : unionParams 32 = up32 := rfl
theorem unionParams_33 : unionParams 33 = up33 := rfl
theorem unionParams_34 : unionParams 34 = up34 := rfl
theorem unionParams_35 : unionParams 35 = up35 := rfl
theorem unionParams_36 : unionParams 36 = up36 := rfl
theorem unionParams_37 : unionParams 37 = up37 := rfl
theorem unionParams_38 : unionParams 38 = up38 := rfl
theorem unionParams_39 : unionParams 39 = up39 := rfl
theorem unionParams_40 : unionParams 40 = up40 := rfl
theorem unionParams_41 : unionParams 41 = up41 := rfl
theorem unionParams_42 : unionParams 42 = up42 := rfl
theorem unionParams_43 : unionParams 43 = up43 := rfl
theorem unionParams_44 : unionParams 44 = up44 := rfl
theorem unionParams_45 : unionParams 45 = up45 := rfl
theorem unionParams_46 : unionParams 46 = up46 := rfl
theorem unionParams_47 : unionParams 47 = up47 := rfl
theorem unionParams_48 : unionParams 48 = up48 := rfl
theorem unionParams_49 : unionParams 49 = up49 := rfl
theorem unionParams_50 : unionParams 50 = up50 := rfl
theorem unionParams_51 : unionParams 51 = up51 := rfl
theorem unionParams_52 : unionParams 52 = up52 := rfl
theorem unionParams_53 : unionParams 53 = up53 := rfl
theorem unionParams_54 : unionParams 54 = up54 := rfl
theorem unionParams_55 : unionParams 55 = up55 := rfl
theorem unionParams_56 : unionParams 56 = up56 := rfl
theorem unionParams_57 : unionParams 57 = up57 := rfl
theorem unionParams_58 : unionParams 58 = up58 := rfl
theorem unionParams_59 : unionParams 59 = up59 := rfl
theorem unionParams_60 : unionParams 60 = up60 := rfl
theorem unionParams_61 : unionParams 61 = up61 := rfl
theorem unionParams_62 : unionParams 62 = up62 := rfl
theorem unionParams_63 : unionParams 63 = up63 := rfl


def pext (p : Word) : List Nat → Word
  | [] => 0
  | cell :: rest => (if p.getLsbD cell then (1 : Word) else 0) ||| (pext p rest <<< 1)
def projectIndex (idx : Index) : List (Nat × Nat) → Byte
  | [] => 0
  | (src,dst) :: rest =>
      (if idx.getLsbD src then ((1 : Byte) <<< dst) else 0) ||| projectIndex idx rest

def refs (p : Word) (square : Nat) : Nat :=
  let u := unionParams square
  LineTable.reference ⟨u.rowPos % 8, Nat.mod_lt _ (by decide)⟩ ((p >>> u.rankShift).setWidth 8) +
  LineTable.reference ⟨u.filePos % 8, Nat.mod_lt _ (by decide)⟩ ((pext p u.fileCells).setWidth 8) +
  LineTable.reference ⟨u.d0Pos % 8, Nat.mod_lt _ (by decide)⟩ (projectIndex ((pext p u.unionCells).setWidth 16) u.d0Map) +
  LineTable.reference ⟨u.d1Pos % 8, Nat.mod_lt _ (by decide)⟩ (projectIndex ((pext p u.unionCells).setWidth 16) u.d1Map)
def refsOpp (p : Word) (square : Nat) : Nat :=
  let u := unionParams square
  LineTable.reference ⟨u.rowPos % 8, Nat.mod_lt _ (by decide)⟩ (((p >>> u.rankShift).setWidth 8) ^^^ 255) +
  LineTable.reference ⟨u.filePos % 8, Nat.mod_lt _ (by decide)⟩ (((pext p u.fileCells).setWidth 8) ^^^ 255) +
  LineTable.reference ⟨u.d0Pos % 8, Nat.mod_lt _ (by decide)⟩ (projectIndex (((pext p u.unionCells).setWidth 16) ^^^ (BitVec.ofNat 16 (u.size - 1))) u.d0Map) +
  LineTable.reference ⟨u.d1Pos % 8, Nat.mod_lt _ (by decide)⟩ (projectIndex (((pext p u.unionCells).setWidth 16) ^^^ (BitVec.ofNat 16 (u.size - 1))) u.d1Map)
def ray (p : Word) (acc : Nat) : List Nat → Nat
  | [] => 0
  | i :: rest => if p.getLsbD i then acc else ray p (acc + 1) rest
def raySquares (square : Nat) (dr dc : Int) : List Nat :=
  (List.range 7).filterMap fun k =>
    let r : Int := (square / 8 : Nat) + dr * (k + 1 : Nat)
    let c : Int := (square % 8 : Nat) + dc * (k + 1 : Nat)
    if 0 ≤ r ∧ r < 8 ∧ 0 ≤ c ∧ c < 8 then some (r.toNat * 8 + c.toNat) else none
def reference (p : Word) (square : Nat) : Nat :=
  ray p 0 (raySquares square 0 1) +
  ray p 0 (raySquares square 1 0) +
  ray p 0 (raySquares square 1 1) +
  ray p 0 (raySquares square 1 (-1)) +
  ray p 0 (raySquares square 0 (-1)) +
  ray p 0 (raySquares square (-1) 0) +
  ray p 0 (raySquares square (-1) (-1)) +
  ray p 0 (raySquares square (-1) 1)
def lookup (u : UnionParams) (row file : Byte) (diag : Index) : Byte :=
  LineTable.countFlip ⟨u.rowPos % 8, Nat.mod_lt _ (by decide)⟩ row +
  LineTable.countFlip ⟨u.filePos % 8, Nat.mod_lt _ (by decide)⟩ file +
  (LineTable.countFlip ⟨u.d0Pos % 8, Nat.mod_lt _ (by decide)⟩ (projectIndex diag u.d0Map) +
   LineTable.countFlip ⟨u.d1Pos % 8, Nat.mod_lt _ (by decide)⟩ (projectIndex diag u.d1Map))
def count (p : Word) (square : Nat) : Byte :=
  let u := unionParams square
  lookup u ((p >>> u.rankShift).setWidth 8) ((pext p u.fileCells).setWidth 8)
    ((pext p u.unionCells).setWidth 16)
def countOpp (p : Word) (square : Nat) : Byte :=
  let u := unionParams square
  lookup u (((p >>> u.rankShift).setWidth 8) ^^^ 255) (((pext p u.fileCells).setWidth 8) ^^^ 255)
    (((pext p u.unionCells).setWidth 16) ^^^ (BitVec.ofNat 16 (u.size - 1)))

theorem getLsbD_cond (b : Prop) [Decidable b] (x y : BitVec w) (i : Nat) :
    (if b then x else y).getLsbD i = (if b then x.getLsbD i else y.getLsbD i) := by
  split <;> rfl
set_option maxRecDepth 10000
set_option maxHeartbeats 0
set_option linter.unusedSimpArgs false
end NeuralReversi.Solve1.Bmi2
