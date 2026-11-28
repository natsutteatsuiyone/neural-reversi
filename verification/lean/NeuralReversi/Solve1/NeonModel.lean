module
public import NeuralReversi.Solve1.NeonParams
public import NeuralReversi.Solve1.ScalarModel
public import NeuralReversi.Solve1.Packed

@[expose] public section

namespace NeuralReversi.Solve1.Neon

def classId (s : Nat) (lane : Fin 4) : Nat :=
  ((neonMeta s >>> (8 * (lane.val + 1))).setWidth 8).toNat

def classPos (s : Nat) (lane : Fin 4) : Fin 8 :=
  ⟨(pairKey (classId s lane)).1 % 8, Nat.mod_lt _ (by decide)⟩

def laneXor (s : Nat) (lane : Fin 4) : Byte := (pairKey (classId s lane)).2

def hasFourth (s : Nat) : Bool := neonMeta s &&& 0x8000000000000000 != 0

def rankShift (s : Nat) : Nat := ((neonMeta s).setWidth 32 &&& 0xff).toNat

def laneIndex (s : Nat) (lane : Fin 4) (p : Word) : Byte :=
  if lane.val = 1 then (p >>> rankShift s).setWidth 8 else Scalar.laneIndex s lane p

def pairCounts (cls : Nat) (idx : Byte) : Packed.Counts :=
  let key := pairKey cls
  let pos : Fin 8 := ⟨key.1 % 8, Nat.mod_lt _ (by decide)⟩
  ⟨LineTable.countFlip pos idx, LineTable.countFlip pos (idx ^^^ key.2)⟩

def pairLookup (cls : Nat) (idx : Byte) : BitVec 32 :=
  Packed.entry32 (pairCounts cls idx)

def laneCounts (s : Nat) (lane : Fin 4) (p : Word) : Packed.Counts :=
  pairCounts (classId s lane) (laneIndex s lane p)

def entries (s : Nat) (p : Word) : List Packed.Counts :=
  [laneCounts s 0 p, laneCounts s 1 p, laneCounts s 2 p] ++
    if hasFourth s then [laneCounts s 3 p] else []

def countPacked (s : Nat) (p : Word) : BitVec 32 :=
  pairLookup (classId s 0) (laneIndex s 0 p) +
  pairLookup (classId s 1) (laneIndex s 1 p) +
  pairLookup (classId s 2) (laneIndex s 2 p) +
  if hasFourth s then pairLookup (classId s 3) (laneIndex s 3 p) else 0

theorem countPacked_eq (s : Nat) (p : Word) :
    countPacked s p = Packed.packedSum (entries s p) := by
  by_cases h : hasFourth s <;> simp [countPacked, Packed.packedSum, entries, h,
    pairLookup, laneCounts]

private theorem rankShift_correct_all : ∀ s : Fin 64,
    rankShift s.val = (Scalar.params s.val).rowShift := by decide +kernel

theorem laneIndex_eq (s : Fin 64) (lane : Fin 4) (p : Word) :
    laneIndex s.val lane p = Scalar.laneIndex s.val lane p := by
  unfold laneIndex
  split
  · rename_i h
    simp [rankShift_correct_all, Scalar.laneIndex, h]
  · rfl

private theorem classPos_correct_all : ∀ s : Fin 64, ∀ lane : Fin 4,
    classPos s.val lane = Scalar.lanePos s.val lane := by decide +kernel

theorem classPos_correct (s : Fin 64) (lane : Fin 4) :
    classPos s.val lane = Scalar.lanePos s.val lane := classPos_correct_all s lane

theorem hasFourth_correct : ∀ s : Fin 64,
    hasFourth s.val = ((Scalar.params s.val).maskD9 != 0) := by decide +kernel

private theorem masked_row (p : Word) (shift : Nat) :
    ((p &&& ((255 : Word) <<< shift)) >>> shift).setWidth 8 =
      (p >>> shift).setWidth 8 := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  simp only [BitVec.getLsbD_setWidth, BitVec.getLsbD_ushiftRight,
    BitVec.getLsbD_and, BitVec.getLsbD_shiftLeft]
  have hn : (255 : Word).getLsbD i = true := by
    match i with
    | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 => decide
    | k + 8 => omega
  by_cases hs : shift + i < 64
  · simp [hi, hs, Nat.add_sub_cancel_left, -BitVec.getLsbD_eq_getElem]
    exact fun _ => hn
  · have hz : p.getLsbD (shift + i) = false := BitVec.getLsbD_of_ge p _ (by omega)
    simp [hz]

theorem laneIndex_mask (s : Nat) (lane : Fin 4) (p : Word) :
    Scalar.laneIndex s lane (p &&& Scalar.laneMask s lane) = Scalar.laneIndex s lane p := by
  rcases lane with ⟨l, hl⟩
  match l with
  | 0 => simp [Scalar.laneIndex, Scalar.laneMask, BitVec.and_assoc]
  | 1 => exact masked_row p _
  | 2 => simp [Scalar.laneIndex, Scalar.laneMask, BitVec.and_assoc]
  | 3 => simp [Scalar.laneIndex, Scalar.laneMask, BitVec.and_assoc]
  | k + 4 => omega

private theorem not_mask (p m : Word) : (~~~(p &&& m)) &&& m = (~~~p) &&& m := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  simp only [BitVec.getLsbD_and, BitVec.getLsbD_not]
  cases p.getLsbD i <;> cases m.getLsbD i <;> simp

theorem laneIndex_not_mask (s : Nat) (lane : Fin 4) (p : Word) :
    Scalar.laneIndex s lane (~~~(p &&& Scalar.laneMask s lane)) =
      Scalar.laneIndex s lane (~~~p) := by
  rw [← laneIndex_mask s lane (~~~(p &&& Scalar.laneMask s lane)), not_mask,
    laneIndex_mask]

private theorem row_xor_params : ∀ s : Fin 64,
    (Scalar.params s.val).rowShift ≤ 56 ∧ laneXor s.val 1 = 255#8 := by decide +kernel

theorem xor_row_all (s : Fin 64) (p : Word) :
    Scalar.laneIndex s.val 1 p ^^^ laneXor s.val 1 = Scalar.laneIndex s.val 1 (~~~p) := by
  have ⟨hsh, hxor⟩ := row_xor_params s
  rw [hxor]
  change (p >>> (Scalar.params s.val).rowShift).setWidth 8 ^^^ 255#8 =
    ((~~~p) >>> (Scalar.params s.val).rowShift).setWidth 8
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  have h255 : (255#8).getLsbD i = true := by
    match i with
    | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 => decide
    | _ + 8 => omega
  have hlt : (Scalar.params s.val).rowShift + i < 64 := by omega
  simp only [BitVec.getLsbD_xor, BitVec.getLsbD_setWidth, BitVec.getLsbD_ushiftRight,
    BitVec.getLsbD_not, hi, hlt, h255, decide_true, Bool.true_and, Bool.xor_true]

end NeuralReversi.Solve1.Neon
