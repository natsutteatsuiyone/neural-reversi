import NeuralReversi.Stability.EdgeEncoding
import NeuralReversi.Solve1.Projection

namespace NeuralReversi.Stability.Semantics
set_option maxRecDepth 100000
set_option maxHeartbeats 0
set_option linter.unusedSimpArgs false

private abbrev P := NeuralReversi.Solve1.Projection.gather
private abbrev S := NeuralReversi.Solve1.Projection.scatter

def packA1A8 (x : Word) : Byte :=
  (((x &&& 0x0101010101010101) * 0x0102040810204080) >>> 56).setWidth 8

def packH1H8 (x : Word) : Byte :=
  (((x &&& 0x8080808080808080) * 0x0002040810204081) >>> 56).setWidth 8

def unpackA2A7 (x : Byte) : Word :=
  ((x &&& 0x7e).setWidth 64 * 0x0000040810204080) &&& 0x0001010101010100

def unpackH2H7 (x : Byte) : Word :=
  ((x &&& 0x7e).setWidth 64 * 0x0002040810204000) &&& 0x0080808080808000

private theorem packA_finite : ∀ n : Fin 256,
    ((S [0,8,16,24,32,40,48,56] (BitVec.ofNat 8 n.val) * 0x0102040810204080) >>> 56).setWidth 8 =
      BitVec.ofNat 8 n.val := by decide +kernel

private theorem packH_finite : ∀ n : Fin 256,
    ((S [7,15,23,31,39,47,55,63] (BitVec.ofNat 8 n.val) * 0x0002040810204081) >>> 56).setWidth 8 =
      BitVec.ofNat 8 n.val := by decide +kernel

theorem packA_eq_gather (x : Word) : packA1A8 x = P [0,8,16,24,32,40,48,56] x := by
  have hm := NeuralReversi.Solve1.Projection.masked_eq_scatter
    [0,8,16,24,32,40,48,56] x (by decide) 0x0101010101010101 (by decide)
  rw [packA1A8, hm]
  simpa using packA_finite ⟨(P [0,8,16,24,32,40,48,56] x).toNat,
    (P [0,8,16,24,32,40,48,56] x).isLt⟩

theorem packH_eq_gather (x : Word) : packH1H8 x = P [7,15,23,31,39,47,55,63] x := by
  have hm := NeuralReversi.Solve1.Projection.masked_eq_scatter
    [7,15,23,31,39,47,55,63] x (by decide) 0x8080808080808080 (by decide)
  rw [packH1H8, hm]
  simpa using packH_finite ⟨(P [7,15,23,31,39,47,55,63] x).toNat,
    (P [7,15,23,31,39,47,55,63] x).isLt⟩

theorem packA_bit (x : Word) (i : Fin 8) :
    (packA1A8 x).getLsbD i.val = x.getLsbD (8*i.val) := by
  rw [packA_eq_gather]
  have hi : i=0 ∨ i=1 ∨ i=2 ∨ i=3 ∨ i=4 ∨ i=5 ∨ i=6 ∨ i=7 := by omega
  rcases hi with hi | hi | hi | hi | hi | hi | hi | hi <;> subst i <;>
    simp [P, NeuralReversi.Solve1.Projection.gather, -BitVec.getLsbD_eq_getElem] <;> rfl

theorem packH_bit (x : Word) (i : Fin 8) :
    (packH1H8 x).getLsbD i.val = x.getLsbD (8*i.val+7) := by
  rw [packH_eq_gather]
  have hi : i=0 ∨ i=1 ∨ i=2 ∨ i=3 ∨ i=4 ∨ i=5 ∨ i=6 ∨ i=7 := by omega
  rcases hi with hi | hi | hi | hi | hi | hi | hi | hi <;> subst i <;>
    simp [P, NeuralReversi.Solve1.Projection.gather, -BitVec.getLsbD_eq_getElem] <;> rfl

theorem packA_eq_column (x : Word) : packA1A8 x = columnBits x 0 := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  rw [packA_bit x ⟨i,hi⟩, columnBits_bit x 0 ⟨i,hi⟩]
  rfl

theorem packH_eq_column (x : Word) : packH1H8 x = columnBits x 7 := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  rw [packH_bit x ⟨i,hi⟩, columnBits_bit x 7 ⟨i,hi⟩]
  rfl

theorem encode_left (s : State) (c : Color) (p o : Word) (repr : Represents s c p o) :
    encodeLine (fun i => s.board (i,0)) c = packA1A8 p := by
  rw [packA_eq_column]
  exact encode_column s c p o repr 0

theorem encode_right (s : State) (c : Color) (p o : Word) (repr : Represents s c p o) :
    encodeLine (fun i => s.board (i,7)) c = packH1H8 p := by
  rw [packH_eq_column]
  exact encode_column s c p o repr 7

private theorem unpackA_finite_0 : ∀ n : Fin 256, ∀ c : Fin 8,
    (unpackA2A7 (BitVec.ofNat 8 n.val)).getLsbD (8*0+c.val) =
      (decide (c.val = 0 ∧ 0 < 0 ∧ 0 < 7) && (BitVec.ofNat 8 n.val).getLsbD 0) := by decide +kernel
private theorem unpackA_finite_1 : ∀ n : Fin 256, ∀ c : Fin 8,
    (unpackA2A7 (BitVec.ofNat 8 n.val)).getLsbD (8*1+c.val) =
      (decide (c.val = 0 ∧ 0 < 1 ∧ 1 < 7) && (BitVec.ofNat 8 n.val).getLsbD 1) := by decide +kernel
private theorem unpackA_finite_2 : ∀ n : Fin 256, ∀ c : Fin 8,
    (unpackA2A7 (BitVec.ofNat 8 n.val)).getLsbD (8*2+c.val) =
      (decide (c.val = 0 ∧ 0 < 2 ∧ 2 < 7) && (BitVec.ofNat 8 n.val).getLsbD 2) := by decide +kernel
private theorem unpackA_finite_3 : ∀ n : Fin 256, ∀ c : Fin 8,
    (unpackA2A7 (BitVec.ofNat 8 n.val)).getLsbD (8*3+c.val) =
      (decide (c.val = 0 ∧ 0 < 3 ∧ 3 < 7) && (BitVec.ofNat 8 n.val).getLsbD 3) := by decide +kernel
private theorem unpackA_finite_4 : ∀ n : Fin 256, ∀ c : Fin 8,
    (unpackA2A7 (BitVec.ofNat 8 n.val)).getLsbD (8*4+c.val) =
      (decide (c.val = 0 ∧ 0 < 4 ∧ 4 < 7) && (BitVec.ofNat 8 n.val).getLsbD 4) := by decide +kernel
private theorem unpackA_finite_5 : ∀ n : Fin 256, ∀ c : Fin 8,
    (unpackA2A7 (BitVec.ofNat 8 n.val)).getLsbD (8*5+c.val) =
      (decide (c.val = 0 ∧ 0 < 5 ∧ 5 < 7) && (BitVec.ofNat 8 n.val).getLsbD 5) := by decide +kernel
private theorem unpackA_finite_6 : ∀ n : Fin 256, ∀ c : Fin 8,
    (unpackA2A7 (BitVec.ofNat 8 n.val)).getLsbD (8*6+c.val) =
      (decide (c.val = 0 ∧ 0 < 6 ∧ 6 < 7) && (BitVec.ofNat 8 n.val).getLsbD 6) := by decide +kernel
private theorem unpackA_finite_7 : ∀ n : Fin 256, ∀ c : Fin 8,
    (unpackA2A7 (BitVec.ofNat 8 n.val)).getLsbD (8*7+c.val) =
      (decide (c.val = 0 ∧ 0 < 7 ∧ 7 < 7) && (BitVec.ofNat 8 n.val).getLsbD 7) := by decide +kernel
theorem unpackA_bit (x : Byte) (r c : Fin 8) :
    (unpackA2A7 x).getLsbD (8*r.val+c.val) =
      (decide (c.val = 0 ∧ 0 < r.val ∧ r.val < 7) && x.getLsbD r.val) := by
  have hr : r=0 ∨ r=1 ∨ r=2 ∨ r=3 ∨ r=4 ∨ r=5 ∨ r=6 ∨ r=7 := by omega
  rcases hr with hr | hr | hr | hr | hr | hr | hr | hr
  · subst r
    change (unpackA2A7 x).getLsbD (8*0+c.val) =
      (decide (c.val = 0 ∧ 0 < 0 ∧ 0 < 7) && x.getLsbD 0)
    simpa only [BitVec.ofNat_toNat, BitVec.setWidth_eq] using unpackA_finite_0 ⟨x.toNat, x.isLt⟩ c
  · subst r
    change (unpackA2A7 x).getLsbD (8*1+c.val) =
      (decide (c.val = 0 ∧ 0 < 1 ∧ 1 < 7) && x.getLsbD 1)
    simpa only [BitVec.ofNat_toNat, BitVec.setWidth_eq] using unpackA_finite_1 ⟨x.toNat, x.isLt⟩ c
  · subst r
    change (unpackA2A7 x).getLsbD (8*2+c.val) =
      (decide (c.val = 0 ∧ 0 < 2 ∧ 2 < 7) && x.getLsbD 2)
    simpa only [BitVec.ofNat_toNat, BitVec.setWidth_eq] using unpackA_finite_2 ⟨x.toNat, x.isLt⟩ c
  · subst r
    change (unpackA2A7 x).getLsbD (8*3+c.val) =
      (decide (c.val = 0 ∧ 0 < 3 ∧ 3 < 7) && x.getLsbD 3)
    simpa only [BitVec.ofNat_toNat, BitVec.setWidth_eq] using unpackA_finite_3 ⟨x.toNat, x.isLt⟩ c
  · subst r
    change (unpackA2A7 x).getLsbD (8*4+c.val) =
      (decide (c.val = 0 ∧ 0 < 4 ∧ 4 < 7) && x.getLsbD 4)
    simpa only [BitVec.ofNat_toNat, BitVec.setWidth_eq] using unpackA_finite_4 ⟨x.toNat, x.isLt⟩ c
  · subst r
    change (unpackA2A7 x).getLsbD (8*5+c.val) =
      (decide (c.val = 0 ∧ 0 < 5 ∧ 5 < 7) && x.getLsbD 5)
    simpa only [BitVec.ofNat_toNat, BitVec.setWidth_eq] using unpackA_finite_5 ⟨x.toNat, x.isLt⟩ c
  · subst r
    change (unpackA2A7 x).getLsbD (8*6+c.val) =
      (decide (c.val = 0 ∧ 0 < 6 ∧ 6 < 7) && x.getLsbD 6)
    simpa only [BitVec.ofNat_toNat, BitVec.setWidth_eq] using unpackA_finite_6 ⟨x.toNat, x.isLt⟩ c
  · subst r
    change (unpackA2A7 x).getLsbD (8*7+c.val) =
      (decide (c.val = 0 ∧ 0 < 7 ∧ 7 < 7) && x.getLsbD 7)
    simpa only [BitVec.ofNat_toNat, BitVec.setWidth_eq] using unpackA_finite_7 ⟨x.toNat, x.isLt⟩ c
private theorem unpackH_finite_0 : ∀ n : Fin 256, ∀ c : Fin 8,
    (unpackH2H7 (BitVec.ofNat 8 n.val)).getLsbD (8*0+c.val) =
      (decide (c.val = 7 ∧ 0 < 0 ∧ 0 < 7) && (BitVec.ofNat 8 n.val).getLsbD 0) := by decide +kernel
private theorem unpackH_finite_1 : ∀ n : Fin 256, ∀ c : Fin 8,
    (unpackH2H7 (BitVec.ofNat 8 n.val)).getLsbD (8*1+c.val) =
      (decide (c.val = 7 ∧ 0 < 1 ∧ 1 < 7) && (BitVec.ofNat 8 n.val).getLsbD 1) := by decide +kernel
private theorem unpackH_finite_2 : ∀ n : Fin 256, ∀ c : Fin 8,
    (unpackH2H7 (BitVec.ofNat 8 n.val)).getLsbD (8*2+c.val) =
      (decide (c.val = 7 ∧ 0 < 2 ∧ 2 < 7) && (BitVec.ofNat 8 n.val).getLsbD 2) := by decide +kernel
private theorem unpackH_finite_3 : ∀ n : Fin 256, ∀ c : Fin 8,
    (unpackH2H7 (BitVec.ofNat 8 n.val)).getLsbD (8*3+c.val) =
      (decide (c.val = 7 ∧ 0 < 3 ∧ 3 < 7) && (BitVec.ofNat 8 n.val).getLsbD 3) := by decide +kernel
private theorem unpackH_finite_4 : ∀ n : Fin 256, ∀ c : Fin 8,
    (unpackH2H7 (BitVec.ofNat 8 n.val)).getLsbD (8*4+c.val) =
      (decide (c.val = 7 ∧ 0 < 4 ∧ 4 < 7) && (BitVec.ofNat 8 n.val).getLsbD 4) := by decide +kernel
private theorem unpackH_finite_5 : ∀ n : Fin 256, ∀ c : Fin 8,
    (unpackH2H7 (BitVec.ofNat 8 n.val)).getLsbD (8*5+c.val) =
      (decide (c.val = 7 ∧ 0 < 5 ∧ 5 < 7) && (BitVec.ofNat 8 n.val).getLsbD 5) := by decide +kernel
private theorem unpackH_finite_6 : ∀ n : Fin 256, ∀ c : Fin 8,
    (unpackH2H7 (BitVec.ofNat 8 n.val)).getLsbD (8*6+c.val) =
      (decide (c.val = 7 ∧ 0 < 6 ∧ 6 < 7) && (BitVec.ofNat 8 n.val).getLsbD 6) := by decide +kernel
private theorem unpackH_finite_7 : ∀ n : Fin 256, ∀ c : Fin 8,
    (unpackH2H7 (BitVec.ofNat 8 n.val)).getLsbD (8*7+c.val) =
      (decide (c.val = 7 ∧ 0 < 7 ∧ 7 < 7) && (BitVec.ofNat 8 n.val).getLsbD 7) := by decide +kernel
theorem unpackH_bit (x : Byte) (r c : Fin 8) :
    (unpackH2H7 x).getLsbD (8*r.val+c.val) =
      (decide (c.val = 7 ∧ 0 < r.val ∧ r.val < 7) && x.getLsbD r.val) := by
  have hr : r=0 ∨ r=1 ∨ r=2 ∨ r=3 ∨ r=4 ∨ r=5 ∨ r=6 ∨ r=7 := by omega
  rcases hr with hr | hr | hr | hr | hr | hr | hr | hr
  · subst r
    change (unpackH2H7 x).getLsbD (8*0+c.val) =
      (decide (c.val = 7 ∧ 0 < 0 ∧ 0 < 7) && x.getLsbD 0)
    simpa only [BitVec.ofNat_toNat, BitVec.setWidth_eq] using unpackH_finite_0 ⟨x.toNat, x.isLt⟩ c
  · subst r
    change (unpackH2H7 x).getLsbD (8*1+c.val) =
      (decide (c.val = 7 ∧ 0 < 1 ∧ 1 < 7) && x.getLsbD 1)
    simpa only [BitVec.ofNat_toNat, BitVec.setWidth_eq] using unpackH_finite_1 ⟨x.toNat, x.isLt⟩ c
  · subst r
    change (unpackH2H7 x).getLsbD (8*2+c.val) =
      (decide (c.val = 7 ∧ 0 < 2 ∧ 2 < 7) && x.getLsbD 2)
    simpa only [BitVec.ofNat_toNat, BitVec.setWidth_eq] using unpackH_finite_2 ⟨x.toNat, x.isLt⟩ c
  · subst r
    change (unpackH2H7 x).getLsbD (8*3+c.val) =
      (decide (c.val = 7 ∧ 0 < 3 ∧ 3 < 7) && x.getLsbD 3)
    simpa only [BitVec.ofNat_toNat, BitVec.setWidth_eq] using unpackH_finite_3 ⟨x.toNat, x.isLt⟩ c
  · subst r
    change (unpackH2H7 x).getLsbD (8*4+c.val) =
      (decide (c.val = 7 ∧ 0 < 4 ∧ 4 < 7) && x.getLsbD 4)
    simpa only [BitVec.ofNat_toNat, BitVec.setWidth_eq] using unpackH_finite_4 ⟨x.toNat, x.isLt⟩ c
  · subst r
    change (unpackH2H7 x).getLsbD (8*5+c.val) =
      (decide (c.val = 7 ∧ 0 < 5 ∧ 5 < 7) && x.getLsbD 5)
    simpa only [BitVec.ofNat_toNat, BitVec.setWidth_eq] using unpackH_finite_5 ⟨x.toNat, x.isLt⟩ c
  · subst r
    change (unpackH2H7 x).getLsbD (8*6+c.val) =
      (decide (c.val = 7 ∧ 0 < 6 ∧ 6 < 7) && x.getLsbD 6)
    simpa only [BitVec.ofNat_toNat, BitVec.setWidth_eq] using unpackH_finite_6 ⟨x.toNat, x.isLt⟩ c
  · subst r
    change (unpackH2H7 x).getLsbD (8*7+c.val) =
      (decide (c.val = 7 ∧ 0 < 7 ∧ 7 < 7) && x.getLsbD 7)
    simpa only [BitVec.ofNat_toNat, BitVec.setWidth_eq] using unpackH_finite_7 ⟨x.toNat, x.isLt⟩ c

theorem packA_disjoint (p o : Word) (hd : p &&& o = 0) :
    packA1A8 p &&& packA1A8 o = 0 := by
  rw [packA_eq_column, packA_eq_column]
  exact columnBits_disjoint p o hd 0

theorem packH_disjoint (p o : Word) (hd : p &&& o = 0) :
    packH1H8 p &&& packH1H8 o = 0 := by
  rw [packH_eq_column, packH_eq_column]
  exact columnBits_disjoint p o hd 7

def fileA : Word := 0x0101010101010101
def fileH : Word := 0x8080808080808080
def fileAInner : Word := 0x0001010101010100
def fileHInner : Word := 0x0080808080808000

def pextAux (src mask : Word) (rem k : Nat) (acc : Word) : Word :=
  match rem with
  | 0 => acc
  | rem' + 1 =>
    let bit := 63 - rem'
    if mask.getLsbD bit then
      let acc' := if src.getLsbD bit then acc ||| (1#64 <<< k) else acc
      pextAux src mask rem' (k + 1) acc'
    else
      pextAux src mask rem' k acc

def pext64 (src mask : Word) : Word := pextAux src mask 64 0 0

def pdepAux (src mask : Word) (rem k : Nat) (acc : Word) : Word :=
  match rem with
  | 0 => acc
  | rem' + 1 =>
    let bit := 63 - rem'
    if mask.getLsbD bit then
      let acc' := if src.getLsbD k then acc ||| (1#64 <<< bit) else acc
      pdepAux src mask rem' (k + 1) acc'
    else
      pdepAux src mask rem' k acc

def pdep64 (src mask : Word) : Word := pdepAux src mask 64 0 0

def packA1A8Bmi2Word (x : Word) : Word := pext64 x fileA
def packH1H8Bmi2Word (x : Word) : Word := pext64 x fileH
def packA1A8Bmi2 (x : Word) : Byte := (packA1A8Bmi2Word x).setWidth 8
def packH1H8Bmi2 (x : Word) : Byte := (packH1H8Bmi2Word x).setWidth 8

def unpackA2A7Bmi2 (x : Byte) : Word :=
  pdep64 (((x &&& 0x7e) >>> 1).setWidth 64) fileAInner

def unpackH2H7Bmi2 (x : Byte) : Word :=
  pdep64 (((x &&& 0x7e) >>> 1).setWidth 64) fileHInner

theorem pextAux_and_mask (src mask : Word) (rem k : Nat) (acc : Word) :
    pextAux src mask rem k acc = pextAux (src &&& mask) mask rem k acc := by
  induction rem generalizing k acc with
  | zero => rfl
  | succ rem' ih =>
    simp only [pextAux]
    split
    · rename_i hmask
      have hbit : (src &&& mask).getLsbD (63 - rem') = src.getLsbD (63 - rem') := by
        simp [BitVec.getLsbD_and, hmask]
      rw [hbit]
      exact ih _ _
    · exact ih _ _

theorem pext64_and_mask (src mask : Word) :
    pext64 src mask = pext64 (src &&& mask) mask :=
  pextAux_and_mask src mask 64 0 0

private theorem packA_bmi2_finite : ∀ n : Fin 256,
    pext64 (S [0,8,16,24,32,40,48,56] (BitVec.ofNat 8 n.val)) fileA =
      (BitVec.ofNat 8 n.val).setWidth 64 := by decide +kernel

private theorem packH_bmi2_finite : ∀ n : Fin 256,
    pext64 (S [7,15,23,31,39,47,55,63] (BitVec.ofNat 8 n.val)) fileH =
      (BitVec.ofNat 8 n.val).setWidth 64 := by decide +kernel

theorem packA1A8Bmi2Word_eq (x : Word) :
    packA1A8Bmi2Word x = (packA1A8 x).setWidth 64 := by
  unfold packA1A8Bmi2Word
  rw [pext64_and_mask]
  have hm := NeuralReversi.Solve1.Projection.masked_eq_scatter
    [0,8,16,24,32,40,48,56] x (by decide) fileA (by decide)
  rw [hm, packA_eq_gather]
  simpa only [BitVec.ofNat_toNat, BitVec.setWidth_eq] using
    packA_bmi2_finite ⟨(P [0,8,16,24,32,40,48,56] x).toNat,
      (P [0,8,16,24,32,40,48,56] x).isLt⟩

theorem packH1H8Bmi2Word_eq (x : Word) :
    packH1H8Bmi2Word x = (packH1H8 x).setWidth 64 := by
  unfold packH1H8Bmi2Word
  rw [pext64_and_mask]
  have hm := NeuralReversi.Solve1.Projection.masked_eq_scatter
    [7,15,23,31,39,47,55,63] x (by decide) fileH (by decide)
  rw [hm, packH_eq_gather]
  simpa only [BitVec.ofNat_toNat, BitVec.setWidth_eq] using
    packH_bmi2_finite ⟨(P [7,15,23,31,39,47,55,63] x).toNat,
      (P [7,15,23,31,39,47,55,63] x).isLt⟩

theorem packA1A8Bmi2_eq_packA1A8 (x : Word) :
    packA1A8Bmi2 x = packA1A8 x := by
  unfold packA1A8Bmi2
  rw [packA1A8Bmi2Word_eq]
  simp

theorem packH1H8Bmi2_eq_packH1H8 (x : Word) :
    packH1H8Bmi2 x = packH1H8 x := by
  unfold packH1H8Bmi2
  rw [packH1H8Bmi2Word_eq]
  simp

private theorem unpackA_bmi2_finite : ∀ n : Fin 256,
    unpackA2A7Bmi2 (BitVec.ofNat 8 n.val) = unpackA2A7 (BitVec.ofNat 8 n.val) := by
  decide +kernel

private theorem unpackH_bmi2_finite : ∀ n : Fin 256,
    unpackH2H7Bmi2 (BitVec.ofNat 8 n.val) = unpackH2H7 (BitVec.ofNat 8 n.val) := by
  decide +kernel

theorem unpackA2A7Bmi2_eq_unpackA2A7 (x : Byte) :
    unpackA2A7Bmi2 x = unpackA2A7 x := by
  simpa only [BitVec.ofNat_toNat, BitVec.setWidth_eq] using
    unpackA_bmi2_finite ⟨x.toNat, x.isLt⟩

theorem unpackH2H7Bmi2_eq_unpackH2H7 (x : Byte) :
    unpackH2H7Bmi2 x = unpackH2H7 x := by
  simpa only [BitVec.ofNat_toNat, BitVec.setWidth_eq] using
    unpackH_bmi2_finite ⟨x.toNat, x.isLt⟩

end NeuralReversi.Stability.Semantics
