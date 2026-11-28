module
public import NeuralReversi.Stability.EdgeEncoding
public import NeuralReversi.Solve1.Projection

@[expose] public section

namespace NeuralReversi.Stability.Semantics
set_option maxRecDepth 100000
set_option maxHeartbeats 0
set_option linter.unusedSimpArgs false

abbrev P := NeuralReversi.Solve1.Projection.gather
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

private def unpackASpec (x : Byte) : Word :=
  (if x.getLsbD 1 then 1#64 <<< 8 else 0) |||
  (if x.getLsbD 2 then 1#64 <<< 16 else 0) |||
  (if x.getLsbD 3 then 1#64 <<< 24 else 0) |||
  (if x.getLsbD 4 then 1#64 <<< 32 else 0) |||
  (if x.getLsbD 5 then 1#64 <<< 40 else 0) |||
  (if x.getLsbD 6 then 1#64 <<< 48 else 0)

private def unpackHSpec (x : Byte) : Word :=
  (if x.getLsbD 1 then 1#64 <<< 15 else 0) |||
  (if x.getLsbD 2 then 1#64 <<< 23 else 0) |||
  (if x.getLsbD 3 then 1#64 <<< 31 else 0) |||
  (if x.getLsbD 4 then 1#64 <<< 39 else 0) |||
  (if x.getLsbD 5 then 1#64 <<< 47 else 0) |||
  (if x.getLsbD 6 then 1#64 <<< 55 else 0)

private theorem unpackA_eq_spec_64 : ∀ n : Fin 64,
    unpackA2A7 (BitVec.ofNat 8 (2 * n.val)) = unpackASpec (BitVec.ofNat 8 (2 * n.val)) := by
  decide +kernel

private theorem unpackH_eq_spec_64 : ∀ n : Fin 64,
    unpackH2H7 (BitVec.ofNat 8 (2 * n.val)) = unpackHSpec (BitVec.ofNat 8 (2 * n.val)) := by
  decide +kernel

private theorem mask7e_eq (x : Byte) :
    x &&& (0x7e : Byte) = BitVec.ofNat 8 (2 * ((x >>> 1) &&& (0x3f : Byte)).toNat) := by
  have all : ∀ n : Fin 256,
      let b := BitVec.ofNat 8 n.val
      b &&& (0x7e : Byte) = BitVec.ofNat 8 (2 * ((b >>> 1) &&& (0x3f : Byte)).toNat) := by decide +kernel
  simpa only [BitVec.ofNat_toNat, BitVec.setWidth_eq] using all ⟨x.toNat, x.isLt⟩

private theorem mask7e_lt (x : Byte) : ((x >>> 1) &&& (0x3f : Byte)).toNat < 64 := by
  have all : ∀ n : Fin 256, (((BitVec.ofNat 8 n.val) >>> 1) &&& (0x3f : Byte)).toNat < 64 := by decide +kernel
  simpa only [BitVec.ofNat_toNat, BitVec.setWidth_eq] using all ⟨x.toNat, x.isLt⟩

private theorem mask7e_bits (x : Byte) : ∀ k : Fin 6,
    (x &&& (0x7e : Byte)).getLsbD (k.val + 1) = x.getLsbD (k.val + 1) := by
  intro k
  have hbit : ∀ j : Fin 6, (0x7e : Byte).getLsbD (j.val + 1) = true := by decide +kernel
  rw [BitVec.getLsbD_and, hbit k, Bool.and_true]

private theorem unpackA_eq_spec (x : Byte) : unpackA2A7 x = unpackASpec x := by
  have h1 : unpackA2A7 x = unpackA2A7 (x &&& (0x7e : Byte)) := by
    simp [unpackA2A7, BitVec.and_assoc]
  have h2 : unpackASpec (x &&& (0x7e : Byte)) = unpackASpec x := by
    simp only [unpackASpec, mask7e_bits x ⟨0, by decide⟩, mask7e_bits x ⟨1, by decide⟩,
      mask7e_bits x ⟨2, by decide⟩, mask7e_bits x ⟨3, by decide⟩,
      mask7e_bits x ⟨4, by decide⟩, mask7e_bits x ⟨5, by decide⟩]
  rw [h1, ← h2]
  have hm : x &&& (0x7e : Byte) = BitVec.ofNat 8 (2 * ((x >>> 1) &&& (0x3f : Byte)).toNat) := mask7e_eq x
  rw [hm]
  exact unpackA_eq_spec_64 ⟨((x >>> 1) &&& (0x3f : Byte)).toNat, mask7e_lt x⟩

private theorem unpackH_eq_spec (x : Byte) : unpackH2H7 x = unpackHSpec x := by
  have h1 : unpackH2H7 x = unpackH2H7 (x &&& (0x7e : Byte)) := by
    simp [unpackH2H7, BitVec.and_assoc]
  have h2 : unpackHSpec (x &&& (0x7e : Byte)) = unpackHSpec x := by
    simp only [unpackHSpec, mask7e_bits x ⟨0, by decide⟩, mask7e_bits x ⟨1, by decide⟩,
      mask7e_bits x ⟨2, by decide⟩, mask7e_bits x ⟨3, by decide⟩,
      mask7e_bits x ⟨4, by decide⟩, mask7e_bits x ⟨5, by decide⟩]
  rw [h1, ← h2]
  have hm : x &&& (0x7e : Byte) = BitVec.ofNat 8 (2 * ((x >>> 1) &&& (0x3f : Byte)).toNat) := mask7e_eq x
  rw [hm]
  exact unpackH_eq_spec_64 ⟨((x >>> 1) &&& (0x3f : Byte)).toNat, mask7e_lt x⟩

private theorem getLsbD_ite_shift (b : Bool) (k i : Nat) (hk : k < 64) (hi : i < 64) :
    (if b = true then 1#64 <<< k else 0).getLsbD i = (b && decide (i = k)) := by
  cases b with
  | false => simp
  | true =>
    simp only [↓reduceIte, Bool.true_and, BitVec.getLsbD_shiftLeft, hi, decide_true, Bool.true_and]
    by_cases h : i = k
    · subst h; simp
    · by_cases hlt : i < k
      · simp [hlt, h]
      · have hne : i - k ≠ 0 := by omega
        simp [hlt, h, BitVec.getLsbD_one, hne]

theorem unpackA_bit (x : Byte) (r c : Fin 8) :
    (unpackA2A7 x).getLsbD (8*r.val+c.val) =
      (decide (c.val = 0 ∧ 0 < r.val ∧ r.val < 7) && x.getLsbD r.val) := by
  rw [unpackA_eq_spec]
  have hi : 8 * r.val + c.val < 64 := by omega
  simp only [unpackASpec, BitVec.getLsbD_or,
    getLsbD_ite_shift _ 8 _ (by decide) hi, getLsbD_ite_shift _ 16 _ (by decide) hi,
    getLsbD_ite_shift _ 24 _ (by decide) hi, getLsbD_ite_shift _ 32 _ (by decide) hi,
    getLsbD_ite_shift _ 40 _ (by decide) hi, getLsbD_ite_shift _ 48 _ (by decide) hi]
  rcases r with ⟨r, hr⟩
  by_cases hc : c.val = 0
  · match r with
    | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 => simp [hc]
    | _ + 8 => omega
  · have h1 : 8 * r + c.val ≠ 8 := by omega
    have h2 : 8 * r + c.val ≠ 16 := by omega
    have h3 : 8 * r + c.val ≠ 24 := by omega
    have h4 : 8 * r + c.val ≠ 32 := by omega
    have h5 : 8 * r + c.val ≠ 40 := by omega
    have h6 : 8 * r + c.val ≠ 48 := by omega
    simp [hc, h1, h2, h3, h4, h5, h6]

theorem unpackH_bit (x : Byte) (r c : Fin 8) :
    (unpackH2H7 x).getLsbD (8*r.val+c.val) =
      (decide (c.val = 7 ∧ 0 < r.val ∧ r.val < 7) && x.getLsbD r.val) := by
  rw [unpackH_eq_spec]
  have hi : 8 * r.val + c.val < 64 := by omega
  simp only [unpackHSpec, BitVec.getLsbD_or,
    getLsbD_ite_shift _ 15 _ (by decide) hi, getLsbD_ite_shift _ 23 _ (by decide) hi,
    getLsbD_ite_shift _ 31 _ (by decide) hi, getLsbD_ite_shift _ 39 _ (by decide) hi,
    getLsbD_ite_shift _ 47 _ (by decide) hi, getLsbD_ite_shift _ 55 _ (by decide) hi]
  rcases r with ⟨r, hr⟩
  by_cases hc : c.val = 7
  · match r with
    | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 => simp [hc]
    | _ + 8 => omega
  · have h1 : 8 * r + c.val ≠ 15 := by omega
    have h2 : 8 * r + c.val ≠ 23 := by omega
    have h3 : 8 * r + c.val ≠ 31 := by omega
    have h4 : 8 * r + c.val ≠ 39 := by omega
    have h5 : 8 * r + c.val ≠ 47 := by omega
    have h6 : 8 * r + c.val ≠ 55 := by omega
    simp [hc, h1, h2, h3, h4, h5, h6]

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

private def maskBits (mask : Word) : Nat → List Nat
  | 0 => []
  | rem' + 1 =>
    let bit := 63 - rem'
    if mask.getLsbD bit then bit :: maskBits mask rem' else maskBits mask rem'

private def pextList (src : Word) : List Nat → Nat → Word → Word
  | [], _, acc => acc
  | bit :: rest, k, acc =>
    pextList src rest (k + 1) (if src.getLsbD bit then acc ||| (1#64 <<< k) else acc)

private theorem pextAux_eq_pextList (src mask : Word) (rem k : Nat) (acc : Word) :
    pextAux src mask rem k acc = pextList src (maskBits mask rem) k acc := by
  induction rem generalizing k acc with
  | zero => rfl
  | succ rem' ih =>
    simp only [pextAux, maskBits]
    split
    · simp only [pextList]; exact ih _ _
    · exact ih _ _

private theorem pextList_getLsbD (src : Word) (bits : List Nat) (k : Nat) (acc : Word) (i : Nat) (hi : i < 64) :
    (pextList src bits k acc).getLsbD i =
      (acc.getLsbD i || (decide (k ≤ i) &&
        match bits[i - k]? with
        | some bit => src.getLsbD bit
        | none => false)) := by
  induction bits generalizing k acc with
  | nil => simp [pextList]
  | cons bit rest ih =>
    rw [pextList, ih]
    have hbit_shift : (if src.getLsbD bit then acc ||| (1#64 <<< k) else acc).getLsbD i =
        (acc.getLsbD i || (decide (i = k) && src.getLsbD bit)) := by
      cases src.getLsbD bit with
      | false => simp
      | true =>
        simp only [↓reduceIte, Bool.and_true, BitVec.getLsbD_or, BitVec.getLsbD_shiftLeft, hi,
          decide_true, Bool.true_and]
        by_cases heq : i = k
        · subst heq; simp
        · by_cases hlt : i < k
          · simp [hlt, heq]
          · have hne : i - k ≠ 0 := by omega
            simp [hlt, heq, BitVec.getLsbD_one, hne]
    rw [hbit_shift]
    by_cases hlt : i < k
    · have h1 : ¬ (k ≤ i) := by omega
      have h2 : ¬ (k + 1 ≤ i) := by omega
      have h3 : i ≠ k := by omega
      simp [h1, h2, h3]
    · by_cases heq : i = k
      · subst heq
        have h2 : ¬ (i + 1 ≤ i) := by omega
        simp [h2]
      · have hle : k ≤ i := by omega
        have hle1 : k + 1 ≤ i := by omega
        have hsub : i - k = (i - (k + 1)) + 1 := by omega
        simp [heq, hle, hle1, hsub]

private theorem maskBits_fileA : maskBits fileA 64 = [0, 8, 16, 24, 32, 40, 48, 56] := by decide +kernel
private theorem maskBits_fileH : maskBits fileH 64 = [7, 15, 23, 31, 39, 47, 55, 63] := by decide +kernel
private theorem maskBits_fileAInner : maskBits fileAInner 64 = [8, 16, 24, 32, 40, 48] := by decide +kernel
private theorem maskBits_fileHInner : maskBits fileHInner 64 = [15, 23, 31, 39, 47, 55] := by decide +kernel

theorem packA1A8Bmi2Word_eq (x : Word) :
    packA1A8Bmi2Word x = (packA1A8 x).setWidth 64 := by
  unfold packA1A8Bmi2Word pext64
  rw [pextAux_eq_pextList, maskBits_fileA]
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  rw [pextList_getLsbD x _ 0 0 i hi, BitVec.getLsbD_setWidth]
  by_cases hi8 : i < 8
  · rw [packA_bit x ⟨i, hi8⟩]
    match i with
    | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 => simp
    | _ + 8 => omega
  · have hnone : ([0, 8, 16, 24, 32, 40, 48, 56] : List Nat)[i]? = none :=
      List.getElem?_eq_none (by simp; omega)
    have hge : (packA1A8 x).getLsbD i = false := BitVec.getLsbD_of_ge _ _ (by omega)
    simp [hi8, hnone, hge]

theorem packH1H8Bmi2Word_eq (x : Word) :
    packH1H8Bmi2Word x = (packH1H8 x).setWidth 64 := by
  unfold packH1H8Bmi2Word pext64
  rw [pextAux_eq_pextList, maskBits_fileH]
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  rw [pextList_getLsbD x _ 0 0 i hi, BitVec.getLsbD_setWidth]
  by_cases hi8 : i < 8
  · rw [packH_bit x ⟨i, hi8⟩]
    match i with
    | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 => simp
    | _ + 8 => omega
  · have hnone : ([7, 15, 23, 31, 39, 47, 55, 63] : List Nat)[i]? = none :=
      List.getElem?_eq_none (by simp; omega)
    have hge : (packH1H8 x).getLsbD i = false := BitVec.getLsbD_of_ge _ _ (by omega)
    simp [hi8, hnone, hge]

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

private def pdepList (src : Word) : List Nat → Nat → Word → Word
  | [], _, acc => acc
  | bit :: rest, k, acc =>
    pdepList src rest (k + 1) (if src.getLsbD k then acc ||| (1#64 <<< bit) else acc)

private theorem pdepAux_eq_pdepList (src mask : Word) (rem k : Nat) (acc : Word) :
    pdepAux src mask rem k acc = pdepList src (maskBits mask rem) k acc := by
  induction rem generalizing k acc with
  | zero => rfl
  | succ rem' ih =>
    simp only [pdepAux, maskBits]
    split
    · simp only [pdepList]; exact ih _ _
    · exact ih _ _

private theorem pdep_shift_bit (x : Byte) (k : Fin 6) :
    (((x &&& 0x7e) >>> 1).setWidth 64).getLsbD k.val = x.getLsbD (k.val + 1) := by
  rw [BitVec.getLsbD_setWidth, BitVec.getLsbD_ushiftRight, BitVec.getLsbD_and]
  have hk64 : k.val < 64 := by omega
  have hbit : ∀ j : Fin 6, (0x7e : Byte).getLsbD (j.val + 1) = true := by decide +kernel
  have hadd : 1 + k.val = k.val + 1 := Nat.add_comm 1 k.val
  rw [hadd, hbit k]
  simp [hk64]

theorem unpackA2A7Bmi2_eq_unpackA2A7 (x : Byte) :
    unpackA2A7Bmi2 x = unpackA2A7 x := by
  rw [unpackA_eq_spec]
  unfold unpackA2A7Bmi2 pdep64
  rw [pdepAux_eq_pdepList, maskBits_fileAInner]
  simp only [pdepList, pdep_shift_bit x ⟨0, by decide⟩, pdep_shift_bit x ⟨1, by decide⟩,
    pdep_shift_bit x ⟨2, by decide⟩, pdep_shift_bit x ⟨3, by decide⟩,
    pdep_shift_bit x ⟨4, by decide⟩, pdep_shift_bit x ⟨5, by decide⟩, unpackASpec]
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  cases x.getLsbD 1 <;> cases x.getLsbD 2 <;> cases x.getLsbD 3 <;>
    cases x.getLsbD 4 <;> cases x.getLsbD 5 <;> cases x.getLsbD 6 <;> simp

theorem unpackH2H7Bmi2_eq_unpackH2H7 (x : Byte) :
    unpackH2H7Bmi2 x = unpackH2H7 x := by
  rw [unpackH_eq_spec]
  unfold unpackH2H7Bmi2 pdep64
  rw [pdepAux_eq_pdepList, maskBits_fileHInner]
  simp only [pdepList, pdep_shift_bit x ⟨0, by decide⟩, pdep_shift_bit x ⟨1, by decide⟩,
    pdep_shift_bit x ⟨2, by decide⟩, pdep_shift_bit x ⟨3, by decide⟩,
    pdep_shift_bit x ⟨4, by decide⟩, pdep_shift_bit x ⟨5, by decide⟩, unpackHSpec]
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  cases x.getLsbD 1 <;> cases x.getLsbD 2 <;> cases x.getLsbD 3 <;>
    cases x.getLsbD 4 <;> cases x.getLsbD 5 <;> cases x.getLsbD 6 <;> simp

end NeuralReversi.Stability.Semantics
