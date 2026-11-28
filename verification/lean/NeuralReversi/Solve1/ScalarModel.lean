module
public import NeuralReversi.Solve1.ScalarParams
public import NeuralReversi.Solve1.Projection
public import NeuralReversi.Solve1.Geometry

@[expose] public section

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
    laneIndex s lane (p &&& laneMask s lane) = laneIndex s lane p := by
  rcases lane with ⟨l, hl⟩
  match l with
  | 0 => simp [laneIndex, laneMask, BitVec.and_assoc]
  | 1 => exact masked_row p _
  | 2 => simp [laneIndex, laneMask, BitVec.and_assoc]
  | 3 => simp [laneIndex, laneMask, BitVec.and_assoc]
  | k + 4 => omega

private theorem ray_mask (p m : Word) (acc : Nat) (cells : List Nat)
    (h : ∀ i ∈ cells, m.getLsbD i = true) :
    Geometry.ray (p &&& m) acc cells = Geometry.ray p acc cells := by
  induction cells generalizing acc with
  | nil => rfl
  | cons i rest ih =>
    have hi : m.getLsbD i = true := h i (by simp)
    simp only [Geometry.ray, BitVec.getLsbD_and, hi, Bool.and_true,
      ih (acc + 1) (fun j hj => h j (List.mem_cons_of_mem _ hj))]

private theorem ray_mask_zero (p m : Word) (cells : List Nat)
    (h : cells.length ≤ 1 ∨ ∀ i ∈ cells, m.getLsbD i = true) :
    Geometry.ray (p &&& m) 0 cells = Geometry.ray p 0 cells := by
  rcases h with hlen | hall
  · match cells with
    | [] => rfl
    | [i] => simp [Geometry.ray]
    | _ :: _ :: _ => simp at hlen
  · exact ray_mask p m 0 cells hall

private def rayCovered (m : Word) (cells : List Nat) : Prop :=
  cells.length ≤ 1 ∨ ∀ i ∈ cells, m.getLsbD i = true

private instance (m : Word) (cells : List Nat) : Decidable (rayCovered m cells) := by
  unfold rayCovered; infer_instance

private theorem laneMask_covers_rays : ∀ s : Fin 64,
    (rayCovered (laneMask s.val 0) (Geometry.raySquares s.val 1 0) ∧
     rayCovered (laneMask s.val 0) (Geometry.raySquares s.val (-1) 0)) ∧
    (rayCovered (laneMask s.val 1) (Geometry.raySquares s.val 0 1) ∧
     rayCovered (laneMask s.val 1) (Geometry.raySquares s.val 0 (-1))) ∧
    (rayCovered (laneMask s.val 2) (Geometry.raySquares s.val 1 (-1)) ∧
     rayCovered (laneMask s.val 2) (Geometry.raySquares s.val (-1) 1) ∧
     ((params s.val).maskD9 = 0 →
       rayCovered (laneMask s.val 2) (Geometry.raySquares s.val 1 1) ∧
       rayCovered (laneMask s.val 2) (Geometry.raySquares s.val (-1) (-1)))) ∧
    ((params s.val).maskD9 ≠ 0 →
      rayCovered (laneMask s.val 3) (Geometry.raySquares s.val 1 1) ∧
      rayCovered (laneMask s.val 3) (Geometry.raySquares s.val (-1) (-1))) := by
  decide +kernel

theorem laneReference_mask (s : Fin 64) (lane : Fin 4) (p : Word) :
    laneReference s.val lane (p &&& laneMask s.val lane) = laneReference s.val lane p := by
  have hc := laneMask_covers_rays s
  have hl : lane = 0 ∨ lane = 1 ∨ lane = 2 ∨ lane = 3 := by omega
  rcases hl with rfl | rfl | rfl | rfl
  · change Geometry.ray _ 0 _ + Geometry.ray _ 0 _ = Geometry.ray _ 0 _ + Geometry.ray _ 0 _
    rw [ray_mask_zero _ _ _ hc.1.1, ray_mask_zero _ _ _ hc.1.2]
  · change Geometry.ray _ 0 _ + Geometry.ray _ 0 _ = Geometry.ray _ 0 _ + Geometry.ray _ 0 _
    rw [ray_mask_zero _ _ _ hc.2.1.1, ray_mask_zero _ _ _ hc.2.1.2]
  · change (if (params s.val).maskD9 = 0 then _ else _) = (if (params s.val).maskD9 = 0 then _ else _)
    by_cases h0 : (params s.val).maskD9 = 0
    · simp only [h0, ↓reduceIte]
      have ⟨h1, h2⟩ := hc.2.2.1.2.2 h0
      rw [ray_mask_zero _ _ _ h1, ray_mask_zero _ _ _ h2,
        ray_mask_zero _ _ _ hc.2.2.1.1, ray_mask_zero _ _ _ hc.2.2.1.2.1]
    · simp only [h0, ↓reduceIte]
      rw [ray_mask_zero _ _ _ hc.2.2.1.1, ray_mask_zero _ _ _ hc.2.2.1.2.1]
  · change (if (params s.val).maskD9 = 0 then _ else _) = (if (params s.val).maskD9 = 0 then _ else _)
    by_cases h0 : (params s.val).maskD9 = 0
    · simp only [h0, ↓reduceIte]
    · simp only [h0, ↓reduceIte]
      have ⟨h1, h2⟩ := hc.2.2.2 h0
      rw [ray_mask_zero _ _ _ h1, ray_mask_zero _ _ _ h2]

private theorem nat_add_eq_or : ∀ (a b : Nat), a &&& b = 0 → a + b = a ||| b := by
  intro a
  induction a using Nat.strongRecOn with
  | ind a ih =>
    intro b hdisj
    by_cases ha : a = 0
    · subst ha; simp
    · by_cases hb : b = 0
      · subst hb; simp
      · have hmod : (a % 2) &&& (b % 2) = 0 := by
          have h := congrArg (· % 2) hdisj
          simpa [Nat.and_mod_two_pow (n := 1)] using h
        have hdiv : (a / 2) &&& (b / 2) = 0 := by
          have h := congrArg (· / 2) hdisj
          simpa [Nat.and_div_two_pow (n := 1)] using h
        have hrec := ih (a / 2) (by omega) (b / 2) hdiv
        have hor_div : (a ||| b) / 2 = (a / 2) ||| (b / 2) := by
          simpa using Nat.or_div_two_pow (a := a) (b := b) (n := 1)
        have hor_mod : (a ||| b) % 2 = (a % 2) ||| (b % 2) := by
          simpa using Nat.or_mod_two_pow (a := a) (b := b) (n := 1)
        have hbit : a % 2 + b % 2 = (a % 2) ||| (b % 2) := by
          have ha2 : a % 2 = 0 ∨ a % 2 = 1 := by omega
          have hb2 : b % 2 = 0 ∨ b % 2 = 1 := by omega
          rcases ha2 with ha0 | ha1 <;>
          rcases hb2 with hb0 | hb1 <;>
          simp_all
        omega

private theorem byte_add_toNat_eq_or (a b : Byte) (h : a &&& b = 0) :
    a.toNat + b.toNat = (a ||| b).toNat := by
  have h0 : a.toNat &&& b.toNat = 0 := by
    have ht := congrArg BitVec.toNat h
    simpa [BitVec.toNat_and] using ht
  simpa [BitVec.toNat_or] using nat_add_eq_or a.toNat b.toNat h0

def byteOf (w : Word) (k : Nat) : Byte := (w >>> (8 * k)).setWidth 8

private theorem byteOf_toNat (w : Word) (k : Nat) :
    (byteOf w k).toNat = (w.toNat / 2 ^ (8 * k)) % 256 := by
  simp [byteOf, BitVec.toNat_setWidth, BitVec.toNat_ushiftRight, Nat.shiftRight_eq_div_pow]

private theorem word_toNat_bytes (w : Word) :
    w.toNat =
      (byteOf w 0).toNat +
      256 * (byteOf w 1).toNat +
      65536 * (byteOf w 2).toNat +
      16777216 * (byteOf w 3).toNat +
      4294967296 * (byteOf w 4).toNat +
      1099511627776 * (byteOf w 5).toNat +
      281474976710656 * (byteOf w 6).toNat +
      72057594037927936 * (byteOf w 7).toNat := by
  simp only [byteOf_toNat]
  have hw := w.isLt
  omega

private theorem byteOf_and (p m : Word) (k : Nat) :
    byteOf (p &&& m) k = byteOf p k &&& byteOf m k := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  simp [byteOf, hi]

private theorem byteOf_getLsbD (w : Word) (k i : Nat) (hi : i < 8) :
    (byteOf w k).getLsbD i = w.getLsbD (8 * k + i) := by
  simp [byteOf, hi]

private theorem extractLsb'_56_toNat (x : Word) :
    (x.extractLsb' 56 8).toNat = (x.toNat / 2 ^ 56) % 256 := by
  simp [BitVec.extractLsb', Nat.shiftRight_eq_div_pow]

theorem sumDisjointBytes_eq (p mask : Word)
    (hdisj : ∀ j k : Fin 8, j.val < k.val → byteOf mask j.val &&& byteOf mask k.val = 0) :
    ((p &&& mask) * (0x0101010101010101 : Word)).extractLsb' 56 8 =
      byteOf (p &&& mask) 0 ||| byteOf (p &&& mask) 1 ||| byteOf (p &&& mask) 2 |||
      byteOf (p &&& mask) 3 ||| byteOf (p &&& mask) 4 ||| byteOf (p &&& mask) 5 |||
      byteOf (p &&& mask) 6 ||| byteOf (p &&& mask) 7 := by
  let w := p &&& mask
  let b := fun k => byteOf w k
  have hd : ∀ j k : Fin 8, j.val < k.val → b j.val &&& b k.val = 0 := by
    intro j k hlt
    have hm := hdisj j k hlt
    apply BitVec.eq_of_getLsbD_eq
    intro i hi
    have hmi := congrArg (fun x : Byte => x.getLsbD i) hm
    rw [BitVec.getLsbD_and, show (0 : Byte).getLsbD i = false from BitVec.getLsbD_zero] at hmi
    change (byteOf w j.val &&& byteOf w k.val).getLsbD i = (0 : Byte).getLsbD i
    rw [BitVec.getLsbD_and, byteOf_and, byteOf_and, BitVec.getLsbD_and, BitVec.getLsbD_and,
      show (0 : Byte).getLsbD i = false from BitVec.getLsbD_zero]
    cases hm1 : (byteOf mask j.val).getLsbD i <;> cases hm2 : (byteOf mask k.val).getLsbD i <;> simp_all
  have hstep : ∀ (acc x : Byte), acc &&& x = 0 → acc.toNat + x.toNat = (acc ||| x).toNat :=
    byte_add_toNat_eq_or
  have hd01 : b 0 &&& b 1 = 0 := hd ⟨0, by decide⟩ ⟨1, by decide⟩ (by decide)
  have hd012 : (b 0 ||| b 1) &&& b 2 = 0 := by
    rw [BitVec.and_or_distrib_right, hd ⟨0, by decide⟩ ⟨2, by decide⟩ (by decide),
      hd ⟨1, by decide⟩ ⟨2, by decide⟩ (by decide)]; rfl
  have hd0123 : (b 0 ||| b 1 ||| b 2) &&& b 3 = 0 := by
    rw [BitVec.and_or_distrib_right, BitVec.and_or_distrib_right,
      hd ⟨0, by decide⟩ ⟨3, by decide⟩ (by decide), hd ⟨1, by decide⟩ ⟨3, by decide⟩ (by decide),
      hd ⟨2, by decide⟩ ⟨3, by decide⟩ (by decide)]; rfl
  have hd01234 : (b 0 ||| b 1 ||| b 2 ||| b 3) &&& b 4 = 0 := by
    rw [BitVec.and_or_distrib_right, BitVec.and_or_distrib_right, BitVec.and_or_distrib_right,
      hd ⟨0, by decide⟩ ⟨4, by decide⟩ (by decide), hd ⟨1, by decide⟩ ⟨4, by decide⟩ (by decide),
      hd ⟨2, by decide⟩ ⟨4, by decide⟩ (by decide), hd ⟨3, by decide⟩ ⟨4, by decide⟩ (by decide)]; rfl
  have hd012345 : (b 0 ||| b 1 ||| b 2 ||| b 3 ||| b 4) &&& b 5 = 0 := by
    rw [BitVec.and_or_distrib_right, BitVec.and_or_distrib_right, BitVec.and_or_distrib_right,
      BitVec.and_or_distrib_right,
      hd ⟨0, by decide⟩ ⟨5, by decide⟩ (by decide), hd ⟨1, by decide⟩ ⟨5, by decide⟩ (by decide),
      hd ⟨2, by decide⟩ ⟨5, by decide⟩ (by decide), hd ⟨3, by decide⟩ ⟨5, by decide⟩ (by decide),
      hd ⟨4, by decide⟩ ⟨5, by decide⟩ (by decide)]; rfl
  have hd0123456 : (b 0 ||| b 1 ||| b 2 ||| b 3 ||| b 4 ||| b 5) &&& b 6 = 0 := by
    rw [BitVec.and_or_distrib_right, BitVec.and_or_distrib_right, BitVec.and_or_distrib_right,
      BitVec.and_or_distrib_right, BitVec.and_or_distrib_right,
      hd ⟨0, by decide⟩ ⟨6, by decide⟩ (by decide), hd ⟨1, by decide⟩ ⟨6, by decide⟩ (by decide),
      hd ⟨2, by decide⟩ ⟨6, by decide⟩ (by decide), hd ⟨3, by decide⟩ ⟨6, by decide⟩ (by decide),
      hd ⟨4, by decide⟩ ⟨6, by decide⟩ (by decide), hd ⟨5, by decide⟩ ⟨6, by decide⟩ (by decide)]; rfl
  have hd01234567 : (b 0 ||| b 1 ||| b 2 ||| b 3 ||| b 4 ||| b 5 ||| b 6) &&& b 7 = 0 := by
    rw [BitVec.and_or_distrib_right, BitVec.and_or_distrib_right, BitVec.and_or_distrib_right,
      BitVec.and_or_distrib_right, BitVec.and_or_distrib_right, BitVec.and_or_distrib_right,
      hd ⟨0, by decide⟩ ⟨7, by decide⟩ (by decide), hd ⟨1, by decide⟩ ⟨7, by decide⟩ (by decide),
      hd ⟨2, by decide⟩ ⟨7, by decide⟩ (by decide), hd ⟨3, by decide⟩ ⟨7, by decide⟩ (by decide),
      hd ⟨4, by decide⟩ ⟨7, by decide⟩ (by decide), hd ⟨5, by decide⟩ ⟨7, by decide⟩ (by decide),
      hd ⟨6, by decide⟩ ⟨7, by decide⟩ (by decide)]; rfl
  have hsum : (b 0).toNat + (b 1).toNat + (b 2).toNat + (b 3).toNat +
      (b 4).toNat + (b 5).toNat + (b 6).toNat + (b 7).toNat =
      (b 0 ||| b 1 ||| b 2 ||| b 3 ||| b 4 ||| b 5 ||| b 6 ||| b 7).toNat := by
    rw [hstep _ _ hd01, hstep _ _ hd012, hstep _ _ hd0123, hstep _ _ hd01234,
      hstep _ _ hd012345, hstep _ _ hd0123456, hstep _ _ hd01234567]
  have hlt : (b 0 ||| b 1 ||| b 2 ||| b 3 ||| b 4 ||| b 5 ||| b 6 ||| b 7).toNat < 256 :=
    (b 0 ||| b 1 ||| b 2 ||| b 3 ||| b 4 ||| b 5 ||| b 6 ||| b 7).isLt
  have hw_eq := word_toNat_bytes w
  apply BitVec.eq_of_toNat_eq
  rw [extractLsb'_56_toNat, BitVec.toNat_mul]
  have hmul_val : (0x0101010101010101 : Word).toNat = 0x0101010101010101 := rfl
  rw [hmul_val]
  change (w.toNat * 0x0101010101010101 % 2 ^ 64) / 2 ^ 56 % 256 =
    (b 0 ||| b 1 ||| b 2 ||| b 3 ||| b 4 ||| b 5 ||| b 6 ||| b 7).toNat
  let S := (b 0).toNat + (b 1).toNat + (b 2).toNat + (b 3).toNat +
    (b 4).toNat + (b 5).toNat + (b 6).toNat + (b 7).toNat
  let lo := (b 0).toNat +
    256 * ((b 0).toNat + (b 1).toNat) +
    65536 * ((b 0).toNat + (b 1).toNat + (b 2).toNat) +
    16777216 * ((b 0).toNat + (b 1).toNat + (b 2).toNat + (b 3).toNat) +
    4294967296 * ((b 0).toNat + (b 1).toNat + (b 2).toNat + (b 3).toNat + (b 4).toNat) +
    1099511627776 * ((b 0).toNat + (b 1).toNat + (b 2).toNat + (b 3).toNat + (b 4).toNat + (b 5).toNat) +
    281474976710656 * ((b 0).toNat + (b 1).toNat + (b 2).toNat + (b 3).toNat + (b 4).toNat + (b 5).toNat + (b 6).toNat)
  let Q := ((b 1).toNat + (b 2).toNat + (b 3).toNat + (b 4).toNat + (b 5).toNat + (b 6).toNat + (b 7).toNat) +
    256 * ((b 2).toNat + (b 3).toNat + (b 4).toNat + (b 5).toNat + (b 6).toNat + (b 7).toNat) +
    65536 * ((b 3).toNat + (b 4).toNat + (b 5).toNat + (b 6).toNat + (b 7).toNat) +
    16777216 * ((b 4).toNat + (b 5).toNat + (b 6).toNat + (b 7).toNat) +
    4294967296 * ((b 5).toNat + (b 6).toNat + (b 7).toNat) +
    1099511627776 * ((b 6).toNat + (b 7).toNat) +
    281474976710656 * (b 7).toNat
  have hS_lt : S < 256 := by omega
  have hlo_lt : lo < 2 ^ 56 := by dsimp only [lo]; omega
  have hprod : w.toNat * 0x0101010101010101 = lo + 2 ^ 56 * S + 2 ^ 64 * Q := by
    rw [hw_eq]; dsimp only [b, lo, S, Q]; omega
  rw [hprod, Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt (show lo + 2 ^ 56 * S < 2 ^ 64 by omega),
    Nat.add_mul_div_left _ _ (by decide : 0 < 2 ^ 56), Nat.div_eq_of_lt hlo_lt, Nat.zero_add,
    Nat.mod_eq_of_lt hS_lt]
  exact hsum

def colMult (c : Nat) : Word := BitVec.ofNat 64 (0x0102040810204080 / 2 ^ c)

theorem packColumn_getLsbD (p : Word) (c : Fin 8) (i : Nat) (hi : i < 8) :
    (((p &&& (0x0101010101010101#64 <<< c.val)) * colMult c.val).extractLsb' 56 8).getLsbD i =
      p.getLsbD (8 * i + c.val) := by
  let w := p &&& (0x0101010101010101#64 <<< c.val)
  let d : Nat → Nat := fun r => if p.getLsbD (8 * r + c.val) then 1 else 0
  have hd_le : ∀ r, d r ≤ 1 := fun r => by dsimp only [d]; split <;> omega
  have hbyte : ∀ r : Fin 8, (byteOf w r.val).toNat = d r.val * 2 ^ c.val := by
    intro r
    have hbm : ∀ col : Fin 8, ∀ row : Fin 8,
        byteOf (0x0101010101010101#64 <<< col.val) row.val = 1#8 <<< col.val := by decide +kernel
    have hbw : byteOf w r.val = byteOf p r.val &&& (1#8 <<< c.val) := by
      rw [byteOf_and, hbm c r]
    have hbit : (byteOf p r.val).getLsbD c.val = p.getLsbD (8 * r.val + c.val) :=
      byteOf_getLsbD p r.val c.val c.isLt
    have hpow : (1#8 <<< c.val).toNat = 2 ^ c.val := by
      have : ∀ col : Fin 8, (1#8 <<< col.val).toNat = 2 ^ col.val := by decide +kernel
      exact this c
    have hval : ∀ (x : Byte), (x &&& (1#8 <<< c.val)).toNat =
        (if x.getLsbD c.val then 1 else 0) * 2 ^ c.val := by
      intro x
      cases hx : x.getLsbD c.val with
      | true =>
        have heq : x &&& (1#8 <<< c.val) = 1#8 <<< c.val := by
          apply BitVec.eq_of_getLsbD_eq
          intro j hj
          rw [BitVec.getLsbD_and, BitVec.getLsbD_shiftLeft]
          simp only [hj, decide_true, Bool.true_and]
          by_cases hjc : j = c.val
          · subst hjc; rw [hx]; simp
          · by_cases hlt : j < c.val
            · simp [hlt]
            · have hne : j - c.val ≠ 0 := by omega
              simp [hlt, BitVec.getLsbD_one, hne]
        rw [heq, hpow]; simp
      | false =>
        have heq : x &&& (1#8 <<< c.val) = 0#8 := by
          apply BitVec.eq_of_getLsbD_eq
          intro j hj
          rw [BitVec.getLsbD_and, BitVec.getLsbD_shiftLeft, BitVec.getLsbD_zero]
          simp only [hj, decide_true, Bool.true_and]
          by_cases hjc : j = c.val
          · subst hjc; rw [hx]; simp
          · by_cases hlt : j < c.val
            · simp [hlt]
            · have hne : j - c.val ≠ 0 := by omega
              simp [hlt, BitVec.getLsbD_one, hne]
        rw [heq]; simp
    rw [hbw, hval, hbit]
  have hcm : 2 ^ c.val * (colMult c.val).toNat = 0x0102040810204080 := by
    have : ∀ col : Fin 8, 2 ^ col.val * (colMult col.val).toNat = 0x0102040810204080 := by decide +kernel
    exact this c
  have hD : w.toNat * (colMult c.val).toNat =
      (d 0 + 256 * d 1 + 65536 * d 2 + 16777216 * d 3 + 4294967296 * d 4 +
       1099511627776 * d 5 + 281474976710656 * d 6 + 72057594037927936 * d 7) * 0x0102040810204080 := by
    have hw_val : w.toNat =
        (d 0 + 256 * d 1 + 65536 * d 2 + 16777216 * d 3 + 4294967296 * d 4 +
         1099511627776 * d 5 + 281474976710656 * d 6 + 72057594037927936 * d 7) * 2 ^ c.val := by
      rw [word_toNat_bytes w, hbyte ⟨0, by decide⟩, hbyte ⟨1, by decide⟩, hbyte ⟨2, by decide⟩,
        hbyte ⟨3, by decide⟩, hbyte ⟨4, by decide⟩, hbyte ⟨5, by decide⟩, hbyte ⟨6, by decide⟩,
        hbyte ⟨7, by decide⟩]
      simp only [Nat.add_mul, Nat.mul_assoc]
    rw [hw_val, Nat.mul_assoc, hcm]
  have hres : (((w * colMult c.val).extractLsb' 56 8)).toNat =
      d 0 + 2 * d 1 + 4 * d 2 + 8 * d 3 + 16 * d 4 + 32 * d 5 + 64 * d 6 + 128 * d 7 := by
    rw [extractLsb'_56_toNat, BitVec.toNat_mul, hD]
    have h0 := hd_le 0; have h1 := hd_le 1; have h2 := hd_le 2; have h3 := hd_le 3
    have h4 := hd_le 4; have h5 := hd_le 5; have h6 := hd_le 6; have h7 := hd_le 7
    let S := d 0 + 2 * d 1 + 4 * d 2 + 8 * d 3 + 16 * d 4 + 32 * d 5 + 64 * d 6 + 128 * d 7
    let lo := d 0 * 0x0002040810204080 + d 1 * 0x0004081020408000 + d 2 * 0x0008102040800000 +
      d 3 * 0x0010204080000000 + d 4 * 0x0020408000000000 + d 5 * 0x0040800000000000 +
      d 6 * 0x0080000000000000
    let Q := d 1 * 0x01 + d 2 * 0x0102 + d 3 * 0x010204 + d 4 * 0x01020408 +
      d 5 * 0x0102040810 + d 6 * 0x010204081020 + d 7 * 0x01020408102040
    have hS_lt : S < 256 := by dsimp only [S]; omega
    have hlo_lt : lo < 2 ^ 56 := by dsimp only [lo]; omega
    have hprod : (d 0 + 256 * d 1 + 65536 * d 2 + 16777216 * d 3 + 4294967296 * d 4 +
        1099511627776 * d 5 + 281474976710656 * d 6 + 72057594037927936 * d 7) * 0x0102040810204080 =
        lo + 2 ^ 56 * S + 2 ^ 64 * Q := by
      have c0 : 0x0102040810204080 = 0x0002040810204080 + 2 ^ 56 * 1 := by decide
      have c1 : 256 * 0x0102040810204080 = 0x0004081020408000 + 2 ^ 56 * 2 + 2 ^ 64 * 0x01 := by decide
      have c2 : 65536 * 0x0102040810204080 = 0x0008102040800000 + 2 ^ 56 * 4 + 2 ^ 64 * 0x0102 := by decide
      have c3 : 16777216 * 0x0102040810204080 = 0x0010204080000000 + 2 ^ 56 * 8 + 2 ^ 64 * 0x010204 := by decide
      have c4 : 4294967296 * 0x0102040810204080 = 0x0020408000000000 + 2 ^ 56 * 16 + 2 ^ 64 * 0x01020408 := by decide
      have c5 : 1099511627776 * 0x0102040810204080 = 0x0040800000000000 + 2 ^ 56 * 32 + 2 ^ 64 * 0x0102040810 := by decide
      have c6 : 281474976710656 * 0x0102040810204080 = 0x0080000000000000 + 2 ^ 56 * 64 + 2 ^ 64 * 0x010204081020 := by decide
      have c7 : 72057594037927936 * 0x0102040810204080 = 2 ^ 56 * 128 + 2 ^ 64 * 0x01020408102040 := by decide
      have term : ∀ (a m r p q : Nat), m * 0x0102040810204080 = r + 2 ^ 56 * p + 2 ^ 64 * q →
          (m * a) * 0x0102040810204080 = a * r + 2 ^ 56 * (p * a) + 2 ^ 64 * (a * q) := by
        intro a m r p q h
        calc (m * a) * 0x0102040810204080
          _ = a * (m * 0x0102040810204080) := by rw [Nat.mul_comm m a, Nat.mul_assoc]
          _ = a * (r + 2 ^ 56 * p + 2 ^ 64 * q) := by rw [h]
          _ = a * r + a * (2 ^ 56 * p) + a * (2 ^ 64 * q) := by simp only [Nat.mul_add]
          _ = a * r + 2 ^ 56 * (p * a) + 2 ^ 64 * (a * q) := by
            rw [Nat.mul_left_comm a (2 ^ 56) p, Nat.mul_comm a p, Nat.mul_left_comm a (2 ^ 64) q]
      have hmul0 : d 0 * 0x0102040810204080 = d 0 * 0x0002040810204080 + 2 ^ 56 * d 0 := by
        calc d 0 * 0x0102040810204080
          _ = d 0 * (0x0002040810204080 + 2 ^ 56 * 1) := by rw [c0]
          _ = d 0 * 0x0002040810204080 + d 0 * (2 ^ 56 * 1) := Nat.mul_add _ _ _
          _ = d 0 * 0x0002040810204080 + 2 ^ 56 * d 0 := by rw [Nat.mul_one, Nat.mul_comm (d 0) (2 ^ 56)]
      have hmul7 : (72057594037927936 * d 7) * 0x0102040810204080 =
          2 ^ 56 * (128 * d 7) + 2 ^ 64 * (d 7 * 0x01020408102040) := by
        calc (72057594037927936 * d 7) * 0x0102040810204080
          _ = d 7 * (72057594037927936 * 0x0102040810204080) := by rw [Nat.mul_comm _ (d 7), Nat.mul_assoc]
          _ = d 7 * (2 ^ 56 * 128 + 2 ^ 64 * 0x01020408102040) := by rw [c7]
          _ = d 7 * (2 ^ 56 * 128) + d 7 * (2 ^ 64 * 0x01020408102040) := Nat.mul_add _ _ _
          _ = 2 ^ 56 * (128 * d 7) + 2 ^ 64 * (d 7 * 0x01020408102040) := by
            rw [Nat.mul_left_comm (d 7) (2 ^ 56) 128, Nat.mul_comm (d 7) 128, Nat.mul_left_comm (d 7) (2 ^ 64)]
      have hcombine : ∀ (r0 s0 r1 s1 q1 r2 s2 q2 r3 s3 q3 r4 s4 q4 r5 s5 q5 r6 s6 q6 s7 q7 : Nat),
          (r0 + s0) + (r1 + s1 + q1) + (r2 + s2 + q2) + (r3 + s3 + q3) + (r4 + s4 + q4) + (r5 + s5 + q5) + (r6 + s6 + q6) + (s7 + q7) =
          (r0 + r1 + r2 + r3 + r4 + r5 + r6) + (s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7) + (q1 + q2 + q3 + q4 + q5 + q6 + q7) := by
        intros; omega
      dsimp only [lo, S, Q]
      simp only [Nat.add_mul, hmul0, term _ _ _ _ _ c1, term _ _ _ _ _ c2, term _ _ _ _ _ c3,
        term _ _ _ _ _ c4, term _ _ _ _ _ c5, term _ _ _ _ _ c6, hmul7]
      rw [hcombine]
      simp only [Nat.mul_add]
    have hlt64 : lo + 2 ^ 56 * S < 2 ^ 64 := by
      have hle : 2 ^ 56 * S ≤ 2 ^ 56 * 255 := Nat.mul_le_mul_left (2 ^ 56) (by omega)
      have hcap : 2 ^ 56 + 2 ^ 56 * 255 = 2 ^ 64 := by decide
      exact Nat.lt_of_lt_of_le (Nat.add_lt_add_of_lt_of_le hlo_lt hle) (Nat.le_of_eq hcap)
    rw [hprod, Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hlt64,
      Nat.add_mul_div_left _ _ (by decide : 0 < 2 ^ 56), Nat.div_eq_of_lt hlo_lt, Nat.zero_add,
      Nat.mod_eq_of_lt hS_lt]
  have hbit : (((w * colMult c.val).extractLsb' 56 8)).getLsbD i =
      ((((w * colMult c.val).extractLsb' 56 8)).toNat).testBit i := rfl
  rw [hbit, hres, Nat.testBit_eq_decide_div_mod_eq]
  have h0 := hd_le 0; have h1 := hd_le 1; have h2 := hd_le 2; have h3 := hd_le 3
  have h4 := hd_le 4; have h5 := hd_le 5; have h6 := hd_le 6; have h7 := hd_le 7
  have hdiv_mod : (d 0 + 2 * d 1 + 4 * d 2 + 8 * d 3 + 16 * d 4 + 32 * d 5 + 64 * d 6 + 128 * d 7) / 2 ^ i % 2 = d i := by
    match i with
    | 0 => change _ / 1 % 2 = d 0; omega
    | 1 => change _ / 2 % 2 = d 1; omega
    | 2 => change _ / 4 % 2 = d 2; omega
    | 3 => change _ / 8 % 2 = d 3; omega
    | 4 => change _ / 16 % 2 = d 4; omega
    | 5 => change _ / 32 % 2 = d 5; omega
    | 6 => change _ / 64 % 2 = d 6; omega
    | 7 => change _ / 128 % 2 = d 7; omega
    | _ + 8 => omega
  rw [hdiv_mod]
  by_cases hp : p.getLsbD (8 * i + c.val) = true
  · simp [d, hp]
  · simp [d, hp]

def findRowBit (mask : Word) (row : Nat) : Option Nat :=
  (List.range 8).find? (fun col => mask.getLsbD (8 * row + col))

def findColBit (mask : Word) (col : Nat) : Option Nat :=
  (List.range 8).find? (fun row => mask.getLsbD (8 * row + col))

def laneCell (s : Nat) (lane : Fin 4) (i : Nat) : Option Nat :=
  let pp := params s
  match lane.val with
  | 0 => if i < 8 then some (8 * i + s % 8) else none
  | 1 => if i < 8 then some (pp.rowShift + i) else none
  | 2 =>
    if i < 8 then
      if pp.addend7 = 0 then
        (findColBit pp.maskD7 i).map (fun r => 8 * r + i)
      else
        (findRowBit pp.maskD7 i).map (fun c => 8 * i + c)
    else none
  | _ =>
    if i < 8 then
      (findColBit pp.maskD9 i).map (fun r => 8 * r + i)
    else none

private def validRowAddBool (mask addend : Word) : Bool :=
  (List.finRange 8).all fun r =>
    match findRowBit mask r.val with
    | none => byteOf mask r.val == 0#8 && decide ((byteOf addend r.val).toNat < 128)
    | some c => decide (c < 8) && (byteOf mask r.val == 1#8 <<< c) &&
        decide ((byteOf mask r.val).toNat + (byteOf addend r.val).toNat = 128)

private theorem nat_div_mod_byte (a b c m : Nat) (hm : 0 < m) (ha : a < m) (hb : b < 256) :
    (a + m * (b + 256 * c)) / m % 256 = b := by
  rw [Nat.add_mul_div_left _ _ hm, Nat.div_eq_of_lt ha, Nat.zero_add,
    Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hb]

private theorem byteAddMask_getLsbD (p mask addend : Word) (hv : validRowAddBool mask addend = true)
    (r : Fin 8) :
    ((p &&& mask) + addend).getLsbD (8 * r.val + 7) =
      match findRowBit mask r.val with
      | some c => p.getLsbD (8 * r.val + c)
      | none => false := by
  let x := p &&& mask
  have hrow : ∀ k : Fin 8,
      match findRowBit mask k.val with
      | none => byteOf mask k.val = 0#8 ∧ (byteOf addend k.val).toNat < 128
      | some c => c < 8 ∧ byteOf mask k.val = 1#8 <<< c ∧ (byteOf mask k.val).toNat + (byteOf addend k.val).toNat = 128 := by
    intro k
    have hk := List.all_eq_true.mp hv k (List.mem_finRange k)
    cases hf : findRowBit mask k.val with
    | none =>
      rw [hf] at hk
      simpa [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] using hk
    | some c =>
      rw [hf] at hk
      simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hk
      exact ⟨hk.1.1, hk.1.2, hk.2⟩
  have hrow_sum : ∀ k : Fin 8, (byteOf x k.val).toNat + (byteOf addend k.val).toNat < 256 := by
    intro k
    have hvk := hrow k
    cases hf : findRowBit mask k.val with
    | none =>
      rw [hf] at hvk
      have hx0 : byteOf x k.val = 0#8 := by simp [x, byteOf_and, hvk.1]
      rw [hx0]; change 0 + (byteOf addend k.val).toNat < 256; omega
    | some c =>
      rw [hf] at hvk
      have hle : (byteOf x k.val).toNat ≤ (byteOf mask k.val).toNat := by
        rw [byteOf_and, BitVec.toNat_and]
        exact Nat.and_le_right
      omega
  let s : Nat → Nat := fun k => (byteOf x k).toNat + (byteOf addend k).toNat
  have hs_lt : ∀ k : Fin 8, s k.val < 256 := hrow_sum
  have h0 : s 0 < 256 := hs_lt ⟨0, by decide⟩; have h1 : s 1 < 256 := hs_lt ⟨1, by decide⟩
  have h2 : s 2 < 256 := hs_lt ⟨2, by decide⟩; have h3 : s 3 < 256 := hs_lt ⟨3, by decide⟩
  have h4 : s 4 < 256 := hs_lt ⟨4, by decide⟩; have h5 : s 5 < 256 := hs_lt ⟨5, by decide⟩
  have h6 : s 6 < 256 := hs_lt ⟨6, by decide⟩; have h7 : s 7 < 256 := hs_lt ⟨7, by decide⟩
  have hadd_toNat : (x + addend).toNat =
      s 0 + 256 * s 1 + 65536 * s 2 + 16777216 * s 3 + 4294967296 * s 4 +
      1099511627776 * s 5 + 281474976710656 * s 6 + 72057594037927936 * s 7 := by
    rw [BitVec.toNat_add, word_toNat_bytes x, word_toNat_bytes addend]
    dsimp only [s] at h0 h1 h2 h3 h4 h5 h6 h7 ⊢
    omega
  have hbr : (byteOf (x + addend) r.val).toNat = s r.val := by
    rw [byteOf_toNat, hadd_toNat]
    rcases r with ⟨rv, hrv⟩
    match rv with
    | 0 =>
      have heq : s 0 + 256 * s 1 + 65536 * s 2 + 16777216 * s 3 + 4294967296 * s 4 +
        1099511627776 * s 5 + 281474976710656 * s 6 + 72057594037927936 * s 7 =
        0 + 1 * (s 0 + 256 * (s 1 + 256 * s 2 + 65536 * s 3 + 16777216 * s 4 + 4294967296 * s 5 + 1099511627776 * s 6 + 281474976710656 * s 7)) := by omega
      rw [heq]; exact nat_div_mod_byte _ _ _ 1 (by decide) (by decide) h0
    | 1 =>
      have heq : s 0 + 256 * s 1 + 65536 * s 2 + 16777216 * s 3 + 4294967296 * s 4 +
        1099511627776 * s 5 + 281474976710656 * s 6 + 72057594037927936 * s 7 =
        s 0 + 256 * (s 1 + 256 * (s 2 + 256 * s 3 + 65536 * s 4 + 16777216 * s 5 + 4294967296 * s 6 + 1099511627776 * s 7)) := by omega
      rw [heq]; exact nat_div_mod_byte _ _ _ 256 (by decide) h0 h1
    | 2 =>
      have heq : s 0 + 256 * s 1 + 65536 * s 2 + 16777216 * s 3 + 4294967296 * s 4 +
        1099511627776 * s 5 + 281474976710656 * s 6 + 72057594037927936 * s 7 =
        (s 0 + 256 * s 1) + 65536 * (s 2 + 256 * (s 3 + 256 * s 4 + 65536 * s 5 + 16777216 * s 6 + 4294967296 * s 7)) := by omega
      rw [heq]; exact nat_div_mod_byte _ _ _ 65536 (by decide) (by omega) h2
    | 3 =>
      have heq : s 0 + 256 * s 1 + 65536 * s 2 + 16777216 * s 3 + 4294967296 * s 4 +
        1099511627776 * s 5 + 281474976710656 * s 6 + 72057594037927936 * s 7 =
        (s 0 + 256 * s 1 + 65536 * s 2) + 16777216 * (s 3 + 256 * (s 4 + 256 * s 5 + 65536 * s 6 + 16777216 * s 7)) := by omega
      rw [heq]; exact nat_div_mod_byte _ _ _ 16777216 (by decide) (by omega) h3
    | 4 =>
      have heq : s 0 + 256 * s 1 + 65536 * s 2 + 16777216 * s 3 + 4294967296 * s 4 +
        1099511627776 * s 5 + 281474976710656 * s 6 + 72057594037927936 * s 7 =
        (s 0 + 256 * s 1 + 65536 * s 2 + 16777216 * s 3) + 4294967296 * (s 4 + 256 * (s 5 + 256 * s 6 + 65536 * s 7)) := by omega
      rw [heq]; exact nat_div_mod_byte _ _ _ 4294967296 (by decide) (by omega) h4
    | 5 =>
      have heq : s 0 + 256 * s 1 + 65536 * s 2 + 16777216 * s 3 + 4294967296 * s 4 +
        1099511627776 * s 5 + 281474976710656 * s 6 + 72057594037927936 * s 7 =
        (s 0 + 256 * s 1 + 65536 * s 2 + 16777216 * s 3 + 4294967296 * s 4) + 1099511627776 * (s 5 + 256 * (s 6 + 256 * s 7)) := by omega
      rw [heq]; exact nat_div_mod_byte _ _ _ 1099511627776 (by decide) (by omega) h5
    | 6 =>
      have heq : s 0 + 256 * s 1 + 65536 * s 2 + 16777216 * s 3 + 4294967296 * s 4 +
        1099511627776 * s 5 + 281474976710656 * s 6 + 72057594037927936 * s 7 =
        (s 0 + 256 * s 1 + 65536 * s 2 + 16777216 * s 3 + 4294967296 * s 4 + 1099511627776 * s 5) + 281474976710656 * (s 6 + 256 * s 7) := by omega
      rw [heq]; exact nat_div_mod_byte _ _ _ 281474976710656 (by decide) (by omega) h6
    | 7 =>
      have heq : s 0 + 256 * s 1 + 65536 * s 2 + 16777216 * s 3 + 4294967296 * s 4 +
        1099511627776 * s 5 + 281474976710656 * s 6 + 72057594037927936 * s 7 =
        (s 0 + 256 * s 1 + 65536 * s 2 + 16777216 * s 3 + 4294967296 * s 4 + 1099511627776 * s 5 + 281474976710656 * s 6) + 72057594037927936 * (s 7 + 256 * 0) := by omega
      rw [heq]; exact nat_div_mod_byte _ _ _ 72057594037927936 (by decide) (by omega) h7
    | _ + 8 => omega
  have hbit7 : (x + addend).getLsbD (8 * r.val + 7) = decide (128 ≤ (byteOf (x + addend) r.val).toNat) := by
    rw [← byteOf_getLsbD (x + addend) r.val 7 (by decide)]
    change (byteOf (x + addend) r.val).toNat.testBit 7 = _
    have hlt := (byteOf (x + addend) r.val).isLt
    rw [Nat.testBit_eq_decide_div_mod_eq]
    have : (byteOf (x + addend) r.val).toNat / 2 ^ 7 % 2 = 1 ↔ 128 ≤ (byteOf (x + addend) r.val).toNat := by omega
    simp [this]
  rw [hbit7, hbr]
  have hvr := hrow r
  cases hf : findRowBit mask r.val with
  | none =>
    rw [hf] at hvr
    have hx0 : byteOf x r.val = 0#8 := by simp [x, byteOf_and, hvr.1]
    have hlt128 : s r.val < 128 := by dsimp only [s]; rw [hx0]; change 0 + (byteOf addend r.val).toNat < 128; omega
    simp [show ¬ 128 ≤ s r.val by omega]
  | some c =>
    rw [hf] at hvr
    obtain ⟨hc, hm, hsum⟩ := hvr
    have hbx : byteOf x r.val = byteOf p r.val &&& (1#8 <<< c) := by
      rw [byteOf_and, hm]
    have hbp : (byteOf p r.val).getLsbD c = p.getLsbD (8 * r.val + c) :=
      byteOf_getLsbD p r.val c hc
    have hpow : (1#8 <<< c).toNat = 2 ^ c := by
      have : ∀ col : Fin 8, (1#8 <<< col.val).toNat = 2 ^ col.val := by decide +kernel
      exact this ⟨c, hc⟩
    cases hp : p.getLsbD (8 * r.val + c) with
    | true =>
      have heq : byteOf x r.val = 1#8 <<< c := by
        rw [hbx]
        apply BitVec.eq_of_getLsbD_eq
        intro j hj
        rw [BitVec.getLsbD_and, BitVec.getLsbD_shiftLeft]
        simp only [hj, decide_true, Bool.true_and]
        by_cases hjc : j = c
        · subst hjc; rw [hbp, hp]; simp
        · by_cases hlt : j < c
          · simp [hlt]
          · have hne : j - c ≠ 0 := by omega
            simp [hlt, BitVec.getLsbD_one, hne]
      have hge128 : 128 ≤ s r.val := by dsimp only [s]; rw [heq, ← hm, hsum]; exact Nat.le_refl 128
      simp [hge128, hp]
    | false =>
      have heq : byteOf x r.val = 0#8 := by
        rw [hbx]
        apply BitVec.eq_of_getLsbD_eq
        intro j hj
        rw [BitVec.getLsbD_and, BitVec.getLsbD_shiftLeft, BitVec.getLsbD_zero]
        simp only [hj, decide_true, Bool.true_and]
        by_cases hjc : j = c
        · subst hjc; rw [hbp, hp]; simp
        · by_cases hlt : j < c
          · simp [hlt]
          · have hne : j - c ≠ 0 := by omega
            simp [hlt, BitVec.getLsbD_one, hne]
      have hpos : 0 < (byteOf mask r.val).toNat := by rw [hm, hpow]; exact Nat.two_pow_pos c
      have hlt128 : s r.val < 128 := by dsimp only [s]; rw [heq]; change 0 + (byteOf addend r.val).toNat < 128; omega
      simp [show ¬ 128 ≤ s r.val by omega, hp]

private def bytesDisjointBool (mask : Word) : Bool :=
  (List.finRange 8).all fun j =>
    (List.finRange 8).all fun k =>
      !decide (j.val < k.val) || (byteOf mask j.val &&& byteOf mask k.val == 0#8)

private theorem bytesDisjoint_of_bool (mask : Word) (h : bytesDisjointBool mask = true) :
    ∀ j k : Fin 8, j.val < k.val → byteOf mask j.val &&& byteOf mask k.val = 0 := by
  intro j k hlt
  have h1 := List.all_eq_true.mp (List.all_eq_true.mp h j (List.mem_finRange j)) k (List.mem_finRange k)
  simp only [Bool.or_eq_true, Bool.not_eq_true', decide_eq_false_iff_not, beq_iff_eq] at h1
  exact h1.resolve_left (not_not_intro hlt)

private def validParamsBool (s : Fin 64) : Bool :=
  let pp := params s.val
  (pp.maskV == 0x0101010101010101#64 <<< (s.val % 8)) &&
  (pp.multV == colMult (s.val % 8)) &&
  decide (pp.rowShift ≤ 56) &&
  (if pp.addend7 == 0 then
    (pp.postMask7 == 0xffffffffffffffff#64) &&
    (pp.multD7 == (0x0101010101010101 : Word)) &&
    bytesDisjointBool pp.maskD7
  else
    (pp.postMask7 == 0x0101010101010101#64 <<< 7) &&
    (pp.multD7 == colMult 7) &&
    validRowAddBool pp.maskD7 pp.addend7) &&
  bytesDisjointBool pp.maskD9

private theorem validParams_all : ∀ s : Fin 64, validParamsBool s = true := by decide +kernel

private theorem colOr_getLsbD (p mask : Word) (i : Nat) (hi : i < 8)
    (hdisj : ∀ j k : Fin 8, j.val < k.val → byteOf mask j.val &&& byteOf mask k.val = 0) :
    (byteOf (p &&& mask) 0 ||| byteOf (p &&& mask) 1 ||| byteOf (p &&& mask) 2 |||
     byteOf (p &&& mask) 3 ||| byteOf (p &&& mask) 4 ||| byteOf (p &&& mask) 5 |||
     byteOf (p &&& mask) 6 ||| byteOf (p &&& mask) 7).getLsbD i =
      match findColBit mask i with
      | some r => p.getLsbD (8 * r + i)
      | none => false := by
  have hunique : ∀ j k : Fin 8, j.val ≠ k.val →
      mask.getLsbD (8 * j.val + i) = true → mask.getLsbD (8 * k.val + i) = false := by
    intro j k hne hj
    have h0 : (byteOf mask j.val &&& byteOf mask k.val).getLsbD i = false := by
      rcases Nat.lt_or_gt_of_ne hne with hlt | hgt
      · rw [hdisj j k hlt]; simp
      · rw [BitVec.and_comm, hdisj k j hgt]; simp
    simp only [BitVec.getLsbD_and, byteOf_getLsbD _ _ i hi, hj, Bool.true_and] at h0
    exact h0
  simp only [BitVec.getLsbD_or, byteOf_getLsbD _ _ i hi, BitVec.getLsbD_and]
  cases hf : findColBit mask i with
  | none =>
    have hall : ∀ r, r < 8 → mask.getLsbD (8 * r + i) = false := by
      intro r hr
      have := List.find?_eq_none.mp hf r (List.mem_range.mpr hr)
      exact Bool.eq_false_iff.mpr this
    have h0 : mask.getLsbD i = false := by simpa using hall 0 (by decide)
    simp [h0, hall 1 (by decide), hall 2 (by decide), hall 3 (by decide),
      hall 4 (by decide), hall 5 (by decide), hall 6 (by decide), hall 7 (by decide)]
  | some r =>
    have hmem : r ∈ List.range 8 := List.mem_of_find?_eq_some hf
    have hr : r < 8 := List.mem_range.mp hmem
    have htrue : mask.getLsbD (8 * r + i) = true :=
      List.find?_some (p := fun row => mask.getLsbD (8 * row + i)) hf
    have hother : ∀ k : Nat, k < 8 → k ≠ r → mask.getLsbD (8 * k + i) = false :=
      fun k hk hne => hunique ⟨r, hr⟩ ⟨k, hk⟩ (Ne.symm hne) htrue
    have hother0 : 0 ≠ r → mask.getLsbD i = false := fun hne => by simpa using hother 0 (by decide) hne
    match r with
    | 0 => simp [show mask.getLsbD i = true by simpa using htrue, hother 1 (by decide) (by decide), hother 2 (by decide) (by decide), hother 3 (by decide) (by decide), hother 4 (by decide) (by decide), hother 5 (by decide) (by decide), hother 6 (by decide) (by decide), hother 7 (by decide) (by decide)]
    | 1 => simp [htrue, hother0 (by decide), hother 2 (by decide) (by decide), hother 3 (by decide) (by decide), hother 4 (by decide) (by decide), hother 5 (by decide) (by decide), hother 6 (by decide) (by decide), hother 7 (by decide) (by decide)]
    | 2 => simp [htrue, hother0 (by decide), hother 1 (by decide) (by decide), hother 3 (by decide) (by decide), hother 4 (by decide) (by decide), hother 5 (by decide) (by decide), hother 6 (by decide) (by decide), hother 7 (by decide) (by decide)]
    | 3 => simp [htrue, hother0 (by decide), hother 1 (by decide) (by decide), hother 2 (by decide) (by decide), hother 4 (by decide) (by decide), hother 5 (by decide) (by decide), hother 6 (by decide) (by decide), hother 7 (by decide) (by decide)]
    | 4 => simp [htrue, hother0 (by decide), hother 1 (by decide) (by decide), hother 2 (by decide) (by decide), hother 3 (by decide) (by decide), hother 5 (by decide) (by decide), hother 6 (by decide) (by decide), hother 7 (by decide) (by decide)]
    | 5 => simp [htrue, hother0 (by decide), hother 1 (by decide) (by decide), hother 2 (by decide) (by decide), hother 3 (by decide) (by decide), hother 4 (by decide) (by decide), hother 6 (by decide) (by decide), hother 7 (by decide) (by decide)]
    | 6 => simp [htrue, hother0 (by decide), hother 1 (by decide) (by decide), hother 2 (by decide) (by decide), hother 3 (by decide) (by decide), hother 4 (by decide) (by decide), hother 5 (by decide) (by decide), hother 7 (by decide) (by decide)]
    | 7 => simp [htrue, hother0 (by decide), hother 1 (by decide) (by decide), hother 2 (by decide) (by decide), hother 3 (by decide) (by decide), hother 4 (by decide) (by decide), hother 5 (by decide) (by decide), hother 6 (by decide) (by decide)]
    | _ + 8 => omega

theorem laneIndex_getLsbD (s : Fin 64) (lane : Fin 4) (p : Word) (i : Nat) (hi : i < 8) :
    (laneIndex s.val lane p).getLsbD i =
      match laneCell s.val lane i with
      | some sq => p.getLsbD sq
      | none => false := by
  have hv := validParams_all s
  simp only [validParamsBool, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hv
  obtain ⟨⟨⟨⟨hmV, hmulV⟩, _⟩, hd7⟩, hd9b⟩ := hv
  have hd9 := bytesDisjoint_of_bool _ hd9b
  rcases lane with ⟨lv, hlv⟩
  match lv with
  | 0 =>
    change (((p &&& (params s.val).maskV) * (params s.val).multV).extractLsb' 56 8).getLsbD i =
      match if i < 8 then some (8 * i + s.val % 8) else none with
      | some sq => p.getLsbD sq | none => false
    simp only [hi, ↓reduceIte, hmV, hmulV]
    exact packColumn_getLsbD p ⟨s.val % 8, Nat.mod_lt _ (by decide)⟩ i hi
  | 1 =>
    change ((p >>> (params s.val).rowShift).setWidth 8).getLsbD i =
      match if i < 8 then some ((params s.val).rowShift + i) else none with
      | some sq => p.getLsbD sq | none => false
    simp only [hi, ↓reduceIte, BitVec.getLsbD_setWidth, BitVec.getLsbD_ushiftRight,
      decide_true, Bool.true_and]
  | 2 =>
    change (((((p &&& (params s.val).maskD7) + (params s.val).addend7) &&& (params s.val).postMask7) *
        (params s.val).multD7).extractLsb' 56 8).getLsbD i =
      match if i < 8 then
        if (params s.val).addend7 = 0 then (findColBit (params s.val).maskD7 i).map (fun r => 8 * r + i)
        else (findRowBit (params s.val).maskD7 i).map (fun c => 8 * i + c)
      else none with
      | some sq => p.getLsbD sq | none => false
    simp only [hi, ↓reduceIte]
    by_cases h0 : (params s.val).addend7 = 0
    · rw [ite_eq_left h0] at hd7
      simp only [Bool.and_eq_true, beq_iff_eq] at hd7
      rw [h0]
      simp only [↓reduceIte]
      obtain ⟨⟨hpm, hmul⟩, hdisjb⟩ := hd7
      have hdisj := bytesDisjoint_of_bool _ hdisjb
      rw [show (p &&& (params s.val).maskD7) + (0 : Word) = p &&& (params s.val).maskD7 from BitVec.add_zero _,
        hpm, show (0xffffffffffffffff#64) = BitVec.allOnes 64 from rfl,
        BitVec.and_allOnes, hmul, sumDisjointBytes_eq p (params s.val).maskD7 hdisj,
        colOr_getLsbD p (params s.val).maskD7 i hi hdisj]
      cases findColBit (params s.val).maskD7 i <;> rfl
    · rw [ite_eq_right h0] at hd7
      simp only [Bool.and_eq_true, beq_iff_eq] at hd7
      rw [ite_eq_right h0]
      obtain ⟨⟨hpm, hmul⟩, hva⟩ := hd7
      rw [hpm, hmul, packColumn_getLsbD ((p &&& (params s.val).maskD7) + (params s.val).addend7)
        ⟨7, by decide⟩ i hi, byteAddMask_getLsbD p (params s.val).maskD7 (params s.val).addend7 hva ⟨i, hi⟩]
      cases findRowBit (params s.val).maskD7 i <;> rfl
  | 3 =>
    change (((p &&& (params s.val).maskD9) * (0x0101010101010101 : Word)).extractLsb' 56 8).getLsbD i =
      match if i < 8 then (findColBit (params s.val).maskD9 i).map (fun r => 8 * r + i) else none with
      | some sq => p.getLsbD sq | none => false
    simp only [hi, ↓reduceIte]
    rw [sumDisjointBytes_eq p (params s.val).maskD9 hd9,
      colOr_getLsbD p (params s.val).maskD9 i hi hd9]
    cases findColBit (params s.val).maskD9 i <;> rfl
  | _ + 4 => omega

end NeuralReversi.Solve1.Scalar
