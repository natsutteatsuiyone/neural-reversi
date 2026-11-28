module
public import NeuralReversi.Solve1.ScalarModel

@[expose] public section

namespace NeuralReversi.Solve1.Scalar

private theorem lineTable_ray_zero_of_all_none (p : Word) (line : Byte) (cellOf : Nat → Option Nat)
    (hbit : ∀ i, i < 8 → line.getLsbD i = match cellOf i with | some sq => p.getLsbD sq | none => false)
    (is : List Nat) (hi : ∀ i ∈ is, i < 8) (hnone : ∀ i ∈ is, cellOf i = none) (acc : Nat) :
    LineTable.ray line acc is = 0 := by
  induction is generalizing acc with
  | nil => rfl
  | cons i rest ih =>
    have hi8 : i < 8 := hi i (by simp)
    have hni : cellOf i = none := hnone i (by simp)
    have hli : line.getLsbD i = false := by rw [hbit i hi8, hni]
    simp only [LineTable.ray, hli, Bool.false_eq_true, ↓reduceIte]
    exact ih (fun j hj => hi j (List.mem_cons_of_mem _ hj))
      (fun j hj => hnone j (List.mem_cons_of_mem _ hj)) (acc + 1)

private theorem lineTable_ray_eq_ray_padded (p : Word) (line : Byte) (cellOf : Nat → Option Nat)
    (hbit : ∀ i, i < 8 → line.getLsbD i = match cellOf i with | some sq => p.getLsbD sq | none => false)
    (is : List Nat) (hi : ∀ i ∈ is, i < 8) (cs : List Nat)
    (hmatch : is.map cellOf = cs.map some ++ List.replicate (is.length - cs.length) none)
    (hlen : cs.length ≤ is.length) (acc : Nat) :
    LineTable.ray line acc is = Geometry.ray p acc cs := by
  induction is generalizing cs acc with
  | nil =>
    cases cs with
    | nil => rfl
    | cons _ _ => simp at hlen
  | cons i rest ih =>
    have hi8 : i < 8 := hi i (by simp)
    have hrest : ∀ j ∈ rest, j < 8 := fun j hj => hi j (List.mem_cons_of_mem _ hj)
    cases cs with
    | nil =>
      simp only [List.map_cons, List.map_nil, List.nil_append, List.length_cons, List.length_nil,
        Nat.sub_zero, List.replicate_succ, List.cons.injEq] at hmatch
      have hli : line.getLsbD i = false := by rw [hbit i hi8, hmatch.1]
      have hnone : ∀ j ∈ rest, cellOf j = none := by
        intro j hj
        have hm : cellOf j ∈ rest.map cellOf := List.mem_map_of_mem hj
        rw [hmatch.2] at hm
        exact List.eq_of_mem_replicate hm
      simp only [LineTable.ray, hli, Bool.false_eq_true, ↓reduceIte, Geometry.ray]
      exact lineTable_ray_zero_of_all_none p line cellOf hbit rest hrest hnone (acc + 1)
    | cons c cs' =>
      simp only [List.map_cons, List.cons_append, List.length_cons, Nat.add_sub_add_right,
        List.cons.injEq] at hmatch hlen
      have hli : line.getLsbD i = p.getLsbD c := by rw [hbit i hi8, hmatch.1]
      simp only [LineTable.ray, Geometry.ray, hli]
      split
      · rfl
      · exact ih hrest cs' hmatch.2 (by omega) (acc + 1)

private theorem lineTable_ray_zero_of_len_le1 (line : Byte) (is : List Nat) (h : is.length ≤ 1) :
    LineTable.ray line 0 is = 0 := by
  match is with
  | [] => rfl
  | [i] => simp [LineTable.ray]
  | _ :: _ :: _ => simp at h

private theorem geom_ray_zero_of_len_le1 (p : Word) (cs : List Nat) (h : cs.length ≤ 1) :
    Geometry.ray p 0 cs = 0 := by
  match cs with
  | [] => rfl
  | [c] => simp [Geometry.ray]
  | _ :: _ :: _ => simp at h

private def rayMatches (cellOf : Nat → Option Nat) (is cs : List Nat) : Bool :=
  if cs.length ≤ 1 then
    is.length ≤ 1 || is.all (fun i => (cellOf i).isNone)
  else
    decide (cs.length ≤ is.length) && (is.map cellOf == cs.map some ++ List.replicate (is.length - cs.length) none)

private theorem rayMatches_sound (p : Word) (line : Byte) (cellOf : Nat → Option Nat)
    (hbit : ∀ i, i < 8 → line.getLsbD i = match cellOf i with | some sq => p.getLsbD sq | none => false)
    (is cs : List Nat) (hi : ∀ i ∈ is, i < 8) (hm : rayMatches cellOf is cs = true) :
    LineTable.ray line 0 is = Geometry.ray p 0 cs := by
  unfold rayMatches at hm
  split at hm
  · rename_i hle
    rw [geom_ray_zero_of_len_le1 p cs hle]
    simp only [Bool.or_eq_true, decide_eq_true_eq, List.all_eq_true, Option.isNone_iff_eq_none] at hm
    rcases hm with hlen | hnone
    · exact lineTable_ray_zero_of_len_le1 line is hlen
    · exact lineTable_ray_zero_of_all_none p line cellOf hbit is hi hnone 0
  · simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hm
    exact lineTable_ray_eq_ray_padded p line cellOf hbit is hi cs hm.2 hm.1 0

private def laneRays (s : Nat) (lane : Fin 4) : List (List Nat) :=
  match lane.val with
  | 0 => [Geometry.raySquares s 1 0, Geometry.raySquares s (-1) 0]
  | 1 => [Geometry.raySquares s 0 1, Geometry.raySquares s 0 (-1)]
  | 2 =>
    if (params s).maskD9 = 0 then
      [Geometry.raySquares s 1 1, Geometry.raySquares s (-1) (-1),
       Geometry.raySquares s 1 (-1), Geometry.raySquares s (-1) 1]
    else
      [Geometry.raySquares s 1 (-1), Geometry.raySquares s (-1) 1]
  | _ =>
    if (params s).maskD9 = 0 then []
    else [Geometry.raySquares s 1 1, Geometry.raySquares s (-1) (-1)]

private theorem laneReference_eq_sum (s : Nat) (lane : Fin 4) (p : Word) :
    laneReference s lane p = ((laneRays s lane).map (Geometry.ray p 0)).sum := by
  have hl : lane = 0 ∨ lane = 1 ∨ lane = 2 ∨ lane = 3 := by omega
  rcases hl with rfl | rfl | rfl | rfl
  · simp [laneReference, laneRays]
  · simp [laneReference, laneRays]
  · by_cases h0 : (params s).maskD9 = 0#64 <;> simp [laneReference, laneRays, h0, Nat.add_assoc]
  · by_cases h0 : (params s).maskD9 = 0#64 <;> simp [laneReference, laneRays, h0]

private def activeRays (rs : List (List Nat)) : List (List Nat) :=
  rs.filter (fun cs => decide (1 < cs.length))

private theorem sum_rays_eq_active (p : Word) (rs : List (List Nat)) :
    (rs.map (Geometry.ray p 0)).sum = ((activeRays rs).map (Geometry.ray p 0)).sum := by
  induction rs with
  | nil => rfl
  | cons cs rest ih =>
    unfold activeRays
    rw [List.filter_cons]
    by_cases h : 1 < cs.length
    · simp only [h, decide_true, ↓reduceIte, List.map_cons, List.sum_cons]
      change Geometry.ray p 0 cs + (rest.map (Geometry.ray p 0)).sum =
        Geometry.ray p 0 cs + ((activeRays rest).map (Geometry.ray p 0)).sum
      rw [ih]
    · simp only [h, decide_false, Bool.false_eq_true, ↓reduceIte, List.map_cons, List.sum_cons]
      rw [geom_ray_zero_of_len_le1 p cs (by omega), Nat.zero_add]
      exact ih

private def validLane (s : Fin 64) (lane : Fin 4) : Bool :=
  let pos := lanePos s.val lane
  let hiIdx := (List.range 8).filter (fun i => pos.val < i)
  let loIdx := (List.range pos.val).reverse
  let cellOf := laneCell s.val lane
  let active := activeRays (laneRays s.val lane)
  match active with
  | [] => rayMatches cellOf hiIdx [] && rayMatches cellOf loIdx []
  | [r0] => (rayMatches cellOf hiIdx r0 && rayMatches cellOf loIdx []) ||
            (rayMatches cellOf hiIdx [] && rayMatches cellOf loIdx r0)
  | [r0, r1] => (rayMatches cellOf hiIdx r0 && rayMatches cellOf loIdx r1) ||
                (rayMatches cellOf hiIdx r1 && rayMatches cellOf loIdx r0)
  | _ => false

private theorem validLane_all : ∀ (s : Fin 64) (lane : Fin 4), validLane s lane = true := by
  decide +kernel

theorem lane_correct (s : Fin 64) (lane : Fin 4) (p : Word) :
    LineTable.reference (lanePos s.val lane) (laneIndex s.val lane p) =
      laneReference s.val lane p := by
  rw [laneReference_eq_sum, sum_rays_eq_active]
  unfold LineTable.reference
  let pos := lanePos s.val lane
  let hiIdx := (List.range 8).filter (fun i => pos.val < i)
  let loIdx := (List.range pos.val).reverse
  have hhi8 : ∀ i ∈ hiIdx, i < 8 := fun i hi => by
    simp only [hiIdx, List.mem_filter, List.mem_range] at hi; exact hi.1
  have hlo8 : ∀ i ∈ loIdx, i < 8 := fun i hi => by
    simp only [loIdx, List.mem_reverse, List.mem_range] at hi; have := pos.isLt; omega
  have hbit : ∀ i, i < 8 → (laneIndex s.val lane p).getLsbD i =
      match laneCell s.val lane i with | some sq => p.getLsbD sq | none => false :=
    laneIndex_getLsbD s lane p
  have hv := validLane_all s lane
  unfold validLane at hv
  change (match activeRays (laneRays s.val lane) with
    | [] => rayMatches (laneCell s.val lane) hiIdx [] && rayMatches (laneCell s.val lane) loIdx []
    | [r0] => (rayMatches (laneCell s.val lane) hiIdx r0 && rayMatches (laneCell s.val lane) loIdx []) ||
              (rayMatches (laneCell s.val lane) hiIdx [] && rayMatches (laneCell s.val lane) loIdx r0)
    | [r0, r1] => (rayMatches (laneCell s.val lane) hiIdx r0 && rayMatches (laneCell s.val lane) loIdx r1) ||
                  (rayMatches (laneCell s.val lane) hiIdx r1 && rayMatches (laneCell s.val lane) loIdx r0)
    | _ => false) = true at hv
  cases ha : activeRays (laneRays s.val lane) with
  | nil =>
    rw [ha] at hv
    simp only [Bool.and_eq_true] at hv
    have e1 := rayMatches_sound p _ _ hbit hiIdx [] hhi8 hv.1
    have e2 := rayMatches_sound p _ _ hbit loIdx [] hlo8 hv.2
    simp [hiIdx, loIdx, pos, Geometry.ray] at e1 e2 ⊢
    omega
  | cons r0 rest =>
    cases rest with
    | nil =>
      rw [ha] at hv
      simp only [Bool.or_eq_true, Bool.and_eq_true] at hv
      rcases hv with ⟨h1, h2⟩ | ⟨h1, h2⟩
      · have e1 := rayMatches_sound p _ _ hbit hiIdx r0 hhi8 h1
        have e2 := rayMatches_sound p _ _ hbit loIdx [] hlo8 h2
        simp [hiIdx, loIdx, pos, Geometry.ray] at e1 e2 ⊢
        omega
      · have e1 := rayMatches_sound p _ _ hbit hiIdx [] hhi8 h1
        have e2 := rayMatches_sound p _ _ hbit loIdx r0 hlo8 h2
        simp [hiIdx, loIdx, pos, Geometry.ray] at e1 e2 ⊢
        omega
    | cons r1 rest2 =>
      cases rest2 with
      | nil =>
        rw [ha] at hv
        simp only [Bool.or_eq_true, Bool.and_eq_true] at hv
        rcases hv with ⟨h1, h2⟩ | ⟨h1, h2⟩
        · have e1 := rayMatches_sound p _ _ hbit hiIdx r0 hhi8 h1
          have e2 := rayMatches_sound p _ _ hbit loIdx r1 hlo8 h2
          simp [hiIdx, loIdx, pos] at e1 e2 ⊢
          omega
        · have e1 := rayMatches_sound p _ _ hbit hiIdx r1 hhi8 h1
          have e2 := rayMatches_sound p _ _ hbit loIdx r0 hlo8 h2
          simp [hiIdx, loIdx, pos] at e1 e2 ⊢
          omega
      | cons _ _ =>
        rw [ha] at hv
        simp at hv

def laneCount (s : Nat) (lane : Fin 4) (p : Word) : BitVec 32 :=
  (LineTable.countFlip (lanePos s lane) (laneIndex s lane p)).setWidth 32

def count (s : Nat) (p : Word) : BitVec 32 :=
  laneCount s 0 p + laneCount s 1 p + laneCount s 2 p +
    (if (params s).maskD9 = 0 then 0 else laneCount s 3 p)

theorem laneCount_bound (s : Nat) (lane : Fin 4) (p : Word) :
    (laneCount s lane p).toNat ≤ 12 := by
  simpa [laneCount, BitVec.toNat_setWidth, Nat.mod_eq_of_lt
    (Nat.lt_of_le_of_lt (LineTable.countFlip_bound (lanePos s lane) (laneIndex s lane p))
      (by decide : 12 < 2^32))] using
    LineTable.countFlip_bound (lanePos s lane) (laneIndex s lane p)

theorem laneCount_toNat (s : Fin 64) (lane : Fin 4) (p : Word) :
    (laneCount s.val lane p).toNat = 2 * laneReference s.val lane p := by
  simp only [laneCount, BitVec.toNat_setWidth, LineTable.countFlip_correct,
    lane_correct]
  have h := LineTable.reference_bound (lanePos s.val lane) (laneIndex s.val lane p)
  rw [lane_correct] at h
  omega

theorem laneReference_sum (s : Nat) (p : Word) :
    laneReference s 0 p + laneReference s 1 p + laneReference s 2 p +
      laneReference s 3 p = Geometry.reference p s := by
  by_cases h : (params s).maskD9 = 0#64
  · simp [laneReference, Geometry.reference, h] <;> omega
  · simp [laneReference, Geometry.reference, h] <;> omega

theorem count_toNat (s : Fin 64) (p : Word) :
    (count s.val p).toNat = 2 * Geometry.reference p s.val := by
  have h0 := laneCount_bound s.val 0 p
  have h1 := laneCount_bound s.val 1 p
  have h2 := laneCount_bound s.val 2 p
  have h3 := laneCount_bound s.val 3 p
  have e0 := laneCount_toNat s 0 p
  have e1 := laneCount_toNat s 1 p
  have e2 := laneCount_toNat s 2 p
  have e3 := laneCount_toNat s 3 p
  have hs := laneReference_sum s.val p
  have ab : (laneCount s.val 0 p + laneCount s.val 1 p).toNat =
      (laneCount s.val 0 p).toNat + (laneCount s.val 1 p).toNat := by
    rw [BitVec.toNat_add]
    exact Nat.mod_eq_of_lt (by omega)
  have abc : (laneCount s.val 0 p + laneCount s.val 1 p + laneCount s.val 2 p).toNat =
      (laneCount s.val 0 p).toNat + (laneCount s.val 1 p).toNat + (laneCount s.val 2 p).toNat := by
    rw [BitVec.toNat_add, ab]
    exact Nat.mod_eq_of_lt (by omega)
  unfold count
  split
  · have hz : laneReference s.val 3 p = 0 := by simp [laneReference, *]
    change (laneCount s.val 0 p + laneCount s.val 1 p + laneCount s.val 2 p + 0#32).toNat = _
    rw [BitVec.add_zero, abc]
    omega
  · rw [BitVec.toNat_add, abc, Nat.mod_eq_of_lt (by omega)]
    omega

end NeuralReversi.Solve1.Scalar
