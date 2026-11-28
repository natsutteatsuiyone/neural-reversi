module
public import NeuralReversi.Solve1.Bmi2Model

@[expose] public section

namespace NeuralReversi.Solve1.Bmi2

private theorem pext_getLsbD (p : Word) :
    ∀ (cells : List Nat) (i : Nat), i < 64 →
      (pext p cells).getLsbD i =
        match cells[i]? with
        | some c => p.getLsbD c
        | none => false
  | [], _, _ => by simp [pext]
  | c :: rest, 0, _ => by
    simp only [pext, BitVec.getLsbD_or, BitVec.getLsbD_shiftLeft, List.getElem?_cons_zero]
    cases p.getLsbD c <;> simp
  | c :: rest, i + 1, hi => by
    simp only [pext, BitVec.getLsbD_or, BitVec.getLsbD_shiftLeft, List.getElem?_cons_succ]
    have hlo : (if p.getLsbD c = true then (1 : Word) else 0).getLsbD (i + 1) = false := by
      cases p.getLsbD c <;> simp [BitVec.getLsbD_one]
    have hgt : (1 ≤ i + 1) := Nat.succ_le_succ (Nat.zero_le i)
    simp only [hlo, Bool.false_or, hi, decide_true, Bool.true_and, Nat.not_lt.mpr hgt,
      decide_false, Bool.not_false, Nat.add_sub_cancel]
    exact pext_getLsbD p rest i (by omega)

private def lookupDst (map : List (Nat × Nat)) (dst : Nat) : Option Nat :=
  (map.find? (fun pair => pair.2 == dst)).map Prod.fst

private def mapDstUnique : List (Nat × Nat) → Bool
  | [] => true
  | (_, d) :: rest => rest.all (fun p => p.2 != d) && mapDstUnique rest

private theorem bit_ite_shift_false (b : Bool) (d i : Nat) (hi : i < 8) (hne : d ≠ i) :
    (if b = true then ((1 : Byte) <<< d) else 0).getLsbD i = false := by
  cases b
  · simp
  · simp only [↓reduceIte, BitVec.getLsbD_shiftLeft, hi, decide_true, Bool.true_and]
    by_cases hlt : i < d
    · simp [hlt]
    · have hsub : i - d ≠ 0 := by omega
      simp [hlt, BitVec.getLsbD_one, hsub]

private theorem projectIndex_getLsbD_none (idx : Index) :
    ∀ (map : List (Nat × Nat)) (i : Nat), i < 8 →
      map.all (fun pair => pair.2 != i) = true →
      (projectIndex idx map).getLsbD i = false
  | [], _, _, _ => by simp [projectIndex]
  | (s, d) :: rest, i, hi, hall => by
    simp only [List.all_cons, Bool.and_eq_true, bne_iff_ne] at hall
    simp only [projectIndex, BitVec.getLsbD_or, bit_ite_shift_false _ d i hi hall.1, Bool.false_or]
    exact projectIndex_getLsbD_none idx rest i hi hall.2

private theorem projectIndex_getLsbD (idx : Index) :
    ∀ (map : List (Nat × Nat)) (i : Nat), i < 8 →
      mapDstUnique map = true →
      (projectIndex idx map).getLsbD i =
        match lookupDst map i with
        | some s => idx.getLsbD s
        | none => false
  | [], _, _, _ => by simp [projectIndex, lookupDst]
  | (s, d) :: rest, i, hi, huniq => by
    simp only [mapDstUnique, Bool.and_eq_true] at huniq
    by_cases hdi : d = i
    · subst hdi
      have hrest := projectIndex_getLsbD_none idx rest d hi huniq.1
      have hself : (if idx.getLsbD s = true then ((1 : Byte) <<< d) else 0).getLsbD d = idx.getLsbD s := by
        cases idx.getLsbD s
        · simp
        · simp [hi]
      have hfind : lookupDst ((s, d) :: rest) d = some s := by
        simp [lookupDst]
      rw [projectIndex, BitVec.getLsbD_or, hself, hrest, Bool.or_false, hfind]
    · have hne : (d == i) = false := beq_false_of_ne hdi
      have hfind : lookupDst ((s, d) :: rest) i = lookupDst rest i := by
        simp [lookupDst, List.find?, hne]
      rw [projectIndex, BitVec.getLsbD_or, bit_ite_shift_false _ d i hi hdi, Bool.false_or, hfind]
      exact projectIndex_getLsbD idx rest i hi huniq.2

private theorem lineTable_ray_zero_of_none (b : Byte) (cellOf : Nat → Option Nat)
    (hbit : ∀ i, i < 8 → b.getLsbD i = match cellOf i with | some _ => b.getLsbD i | none => false)
    (acc : Nat) :
    ∀ (is : List Nat), (∀ i ∈ is, i < 8) → (∀ i ∈ is, cellOf i = none) →
      LineTable.ray b acc is = 0
  | [], _, _ => rfl
  | i :: is, hlt, hnone => by
    have hi : i < 8 := hlt i (by simp)
    have hbi : b.getLsbD i = false := by
      have h := hbit i hi
      rw [hnone i (by simp)] at h
      exact h
    simp only [LineTable.ray, hbi]
    exact lineTable_ray_zero_of_none b cellOf hbit (acc + 1) is
      (fun j hj => hlt j (by simp [hj])) (fun j hj => hnone j (by simp [hj]))

private theorem lineTable_ray_eq_ray_padded (b : Byte) (p : Word) (cellOf : Nat → Option Nat)
    (hbit : ∀ i, i < 8 → b.getLsbD i = match cellOf i with | some c => p.getLsbD c | none => false)
    (acc : Nat) :
    ∀ (is cs : List Nat),
      (∀ i ∈ is, i < 8) →
      is.map cellOf = cs.map some ++ List.replicate (is.length - cs.length) none →
      LineTable.ray b acc is = ray p acc cs
  | [], [], _, _ => rfl
  | x :: xs, [], hlt, hmap => by
    have hbit' : ∀ i, i < 8 → b.getLsbD i = match cellOf i with | some _ => b.getLsbD i | none => false := by
      intro i hi
      have h := hbit i hi
      cases hc : cellOf i
      · simpa [hc] using h
      · simp
    apply lineTable_ray_zero_of_none b cellOf hbit' acc (x :: xs) hlt
    intro j hj
    have hmem : cellOf j ∈ List.replicate (x :: xs).length (none : Option Nat) := by
      have h1 : cellOf j ∈ (x :: xs).map cellOf := List.mem_map_of_mem hj
      simpa [hmap] using h1
    exact List.eq_of_mem_replicate hmem
  | [], _ :: _, _, hmap => by simp at hmap
  | i :: is, c :: cs, hlt, hmap => by
    simp only [List.map_cons, List.cons_append, List.cons.injEq, List.length_cons,
      Nat.succ_sub_succ_eq_sub] at hmap
    have hi : i < 8 := hlt i (by simp)
    have hbi : b.getLsbD i = p.getLsbD c := by rw [hbit i hi, hmap.1]
    simp only [LineTable.ray, ray, hbi]
    cases p.getLsbD c
    · exact lineTable_ray_eq_ray_padded b p cellOf hbit (acc + 1) is cs
        (fun j hj => hlt j (by simp [hj])) hmap.2
    · rfl

private def validLine (pos : Fin 8) (cellOf : Nat → Option Nat) (hiRay loRay : List Nat) : Bool :=
  let hiIdx := (List.range 8).filter (fun i => pos.val < i)
  let loIdx := (List.range pos.val).reverse
  (hiIdx.map cellOf == hiRay.map some ++ List.replicate (hiIdx.length - hiRay.length) none) &&
  (loIdx.map cellOf == loRay.map some ++ List.replicate (loIdx.length - loRay.length) none)

private theorem reference_eq_rays (b : Byte) (p : Word) (pos : Fin 8)
    (cellOf : Nat → Option Nat) (hiRay loRay : List Nat)
    (hbit : ∀ i, i < 8 → b.getLsbD i = match cellOf i with | some c => p.getLsbD c | none => false)
    (hvalid : validLine pos cellOf hiRay loRay = true) :
    LineTable.reference pos b = ray p 0 hiRay + ray p 0 loRay := by
  simp only [validLine, Bool.and_eq_true, beq_iff_eq] at hvalid
  have hhiLt : ∀ i ∈ (List.range 8).filter (fun i => pos.val < i), i < 8 := by
    intro i hi
    simp only [List.mem_filter, List.mem_range] at hi
    exact hi.1
  have hloLt : ∀ i ∈ (List.range pos.val).reverse, i < 8 := by
    intro i hi
    simp only [List.mem_reverse, List.mem_range] at hi
    omega
  unfold LineTable.reference
  rw [lineTable_ray_eq_ray_padded b p cellOf hbit 0 _ hiRay hhiLt hvalid.1,
      lineTable_ray_eq_ray_padded b p cellOf hbit 0 _ loRay hloLt hvalid.2]

private def unionCellOf (u : UnionParams) (map : List (Nat × Nat)) (i : Nat) : Option Nat :=
  match lookupDst map i with
  | some s => u.unionCells[s]?
  | none => none

private def validSquare (sq : Fin 64) : Bool :=
  let u := unionParams sq.val
  let rowPos : Fin 8 := ⟨u.rowPos % 8, Nat.mod_lt _ (by decide)⟩
  let filePos : Fin 8 := ⟨u.filePos % 8, Nat.mod_lt _ (by decide)⟩
  let d0Pos : Fin 8 := ⟨u.d0Pos % 8, Nat.mod_lt _ (by decide)⟩
  let d1Pos : Fin 8 := ⟨u.d1Pos % 8, Nat.mod_lt _ (by decide)⟩
  let rowOk := decide (u.rankShift ≤ 56) &&
    validLine rowPos (fun i => some (u.rankShift + i)) (raySquares sq.val 0 1) (raySquares sq.val 0 (-1))
  let fileOk := decide (u.fileCells.length = 8) &&
    u.fileCells.all (fun c => decide (c < 64)) &&
    validLine filePos (fun i => u.fileCells[i]?) (raySquares sq.val 1 0) (raySquares sq.val (-1) 0)
  let unionOk := decide (u.unionCells.length ≤ 16) &&
    decide (u.size = 2 ^ u.unionCells.length) &&
    u.unionCells.all (fun c => decide (c < 64)) &&
    mapDstUnique u.d0Map && mapDstUnique u.d1Map &&
    u.d0Map.all (fun p => decide (p.1 < 16)) &&
    u.d1Map.all (fun p => decide (p.1 < 16))
  let d0Ok := validLine d0Pos (unionCellOf u u.d0Map) (raySquares sq.val 1 (-1)) (raySquares sq.val (-1) 1)
  let d1Ok := validLine d1Pos (unionCellOf u u.d1Map) (raySquares sq.val 1 1) (raySquares sq.val (-1) (-1))
  rowOk && fileOk && unionOk && d0Ok && d1Ok

private theorem all_squares_valid : ∀ sq : Fin 64, validSquare sq = true := by decide +kernel

theorem refs_correct (p : Word) (square : Fin 64) : refs p square.val = reference p square.val := by
  have hv := all_squares_valid square
  simp only [validSquare, Bool.and_eq_true, decide_eq_true_eq] at hv
  obtain ⟨⟨⟨⟨⟨_, hrow⟩, ⟨⟨_, _⟩, hfile⟩⟩, ⟨⟨⟨⟨⟨⟨_, _⟩, _⟩, hd0u⟩, hd1u⟩, hd0s⟩, hd1s⟩⟩, hd0⟩, hd1⟩ := hv
  let u := unionParams square.val
  have hRowRef : LineTable.reference ⟨u.rowPos % 8, Nat.mod_lt _ (by decide)⟩ ((p >>> u.rankShift).setWidth 8) =
      ray p 0 (raySquares square.val 0 1) + ray p 0 (raySquares square.val 0 (-1)) := by
    apply reference_eq_rays _ p _ (fun i => some (u.rankShift + i)) _ _ _ hrow
    intro i hi
    simp only [BitVec.getLsbD_setWidth, BitVec.getLsbD_ushiftRight, hi, decide_true, Bool.true_and]
  have hFileRef : LineTable.reference ⟨u.filePos % 8, Nat.mod_lt _ (by decide)⟩ ((pext p u.fileCells).setWidth 8) =
      ray p 0 (raySquares square.val 1 0) + ray p 0 (raySquares square.val (-1) 0) := by
    apply reference_eq_rays _ p _ (fun i => u.fileCells[i]?) _ _ _ hfile
    intro i hi
    simp only [BitVec.getLsbD_setWidth, hi, decide_true, Bool.true_and]
    exact pext_getLsbD p u.fileCells i (by omega)
  have hD0Ref : LineTable.reference ⟨u.d0Pos % 8, Nat.mod_lt _ (by decide)⟩
        (projectIndex ((pext p u.unionCells).setWidth 16) u.d0Map) =
      ray p 0 (raySquares square.val 1 (-1)) + ray p 0 (raySquares square.val (-1) 1) := by
    apply reference_eq_rays _ p _ (unionCellOf u u.d0Map) _ _ _ hd0
    intro i hi
    rw [projectIndex_getLsbD _ u.d0Map i hi hd0u]
    unfold unionCellOf
    cases hfind : lookupDst u.d0Map i with
    | none => rfl
    | some s =>
      have hs16 : s < 16 := by
        unfold lookupDst at hfind
        cases hq : u.d0Map.find? (fun pair => pair.2 == i) with
        | none => simp [hq] at hfind
        | some pair =>
          simp only [hq, Option.map_some, Option.some.injEq] at hfind
          have hmem := List.mem_of_find?_eq_some hq
          have := List.all_eq_true.mp hd0s pair hmem
          simpa [hfind] using this
      simp only [BitVec.getLsbD_setWidth, hs16, decide_true, Bool.true_and]
      exact pext_getLsbD p u.unionCells s (by omega)
  have hD1Ref : LineTable.reference ⟨u.d1Pos % 8, Nat.mod_lt _ (by decide)⟩
        (projectIndex ((pext p u.unionCells).setWidth 16) u.d1Map) =
      ray p 0 (raySquares square.val 1 1) + ray p 0 (raySquares square.val (-1) (-1)) := by
    apply reference_eq_rays _ p _ (unionCellOf u u.d1Map) _ _ _ hd1
    intro i hi
    rw [projectIndex_getLsbD _ u.d1Map i hi hd1u]
    unfold unionCellOf
    cases hfind : lookupDst u.d1Map i with
    | none => rfl
    | some s =>
      have hs16 : s < 16 := by
        unfold lookupDst at hfind
        cases hq : u.d1Map.find? (fun pair => pair.2 == i) with
        | none => simp [hq] at hfind
        | some pair =>
          simp only [hq, Option.map_some, Option.some.injEq] at hfind
          have hmem := List.mem_of_find?_eq_some hq
          have := List.all_eq_true.mp hd1s pair hmem
          simpa [hfind] using this
      simp only [BitVec.getLsbD_setWidth, hs16, decide_true, Bool.true_and]
      exact pext_getLsbD p u.unionCells s (by omega)
  unfold refs reference
  dsimp only
  change LineTable.reference ⟨u.rowPos % 8, _⟩ _ +
      LineTable.reference ⟨u.filePos % 8, _⟩ _ +
      LineTable.reference ⟨u.d0Pos % 8, _⟩ _ +
      LineTable.reference ⟨u.d1Pos % 8, _⟩ _ = _
  rw [hRowRef, hFileRef, hD0Ref, hD1Ref]
  omega

private theorem bitvec_pow2_sub_one_getLsbD (k i : Nat) (hk : k ≤ 16) (hi : i < 16) :
    (BitVec.ofNat 16 (2 ^ k - 1)).getLsbD i = decide (i < k) := by
  have hall : ∀ (k : Fin 17) (i : Fin 16),
      (BitVec.ofNat 16 (2 ^ k.val - 1)).getLsbD i.val = decide (i.val < k.val) := by decide +kernel
  exact hall ⟨k, by omega⟩ ⟨i, hi⟩

private theorem refsOpp_eq_refs_not (p : Word) (square : Fin 64) :
    refsOpp p square.val = refs (~~~p) square.val := by
  have hv := all_squares_valid square
  simp only [validSquare, Bool.and_eq_true, decide_eq_true_eq] at hv
  obtain ⟨⟨⟨⟨⟨hshift, _⟩, ⟨⟨hflen, hfc⟩, _⟩⟩, ⟨⟨⟨⟨⟨⟨hulen, husize⟩, huc⟩, _⟩, _⟩, _⟩, _⟩⟩, _⟩, _⟩ := hv
  let u := unionParams square.val
  have hrow : ((p >>> u.rankShift).setWidth 8) ^^^ 255 = ((~~~p) >>> u.rankShift).setWidth 8 := by
    apply BitVec.eq_of_getLsbD_eq
    intro i hi
    have h255 : (255 : Byte).getLsbD i = true := by
      have : ∀ j : Fin 8, (255 : Byte).getLsbD j.val = true := by decide
      exact this ⟨i, hi⟩
    have hlt64 : u.rankShift + i < 64 := by change (unionParams square.val).rankShift + i < 64; omega
    simp only [BitVec.getLsbD_xor, BitVec.getLsbD_setWidth, BitVec.getLsbD_ushiftRight,
      BitVec.getLsbD_not, hi, hlt64, h255, decide_true, Bool.true_and, Bool.xor_true]
  have hfile : ((pext p u.fileCells).setWidth 8) ^^^ 255 = (pext (~~~p) u.fileCells).setWidth 8 := by
    apply BitVec.eq_of_getLsbD_eq
    intro i hi
    have h255 : (255 : Byte).getLsbD i = true := by
      have : ∀ j : Fin 8, (255 : Byte).getLsbD j.val = true := by decide
      exact this ⟨i, hi⟩
    simp only [BitVec.getLsbD_xor, BitVec.getLsbD_setWidth, hi, h255, decide_true, Bool.true_and, Bool.xor_true]
    rw [pext_getLsbD p u.fileCells i (by omega), pext_getLsbD (~~~p) u.fileCells i (by omega)]
    have hilt : i < u.fileCells.length := by change i < (unionParams square.val).fileCells.length; omega
    rw [List.getElem?_eq_getElem hilt]
    have hclt : u.fileCells[i] < 64 := by
      have := List.all_eq_true.mp hfc u.fileCells[i] (List.getElem_mem hilt)
      simpa using this
    simp [hclt]
  have hunion : ((pext p u.unionCells).setWidth 16) ^^^ BitVec.ofNat 16 (u.size - 1) =
      (pext (~~~p) u.unionCells).setWidth 16 := by
    apply BitVec.eq_of_getLsbD_eq
    intro i hi
    simp only [BitVec.getLsbD_xor, BitVec.getLsbD_setWidth, hi, decide_true, Bool.true_and]
    have hsz : u.size = 2 ^ u.unionCells.length := husize
    have hul : u.unionCells.length ≤ 16 := hulen
    rw [hsz, bitvec_pow2_sub_one_getLsbD u.unionCells.length i hul hi,
      pext_getLsbD p u.unionCells i (by omega), pext_getLsbD (~~~p) u.unionCells i (by omega)]
    by_cases hilt : i < u.unionCells.length
    · rw [List.getElem?_eq_getElem hilt]
      have hclt : u.unionCells[i] < 64 := by
        have := List.all_eq_true.mp huc u.unionCells[i] (List.getElem_mem hilt)
        simpa using this
      simp [hilt, hclt]
    · rw [List.getElem?_eq_none (by omega)]
      simp [hilt]
  unfold refsOpp refs
  dsimp only
  rw [hrow, hfile, hunion]

theorem refsOpp_correct (p : Word) (square : Fin 64) : refsOpp p square.val = reference (~~~p) square.val := by
  rw [refsOpp_eq_refs_not p square]
  exact refs_correct (~~~p) square

theorem count_refs (p : Word) (square : Nat) : count p square = BitVec.ofNat 8 (2 * refs p square) := by
  simp only [count, lookup, LineTable.countFlip_eq_ofNat, refs, Nat.mul_add, BitVec.ofNat_add]
  simp only [BitVec.add_assoc]

theorem countOpp_refs (p : Word) (square : Nat) : countOpp p square = BitVec.ofNat 8 (2 * refsOpp p square) := by
  simp only [countOpp, lookup, LineTable.countFlip_eq_ofNat, refsOpp, Nat.mul_add, BitVec.ofNat_add]
  simp only [BitVec.add_assoc]

theorem count_correct (p : Word) (square : Fin 64) :
    count p square.val = BitVec.ofNat 8 (2 * reference p square.val) := by
  rw [count_refs, refs_correct]
theorem countOpp_correct (p : Word) (square : Fin 64) :
    countOpp p square.val = BitVec.ofNat 8 (2 * reference (~~~p) square.val) := by
  rw [countOpp_refs, refsOpp_correct]

theorem refs_bound (p : Word) (square : Nat) : refs p square ≤ 24 := by
  unfold refs
  exact Nat.add_le_add
    (Nat.add_le_add (Nat.add_le_add (LineTable.reference_bound _ _) (LineTable.reference_bound _ _))
      (LineTable.reference_bound _ _)) (LineTable.reference_bound _ _)

theorem reference_bound (p : Word) (square : Fin 64) : reference p square.val ≤ 24 := by
  rw [← refs_correct]
  exact refs_bound p square.val

theorem count_toNat (p : Word) (square : Fin 64) :
    (count p square.val).toNat = 2 * reference p square.val := by
  rw [count_correct, BitVec.toNat_ofNat]
  have h := reference_bound p square
  apply Nat.mod_eq_of_lt
  omega

theorem countOpp_toNat (p : Word) (square : Fin 64) :
    (countOpp p square.val).toNat = 2 * reference (~~~p) square.val := by
  rw [countOpp_correct, BitVec.toNat_ofNat]
  have h := reference_bound (~~~p) square
  apply Nat.mod_eq_of_lt
  omega

theorem metadata_positions (square : Fin 64) :
    (unionParams square.val).rowPos < 8 ∧ (unionParams square.val).filePos < 8 ∧
    (unionParams square.val).d0Pos < 8 ∧ (unionParams square.val).d1Pos < 8 := by
  have all : ∀ s : Fin 64,
    (unionParams s.val).rowPos < 8 ∧ (unionParams s.val).filePos < 8 ∧
    (unionParams s.val).d0Pos < 8 ∧ (unionParams s.val).d1Pos < 8 := by decide
  exact all square

def lookupWide (u : UnionParams) (row file : Byte) (diag : Index) : BitVec 32 :=
  (LineTable.countFlip ⟨u.rowPos % 8, Nat.mod_lt _ (by decide)⟩ row).setWidth 32 +
  (LineTable.countFlip ⟨u.filePos % 8, Nat.mod_lt _ (by decide)⟩ file).setWidth 32 +
  (LineTable.countFlip ⟨u.d0Pos % 8, Nat.mod_lt _ (by decide)⟩ (projectIndex diag u.d0Map) +
   LineTable.countFlip ⟨u.d1Pos % 8, Nat.mod_lt _ (by decide)⟩ (projectIndex diag u.d1Map)).setWidth 32

theorem lookupWide_toNat (u : UnionParams) (row file : Byte) (diag : Index) :
    (lookupWide u row file diag).toNat = (lookup u row file diag).toNat := by
  have h0 := LineTable.countFlip_bound ⟨u.rowPos % 8, Nat.mod_lt _ (by decide)⟩ row
  have h1 := LineTable.countFlip_bound ⟨u.filePos % 8, Nat.mod_lt _ (by decide)⟩ file
  have h2 := LineTable.countFlip_bound ⟨u.d0Pos % 8, Nat.mod_lt _ (by decide)⟩ (projectIndex diag u.d0Map)
  have h3 := LineTable.countFlip_bound ⟨u.d1Pos % 8, Nat.mod_lt _ (by decide)⟩ (projectIndex diag u.d1Map)
  simp only [lookupWide, lookup, BitVec.toNat_add, BitVec.toNat_setWidth]
  omega

def countWide (p : Word) (square : Nat) : BitVec 32 :=
  let u := unionParams square
  lookupWide u ((p >>> u.rankShift).setWidth 8) ((pext p u.fileCells).setWidth 8)
    ((pext p u.unionCells).setWidth 16)
def countOppWide (p : Word) (square : Nat) : BitVec 32 :=
  let u := unionParams square
  lookupWide u (((p >>> u.rankShift).setWidth 8) ^^^ 255) (((pext p u.fileCells).setWidth 8) ^^^ 255)
    (((pext p u.unionCells).setWidth 16) ^^^ (BitVec.ofNat 16 (u.size - 1)))

theorem countWide_toNat (p : Word) (square : Fin 64) :
    (countWide p square.val).toNat = 2 * reference p square.val := by
  rw [countWide, lookupWide_toNat]
  exact count_toNat p square

theorem countOppWide_toNat (p : Word) (square : Fin 64) :
    (countOppWide p square.val).toNat = 2 * reference (~~~p) square.val := by
  rw [countOppWide, lookupWide_toNat]
  exact countOpp_toNat p square

theorem ray_eq_geometry (p : Word) (acc : Nat) (cells : List Nat) :
    ray p acc cells = Geometry.ray p acc cells := by
  induction cells generalizing acc with
  | nil => rfl
  | cons i cells ih => simp only [ray, Geometry.ray, ih]

theorem reference_eq_geometry (p : Word) (square : Nat) :
    reference p square = Geometry.reference p square := by
  simp only [reference, Geometry.reference, ray_eq_geometry]
  rfl

end NeuralReversi.Solve1.Bmi2
