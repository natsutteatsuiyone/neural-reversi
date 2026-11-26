import NeuralReversi.PvsParallel.EndgameOrderMir.Program
import NeuralReversi.PvsParallel.GeneralNonPvMir
import NeuralReversi.PvsParallel.RootFrameMir

namespace NeuralReversi.PvsParallel.EndgameOrderMir

open Bitboard.Moves (X86Config)
private abbrev FlipConfig := Flip.Dispatch.X86Config
open PvsBoardTree (Position GeneratedSquare legalSquares play)
open MoveList.Ordering (ttMoveValue SortedDesc sortByKeyDesc)
open MoveList.OrderWeights

/-- The extracted endgame depth table selects the fast evaluator at full depth
    exactly through thirteen empties. -/
theorem fullDepth_fast : ∀ n : Fin 64, useFast (BitVec.ofNat 32 n.val) n = decide (n.val ≤ 13) := by decide

theorem fullDepth_key (mc : X86Config) (fc : FlipConfig) (b : Position)
    (depth : BitVec 32) (empty : Fin 64) (full : depth.toNat = empty.val) (small : empty.val ≤ 13)
    (slower : Heuristic) (sq : Bitboard.Square) (hint : Option Bitboard.Square) :
    let board := Board.place b.val (Flip.Dispatch.resultValue fc sq b.val.player b.val.opponent) sq
    actualKey (fun i => raw (if useFast depth empty then .fast else slower) mc board i) hint sq =
      (if some sq = hint then tt else raw .fast mc board sq) := by
  have same : depth = BitVec.ofNat 32 empty.val := by rw [← full]; simp
  simp [same,fullDepth_fast,small,actualKey]

theorem estimate_range (e : Estimate) : -32768 ≤ e.read ∧ e.read ≤ 32768 := by
  induction e with
  | network raw => simp only [Estimate.read]; omega
  | cacheHit bits =>
    have lower := BitVec.le_toInt bits
    have upper := BitVec.toInt_lt (x := bits)
    change -32768 ≤ bits.toInt ∧ bits.toInt ≤ 32768
    simp at lower upper
    omega
  | terminal b =>
    have v := (PvsBoardTree.terminalScore b).property
    unfold PvsMoveLoop.scoreMax at v
    change -32768 ≤ (PvsBoardTree.terminalScore b).val ∧ (PvsBoardTree.terminalScore b).val ≤ 32768
    omega
  | finalCount b =>
    have p := (Bitboard.populationCountContract_proven b.player).2.1

    simp only [Estimate.read]
    omega
  | maximum | infinity => decide
  | neg _ ih => simp only [Estimate.read]; omega
  | higher x y hx hy => simp only [Estimate.read]; omega

/-- The hand-written deep-order key agrees with the extracted fast key on the successor's moves. -/
theorem fastMoveValue_raw (mc : X86Config) (next : Board.Board) (sq : Bitboard.Square) :
    (EndgameSearch.fastMoveValue sq next (Board.getMoves mc next)).toInt = raw .fast mc next sq := by
  rw [EndgameSearch.fastMoveValue_exact]
  rfl

/-- Bounds of `raw` over every heuristic, derived from the extracted weights. Corner stability is
    at most 12 and a corner-weighted move count is at most 68. -/
def priorityMin : Int :=
  min (-32 * mobilityWeight) (-(32768 + 68 * searchMobilityWeight + 68 * searchPotentialWeight))
def priorityMax : Int :=
  max (squareMax * squareWeight + 12 * stabilityWeight + 36 * mobilityWeight) (32768 + 12 * searchStabilityWeight)

/-- The bounds above assume nonnegative weights. -/
theorem weights_nonneg : 0 ≤ squareWeight ∧ 0 ≤ stabilityWeight ∧ 0 ≤ mobilityWeight ∧
    0 ≤ searchStabilityWeight ∧ 0 ≤ searchMobilityWeight ∧ 0 ≤ searchPotentialWeight := by decide

/-- Every key fits in `i32` and stays below the TT hint after ranking. -/
theorem keys_fit : -2147483648 ≤ priorityMin ∧ 64 * priorityMax + 63 < tt := by decide

private theorem fast_bounds : priorityMin ≤ -32 * mobilityWeight ∧
    squareMax * squareWeight + 12 * stabilityWeight + 36 * mobilityWeight ≤ priorityMax :=
  ⟨Int.min_le_left _ _, Int.le_max_left _ _⟩

private theorem search_bounds :
    priorityMin ≤ -(32768 + 68 * searchMobilityWeight + 68 * searchPotentialWeight) ∧
      32768 + 12 * searchStabilityWeight ≤ priorityMax :=
  ⟨Int.min_le_right _ _, Int.le_max_right _ _⟩

theorem priority_range (method : Heuristic) (mc : X86Config) (next : Board.Board) (sq : Bitboard.Square) :
    priorityMin ≤ raw method mc next sq ∧ raw method mc next sq ≤ priorityMax := by
  have st := (Bitboard.cornerStabilityContract_proven next.opponent).2
  have mob (bits : Bitboard.Bits) := (Bitboard.cornerWeightedCountContract_proven bits).2.1
  have f := fast_bounds
  have r := search_bounds
  simp only [squareMax, squareWeight, stabilityWeight, mobilityWeight, searchStabilityWeight,
    searchMobilityWeight, searchPotentialWeight] at f r

  cases method with
  | fast =>
    have s := MoveList.Ordering.squareValue_range sq
    have m := mob (Board.getMoves mc next)
    simp only [squareMax] at s
    simp only [raw, squareWeight, stabilityWeight, mobilityWeight]
    omega
  | skip =>
    have m := mob (Board.getMoves mc next)
    simp only [raw]
    omega
  | search e =>
    have s := estimate_range e
    have m := mob (Board.getMovesAndPotential mc next).1
    have p := mob (Board.getMovesAndPotential mc next).2
    simp only [raw, searchStabilityWeight, searchMobilityWeight, searchPotentialWeight]
    omega

def priority (mc : X86Config) (fc : FlipConfig) (b : Position) (mode : Bitboard.Square → Heuristic)
    (sq : Bitboard.Square) : Int :=
  raw (mode sq) mc (Board.place b.val (Flip.Dispatch.resultValue fc sq b.val.player b.val.opponent) sq) sq

theorem priority_low (mc : X86Config) (fc : FlipConfig) (b : Position) (mode : Bitboard.Square → Heuristic)
    (sq : Bitboard.Square) : priority mc fc b mode sq < ttMoveValue := by
  have w := keys_fit.2
  simp only [tt] at w
  exact Int.lt_of_le_of_lt (priority_range _ _ _ _).2 (by simp only [ttMoveValue]; omega)

theorem checked_operands (mc : X86Config) (next : Board.Board) (sq : Bitboard.Square) (e : Estimate) :
    let sv := squareValue[sq]
    let s : Int := (Bitboard.cornerStability next.opponent).toNat
    let m : Int := (Bitboard.cornerWeightedCount (Board.getMoves mc next)).toNat
    let q := Board.getMovesAndPotential mc next
    let p : Int := (Bitboard.cornerWeightedCount q.1).toNat
    let z : Int := (Bitboard.cornerWeightedCount q.2).toNat
    ∀ x ∈ [sv*squareWeight, s*stabilityWeight, 36-m, (36-m)*mobilityWeight, sv*squareWeight+s*stabilityWeight,
      sv*squareWeight+s*stabilityWeight+(36-m)*mobilityWeight,
      -m, s*searchStabilityWeight, e.read+s*searchStabilityWeight, p*searchMobilityWeight,
      e.read+s*searchStabilityWeight-p*searchMobilityWeight, z*searchPotentialWeight,
      e.read+s*searchStabilityWeight-p*searchMobilityWeight-z*searchPotentialWeight],
      -2147483648 ≤ x ∧ x < 2147483648 := by
  have st := (Bitboard.cornerStabilityContract_proven next.opponent).2
  have mob (bits : Bitboard.Bits) := (Bitboard.cornerWeightedCountContract_proven bits).2.1
  have sqv := MoveList.Ordering.squareValue_range sq
  have m := mob (Board.getMoves mc next)
  have p := mob (Board.getMovesAndPotential mc next).1
  have z := mob (Board.getMovesAndPotential mc next).2
  have ev := estimate_range e
  have w := keys_fit
  have f := fast_bounds
  have r := search_bounds
  simp only [squareMax, squareWeight, stabilityWeight, mobilityWeight, searchStabilityWeight,
    searchMobilityWeight, searchPotentialWeight, tt] at sqv w f r
  dsimp only
  simp only [squareWeight, stabilityWeight, mobilityWeight, searchStabilityWeight, searchMobilityWeight,
    searchPotentialWeight]
  intro x member
  simp only [List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> omega

theorem key_bits_eq (mc : X86Config) (fc : FlipConfig) (b : Position) (mode : Bitboard.Square → Heuristic)
    (hint : Option Bitboard.Square) (sq : Bitboard.Square) :
    (BitVec.ofInt 32 (actualKey (priority mc fc b mode) hint sq)).toInt =
      actualKey (priority mc fc b mode) hint sq := by
  apply BitVec.toInt_ofInt_eq_self (by decide)
  all_goals
    have range := priority_range (mode sq) mc (Board.place b.val (Flip.Dispatch.resultValue fc sq b.val.player b.val.opponent) sq) sq
    have w := keys_fit
    simp only [tt] at w
    simp only [actualKey, tt, priority]
    split <;> omega

theorem sort_ok (key : Move → Int) (tie : Move → Fin 64) (xs : List Move) :
    (sortMoves key tie xs).Perm xs ∧ SortedDesc key (sortMoves key tie xs) := by
  unfold sortMoves
  split
  · exact MoveList.Ordering.sortContract_proven _ _ _
  · obtain ⟨perm, ordered⟩ := MoveList.Ordering.sortContract_proven _ (fun m => 64 * key m + (tie m).val) xs
    refine ⟨perm, ordered.imp (fun {x y} high => ?_)⟩
    have a := (tie x).isLt
    have b := (tie y).isLt
    dsimp only at high
    omega

private def reverseIndex [DecidableEq Move] (xs : List Move) (m : Move) : Fin 64 :=
  ⟨63 - xs.idxOf m, by omega⟩

private theorem rank_injective [DecidableEq Move] (xs : List Move) (small : xs.length ≤ 64)
    (key : Move → Int) (a b : Move) (ha : a ∈ xs) (hb : b ∈ xs)
    (eq : 64 * key a + (reverseIndex xs a).val = 64 * key b + (reverseIndex xs b).val) : a = b := by
  have ia := List.idxOf_lt_length_iff.mpr ha
  have ib := List.idxOf_lt_length_iff.mpr hb
  dsimp [reverseIndex] at eq
  have ab : xs.idxOf a = xs.idxOf b := by omega
  have cells := congrArg (fun i => xs[i]?) ab
  simp only [List.getElem?_eq_getElem ia, List.getElem?_eq_getElem ib, List.getElem_idxOf, Option.some.injEq] at cells
  exact cells

/-- Every descending permutation returned by the standard library on a long initialized slice is representable. -/
theorem library_realizable [DecidableEq Move] (key : Move → Int) (xs output : List Move)
    (long : 3 < xs.length) (fits : xs.length ≤ 64) (unique : xs.Nodup)
    (perm : output.Perm xs) (descending : SortedDesc key output) :
    ∃ tie : Move → Fin 64, sortMoves key tie xs = output := by
  let tie := reverseIndex output
  let indexed := fun m => 64 * key m + (tie m).val
  have outFit : output.length ≤ 64 := perm.length_eq ▸ fits
  have outUnique := perm.nodup_iff.mpr unique
  have indices : SortedDesc indexed output := by
    apply List.pairwise_iff_getElem.mpr
    intro i j hi hj before
    have sorted := descending.rel_getElem_of_lt hi hj before
    change 64 * key output[j] + ((63 - output.idxOf output[j] : Nat) : Int) ≤
      64 * key output[i] + ((63 - output.idxOf output[i] : Nat) : Int)
    rw [outUnique.idxOf_getElem j hj, outUnique.idxOf_getElem i hi]
    change key output[j] ≤ key output[i] at sorted
    omega
  obtain ⟨covers, ordered⟩ := MoveList.Ordering.sortContract_proven Move indexed xs
  have same := covers.trans perm.symm
  refine ⟨tie, ?_⟩
  rw [sortMoves, ite_eq_right (by omega)]
  apply List.Perm.eq_of_pairwise _ ordered indices same
  intro a b ha hb ab ba
  exact rank_injective output outFit key a b (same.mem_iff.mp ha) hb (Int.le_antisymm ba ab)

theorem ordinary_perm [DecidableEq Move] (key : Move → Int) (tie : Move → Fin 64)
    (used : Bool) (hint : Option Move) (xs : List Move) :
    (ordinary key tie used hint xs).Perm xs := by
  unfold ordinary
  generalize (if used then 1 else 0) = seed
  split
  · exact MoveList.Ordering.sortByKeyDesc_perm _ _
  · cases xs with
    | nil => rfl
    | cons a tail =>
      cases tail with
      | nil => rfl
      | cons b rest =>
        cases rest with
        | cons => rfl
        | nil =>
          dsimp only
          split
          · rfl
          · exact MoveList.Ordering.swapList_perm _ _ _

theorem ranked_low (mc : X86Config) (fc : FlipConfig) (b : Position) (mode : Bitboard.Square → Heuristic)
    (tie : GeneratedSquare mc b.val → Fin 64) (length : Nat) (m : GeneratedSquare mc b.val) :
    ranked (fun x : GeneratedSquare mc b.val => priority mc fc b mode x.val) tie length m < ttMoveValue := by
  have bound := (priority_range (mode m.val) mc (Board.place b.val (Flip.Dispatch.resultValue fc m.val b.val.player b.val.opponent) m.val) m.val).2
  have small := (tie m).isLt
  have w := keys_fit.2
  simp only [tt] at w
  simp only [ranked, priority, ttMoveValue]
  split <;> omega

def framePriority (mc : X86Config) (fc : FlipConfig) (b : Position) (mode : Bitboard.Square → Heuristic)
    (tie : GeneratedSquare mc b.val → Fin 64) : GeneratedSquare mc b.val → Int :=
  ranked (fun m => priority mc fc b mode m.val) tie (legalSquares mc b.val).length

theorem framePriority_low (mc : X86Config) (fc : FlipConfig) (b : Position) (mode : Bitboard.Square → Heuristic)
    (tie : GeneratedSquare mc b.val → Fin 64) (m : GeneratedSquare mc b.val) :
    framePriority mc fc b mode tie m < ttMoveValue :=
  ranked_low mc fc b mode tie _ m

theorem concrete_order [DecidableEq Move] (key : Move → Int) (tie : Move → Fin 64) (hint : Option Move) (xs : List Move) :
    ordinary key tie (xs.length == 1 || hint.isSome) hint xs = GeneralNonPvMir.order (ranked key tie xs.length) hint xs := by
  cases xs with
  | nil => cases hint <;> rfl
  | cons a rest =>
    cases rest with
    | nil => cases hint <;> rfl
    | cons b more =>
      cases more with
      | nil =>
        cases hint with
        | none => rfl
        | some selected => simp [ordinary, GeneralNonPvMir.order]
      | cons c tail =>
        cases hint <;> simp [ordinary, GeneralNonPvMir.order, ttMoveValue] <;> congr 1 <;> funext m <;> simp [actualKey, tt]

private theorem sort_congr (key other : Move → Int) (cmp : ∀ a b, key a < key b ↔ other a < other b)
    (xs : List Move) : sortByKeyDesc key xs = sortByKeyDesc other xs := by
  have swap (ys : List Move) (a b : Nat) :
      MoveList.Ordering.compareSwap key ys a b = MoveList.Ordering.compareSwap other ys a b := by
    unfold MoveList.Ordering.compareSwap
    cases ys[a]? <;> cases ys[b]? <;> simp [cmp]
  have insert (m : Move) (ys : List Move) :
      MoveList.Ordering.insertDesc key m ys = MoveList.Ordering.insertDesc other m ys := by
    induction ys with
    | nil => rfl
    | cons y ys ih => simp only [MoveList.Ordering.insertDesc, cmp, ih]
  have insertion (ys : List Move) :
      MoveList.Ordering.insertionSortDesc key ys = MoveList.Ordering.insertionSortDesc other ys := by
    induction ys with
    | nil => rfl
    | cons y ys ih => simp only [MoveList.Ordering.insertionSortDesc, ih, insert]
  unfold sortByKeyDesc
  split <;> simp only [swap, insertion]

private theorem hint_tie_comparison [DecidableEq Move] (key : Move → Int) (tie : Move → Fin 64)
    (small : ∀ m, key m ≤ priorityMax) (hint : Option Move) (a b : Move) :
    64 * actualKey key hint a + (tie a).val < 64 * actualKey key hint b + (tie b).val ↔
      actualKey (fun m => 64 * key m + (tie m).val) hint a < actualKey (fun m => 64 * key m + (tie m).val) hint b := by
  have ka := small a
  have kb := small b
  have ta := (tie a).isLt
  have tb := (tie b).isLt
  have w := keys_fit.2
  simp only [tt] at w
  by_cases x : some a = hint <;> by_cases y : some b = hint
  · have same := Option.some.inj (x.trans y.symm)
    subst b
    simp
  all_goals simp only [actualKey, x, y, ↓reduceIte, tt] <;> omega

theorem ordinary_realizable {Move : Type} [DecidableEq Move] (key : Move → Int) (small : ∀ m, key m ≤ priorityMax)
    (hint : Option Move) (xs output : List Move) (long : 3 < xs.length) (fits : xs.length ≤ 64)
    (unique : xs.Nodup) (perm : output.Perm xs) (descending : SortedDesc (actualKey key hint) output) :
    ∃ tie : Move → Fin 64, GeneralNonPvMir.order (ranked key tie xs.length) hint xs = output := by
  obtain ⟨tie, library⟩ := library_realizable (actualKey key hint) xs output long fits unique perm descending
  have source := concrete_order key tie hint xs
  refine ⟨tie, ?_⟩
  rw [← source]
  have count : 1 < xs.length - (if xs.length == 1 || hint.isSome then 1 else 0) := by split <;> omega
  unfold ordinary
  rw [ite_eq_left count]
  rw [sortMoves, ite_eq_right (by omega)] at library
  have equal := sort_congr _ _ (hint_tie_comparison key tie small hint) xs
  have large : ranked key tie xs.length = fun m => 64 * key m + (tie m).val := by
    funext m
    simp [ranked, show ¬xs.length ≤ 3 by omega]
  rw [large]
  exact equal.symm.trans library

theorem generated_layout (mc : X86Config) (b : Position) :
    (legalSquares mc b.val).Nodup ∧ (legalSquares mc b.val).length ≤ 64 := by
  let p := fun sq : Bitboard.Square => (Board.getMoves mc b.val).getLsbD sq.val
  have eq : (legalSquares mc b.val).map Subtype.val = (List.finRange 64).filter p := PvsBoardTree.legalSquares_values mc b.val
  have unique : ((legalSquares mc b.val).map Subtype.val).Nodup := eq ▸ (List.nodup_finRange 64).filter p
  have size := List.length_filter_le p (List.finRange 64)
  rw [← eq, List.length_map, List.length_finRange] at size
  exact ⟨unique.of_map Subtype.val (fun _ _ neq same => neq (congrArg Subtype.val same)),size⟩

theorem compiled_library_order (mc : X86Config) (fc : FlipConfig) (b : Position) (mode : Bitboard.Square → Heuristic)
    (hint : Option (GeneratedSquare mc b.val)) (output : List (GeneratedSquare mc b.val))
    (long : 3 < (legalSquares mc b.val).length) (perm : output.Perm (legalSquares mc b.val))
    (sorted : SortedDesc (actualKey (fun m : GeneratedSquare mc b.val => priority mc fc b mode m.val) hint) output) :
    ∃ tie : GeneratedSquare mc b.val → Fin 64, GeneralNonPvMir.order (framePriority mc fc b mode tie) hint
      (legalSquares mc b.val) = output := by
  obtain ⟨unique, fits⟩ := generated_layout mc b
  exact ordinary_realizable (fun m : GeneratedSquare mc b.val => priority mc fc b mode m.val)
    (fun m => (priority_range (mode m.val) mc _ m.val).2) hint _ output long fits unique perm sorted

theorem presearch_head (mc : X86Config) (fc : FlipConfig) (b : Position) (mode : Bitboard.Square → Heuristic)
    (tie : GeneratedSquare mc b.val → Fin 64) (sq : GeneratedSquare mc b.val) (xs : List (GeneratedSquare mc b.val))
    (member : sq ∈ xs) :
    let key := fun m : GeneratedSquare mc b.val => priority mc fc b mode m.val
    let result := ordinary key tie true (some sq) xs
    result.Perm xs ∧ result.head? = some sq := by
  let key := fun m : GeneratedSquare mc b.val => priority mc fc b mode m.val
  have eq := concrete_order key tie (some sq) xs
  have front := GeneralNonPvMir.orderHint (ranked key tie xs.length) xs sq member
    (fun m _ _ => ranked_low mc fc b mode tie xs.length m)
  simp only [Option.isSome_some, Bool.or_true] at eq
  rw [← eq] at front
  exact front

/-- The PV owner starts the loop at index zero. The raw table square is only
    compared with squares already in its move list. -/
def pvOrder (mc : X86Config) (fc : FlipConfig) (b : Position) (mode : Bitboard.Square → Heuristic)
    (tie : GeneratedSquare mc b.val → Fin 64) (hint : Option Bitboard.Square) :=
  let moves := legalSquares mc b.val
  if 1 < moves.length then
    sortMoves (fun m : GeneratedSquare mc b.val => actualKey (priority mc fc b mode) hint m.val) tie moves
  else moves

theorem pvOrder_perm (mc : X86Config) (fc : FlipConfig) (b : Position) (mode tie hint) :
    (pvOrder mc fc b mode tie hint).Perm (legalSquares mc b.val) := by
  dsimp only [pvOrder]
  split
  · exact (sort_ok _ _ _).1
  · rfl

theorem pvOrder_illegal (mc : X86Config) (fc : FlipConfig) (b : Position) (mode tie sq)
    (illegal : (Board.getMoves mc b.val).getLsbD sq.val = false) :
    pvOrder mc fc b mode tie (some sq) = pvOrder mc fc b mode tie none := by
  have key (m : GeneratedSquare mc b.val) : actualKey (priority mc fc b mode) (some sq) m.val =
      actualKey (priority mc fc b mode) none m.val := by
    have different : m.val ≠ sq := by
      intro h
      have legal := m.property
      rw [h,illegal] at legal
      contradiction
    simp [actualKey,different]
  unfold pvOrder
  simp only [key]

def rootRecords (mc : X86Config) (fc : FlipConfig) (b : Position) (roots : List Bitboard.Square)
    (idx : Nat) (mode : Bitboard.Square → Heuristic) (tie : Move.Move → Fin 64) (hint : Option Bitboard.Square) :=
  let records := MoveList.Generation.records fc b.val (Root.remainingMask mc b.val roots idx)
  let k := actualKey (priority mc fc b mode) hint
  if 1 < records.length then
    sortMoves (fun m : Move.Move => m.value.toInt) tie
      (records.map (RootFrameMir.setKey (fun sq => BitVec.ofInt 32 (k sq))))
  else records

theorem rootRecords_excluded (mc : X86Config) (fc : FlipConfig) (b : Position) (roots idx mode tie)
    (sq : Bitboard.Square) (outside : (Root.remainingMask mc b.val roots idx).getLsbD sq.val = false) :
    rootRecords mc fc b roots idx mode tie (some sq) = rootRecords mc fc b roots idx mode tie none := by
  let mask := Root.remainingMask mc b.val roots idx
  have keys : ∀ m ∈ MoveList.Generation.records fc b.val mask,
      actualKey (priority mc fc b mode) (some sq) m.sq = actualKey (priority mc fc b mode) none m.sq := by
    intro m mem
    obtain ⟨s,inside,rfl⟩ := List.mem_map.mp mem
    have different : s ≠ sq := by
      intro eq
      have bit := (List.mem_filter.mp inside).2
      subst s
      exact Bool.noConfusion (bit.symm.trans outside)
    simp [MoveList.Generation.entry,actualKey,different]
  dsimp only [rootRecords]
  split
  · congr 1
    apply List.map_congr_left
    intro m mem
    simp [RootFrameMir.setKey,keys m mem]
  · rfl

theorem root_perm (mc : X86Config) (fc : FlipConfig) (b : Position) (roots : List Bitboard.Square)
    (idx : Nat) (mode : Bitboard.Square → Heuristic) (tie : Move.Move → Fin 64) (hint : Option Bitboard.Square) :
    let output := rootRecords mc fc b roots idx mode tie hint
    (output.filterMap (RootFrameMir.asChoice mc b)).Perm (Root.choices mc b roots idx) ∧
    ∀ m ∈ output, m.flipped = Flip.Dispatch.resultValue fc m.sq b.val.player b.val.opponent := by
  dsimp only [rootRecords]
  split
  · exact RootFrameMir.order_of_permutation mc fc b roots idx _ _ (sort_ok _ _ _).1
  · apply RootFrameMir.order_of_permutation mc fc b roots idx (fun _ => 0)
    unfold MoveList.Generation.records
    simp [List.map_map, RootFrameMir.setKey, MoveList.Generation.entry, Function.comp_def]
    exact .refl _

theorem root_library_order (mc : X86Config) (fc : FlipConfig) (b : Position) (roots : List Bitboard.Square)
    (idx : Nat) (mode : Bitboard.Square → Heuristic) (hint : Option Bitboard.Square) (output : List Move.Move) :
    let choices := Root.choices mc b roots idx
    let records := MoveList.Generation.records fc b.val (Root.remainingMask mc b.val roots idx)
    let set := RootFrameMir.setKey (fun sq => BitVec.ofInt 32 (actualKey (priority mc fc b mode) hint sq))
    3 < choices.length → output.Perm (records.map set) → SortedDesc (fun m : Move.Move => m.value.toInt) output →
      ∃ tie, rootRecords mc fc b roots idx mode tie hint = output := by
  classical
  intro choices records set long perm sorted
  have layout : records = choices.map (fun m => MoveList.Generation.entry fc b.val m.val) := by
    simp [records, choices, MoveList.Generation.records, RootFrameMir.generated_squares]
  have len : (records.map set).length = choices.length := by simp [layout]
  have unique : choices.Nodup := (generated_layout mc b).1.filter _
  have fits : choices.length ≤ 64 := Nat.le_trans (List.length_filter_le _ _) (generated_layout mc b).2
  have uniqueRecords : (records.map set).Nodup := by
    rw [layout, List.map_map]
    apply List.pairwise_map.mpr
    exact unique.imp (fun {a b} different same => different (Subtype.ext (congrArg Move.Move.sq same)))
  obtain ⟨tie, eq⟩ := library_realizable (fun m : Move.Move => m.value.toInt) (records.map set) output (len ▸ long)
    (len ▸ fits) uniqueRecords perm sorted
  refine ⟨tie,?_⟩
  have many : 1 < records.length := by simp only [List.length_map] at len; omega
  change (if 1 < records.length then sortMoves _ tie (records.map set) else records) = output
  rw [ite_eq_left many]
  exact eq

end NeuralReversi.PvsParallel.EndgameOrderMir
