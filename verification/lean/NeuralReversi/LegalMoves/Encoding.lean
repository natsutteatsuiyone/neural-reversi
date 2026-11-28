module
public import NeuralReversi.LegalMoves.Stencils
public import NeuralReversi.LegalMoves.Preservation
public import NeuralReversi.Bitboard.NeonMoves
public import Std.Sat.CNF.Dimacs
public import Std.Tactic.BVDecide.LRAT.Checker

@[expose] public section

namespace NeuralReversi.LegalMoves.MoveBound

open Std.Sat (CNF)
open Bitboard (Bits)
open Bitboard.Moves (getMovesPortable legalMovesTarget targetMoveGenerationContract_proven)

inductive Atom where
  | player (square : Nat)
  | opponent (square : Nat)
  | move (square : Nat)
  | bracket (square : Nat) (d : Direction) (distance : Nat)
  | count (start depth index : Nat)
  deriving DecidableEq

instance : BEq Atom := ⟨fun a b => decide (a = b)⟩
instance : LawfulBEq Atom where
  eq_of_beq := of_decide_eq_true
  rfl := decide_eq_true rfl

abbrev Lit := Std.Sat.Literal Atom
abbrev Rows := List (List Lit)

def clause : List Lit → CNF.Clause Atom
  | [] => .empty
  | (atom, sign) :: rest => (clause rest).add atom sign

private theorem clause_sat (assignment : Atom → Bool) (lits : List Lit) :
    (clause lits).Sat assignment ↔ ∃ l ∈ lits, assignment l.1 = l.2 := by
  induction lits with
  | nil => simp [clause]
  | cons lit rest ih =>
    obtain ⟨atom, sign⟩ := lit
    simp [clause, ih]

def fromRows (rows : Rows) : CNF Atom := ⟨(rows.map clause).toArray⟩

private theorem fromRows_sat (assignment : Atom → Bool) (rows : Rows) :
    (fromRows rows).Sat assignment ↔ ∀ row ∈ rows, ∃ l ∈ row, assignment l.1 = l.2 := by
  simp only [CNF.sat_iff_all_mem_sat, CNF.Internal.mem_iff, fromRows, List.mem_toArray,
    List.mem_map, forall_exists_index, and_imp]
  constructor
  · intro h row hm
    exact (clause_sat _ _).mp (h _ row hm rfl)
  · rintro h _ row hm rfl
    exact (clause_sat _ _).mpr (h row hm)

def keys (target : Nat) : List (Direction × Nat) :=
  ((List.finRange directions.length).flatMap (fun d =>
    (List.range' 2 6).map (fun k => (d, k)))).filter (fun (d, k) => fits target k d)

private theorem mem_keys (target : Nat) (d : Direction) (k : Nat) :
    (d, k) ∈ keys target ↔ 2 ≤ k ∧ k ≤ 7 ∧ fits target k d = true := by
  simp only [keys, List.mem_filter, List.mem_flatMap, List.mem_map, List.mem_range'_1]
  constructor
  · rintro ⟨⟨other, _, j, bounds, eq⟩, fits⟩
    obtain ⟨rfl, rfl⟩ := Prod.mk.inj eq
    exact ⟨bounds.1, by omega, fits⟩
  · rintro ⟨lo, hi, fits⟩
    exact ⟨⟨d, by simp, k, ⟨lo, by omega⟩, rfl⟩, fits⟩

def geometryRows (i : Nat) : Rows :=
  let ks := keys i
  [[(.player i, false), (.opponent i, false)],
    [(.move i, false), (.player i, false)],
    [(.move i, false), (.opponent i, false)],
    (.move i, false) :: ks.map (fun (d, k) => (.bracket i d k, true))] ++
  ks.flatMap (fun (d, k) =>
    [[(.bracket i d k, false), (.player (rayCell i k d), true)]] ++
    (middleSteps k).map (fun j => [(.bracket i d k, false), (.opponent (rayCell i j d), true)]))

def blockCount (played : Nat → Bool) (start depth : Nat) : Nat :=
  (List.range' start (2^depth)).countP played

def boardAssignment (p o : Bits) : Atom → Bool
  | .player i => p.getLsbD i
  | .opponent i => o.getLsbD i
  | .move i => (getMovesPortable p o).getLsbD i
  | .bracket i d k => rayMatches p o i k d
  | .count start depth index =>
    decide (index + 1 ≤ blockCount (getMovesPortable p o).getLsbD start depth)

private theorem geometryRows_sat (p o : Bits) (i : Nat) (disjoint : p &&& o = 0#64)
    (row : List Lit) (hr : row ∈ geometryRows i) :
    ∃ l ∈ row, boardAssignment p o l.1 = l.2 := by
  let a := boardAssignment p o
  have move := ((Bitboard.Moves.portableMovesContract_proven p o i).trans
    (stencilContract_proven p o i)).mp
  simp only [geometryRows, List.mem_append, List.mem_cons, List.not_mem_nil, or_false] at hr
  rcases hr with (rfl | rfl | rfl | rfl) | h
  · by_cases hp : a (.player i) = false
    · exact ⟨(.player i, false), by simp, hp⟩
    · have bit := congrArg (fun x : Bits => x.getLsbD i) disjoint
      have hp' : p.getLsbD i = true := Bool.eq_true_of_ne_false hp
      exact ⟨(.opponent i, false), by simp, by simpa [a, boardAssignment, hp'] using bit⟩
  · by_cases hm : a (.move i) = false
    · exact ⟨(.move i, false), by simp, hm⟩
    · exact ⟨(.player i, false), by simp, (move (Bool.eq_true_of_ne_false hm)).2.1⟩
  · by_cases hm : a (.move i) = false
    · exact ⟨(.move i, false), by simp, hm⟩
    · exact ⟨(.opponent i, false), by simp, (move (Bool.eq_true_of_ne_false hm)).2.2.1⟩
  · by_cases hm : a (.move i) = false
    · exact ⟨(.move i, false), by simp, hm⟩
    · obtain ⟨d, k, lo, hi, shape, colors⟩ := (move (Bool.eq_true_of_ne_false hm)).2.2.2
      have mem := (mem_keys i d k).mpr ⟨lo, hi, shape⟩
      exact ⟨(.bracket i d k, true), List.mem_cons_of_mem _ (List.mem_map_of_mem mem), colors⟩
  · obtain ⟨⟨d, k⟩, _, member⟩ := List.mem_flatMap.mp h
    simp only [List.singleton_append, List.mem_cons, List.mem_map] at member
    by_cases colors : a (.bracket i d k) = false
    · obtain rfl | ⟨_, _, rfl⟩ := member <;> exact ⟨(.bracket i d k, false), by simp, colors⟩
    · have matched : rayMatches p o i k d = true := Bool.eq_true_of_ne_false colors
      obtain ⟨hp, ho⟩ := Bool.and_eq_true_iff.mp matched
      obtain rfl | ⟨j, hmem, rfl⟩ := member
      · exact ⟨(.player (rayCell i k d), true), by simp, hp⟩
      · exact ⟨(.opponent (rayCell i j d), true), by simp, List.all_eq_true.mp ho j hmem⟩

def threshold (start depth index : Nat) : Atom :=
  if depth = 0 then .move start else .count start depth index

def childLiteral (start depth index : Nat) : List Lit :=
  if index < 2^depth then [(threshold start depth index, true)] else []

def mergeRow (start depth i j : Nat) : List Lit :=
  (threshold start (depth + 1) (i + j), false) ::
    (childLiteral start depth i ++ childLiteral (start + 2^depth) depth j)

def mergeRows (start depth : Nat) : Rows :=
  let width := 2^depth
  (List.range (width + 1)).flatMap (fun i =>
    ((List.range (width + 1)).filter (fun j => i + j < 2 * width)).map (mergeRow start depth i))

def countRows (start : Nat) : Nat → Rows
  | 0 => []
  | depth + 1 =>
    countRows start depth ++ countRows (start + 2^depth) depth ++ mergeRows start depth

private theorem blockCount_le (played : Nat → Bool) (start depth : Nat) :
    blockCount played start depth ≤ 2^depth := by
  simpa only [blockCount, List.length_range'] using
    (List.countP_le_length (p := played) (l := List.range' start (2^depth)))

private theorem blockCount_merge (played : Nat → Bool) (start depth : Nat) :
    blockCount played start (depth + 1) =
      blockCount played start depth + blockCount played (start + 2^depth) depth := by
  unfold blockCount
  rw [Nat.pow_succ, Nat.mul_two, ← List.range'_append, List.countP_append]
  simp

private theorem threshold_exact (p o : Bits) (start depth index : Nat) (bound : index < 2^depth) :
    boardAssignment p o (threshold start depth index) =
      decide (index + 1 ≤ blockCount (getMovesPortable p o).getLsbD start depth) := by
  cases depth with
  | zero =>
    have empty : index = 0 := by simpa using bound
    subst index
    cases h : (getMovesPortable p o).getLsbD start <;>
      simp [threshold, boardAssignment, blockCount, h]
  | succ depth => rfl

private theorem mergeRow_sat (p o : Bits) (start depth i j : Nat) (sum : i + j < 2 * 2^depth) :
    ∃ l ∈ mergeRow start depth i j, boardAssignment p o l.1 = l.2 := by
  let played := (getMovesPortable p o).getLsbD
  let a := boardAssignment p o
  let parent := threshold start (depth + 1) (i + j)
  have tBound : i + j < 2^(depth + 1) := by rw [Nat.pow_succ, Nat.mul_comm]; exact sum
  by_cases inactive : a parent = false
  · exact ⟨(parent, false), List.mem_cons_self, inactive⟩
  · have active := Bool.eq_true_of_ne_false inactive
    change boardAssignment p o (threshold start (depth + 1) (i + j)) = true at active
    rw [threshold_exact p o start (depth + 1) (i + j) tBound] at active
    have total := of_decide_eq_true active
    rw [blockCount_merge] at total
    change i + j + 1 ≤ blockCount played start depth +
      blockCount played (start + 2^depth) depth at total
    by_cases left : i < blockCount played start depth
    · have bound := Nat.lt_of_lt_of_le left (blockCount_le played start depth)
      refine ⟨(threshold start depth i, true), ?_, ?_⟩
      · simp [mergeRow, childLiteral, bound]
      · rw [threshold_exact p o start depth i bound]
        exact decide_eq_true left
    · have right : j < blockCount played (start + 2^depth) depth := by omega
      have bound := Nat.lt_of_lt_of_le right (blockCount_le played (start + 2^depth) depth)
      refine ⟨(threshold (start + 2^depth) depth j, true), ?_, ?_⟩
      · simp [mergeRow, childLiteral, bound]
      · rw [threshold_exact p o (start + 2^depth) depth j bound]
        exact decide_eq_true right

private theorem countRows_sat (p o : Bits) (start depth : Nat) (row : List Lit)
    (mem : row ∈ countRows start depth) :
    ∃ l ∈ row, boardAssignment p o l.1 = l.2 := by
  induction depth generalizing start with
  | zero => simp [countRows] at mem
  | succ depth ih =>
    simp only [countRows, List.mem_append] at mem
    rcases mem with (left | right) | merged
    · exact ih start left
    · exact ih (start + 2^depth) right
    · obtain ⟨i, _, mem⟩ := List.mem_flatMap.mp merged
      obtain ⟨j, valid, rfl⟩ := List.mem_map.mp mem
      exact mergeRow_sat p o start depth i j (of_decide_eq_true (List.mem_filter.mp valid).2)

def formula (index : Nat) : CNF Atom :=
  fromRows ([(threshold 0 6 index, true)] ::
    (List.range 64).flatMap geometryRows ++ countRows 0 6)

private theorem count_eq_blockCount (p o : Bits) :
    LegalMoves.count ⟨getMovesPortable p o⟩ = blockCount (getMovesPortable p o).getLsbD 0 6 := by
  unfold LegalMoves.count blockCount
  rw [(Bitboard.populationCountContract_proven _).1, List.range_eq_range']

theorem boardAssignment_sat (p o : Bits) (index : Nat) (bound : index < 64)
    (disjoint : p &&& o = 0#64) (moves : index < LegalMoves.count ⟨getMovesPortable p o⟩) :
    (formula index).Sat (boardAssignment p o) := by
  apply (fromRows_sat _ _).mpr
  intro row hrow
  simp only [List.mem_cons, List.mem_append, List.mem_flatMap] at hrow
  rcases hrow with (rfl | ⟨i, _, h⟩) | h
  · refine ⟨(threshold 0 6 index, true), by simp, ?_⟩
    rw [threshold_exact _ _ _ _ _ bound, ← count_eq_blockCount p o]
    exact decide_eq_true moves
  · exact geometryRows_sat p o i disjoint row h
  · exact countRows_sat p o 0 6 row h

def variables (f : CNF Atom) : List Atom :=
  (f.clauses.toList.flatMap (fun c => c.literals.map Prod.fst)).eraseDups

private theorem var_mem (f : CNF Atom) (a : Atom) (hs : f.VarMem a) :
    a ∈ variables f := by
  obtain ⟨c, mem, vl⟩ := hs
  obtain ⟨sign, ml⟩ := CNF.Clause.VarMem_iff_exists_mem_literals.mp vl
  simp only [variables, List.mem_eraseDups, List.mem_flatMap, List.mem_map]
  exact ⟨c, by simpa using mem, (a, sign), ml, rfl⟩

@[noinline] def relabelWith (labels : List Atom) (f : CNF Atom) : CNF Nat :=
  f.relabel labels.idxOf

def dimacsFormula (index : Nat) : CNF Nat :=
  let f := formula index
  relabelWith (variables f) f

theorem dimacs_unsat_iff (index : Nat) : (dimacsFormula index).Unsat ↔ (formula index).Unsat := by
  apply CNF.unsat_relabel_iff
  intro a b va vb eq
  let labels := variables (formula index)
  have ia : labels.idxOf a < labels.length := List.idxOf_lt_length_iff.mpr (var_mem _ _ va)
  have ib : labels.idxOf b < labels.length := List.idxOf_lt_length_iff.mpr (var_mem _ _ vb)
  have eq' := congrArg (fun i : Nat => labels[i]?) eq
  rw [List.getElem?_eq_getElem ia, List.getElem?_eq_getElem ib,
    List.getElem_idxOf ia, List.getElem_idxOf ib] at eq'
  exact Option.some.inj eq'

def BoundReductionContract : Prop :=
  ∀ index, index < 64 → (dimacsFormula index).Unsat →
    ∀ config p o, p &&& o = 0#64 → LegalMoves.count ⟨legalMovesTarget config p o⟩ ≤ index

theorem target_eq_portable (config) (p o : Bits) (disjoint : p &&& o = 0#64) :
    legalMovesTarget config p o = getMovesPortable p o := by
  apply BitVec.eq_of_getLsbD_eq
  intro i _
  exact Bool.eq_iff_iff.mpr ((targetMoveGenerationContract_proven config p o i disjoint).1.trans
    (Bitboard.Moves.portableMovesContract_proven p o i).symm)

theorem boundReductionContract_proven : BoundReductionContract := by
  intro index valid impossible config p o disjoint
  rw [target_eq_portable config p o disjoint]
  apply Nat.le_of_not_gt
  intro above
  have unsat := (dimacs_unsat_iff index).mp impossible
  exact (CNF.unsat_iff_not_sat.mp unsat (boardAssignment p o)) (boardAssignment_sat p o index valid disjoint above)

def CertifiedBoundContract : Prop :=
  ∀ index proof, index < 64 → Std.Tactic.BVDecide.LRAT.check proof (dimacsFormula index) = true →
    ∀ config p o, p &&& o = 0#64 → LegalMoves.count ⟨legalMovesTarget config p o⟩ ≤ index

theorem certifiedBoundContract_proven : CertifiedBoundContract := by
  intro index proof valid checked
  exact boundReductionContract_proven index valid (Std.Tactic.BVDecide.LRAT.check_sound proof _ checked)

def witnessBoards : List (Bits × Bits) :=
  [(0x00043e2480680404#64, 0x006a400a44007a02#64),
   (0x00816600041c1020#64, 0x0066804a1a226a00#64)]

def BoundWitnessContract : Prop :=
  ∀ config p o, (p, o) ∈ witnessBoards →
    p &&& o = 0#64 ∧ LegalMoves.count ⟨legalMovesTarget config p o⟩ = 35

theorem boundWitnessContract_proven : BoundWitnessContract := by
  intro config p o witness
  have h : ∀ pp ∈ witnessBoards, pp.1 &&& pp.2 = 0#64 ∧
      LegalMoves.count ⟨getMovesPortable pp.1 pp.2⟩ = 35 := by
    intro pp mem
    simp only [witnessBoards, List.mem_cons, List.not_mem_nil, or_false] at mem
    rcases mem with rfl | rfl <;> decide
  obtain ⟨disjoint, count⟩ := h (p, o) witness
  exact ⟨disjoint, target_eq_portable config p o disjoint ▸ count⟩

end NeuralReversi.LegalMoves.MoveBound
