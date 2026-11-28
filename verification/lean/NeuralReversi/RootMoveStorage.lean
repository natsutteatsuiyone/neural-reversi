module
public import NeuralReversi.PvsParallel.Root

@[expose] public section

namespace NeuralReversi.RootMoveStorage

open Bitboard (Bits Square)
open PvsMoveLoop (scoreInf)

structure Row where
  square : Square
  score : Int
  average : Int
  meanSquared : Int
  pv : List Square

def fresh (square : Square) : Row := ⟨square, -scoreInf, -scoreInf, 0, []⟩

def invalidated (row : Row) : Row := { row with score := -scoreInf }

def resetFrom (rows : List Row) (index : Nat) : List Row :=
  rows.take index ++ (rows.drop index).map invalidated

def metadata (row : Row) : Square × Int × Int × List Square :=
  (row.square, row.average, row.meanSquared, row.pv)

def ResetContract : Prop :=
  ∀ rows index,
    let reset := resetFrom rows index
    reset.take index = rows.take index ∧
    reset.drop index = (rows.drop index).map invalidated ∧
    reset.map metadata = rows.map metadata ∧
    (∀ row ∈ reset.drop index, row.score = -scoreInf) ∧
    (rows.length ≤ index → reset = rows)

theorem resetContract_proven : ResetContract := by
  intro rows index reset
  by_cases within : index ≤ rows.length
  · have size : (rows.take index).length = index := by simp [Nat.min_eq_left within]
    have front : reset.take index = rows.take index := List.take_left' size
    have suffix : reset.drop index = (rows.drop index).map invalidated := List.drop_left' size
    refine ⟨front, suffix, ?_, ?_, ?_⟩
    · simp only [reset, resetFrom, List.map_append, List.map_map]
      have unchanged : metadata ∘ invalidated = metadata := by rfl
      rw [unchanged, ← List.map_append, List.take_append_drop]
    · intro row inSuffix
      rw [suffix] at inSuffix
      obtain ⟨old, _, rfl⟩ := List.mem_map.mp inSuffix
      rfl
    · intro outside
      simp [reset, resetFrom, List.take_of_length_le outside, List.drop_eq_nil_of_le outside]
  · have outside : rows.length ≤ index := by omega
    simp [reset, resetFrom, List.take_of_length_le outside, List.drop_eq_nil_of_le outside]

structure Storage where
  rows : List Row
  pvIndex : Nat

def start (squares : List Square) : Storage := ⟨squares.map fresh, 0⟩

def withIndex (state : Storage) (index : Nat) : Storage := { state with pvIndex := index }

def remaining (state : Storage) (readIndex : Nat) : Bits :=
  PvsParallel.Root.mask ((state.rows.drop readIndex).map Row.square)

def LookupContract : Prop :=
  (∀ squares,
    let state := start squares
    state.pvIndex = 0 ∧ state.rows.map Row.square = squares ∧
    ∀ row ∈ state.rows, row.average = -scoreInf ∧ row.score = -scoreInf ∧
      row.meanSquared = 0 ∧ row.pv = []) ∧
  ∀ state index square,
    let indexed := withIndex state index
    indexed.rows = state.rows ∧ indexed.pvIndex = index ∧
    (indexed.rows[indexed.pvIndex]? = none ↔ state.rows.length ≤ index) ∧
    ((remaining indexed index).getLsbD square.val = true ↔
      ∃ row ∈ state.rows.drop index, row.square = square)

theorem lookupContract_proven : LookupContract := by
  constructor
  · intro squares
    simp [start, fresh, List.map_map, Function.comp_def]
  · intro state index square
    refine ⟨rfl, rfl, List.getElem?_eq_none_iff, ?_⟩
    change (PvsParallel.Root.mask ((state.rows.drop index).map Row.square)).getLsbD square.val = true ↔ _
    rw [PvsParallel.Root.mask_bit]
    exact List.mem_map

def sortRows (rows : List Row) (index : Nat) : List Row :=
  rows.take index ++ (rows.drop index).mergeSort (fun a b => decide (b.score ≤ a.score))

def SortRowsContract : Prop :=
  ∀ rows index,
    let sorted := sortRows rows index
    sorted.Perm rows ∧ sorted.take index = rows.take index ∧
    (sorted.drop index).Pairwise (fun a b => b.score ≤ a.score) ∧
    (∀ a b, a.score = b.score → [a, b].Sublist (rows.drop index) →
      [a, b].Sublist (sorted.drop index)) ∧
    (rows.length ≤ index → sorted = rows)

theorem sortRowsContract_proven : SortRowsContract := by
  intro rows index sorted
  let cmp := fun (a b : Row) => decide (b.score ≤ a.score)
  have trans (a b c : Row) (ab : cmp a b) (bc : cmp b c) : cmp a c := by
    simp only [cmp, decide_eq_true_eq] at ab bc ⊢
    omega
  have total (a b : Row) : cmp a b || cmp b a := by
    simp only [cmp, Bool.or_eq_true, decide_eq_true_eq]
    exact Int.le_total _ _
  have unchanged (outside : rows.length ≤ index) : sorted = rows := by
    simp [sorted, sortRows, List.take_of_length_le outside, List.drop_eq_nil_of_le outside]
  by_cases within : index ≤ rows.length
  · have size : (rows.take index).length = index := by simp [Nat.min_eq_left within]
    have front : sorted.take index = rows.take index := List.take_left' size
    have back : sorted.drop index = (rows.drop index).mergeSort cmp := List.drop_left' size
    have perm : sorted.Perm rows :=
      (List.Perm.append_left (rows.take index) (List.mergeSort_perm _ cmp)).trans
        (List.Perm.of_eq (List.take_append_drop index rows))
    refine ⟨perm, front, ?_, ?_, unchanged⟩
    · rw [back]
      exact (List.pairwise_mergeSort trans total (rows.drop index)).imp of_decide_eq_true
    · intro a b tie before
      rw [back]
      exact List.pair_sublist_mergeSort trans total (by simp [cmp, tie]) before
  · have outside : rows.length ≤ index := by omega
    rw [unchanged outside]
    simp [List.drop_eq_nil_of_le outside]

end NeuralReversi.RootMoveStorage
