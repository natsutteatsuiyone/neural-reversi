module
public import NeuralReversi.PublicationMir.Program

@[expose] public section

namespace NeuralReversi.PublicationMir

open RootMoveStorage (Row)

def asSnapshot (it : Iteration) : SearchPublication.Snapshot :=
  ⟨it.rows.map project, ⟨it.depth, it.selectivity⟩, it.endgame⟩

theorem intoResult_fromSnapshot (it : Iteration) (nodes : Nat) (fallback : Unit → SearchResult) :
    intoResult (some it) nodes fallback =
      (SearchFallback.fromSnapshot (asSnapshot it) nodes).map SearchResult.bestMove := by
  cases it with
  | mk rows depth selectivity endgame =>
    cases rows <;> rfl

def completed (sort : SliceSort) (rows : List Row) (depth sel : Nat) (endgame : Bool)
    (nodes : Nat) (fallback : Unit → SearchResult) : Option SearchResult := do
  let (last, _) ← record sort rows depth sel endgame
  intoResult last nodes fallback

theorem completed_fromSnapshot (sort : SliceSort) (rows : List Row) (depth sel : Nat) (endgame : Bool)
    (nodes : Nat) (fallback : Unit → SearchResult) :
    completed sort rows depth sel endgame nodes fallback =
      (SearchFallback.fromSnapshot (asSnapshot (capture sort rows depth sel endgame)) nodes).map
        SearchResult.bestMove := by
  generalize saved : capture sort rows depth sel endgame = it
  unfold completed record
  rw [saved]
  cases it with
  | mk moves d s ending =>
    cases moves <;> rfl

private theorem headMax {r : SearchPublication.Row} {tail : List SearchPublication.Row}
    (order : (r :: tail).Pairwise (fun a b => b.score ≤ a.score)) :
    ∀ x ∈ r :: tail, x.score ≤ r.score := by
  intro x present
  rcases List.mem_cons.mp present with rfl | rest
  · exact Int.le_refl _
  · exact (List.pairwise_cons.mp order).1 x rest

/-- The public score agrees with the model capture for every input order and tie pattern,
including an empty vector where the Rust `expect` has no normal return. -/
theorem completedScore (sort : SliceSort) (rows : List Row) (depth sel : Nat) (endgame : Bool)
    (nodes : Nat) (fallback : Unit → SearchResult) :
    (completed sort rows depth sel endgame nodes fallback).map score =
      (SearchFallback.fromSnapshot (SearchPublication.capture (rows.map project) ⟨depth, sel⟩ endgame)
        nodes).map (fun r => some r.best.score) := by
  rw [completed_fromSnapshot]
  let rust := asSnapshot (capture sort rows depth sel endgame)
  let model := SearchPublication.capture (rows.map project) ⟨depth, sel⟩ endgame
  have libPerm := (sort.permutation key rows).map project
  have spec := SearchPublication.captureContract_proven (rows.map project) ⟨depth, sel⟩ endgame
  have perm : rust.rows.Perm model.rows := libPerm.trans spec.2.2.1.symm
  have leftOrder : rust.rows.Pairwise (fun a b => b.score ≤ a.score) := by
    apply List.pairwise_map.mpr
    exact sort.ordered key rows
  have rightOrder : model.rows.Pairwise (fun a b => b.score ≤ a.score) := spec.2.2.2.1
  change ((SearchFallback.fromSnapshot rust nodes).map SearchResult.bestMove).map score =
    (SearchFallback.fromSnapshot model nodes).map _
  unfold SearchFallback.fromSnapshot
  cases left : rust.rows with
  | nil =>
    have right : model.rows = [] := List.length_eq_zero_iff.mp (by simpa [left] using perm.length_eq.symm)
    rw [right]
    rfl
  | cons a as =>
    cases right : model.rows with
    | nil =>
      have len := perm.length_eq
      simp [left, right] at len
    | cons b bs =>
      rw [left, right] at perm
      rw [left] at leftOrder
      rw [right] at rightOrder
      have equal := Int.le_antisymm (headMax rightOrder a (perm.mem_iff.mp (by simp)))
        (headMax leftOrder b (perm.mem_iff.mpr (by simp)))
      change some (some a.score) = some (some b.score)
      rw [equal]

theorem completedScore_of_report (sort : SliceSort) (rows : List Row) (depth sel : Nat) (endgame : Bool)
    (nodes : Nat) (fallback : Unit → SearchResult) {report : SearchFallback.Report}
    (produced : SearchFallback.fromSnapshot
      (SearchPublication.capture (rows.map project) ⟨depth, sel⟩ endgame) nodes = some report) :
    ∃ result, completed sort rows depth sel endgame nodes fallback = some result ∧
      score result = some report.best.score := by
  have eq := completedScore sort rows depth sel endgame nodes fallback
  rw [produced] at eq
  cases h : completed sort rows depth sel endgame nodes fallback with
  | none => rw [h] at eq; contradiction
  | some result =>
    rw [h] at eq
    exact ⟨result, rfl, Option.some.inj eq⟩

end NeuralReversi.PublicationMir
