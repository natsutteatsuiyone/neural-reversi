import NeuralReversi.MoveList.Initialization
import NeuralReversi.Board.Move.Wipeout
import NeuralReversi.Flip.Avx512.WidePair

namespace NeuralReversi.MoveList.SmallGenerate

private def batches : Bool → List α → List (List α)
  | _, [] => []
  | false, x :: xs => [x] :: batches false xs
  | true, [x] => [[x]]
  | true, x :: y :: xs => [x, y] :: batches true xs

private theorem flatten_batches (paired : Bool) (xs : List α) :
    (batches paired xs).flatten = xs := by
  induction paired, xs using batches.induct <;> simp_all [batches]

private def visit (wipeout : α → Bool) : List (List α) → Bool × List α
  | [] => (true, [])
  | batch :: rest =>
    if batch.any wipeout then (false, []) else
      let tail := visit wipeout rest
      (tail.1, batch ++ tail.2)

def generate (paired : Bool) (wipeout : α → Bool) (xs : List α) : Bool × List α :=
  visit wipeout (batches paired xs)

private theorem visit_spec (wipeout : α → Bool) (bs : List (List α)) :
    let r := visit wipeout bs
    (r.1 = true ↔ ∀ x ∈ bs.flatten, wipeout x = false) ∧
    (r.1 = true → r.2 = bs.flatten) ∧ (r.2 <+: bs.flatten) ∧
    (∀ x ∈ r.2, wipeout x = false) := by
  induction bs with
  | nil => simp [visit]
  | cons batch rest ih =>
    dsimp only at ih ⊢
    obtain ⟨ok, complete, pre, clean⟩ := ih
    rw [visit, List.flatten_cons]
    cases hit : batch.any wipeout with
    | false =>
      have allClear : ∀ x ∈ batch, wipeout x = false := by
        simpa using List.any_eq_false.mp hit
      simp only [Bool.false_eq_true, ↓reduceIte, List.prefix_append_right_inj, pre, true_and,
        List.mem_append, or_imp, forall_and]
      exact ⟨⟨fun h => ⟨allClear, ok.mp h⟩, fun h => ok.mpr h.2⟩,
        fun h => congrArg (batch ++ ·) (complete h), allClear, clean⟩
    | true =>
      obtain ⟨x, member, flagged⟩ := List.any_eq_true.mp hit
      simp
      exact ⟨x, Or.inl member, flagged⟩

theorem generate_spec (paired : Bool) (wipeout : α → Bool) (xs : List α) :
    let r := generate paired wipeout xs
    (r.1 = true ↔ ∀ x ∈ xs, wipeout x = false) ∧
    (r.1 = true → r.2 = xs) ∧ (r.2 <+: xs) ∧
    (∀ x ∈ r.2, wipeout x = false) ∧ r.2.length ≤ xs.length := by
  have h := visit_spec wipeout (batches paired xs)
  rw [flatten_batches] at h
  exact ⟨h.1, h.2.1, h.2.2.1, h.2.2.2, h.2.2.1.length_le⟩

example : generate true (· == (2 : Nat)) [1, 2, 3] = (false, []) ∧
    generate false (· == (2 : Nat)) [1, 2, 3] = (false, [1]) := by decide

open Bitboard.Moves (X86Config)
open Board (Board)

def candidates (mc : X86Config) (b : Board) : List (Fin 64) :=
  (Bitboard.setIndices (Board.getMoves mc b)).attach.map fun ⟨i, mem⟩ =>
    ⟨i, List.mem_range.mp (List.mem_filter.mp mem).1⟩

theorem candidates_eq_filter (mc : X86Config) (b : Board) :
    candidates mc b = (List.finRange 64).filter (fun sq => (Board.getMoves mc b).getLsbD sq.val) := by
  apply (List.map_inj_right (f := Fin.val) (fun _ _ => Fin.ext)).mp
  have indices : (List.finRange 64).map Fin.val = List.range 64 := by decide
  change (candidates mc b).map Fin.val =
    ((List.finRange 64).filter ((Board.getMoves mc b).getLsbD ∘ Fin.val)).map Fin.val
  rw [← List.filter_map, indices]
  simp [candidates, Bitboard.setIndices, List.map_map, Function.comp_def]

private theorem mem_candidates (mc : X86Config) (b : Board) (sq : Fin 64) :
    sq ∈ candidates mc b ↔ (Board.getMoves mc b).getLsbD sq.val = true := by
  constructor
  · intro h
    obtain ⟨⟨_, mem⟩, _, rfl⟩ := List.mem_map.mp h
    exact (List.mem_filter.mp mem).2
  · intro bit
    exact List.mem_map.mpr ⟨⟨sq.val, List.mem_filter.mpr ⟨List.mem_range.mpr sq.isLt, bit⟩⟩, by simp, rfl⟩

private theorem candidates_fit (mc : X86Config) (b : Board) :
    (candidates mc b).length ≤ (Board.emptyCount b).toNat := by
  simp only [candidates, List.length_map, List.length_attach, Bitboard.setIndices,
    ← List.countP_eq_length_filter, (Board.squareCountsContract_proven b).2.2]
  apply List.countP_mono_left
  intro i _ bit
  obtain ⟨_, player, opponent, _⟩ := (Board.getMoves_exact mc b i).mp bit
  simp [player, opponent]

private def slots (records : List α) (fallback : α) : Initialization.Memory α :=
  Initialization.scalar (fun i => records[i]?.getD fallback) records.length 0 (fun _ => none)

private theorem slots_exact (records : List α) (fallback : α) (i : Nat) :
    slots records fallback i = records[i]? := by
  rw [slots, Initialization.scalar_prefix]
  split
  next hit => simp [List.getElem?_eq_getElem hit]
  next miss => simp [List.getElem?_eq_none_iff.mpr (by omega : records.length ≤ i)]

theorem widePair_eq_generatedMasks (fc : Flip.Dispatch.X86Config) (b : Board)
    (wf : Board.wellFormed b) (first second : Fin 64) :
    Flip.Avx512.WidePair.flipPair first second b.player b.opponent =
      (Flip.Dispatch.resultValue fc first b.player b.opponent,
       Flip.Dispatch.resultValue fc second b.player b.opponent) := by
  rw [Flip.Avx512.WidePair.pairContract_proven,
    Flip.Dispatch.value_disjoint_scalar _ _ _ _ wf,
    Flip.Dispatch.value_disjoint_scalar _ _ _ _ wf]

def GenerationContract : Prop :=
  ∀ (mc : X86Config) (fc : Flip.Dispatch.X86Config) (paired checked overflow : Bool) (b : Board),
    Board.wellFormed b → (Board.emptyCount b).toNat ≤ 12 →
    let xs := candidates mc b
    let flipped := fun sq => Flip.Dispatch.resultValue fc sq b.player b.opponent
    let r := generate paired (fun sq => flipped sq == b.opponent) xs
    let records := r.2.map fun sq => (sq, Board.place b (flipped sq) sq)
    (Bitboard.count (Board.getMoves mc b)).toNat ≤ 12 ∧ records.length ≤ 12 ∧ (r.2 <+: xs) ∧
      (∀ i, slots records (0, b) i = records[i]?) ∧
      (∀ sq child, (sq, child) ∈ records →
        flipped sq ≠ b.opponent ∧
        Board.makeMoveWithFlippedAny checked overflow b (flipped sq) sq = .ok child ∧
        Board.makeMove fc checked overflow b sq = .ok child) ∧
      (r.1 = true ↔ ∀ sq ∈ xs, flipped sq ≠ b.opponent) ∧
      (r.1 = true → r.2 = xs) ∧
      (r.1 = false → ∃ sq ∈ xs, flipped sq = b.opponent ∧ ∃ child,
        Board.makeMove fc checked overflow b sq = .ok child ∧ Board.isGameOver mc child = true ∧
        ScoreRefresh.BoardBridge.solve checked child (Board.emptyCount child) = .ok ScoreRefresh.scoreMin)

theorem generationContract_proven : GenerationContract := by
  intro mc fc paired checked overflow b wf depth
  dsimp only
  let xs := candidates mc b
  let flipped := fun sq => Flip.Dispatch.resultValue fc sq b.player b.opponent
  have ⟨clear, complete, prefixOf, nonWipeout, bound⟩ :=
    generate_spec paired (fun sq => flipped sq == b.opponent) xs
  have fits := Nat.le_trans (candidates_fit mc b) depth
  have count : xs.length = (Bitboard.count (Board.getMoves mc b)).toNat := by
    simpa only [xs, candidates, List.length_map, List.length_attach] using
      (Bitboard.iteratorEnumerationContract_proven (Board.getMoves mc b) false).2.2.2.1
  refine ⟨count ▸ fits, ?_, prefixOf, fun i => slots_exact _ _ i, ?_, ?_, complete, ?_⟩
  · simpa only [List.length_map] using Nat.le_trans bound fits
  · intro sq child mem
    obtain ⟨s, hs, eq⟩ := List.mem_map.mp mem
    cases eq
    refine ⟨by simpa using nonWipeout sq hs, ?_⟩
    exact Board.precomputedMoveContract_proven mc fc checked overflow b sq wf
      ((mem_candidates mc b sq).mp (prefixOf.subset hs))
  · simpa using clear
  · intro stopped
    have bad : ¬ ∀ sq ∈ xs, flipped sq ≠ b.opponent := by
      intro clean
      have ok := clear.mpr (by simpa using clean)
      rw [stopped] at ok
      contradiction
    obtain ⟨sq, hs, heq⟩ := Classical.byContradiction (p := ∃ sq ∈ xs, flipped sq = b.opponent)
      (fun no => bad (fun s mem eq => no ⟨s, mem, eq⟩))
    exact ⟨sq, hs, heq, Board.wipeoutScoreContract_proven mc fc checked overflow b sq wf
      ((mem_candidates mc b sq).mp hs) heq⟩

end NeuralReversi.MoveList.SmallGenerate
