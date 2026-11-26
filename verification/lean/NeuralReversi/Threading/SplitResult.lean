import NeuralReversi.MoveList.Concurrent
import NeuralReversi.PvsMoveLoop

namespace NeuralReversi.Threading.SplitResult

open PvsMoveLoop (Bound bestValue scoreMax)
open MoveList.Concurrent (Claims advance)

structure Report where
  index : Nat
  searchedAlpha : Int
  score : Int

inductive Merge (value : Nat → Int) (alpha beta : Int) : Int → List Report → Int → Prop
  | done : Merge value alpha beta best [] best
  | accept : report.searchedAlpha ≤ max alpha best →
      Bound (value report.index) report.searchedAlpha beta report.score →
      Merge value alpha beta (max best report.score) rest final →
      Merge value alpha beta best (report :: rest) final

private theorem Merge.seed_le (merge : Merge value alpha beta seed reports final) : seed ≤ final := by
  induction merge with
  | done => exact Int.le_refl _
  | accept _ _ _ ih => exact Int.le_trans (Int.le_max_left _ _) ih

private theorem Merge.upper (merge : Merge value alpha beta seed reports final)
    (low : final < beta) (member : report ∈ reports) : value report.index ≤ final := by
  induction merge with
  | done => simp at member
  | accept _ bound rest ih =>
    rcases List.mem_cons.mp member with same | later
    · subst report
      have scored := Int.le_trans (Int.le_max_right _ _) rest.seed_le
      exact Int.le_trans (bound.1 (Int.lt_of_le_of_lt scored low)) scored
    · exact ih low later

private theorem Merge.lower (merge : Merge value alpha beta seed reports final)
    (base : alpha < seed → seed ≤ truth)
    (members : ∀ report ∈ reports, value report.index ≤ truth) :
    alpha < final → final ≤ truth := by
  induction merge with
  | done => exact base
  | @accept best rest final report earlier bound tail ih =>
    apply ih
    · intro improved
      by_cases same : report.score ≤ best
      · rw [Int.max_eq_left same] at improved ⊢
        exact base improved
      · have better : best < report.score := by omega
        rw [Int.max_eq_right (Int.le_of_lt better)] at improved ⊢
        exact Int.le_trans (bound.2 (by omega)) (members report (by simp))
    · intro other member
      exact members other (List.mem_cons_of_mem _ member)

def slots (start len : Nat) : List Nat :=
  List.range' start (len - start)

theorem mem_slots : i ∈ slots start len ↔ start ≤ i ∧ i < len := by
  simp only [slots, List.mem_range', Nat.one_mul]
  constructor
  · rintro ⟨offset, beforeEnd, rfl⟩
    omega
  · intro ⟨afterStart, beforeEnd⟩
    exact ⟨i - start, by omega, by omega⟩

def nodeValue (value : Nat → Int) (start len : Nat) (prefixValue : Int) : Int :=
  bestValue value (slots start len) prefixValue

theorem bestValue_append (value : Move → Int) (first rest : List Move) (seed : Int) :
    bestValue value (first ++ rest) seed = bestValue value rest (bestValue value first seed) := by
  induction first generalizing seed with
  | nil => rfl
  | cons move tail ih => exact ih (max seed (value move))

theorem suffixLabels (first rest : List Move) (fallback : Move) :
    ((slots first.length (first ++ rest).length).map
      (fun i => (first ++ rest)[i]?.getD fallback)) = rest := by
  apply List.ext_getElem
  · simp [slots]
  · intro i _ hi
    simp only [List.getElem_map, slots, List.getElem_range', Nat.one_mul]
    rw [List.getElem?_append_right (by omega)]
    simp [List.getElem?_eq_getElem hi]

theorem partitionLabels (first rest : List Move) (fallback : Move) (value : Move → Int) (seed : Int) :
    nodeValue (fun i => value ((first ++ rest)[i]?.getD fallback))
      first.length (first ++ rest).length (bestValue value first seed) =
      bestValue value (first ++ rest) seed := by
  rw [nodeValue, ← PvsMoveLoop.bestValue_map, suffixLabels, bestValue_append]

theorem Merge.bound_of_coverage (merge : Merge value alpha beta seed reports final)
    (base : Bound prefixValue alpha beta seed)
    (caps : ∀ i ∈ slots start len, value i ≤ scoreMax) (cap : prefixValue ≤ scoreMax)
    (valid : ∀ report ∈ reports, report.index ∈ slots start len)
    (covered : beta ≤ final ∨ scoreMax ≤ final ∨
      ∀ i ∈ slots start len, ∃ report ∈ reports, report.index = i) :
    Bound (nodeValue value start len prefixValue) alpha beta final := by
  constructor
  · intro low
    rcases covered with high | maximal | allDone
    · omega
    · exact Int.le_trans (PvsMoveLoop.bestValue_le value _ _ _ cap caps) maximal
    · apply PvsMoveLoop.bestValue_le
      · exact Int.le_trans (base.1 (Int.lt_of_le_of_lt merge.seed_le low)) merge.seed_le
      · intro index member
        obtain ⟨report, published, same⟩ := allDone index member
        exact same ▸ merge.upper low published
  · apply merge.lower
    · intro higher
      exact Int.le_trans (base.2 higher) (PvsMoveLoop.bestValue_ge_seed value _ _)
    · intro report member
      exact PvsMoveLoop.bestValue_ge_move value _ _ (valid report member)

def SplitResultContract : Prop :=
  ∀ (w len : Nat) (start last : BitVec w) (issued : List (BitVec w))
    (value : Nat → Int) (prefixValue seed alpha beta final : Int) (reports : List Report),
    len < 2 ^ w → Claims len start issued last →
    Bound prefixValue alpha beta seed →
    (∀ i ∈ slots start.toNat len, value i ≤ scoreMax) → prefixValue ≤ scoreMax →
    Merge value alpha beta seed reports final →
    (∀ report ∈ reports, report.index ∈ slots start.toNat len) →
    ((advance len last = none ∧
        ∀ i ∈ issued.map BitVec.toNat, ∃ report ∈ reports, report.index = i) ∨
      beta ≤ final ∨ scoreMax ≤ final) →
    Bound (nodeValue value start.toNat len prefixValue) alpha beta final

theorem splitResultContract_proven : SplitResultContract := by
  intro w len start last issued value prefixValue seed alpha beta final reports capacity claims
    base caps cap merge valid covered
  apply merge.bound_of_coverage base caps cap valid
  rcases covered with ⟨exhausted, allDone⟩ | high | maximal
  · have allIssued := (MoveList.Concurrent.exhaustionContract_proven w len start last issued
      capacity claims exhausted).1
    exact Or.inr (Or.inr (by simpa only [allIssued, slots] using allDone))
  · exact Or.inl high
  · exact Or.inr (Or.inl maximal)

example :
    Claims 1 (0#32) [0#32] (1#32) ∧ advance 1 (1#32) = none ∧
      ¬ Bound (nodeValue (fun _ => 4) 0 1 0) 0 16 0 := by
  refine ⟨.take rfl .done, rfl, ?_⟩
  intro result
  have tooLow := result.1 (by decide)
  change (4 : Int) ≤ 0 at tooLow
  omega

end NeuralReversi.Threading.SplitResult
