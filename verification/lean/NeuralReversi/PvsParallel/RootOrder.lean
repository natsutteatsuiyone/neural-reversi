import NeuralReversi.PvsParallel.Root

namespace NeuralReversi.PvsParallel.RootOrder

open Aspiration (Window)
open Bitboard (Square)
open PvsMoveLoop (Bound scoreInf)
open Threading.SplitWindow (State publish)

structure Report where
  square : Square
  searchedAlpha : Int
  score : Int

def visibleScore (first : Bool) (alpha score : Int) : Int :=
  if first = true ∨ alpha < score then score else -scoreInf

def writeScore (scores : Square → Int) (square : Square) (score : Int) : Square → Int :=
  fun query => if query = square then score else scores query

inductive Records (candidate : List Square) (value : Square → Int) (original beta : Int) :
    List Square → State → (Square → Int) → Prop
  | start : Records candidate value original beta [] ⟨original, -scoreInf⟩ (fun _ => -scoreInf)
  | push {seen : List Square} {state : State} {scores : Square → Int} (report : Report) :
      Records candidate value original beta seen state scores →
      report.square ∈ candidate → report.square ∉ seen →
      report.searchedAlpha ≤ state.alpha →
      Bound (value report.square) report.searchedAlpha beta report.score →
      Records candidate value original beta (report.square :: seen) (publish true beta state report.score)
        (writeScore scores report.square (visibleScore seen.isEmpty state.alpha report.score))

private theorem first_alpha (reports : Records candidate value original beta seen state scores)
    (first : seen.isEmpty = true) : state.alpha = original := by
  cases reports <;> simp_all

private structure Invariant (candidate : List Square) (value : Square → Int) (original : Int)
    (seen : List Square) (state : State) (scores : Square → Int) : Prop where
  alpha : state.alpha = max original state.best
  floor : -scoreInf ≤ state.best
  upper : ∀ sq, scores sq ≤ state.best
  exactScore : ∀ sq, original < scores sq → value sq = scores sq
  unseen : ∀ sq, sq ∉ seen → scores sq = -scoreInf
  winner : original < state.best → ∃ sq ∈ candidate, scores sq = state.best

private theorem records_valid (lo : -scoreInf ≤ original)
    (reports : Records candidate value original beta seen state scores) (below : state.best < beta) :
    Invariant candidate value original seen state scores := by
  induction reports with
  | start =>
    exact ⟨(Int.max_eq_left lo).symm, Int.le_refl _, fun _ => Int.le_refl _,
      fun _ h => False.elim (Int.not_lt.mpr lo h), fun _ _ => rfl,
      fun h => False.elim (Int.not_lt.mpr lo h)⟩
  | @push seen state scores report prior member fresh ordered sound ih =>
    let best := max state.best report.score
    let recorded := visibleScore seen.isEmpty state.alpha report.score
    change best < beta at below
    have oldLe : state.best ≤ best := Int.le_max_left _ _
    have scoreLe : report.score ≤ best := Int.le_max_right _ _
    have prev := ih (Int.lt_of_le_of_lt oldLe below)
    have scoreLo := Int.le_trans prev.floor oldLe
    have scoreBelow := Int.lt_of_le_of_lt scoreLe below
    have alpha : (publish true beta state report.score).alpha = max original best := by
      change (if state.best < report.score ∧ state.alpha < report.score ∧ true = true ∧
        report.score < beta then report.score else state.alpha) = max original best
      simp only [and_true, scoreBelow]
      have := prev.alpha
      dsimp only [best]
      split <;> omega
    have recordBound : recorded ≤ best := by
      dsimp only [recorded, visibleScore]
      split
      · exact scoreLe
      · exact scoreLo
    have recordExact : original < recorded → value report.square = recorded := by
      by_cases keep : seen.isEmpty = true ∨ state.alpha < report.score
      · simp only [recorded, visibleScore, ite_eq_left keep]
        intro high
        have usable : report.searchedAlpha < report.score := by
          rcases keep with first | raises
          · have initial := first_alpha prior first
            omega
          · omega
        exact Int.le_antisymm (sound.1 scoreBelow) (sound.2 usable)
      · dsimp only [recorded, visibleScore]
        rw [ite_eq_right keep]
        exact fun high => False.elim (Int.not_lt.mpr lo high)
    refine ⟨alpha, scoreLo, ?_, ?_, ?_, ?_⟩
    · intro sq
      change writeScore scores report.square recorded sq ≤ best
      by_cases same : sq = report.square
      · simp [writeScore, same, recordBound]
      · simpa [writeScore, same] using Int.le_trans (prev.upper sq) oldLe
    · intro sq high
      change original < writeScore scores report.square recorded sq at high
      change value sq = writeScore scores report.square recorded sq
      by_cases same : sq = report.square
      · subst sq
        simp only [writeScore] at high ⊢
        exact recordExact high
      · simp only [writeScore, ite_eq_right same] at high ⊢
        exact prev.exactScore sq high
    · intro sq unseen
      have different : sq ≠ report.square := fun same => unseen (List.mem_cons.mpr (.inl same))
      have older : sq ∉ seen := fun member => unseen (List.mem_cons.mpr (.inr member))
      simpa [writeScore, different] using prev.unseen sq older
    · intro high
      change original < best at high
      by_cases improves : state.best < report.score
      · have same : best = report.score := Int.max_eq_right (Int.le_of_lt improves)
        have raises : state.alpha < report.score := by have := prev.alpha; omega
        refine ⟨report.square, member, ?_⟩
        change writeScore scores report.square recorded report.square = best
        rw [same]
        simp [writeScore, recorded, visibleScore, raises]
      · have same : best = state.best := Int.max_eq_left (Int.le_of_not_gt improves)
        obtain ⟨sq, allowed, winner⟩ := prev.winner (by omega)
        have different : sq ≠ report.square := by
          intro equal
          have absent := prev.unseen report.square fresh
          rw [equal, absent] at winner
          omega
        refine ⟨sq, allowed, ?_⟩
        change writeScore scores report.square recorded sq = best
        simpa [writeScore, different, same] using winner

def sortFrom (roots : List Square) (idx : Nat) (score : Square → Int) : List Square :=
  roots.take idx ++ (roots.drop idx).mergeSort (fun a b => decide (score b ≤ score a))

def SortContract : Prop :=
  ∀ (roots : List Square) (idx : Nat) (score : Square → Int),
    idx ≤ roots.length →
    let reordered := sortFrom roots idx score
    reordered.Perm roots ∧ reordered.take idx = roots.take idx ∧
      (reordered.drop idx).Perm (roots.drop idx) ∧
      (reordered.drop idx).Pairwise (fun a b => score b ≤ score a) ∧
      ∀ a b, score a = score b → [a, b].Sublist (roots.drop idx) → [a, b].Sublist (reordered.drop idx)

theorem sortContract_proven : SortContract := by
  intro roots idx score inside reordered
  let comparator := fun a b => decide (score b ≤ score a)
  have suffix := List.mergeSort_perm (roots.drop idx) comparator
  have size : (roots.take idx).length = idx := by simp [Nat.min_eq_left inside]
  have front : reordered.take idx = roots.take idx := List.take_left' size
  have back : reordered.drop idx = (roots.drop idx).mergeSort comparator := List.drop_left' size
  have perm : reordered.Perm roots :=
    (List.Perm.append_left (roots.take idx) suffix).trans (List.Perm.of_eq (List.take_append_drop idx roots))
  have transitive (a b c : Square) (ab : comparator a b) (bc : comparator b c) :
      comparator a c := by
    simp only [comparator, decide_eq_true_eq] at ab bc ⊢
    exact Int.le_trans bc ab
  have total (a b : Square) : comparator a b || comparator b a := by
    simp only [comparator, Bool.or_eq_true, decide_eq_true_eq]
    exact Int.le_total _ _
  have ordered := List.pairwise_mergeSort transitive total (roots.drop idx)
  rw [back]
  refine ⟨perm, front, suffix, ordered.imp of_decide_eq_true, fun a b tied pair => ?_⟩
  exact List.pair_sublist_mergeSort transitive total (by simp [comparator, tied]) pair

def SelectedContract : Prop :=
  ∀ (roots : List Square) (idx : Nat) (value scores : Square → Int)
    (original beta : Int) (seen : List Square) (state : State) (head : Square) (tail : List Square),
    idx ≤ roots.length → Window original beta →
    Records (roots.drop idx) value original beta seen state scores →
    original < state.best → state.best < beta →
    (sortFrom roots idx scores).drop idx = head :: tail →
    head ∈ roots.drop idx ∧ scores head = state.best ∧ value head = state.best

theorem selectedContract_proven : SelectedContract := by
  intro roots idx value scores original beta seen state head tail inside window reports exactHigh exactLow sorted
  have lo : -scoreInf ≤ original := window.lower
  have valid := records_valid lo reports exactLow
  obtain ⟨_, _, perm, ordered, _⟩ := sortContract_proven roots idx scores inside
  rw [sorted] at perm ordered
  have member : head ∈ roots.drop idx := perm.mem_iff.mp (by simp)
  obtain ⟨winner, legal, stored⟩ := valid.winner exactHigh
  have ranked : scores winner ≤ scores head := by
    rcases List.mem_cons.mp (perm.mem_iff.mpr legal) with same | later
    · exact same ▸ Int.le_refl _
    · exact (List.pairwise_cons.mp ordered).1 winner later
  have score : scores head = state.best := by have := valid.upper head; omega
  exact ⟨member, score, (valid.exactScore head (by omega)).trans score⟩

def SelectedLine (mc : Bitboard.Moves.X86Config) (fc : Root.FlipConfig) (b : PvsBoardTree.Position)
    (roots : List Square) (idx : Nat) (scores : Square → Int) (head : Square) : Prop :=
  head ∈ roots.drop idx ∧ scores head = Root.value mc fc b roots idx ∧
  Root.squareValue mc fc b head = Root.value mc fc b roots idx ∧
  (∀ sq ∈ roots.drop idx, Root.squareValue mc fc b sq ≤ Root.squareValue mc fc b head) ∧
  (∀ sq, (Root.remainingMask mc b.val (sortFrom roots idx scores) (idx + 1)).getLsbD sq.val = true ↔
    sq ∈ roots.drop idx ∧ sq ≠ head)

def SoundAttempt (mc : Bitboard.Moves.X86Config) (fc : Root.FlipConfig) (b : PvsBoardTree.Position)
    (roots : List Square) (idx : Nat) (alpha beta score : Int) (scores : Square → Int) : Prop :=
  Aspiration.SoundResult (Root.value mc fc b roots idx) alpha beta score ∧
    (alpha < score → score < beta →
      ∃ head tail, (sortFrom roots idx scores).drop idx = head :: tail ∧
        SelectedLine mc fc b roots idx scores head)

def ExactLineContract : Prop :=
  ∀ (completed : Search), Follows completed →
    ∀ (mc : Bitboard.Moves.X86Config) (fc : Root.FlipConfig) (b : PvsBoardTree.Position)
      (roots : List Square) (idx : Nat) (scores : Square → Int) (original beta : Int)
      (seen : List Square) (state : State) (head : Square) (tail : List Square),
      Root.Covers mc b roots → idx < roots.length → Window original beta →
      Root.Execution completed mc fc b roots idx original beta state.best →
      Records (roots.drop idx) (Root.squareValue mc fc b) original beta seen state scores →
      original < state.best → state.best < beta →
      (sortFrom roots idx scores).drop idx = head :: tail →
      SelectedLine mc fc b roots idx scores head

theorem selectedLine_of_exact (covers : Root.Covers mc b roots) (inside : idx < roots.length)
    (window : Window original beta) (exactBest : state.best = Root.value mc fc b roots idx)
    (records : Records (roots.drop idx) (Root.squareValue mc fc b) original beta seen state scores)
    (high : original < state.best) (low : state.best < beta)
    (sorted : (sortFrom roots idx scores).drop idx = head :: tail) :
    SelectedLine mc fc b roots idx scores head := by
  have bounded := Nat.le_of_lt inside
  obtain ⟨member, stored, exactHead⟩ := selectedContract_proven roots idx _ scores original beta seen
    state head tail bounded window records high low sorted
  have bestHead := exactHead.trans exactBest
  refine ⟨member, stored.trans exactBest, bestHead, fun sq kept => ?_, ?_⟩
  · rw [bestHead]
    exact Root.square_value_le covers kept
  · intro sq
    obtain ⟨perm, _, suffix, _, _⟩ := sortContract_proven roots idx scores bounded
    have reordered := perm.trans covers
    have distinct := (List.drop_sublist idx (sortFrom roots idx scores)).nodup (Root.roots_nodup reordered)
    rw [sorted, List.nodup_cons] at distinct
    rw [sorted] at suffix
    have mask := (Root.maskContract_proven mc b (sortFrom roots idx scores) (idx + 1) reordered).1
    rw [mask, List.drop_add_one_eq_tail_drop, sorted, ← suffix.mem_iff, List.mem_cons]
    constructor
    · intro kept
      exact ⟨.inr kept, fun same => distinct.1 (same ▸ kept)⟩
    · rintro ⟨rfl | kept, different⟩
      · exact False.elim (different rfl)
      · exact kept

theorem exactLineContract_proven : ExactLineContract := by
  intro completed follows mc fc b roots idx scores original beta seen state head tail covers inside
    window execution records high low sorted
  have exactBest :=
    (Root.scoreContract_proven completed follows mc fc b roots idx original beta state.best
      window execution).exact high low
  exact selectedLine_of_exact covers inside window exactBest records high low sorted

private def twoWipeouts : PvsBoardTree.Position := ⟨⟨0x04080000#64, 0x08000000#64⟩, by rfl⟩

example :
    let mc : Bitboard.Moves.X86Config := ⟨false, false⟩
    let fc : Root.FlipConfig := ⟨false, false, false⟩
    let roots : List Square := [35, 28]
    let scores := writeScore (fun _ => -scoreInf) 28 PvsMoveLoop.scoreMax
    scores 28 = Root.value mc fc twoWipeouts roots 0 ∧
      Root.remainingMask mc twoWipeouts.val (sortFrom roots 0 scores) 1 = Bitboard.squareMask 35 := by
  intro mc fc roots scores
  let move : PvsBoardTree.GeneratedSquare mc twoWipeouts.val := ⟨28, by decide⟩
  let state := publish true 32767 ⟨-32767, -scoreInf⟩ PvsMoveLoop.scoreMax
  have legal : (PvsBoardTree.legalSquares mc twoWipeouts.val).map Subtype.val = [28, 35] := by decide
  have covered : Root.Covers mc twoWipeouts roots := by
    unfold Root.Covers
    rw [legal]
    exact List.Perm.swap 28 35 []
  have sorted : sortFrom roots 0 scores = [28, 35] := by
    change ([35, 28] : List Square).mergeSort _ = _
    rw [List.mergeSort_pair]
    decide
  have wipe := EndgameCutoffs.wipeout_child_value mc fc twoWipeouts move (by decide)
  have exactMove : Root.squareValue mc fc twoWipeouts 28 = PvsMoveLoop.scoreMax := by
    rw [Root.squareValue, dite_eq_left move.property, Root.childValue, wipe]
    decide
  have recorded : Records (roots.drop 0) (Root.squareValue mc fc twoWipeouts) (-32767) 32767
      [28] state scores :=
    .push ⟨28, -32767, PvsMoveLoop.scoreMax⟩ .start (by decide) (by simp) (by decide)
      ⟨fun _ => Int.le_of_eq exactMove, fun _ => Int.le_of_eq exactMove.symm⟩
  have chosen := exactLineContract_proven _ sequential_follows mc fc twoWipeouts roots 0 scores
    (-32767) 32767 [28] state 28 [35] covered (by decide)
    ⟨by decide, by decide, by decide⟩ (.wipeout move (by decide) (by decide))
    recorded (by decide) (by decide) sorted
  exact ⟨chosen.2.1, by rw [sorted]; decide⟩

end NeuralReversi.PvsParallel.RootOrder
