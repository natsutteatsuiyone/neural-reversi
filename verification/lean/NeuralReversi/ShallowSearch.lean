import NeuralReversi.PvsMoveLoop

namespace NeuralReversi.ShallowSearch

open Aspiration (SoundResult Window)
open PvsMoveLoop (Bound Tree bestValue minimax scoreInf scoreMax)

def scan (value : Move → Int → Int) (wipeout : Move → Bool) (beta : Int) :
    List Move → Int → Int → Int
  | [], _, best => best
  | move :: rest, alpha, best =>
    if wipeout move then scoreMax
    else
      let score := value move alpha
      if beta ≤ score then score
      else scan value wipeout beta rest (max alpha score) (max best score)

private theorem scan_sound (score : Move → Int → Int) (wipeout : Move → Bool) (value : Move → Int)
    (moves : List Move) (original alpha beta prev best : Int)
    (window : Window original beta)
    (raised : alpha = max original best) (bestLt : best < beta)
    (seed : Bound prev original beta best) (seedMax : prev ≤ scoreMax)
    (bounds : ∀ move ∈ moves, value move ≤ scoreMax)
    (wipes : ∀ move ∈ moves, wipeout move = true → value move = scoreMax)
    (sound : ∀ move ∈ moves, ∀ a, Window a beta → wipeout move = false →
      SoundResult (value move) a beta (score move a))
    (searched : -scoreInf < best ∨ moves ≠ []) :
    let result := scan score wipeout beta moves alpha best
    (-scoreInf < result) ∧ result < scoreInf ∧ Bound (bestValue value moves prev) original beta result := by
  induction moves generalizing alpha prev best with
  | nil => exact ⟨searched.resolve_right (by simp), Int.lt_of_lt_of_le bestLt window.upper, seed⟩
  | cons move rest ih =>
    let next := max prev (value move)
    have member : move ∈ move :: rest := by simp
    have tail (m : Move) (h : m ∈ rest) : m ∈ move :: rest := List.mem_cons_of_mem _ h
    have nextMax : next ≤ scoreMax := Int.max_le.mpr ⟨seedMax, bounds move member⟩
    have upper : bestValue value rest next ≤ scoreMax :=
      PvsMoveLoop.bestValue_le _ _ _ _ nextMax (fun m h => bounds m (tail m h))
    have lower : value move ≤ bestValue value rest next :=
      Int.le_trans (Int.le_max_right _ _) (PvsMoveLoop.bestValue_ge_seed _ _ _)
    change -scoreInf < scan score wipeout beta (move :: rest) alpha best ∧
      scan score wipeout beta (move :: rest) alpha best < scoreInf ∧
      Bound (bestValue value rest next) original beta (scan score wipeout beta (move :: rest) alpha best)
    by_cases wipe : wipeout move = true
    · rw [scan, ite_eq_left wipe]
      have reaches := wipes move member wipe
      exact ⟨by decide, by decide, fun _ => upper, fun _ => reaches ▸ lower⟩
    · have ordinary : wipeout move = false := Bool.eq_false_iff.mpr wipe
      rw [scan, ite_eq_right wipe]
      let current := score move alpha
      let result := if beta ≤ current then current else
        scan score wipeout beta rest (max alpha current) (max best current)
      change -scoreInf < result ∧ result < scoreInf ∧ Bound (bestValue value rest next) original beta result
      dsimp only [result]
      have active : Window alpha beta :=
        ⟨by have := window.lower; omega, by have := window.ordered; omega, window.upper⟩
      have honest := sound move member alpha active ordinary
      have bound := Bound.ofSoundResult honest
      by_cases high : beta ≤ current
      · simp only [high, ite_true]
        exact ⟨honest.lower, honest.upper, fun h => False.elim (by omega),
          fun _ => Int.le_trans (honest.failHigh high) lower⟩
      · simp only [high, ite_false]
        have upperMove := bound.1 (by omega)
        have upperPrefix := seed.1 bestLt
        have nextBound : Bound next original beta (max best current) := by
          refine ⟨fun _ => by dsimp [next]; omega, fun above => ?_⟩
          by_cases improves : best < current
          · have overAlpha : alpha < current := by omega
            have moveLower := bound.2 overAlpha
            dsimp [next]
            omega
          · have prevLower := seed.2 (by omega)
            dsimp [next]
            omega
        have nextLower := Int.lt_of_lt_of_le honest.lower (Int.le_max_right best current)
        exact ih (max alpha current) next (max best current) (by omega) (by omega) nextBound nextMax
          (fun m h => bounds m (tail m h)) (fun m h => wipes m (tail m h))
          (fun m h => sound m (tail m h)) (.inl nextLower)

def ScanContract : Prop :=
  ∀ (Move : Type) (score : Move → Int → Int) (wipeout : Move → Bool) (value : Move → Int)
    (moves : List Move) (alpha beta : Int),
    moves ≠ [] → Window alpha beta →
    (∀ move ∈ moves, value move ≤ scoreMax) →
    (∀ move ∈ moves, wipeout move = true → value move = scoreMax) →
    (∀ move ∈ moves, ∀ a, Window a beta → wipeout move = false →
      SoundResult (value move) a beta (score move a)) →
    SoundResult (bestValue value moves (-scoreInf)) alpha beta
      (scan score wipeout beta moves alpha (-scoreInf))

theorem scanContract_proven : ScanContract := by
  intro Move score wipeout value moves alpha beta ne window bounds wipes sound
  obtain ⟨lo, hi, correct⟩ := scan_sound score wipeout value moves alpha alpha beta (-scoreInf) (-scoreInf)
    window (by have := window.lower; unfold scoreInf; omega)
    (by have := window.lower; have := window.ordered; unfold scoreInf; omega)
    ⟨fun _ => Int.le_refl _, fun _ => Int.le_refl _⟩ (by decide) bounds wipes sound (.inr ne)
  exact correct.toSoundResult window lo hi

def run (tree : Tree) (alpha beta : Int) : Int :=
  match tree with
  | .leaf value => value.val
  | .pass child => -run child (-beta) (-alpha)
  | .turn first rest =>
    let children := (first :: rest).attach
    scan (fun (child : {t : Tree // t ∈ first :: rest}) a =>
      -(run child.val (-beta) (-a))) (fun _ => false) beta children alpha (-scoreInf)
termination_by sizeOf tree
decreasing_by
  · simp_wf
  · exact Tree.earlier child

private theorem run_sound (tree : Tree) (alpha beta : Int) (window : Window alpha beta) :
    SoundResult (minimax tree) alpha beta (run tree alpha beta) := by
  rw [run.eq_def]
  cases tree with
  | leaf value =>
    rw [PvsMoveLoop.minimax.eq_def]
    change SoundResult value.val alpha beta value.val
    obtain ⟨lo, hi⟩ := value.property
    exact Bound.toSoundResult window ⟨fun _ => Int.le_refl _, fun _ => Int.le_refl _⟩
      (by unfold scoreInf; unfold scoreMax at lo; omega) (by unfold scoreInf; unfold scoreMax at hi; omega)
  | pass child =>
    rw [PvsMoveLoop.minimax.eq_def]
    change SoundResult (-minimax child) alpha beta (-run child (-beta) (-alpha))
    have sound := run_sound child (-beta) (-alpha) (PvsMoveLoop.reflectedWindow window)
    have bound := (Bound.ofSoundResult sound).negate
    exact bound.toSoundResult window (by have := sound.upper; unfold scoreInf; omega)
      (by have := sound.lower; unfold scoreInf; omega)
  | turn first rest =>
    rw [PvsMoveLoop.minimax_turn]
    let children := (first :: rest).attach
    let value := fun child : {t : Tree // t ∈ first :: rest} => -minimax child.val
    have nonempty : children ≠ [] := by simp [children]
    have same : bestValue value children (-scoreInf) =
        bestValue (fun child => -minimax child) (first :: rest) (-scoreInf) := by
      let evaluation := fun child => -minimax child
      rw [← PvsMoveLoop.bestValue_map evaluation Subtype.val children]
      simp [children, evaluation]
    rw [← same]
    apply scanContract_proven _ _ _ value children alpha beta nonempty window
    · intro child _
      have lo := (PvsMoveLoop.minimax_range child.val).1
      dsimp [value]
      omega
    · simp
    · intro child _ a active _
      have smaller := Tree.earlier child
      have sound := run_sound child.val (-beta) (-a) (PvsMoveLoop.reflectedWindow active)
      exact (Bound.ofSoundResult sound).negate.toSoundResult active
        (by have := sound.upper; unfold scoreInf; omega)
        (by have := sound.lower; unfold scoreInf; omega)
termination_by sizeOf tree
decreasing_by
  · subst_vars
    simp_wf
  · subst_vars
    exact Tree.earlier child

def Contract : Prop :=
  ∀ (tree : Tree) (alpha beta : Int), Window alpha beta →
    SoundResult (minimax tree) alpha beta (run tree alpha beta)

theorem contract_proven : Contract := run_sound

theorem reachesThreshold (verify shallow limit : Int)
    (sound : SoundResult shallow (limit - 1) limit verify) : limit ≤ verify ↔ limit ≤ shallow := by
  refine ⟨fun hit => Int.le_trans hit (sound.failHigh hit), fun hit => ?_⟩
  apply Int.le_of_not_gt
  intro miss
  have upperBound := sound.failLow (by omega)
  omega

example : scan (fun score _ => score) (fun _ => false) 3 [-1, -2] (-4) (-scoreInf) = -1 := by decide

end NeuralReversi.ShallowSearch
