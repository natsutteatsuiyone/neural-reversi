module
public import NeuralReversi.Aspiration

@[expose] public section

namespace NeuralReversi.PvsMoveLoop

open Aspiration (SoundResult Window)

abbrev ChildSearch (Move : Type) := Move → Bool → Int → Int → Int

def scoreMax : Int := 16384
def scoreInf : Int := 32767

def Bound (value alpha beta score : Int) : Prop :=
  (score < beta → value ≤ score) ∧ (alpha < score → score ≤ value)

theorem Bound.ofSoundResult {value alpha beta score : Int}
    (s : SoundResult value alpha beta score) :
    Bound value alpha beta score := by
  constructor
  · intro hi
    by_cases lo : score ≤ alpha
    · exact s.failLow lo
    · exact Int.le_of_eq (s.exact (by omega) hi).symm
  · intro lo
    by_cases hi : beta ≤ score
    · exact s.failHigh hi
    · exact Int.le_of_eq (s.exact lo (by omega))

theorem Bound.toSoundResult {value alpha beta score : Int}
    (w : Window alpha beta) (s : Bound value alpha beta score)
    (lo : -scoreInf < score) (hi : score < scoreInf) :
    SoundResult value alpha beta score := by
  refine ⟨lo, hi, ?_, ?_, ?_⟩
  · intro h
    exact s.1 (Int.lt_of_le_of_lt h w.ordered)
  · intro h
    exact s.2 (Int.lt_of_lt_of_le w.ordered h)
  · intro h₁ h₂
    exact Int.le_antisymm (s.2 h₁) (s.1 h₂)

theorem Bound.negate {value alpha beta score : Int} (s : Bound value (-beta) (-alpha) score) :
    Bound (-value) alpha beta (-score) := by
  constructor
  · intro h
    have := s.2 (by omega)
    omega
  · intro h
    have := s.1 (by omega)
    omega

theorem reflectedWindow {alpha beta : Int} (w : Window alpha beta) : Window (-beta) (-alpha) :=
  ⟨by have := w.upper; omega, by have := w.ordered; omega, by have := w.lower; omega⟩

theorem scoutWindow {alpha beta : Int} (w : Window alpha beta) :
    Window (-(alpha + 1)) (-alpha) :=
  reflectedWindow ⟨w.lower, by omega, by have := w.ordered; have := w.upper; omega⟩

def moveScore {Move : Type} (pv first : Bool) (search : ChildSearch Move)
    (move : Move) (alpha beta : Int) : Int :=
  if pv && first then -(search move true (-beta) (-alpha))
  else
    let scout := -(search move false (-(alpha + 1)) (-alpha))
    if pv = true ∧ alpha < scout then -(search move true (-beta) (-alpha))
    else scout

abbrev ChildRelation (Move : Type) := Move → Bool → Int → Int → Int → Prop

def MoveCalls (completed : ChildRelation Move) (search : ChildSearch Move)
    (pv first : Bool) (move : Move) (alpha beta : Int) : Prop :=
  if pv && first then completed move true (-beta) (-alpha) (search move true (-beta) (-alpha))
  else
    let scout := search move false (-(alpha + 1)) (-alpha)
    completed move false (-(alpha + 1)) (-alpha) scout ∧
      (pv = true ∧ alpha < -scout → completed move true (-beta) (-alpha) (search move true (-beta) (-alpha)))

theorem MoveCalls.weaken {done more : ChildRelation Move}
    (lift : ∀ move pv alpha beta score, done move pv alpha beta score → more move pv alpha beta score)
    (calls : MoveCalls done search pv first move alpha beta) : MoveCalls more search pv first move alpha beta := by
  unfold MoveCalls at calls ⊢
  split at calls <;> rename_i branch
  · rw [ite_eq_left branch]
    exact lift _ _ _ _ _ calls
  · rw [ite_eq_right branch]
    exact ⟨lift _ _ _ _ _ calls.1, fun retry => lift _ _ _ _ _ (calls.2 retry)⟩

theorem MoveCalls.ofAll (all : ∀ childPv a b, completed move childPv a b (search move childPv a b)) :
    MoveCalls completed search pv first move alpha beta := by
  unfold MoveCalls
  split
  · exact all true _ _
  · exact ⟨all false _ _, fun _ => all true _ _⟩

theorem moveScore_boundSelected {Move : Type} {search : ChildSearch Move} {value : Move → Int}
    {pv first : Bool} {move : Move} {alpha beta : Int}
    (w : Window alpha beta) (null : pv = false → beta = alpha + 1)
    {completed : ChildRelation Move} (calls : MoveCalls completed search pv first move alpha beta)
    (sound : ∀ childPv a b s, Window a b → (childPv = false → b = a + 1) →
      completed move childPv a b s → SoundResult (-(value move)) a b s) :
    let score := moveScore pv first search move alpha beta
    (-scoreInf < score) ∧ score < scoreInf ∧ Bound (value move) alpha beta score := by
  have wideOK (heard : completed move true (-beta) (-alpha) (search move true (-beta) (-alpha))) :
      -scoreInf < -(search move true (-beta) (-alpha)) ∧
      -(search move true (-beta) (-alpha)) < scoreInf ∧
        Bound (value move) alpha beta (-(search move true (-beta) (-alpha))) := by
    have wide := sound true _ _ _ (reflectedWindow w) (by simp) heard
    have wideBound : Bound (value move) alpha beta (-(search move true (-beta) (-alpha))) := by
      simpa only [Int.neg_neg] using (Bound.ofSoundResult wide).negate
    exact ⟨by have := wide.upper; unfold scoreInf; omega,
      by have := wide.lower; unfold scoreInf; omega, wideBound⟩
  dsimp only [moveScore, MoveCalls] at calls ⊢
  split at calls <;> rename_i eldest
  · rw [ite_eq_left eldest]
    exact wideOK calls
  · rw [ite_eq_right eldest]
    have narrow := sound false _ _ _ (scoutWindow w) (by omega) calls.1
    have narrowBound : Bound (value move) alpha (alpha + 1)
        (-(search move false (-(alpha + 1)) (-alpha))) := by
      simpa only [Int.neg_neg] using (Bound.ofSoundResult narrow).negate
    split
    · rename_i retry
      exact wideOK (calls.2 retry)
    · rename_i noResearch
      refine ⟨by have := narrow.upper; unfold scoreInf; omega,
        by have := narrow.lower; unfold scoreInf; omega, ?_⟩
      cases pv
      · rw [null rfl]
        exact narrowBound
      · simp only [true_and] at noResearch
        constructor
        · intro _
          exact narrowBound.1 (by omega)
        · omega

def scan {Move : Type} (pv : Bool) (search : ChildSearch Move) (beta : Int) :
    List Move → Bool → Int → Int → Int
  | [], _, _, best => best
  | move :: rest, first, alpha, best =>
    let score := moveScore pv first search move alpha beta
    if best < score then
      if alpha < score then
        if pv = true ∧ score < beta ∧ score < scoreMax then
          scan pv search beta rest false score score
        else score
      else scan pv search beta rest false alpha score
    else scan pv search beta rest false alpha best

def nextScanState (pv : Bool) (beta alpha best score : Int) : Option (Int × Int) :=
  if best < score then
    if alpha < score then
      if pv = true ∧ score < beta ∧ score < scoreMax then some (score, score) else none
    else some (alpha, score)
  else some (alpha, best)

theorem scan_step (pv : Bool) (search : ChildSearch Move) (beta : Int)
    (move : Move) (rest : List Move) (first : Bool) (alpha best : Int) :
    let score := moveScore pv first search move alpha beta
    scan pv search beta (move :: rest) first alpha best =
      match nextScanState pv beta alpha best score with
      | none => score
      | some (a, s) => scan pv search beta rest false a s := by
  dsimp only
  rw [scan]
  unfold nextScanState
  split
  · split
    · split <;> rfl
    · rfl
  · rfl

def ScanCalls (completed : ChildRelation Move) (search : ChildSearch Move) (pv : Bool) (beta : Int) :
    List Move → Bool → Int → Int → Prop
  | [], _, _, _ => True
  | move :: rest, first, alpha, best =>
    MoveCalls completed search pv first move alpha beta ∧
      match nextScanState pv beta alpha best (moveScore pv first search move alpha beta) with
      | none => True
      | some (nextAlpha, nextBest) => ScanCalls completed search pv beta rest false nextAlpha nextBest

theorem ScanCalls.weaken {done more : ChildRelation Move}
    (lift : ∀ move pv alpha beta score, done move pv alpha beta score → more move pv alpha beta score)
    (calls : ScanCalls done search pv beta moves first alpha best) :
    ScanCalls more search pv beta moves first alpha best := by
  induction moves generalizing first alpha best with
  | nil => trivial
  | cons move rest ih =>
    refine ⟨calls.1.weaken lift, ?_⟩
    have after := calls.2
    cases next : nextScanState pv beta alpha best (moveScore pv first search move alpha beta) with
    | none => trivial
    | some state =>
      rw [next] at after
      rcases state with ⟨_, _⟩
      exact ih after

theorem ScanCalls.ofAll
    (calls : ∀ m ∈ moves, ∀ childPv a b,
      completed m childPv a b (search m childPv a b)) : ScanCalls completed search pv beta moves first alpha best := by
  induction moves generalizing first alpha best with
  | nil => trivial
  | cons move rest ih =>
    refine ⟨MoveCalls.ofAll (calls move (by simp)), ?_⟩
    cases nextScanState pv beta alpha best (moveScore pv first search move alpha beta) with
    | none => trivial
    | some state =>
      rcases state with ⟨_, _⟩
      exact ih (fun m member => calls m (List.mem_cons_of_mem _ member))

def nonPvPresearch {Move : Type} (search : ChildSearch Move)
    (first : Move) (rest : List Move) (alpha beta : Int) : Int :=
  let score := -(search first false (-(alpha + 1)) (-alpha))
  if alpha < score ∧ beta ≤ score then score
  else scan false search beta rest false alpha score

def NonPvPresearchContract : Prop :=
  ∀ (Move : Type) (completed : ChildRelation Move) (search : ChildSearch Move)
    (first : Move) (rest : List Move) (alpha beta : Int),
    let scout := search first false (-(alpha + 1)) (-alpha)
    beta = alpha + 1 → -scoreInf < -scout →
    completed first false (-(alpha + 1)) (-alpha) scout →
    (-scout < beta → ScanCalls completed search false beta rest false alpha (-scout)) →
    ScanCalls completed search false beta (first :: rest) true alpha (-scoreInf) ∧
      nonPvPresearch search first rest alpha beta = scan false search beta (first :: rest) true alpha (-scoreInf)

theorem nonPvPresearchContract_proven : NonPvPresearchContract := by
  intro Move completed search first rest alpha beta scout null above returned continues
  let score := -scout
  have firstScore : moveScore false true search first alpha beta = score := rfl
  have step :
      nextScanState false beta alpha (-scoreInf) (moveScore false true search first alpha beta) =
        if alpha < score then none else some (alpha, score) := by
    rw [firstScore]
    have valid : -scoreInf < score := above
    simp [nextScanState, valid]
  refine ⟨⟨⟨returned, by simp⟩, ?_⟩, ?_⟩
  · rw [step]
    by_cases high : alpha < score
    · simp [high]
    · simpa [high] using continues (show score < beta by omega)
  rw [scan_step, step, firstScore, nonPvPresearch]
  change (if alpha < score ∧ beta ≤ score then score
    else scan false search beta rest false alpha score) = _
  by_cases high : alpha < score
  · have cut : alpha < score ∧ beta ≤ score := ⟨high, by omega⟩
    simp [cut]
  · simp [high]

def bestValue {Move : Type} (value : Move → Int) : List Move → Int → Int
  | [], seed => seed
  | move :: rest, seed => bestValue value rest (max seed (value move))

theorem bestValue_ge_seed {Move : Type} (value : Move → Int) (moves : List Move) (seed : Int) :
    seed ≤ bestValue value moves seed := by
  induction moves generalizing seed with
  | nil => exact Int.le_refl _
  | cons move rest ih =>
    exact Int.le_trans (Int.le_max_left _ _) (ih _)

theorem bestValue_le {Move : Type} (value : Move → Int) (moves : List Move) (seed bound : Int)
    (seedLe : seed ≤ bound) (all : ∀ m ∈ moves, value m ≤ bound) :
    bestValue value moves seed ≤ bound := by
  induction moves generalizing seed with
  | nil => exact seedLe
  | cons move rest ih =>
    apply ih _ (Int.max_le.mpr ⟨seedLe, all move (by simp)⟩)
    intro m h
    exact all m (List.mem_cons_of_mem _ h)

theorem bestValue_ge_move {Move : Type} (value : Move → Int) (moves : List Move) (seed : Int)
    {move : Move} (member : move ∈ moves) : value move ≤ bestValue value moves seed := by
  induction moves generalizing seed with
  | nil => cases member
  | cons first rest ih =>
    rw [bestValue]
    rcases List.mem_cons.mp member with equal | later
    · subst first
      exact Int.le_trans (Int.le_max_right _ _) (bestValue_ge_seed _ _ _)
    · exact ih _ later

theorem bestValue_map {Move Child : Type} (value : Child → Int) (child : Move → Child)
    (moves : List Move) (seed : Int) :
    bestValue value (moves.map child) seed = bestValue (fun move => value (child move)) moves seed := by
  induction moves generalizing seed with
  | nil => rfl
  | cons move rest ih => exact ih _

theorem bestValue_eq_of_same_moves {Move : Type} (value : Move → Int) (seed : Int)
    {xs ys : List Move} (same : ∀ move, move ∈ xs ↔ move ∈ ys) :
    bestValue value xs seed = bestValue value ys seed := by
  apply Int.le_antisymm
  · exact bestValue_le _ _ _ _ (bestValue_ge_seed _ _ _)
      (fun move mem => bestValue_ge_move _ _ _ ((same move).mp mem))
  · exact bestValue_le _ _ _ _ (bestValue_ge_seed _ _ _)
      (fun move mem => bestValue_ge_move _ _ _ ((same move).mpr mem))

private theorem scan_sound {Move : Type} (pv : Bool) (search : ChildSearch Move) (value : Move → Int)
    (completed : ChildRelation Move)
    (moves : List Move) (first : Bool) (original alpha beta prev best : Int)
    (window : Window original beta) (null : pv = false → beta = original + 1)
    (raised : alpha = max original best) (bestLt : best < beta)
    (seed : Bound prev original beta best) (seedMax : prev ≤ scoreMax)
    (bounds : ∀ m ∈ moves, value m ≤ scoreMax)
    (calls : ScanCalls completed search pv beta moves first alpha best)
    (sound : ∀ m ∈ moves, ∀ childPv a b s, Window a b → (childPv = false → b = a + 1) →
      completed m childPv a b s → SoundResult (-(value m)) a b s)
    (searched : -scoreInf < best ∨ moves ≠ []) :
    let result := scan pv search beta moves first alpha best
    (-scoreInf < result) ∧ result < scoreInf ∧ Bound (bestValue value moves prev) original beta result := by
  induction moves generalizing first alpha prev best with
  | nil =>
    exact ⟨searched.resolve_right (by simp), Int.lt_of_lt_of_le bestLt window.upper, seed⟩
  | cons move rest ih =>
    let score := moveScore pv first search move alpha beta
    have w : Window alpha beta :=
      ⟨by have := window.lower; omega, by have := window.ordered; omega, window.upper⟩
    have narrow : pv = false → beta = alpha + 1 := by
      intro h
      have := null h
      omega
    have ⟨lo, hi, current⟩ := moveScore_boundSelected (first := first) w narrow calls.1 (sound move (by simp))
    have after : (match nextScanState pv beta alpha best score with
        | none => True
        | some (nextAlpha, nextBest) => ScanCalls completed search pv beta rest false nextAlpha nextBest) := calls.2
    change -scoreInf < score at lo
    change score < scoreInf at hi
    change Bound (value move) alpha beta score at current
    let next := max prev (value move)
    have nextMax : next ≤ scoreMax := Int.max_le.mpr ⟨seedMax, bounds move (by simp)⟩
    have restMax (m : Move) (mem : m ∈ rest) := bounds m (List.mem_cons_of_mem _ mem)
    have restSound (m : Move) (mem : m ∈ rest) := sound m (List.mem_cons_of_mem _ mem)
    have prevLe : prev ≤ best := seed.1 bestLt
    let result := if best < score then
        if alpha < score then
          if pv = true ∧ score < beta ∧ score < scoreMax then
            scan pv search beta rest false score score
          else score
        else scan pv search beta rest false alpha score
      else scan pv search beta rest false alpha best
    change -scoreInf < result ∧ result < scoreInf ∧
      Bound (bestValue value rest next) original beta result
    have alphaLt := w.ordered
    dsimp only [result]
    by_cases improves : best < score
    · rw [ite_eq_left improves]
      by_cases above : alpha < score
      · rw [ite_eq_left above]
        by_cases continueSearch : pv = true ∧ score < beta ∧ score < scoreMax
        · rw [ite_eq_left continueSearch]
          have scoreLt := continueSearch.2.1
          have valEq := Int.le_antisymm (current.1 scoreLt) (current.2 above)
          have seed' : Bound next original beta score := by
            unfold next
            constructor <;> intro h <;> omega
          have seen : ScanCalls completed search pv beta rest false score score := by
            simpa only [nextScanState, ite_eq_left improves, ite_eq_left above, ite_eq_left continueSearch]
              using after
          exact ih false score next score (by omega) scoreLt seed' nextMax restMax seen restSound (.inl lo)
        · rw [ite_eq_right continueSearch]
          have highOrMax : beta ≤ score ∨ scoreMax ≤ score := by
            by_cases h : pv = false
            · exact .inl (by have := narrow h; omega)
            · have pvTrue : pv = true := by cases pv <;> simp_all
              by_cases hi : beta ≤ score
              · exact .inl hi
              · exact .inr (Int.le_of_not_gt fun h => continueSearch ⟨pvTrue, by omega, h⟩)
          have bestAtLeast : score ≤ bestValue value rest next :=
            Int.le_trans (current.2 above) (Int.le_trans (Int.le_max_right _ _) (bestValue_ge_seed _ _ _))
          have upper : score < beta → bestValue value rest next ≤ score := by
            intro h
            have maxLe := highOrMax.resolve_left (by omega)
            exact Int.le_trans (bestValue_le _ _ _ _ nextMax restMax) maxLe
          exact ⟨lo, hi, upper, fun _ => bestAtLeast⟩
      · rw [ite_eq_right above]
        have scoreLt : score < beta := by omega
        have moveLe := current.1 scoreLt
        have seed' : Bound next original beta score := by
          unfold next
          constructor <;> intro h <;> omega
        have seen : ScanCalls completed search pv beta rest false alpha score := by
          simpa only [nextScanState, ite_eq_left improves, ite_eq_right above] using after
        exact ih false alpha next score (by omega) scoreLt seed' nextMax restMax seen restSound (.inl lo)
    · rw [ite_eq_right improves]
      have moveLe := current.1 (by omega)
      have seed' : Bound next original beta best := by
        unfold next
        exact ⟨fun _ => Int.max_le.mpr ⟨prevLe, by omega⟩,
          fun h => Int.le_trans (seed.2 h) (Int.le_max_left _ _)⟩
      have seen : ScanCalls completed search pv beta rest false alpha best := by
        simpa only [nextScanState, ite_eq_right improves] using after
      exact ih false alpha next best raised bestLt seed' nextMax restMax seen restSound (.inl (by omega))

def SelectedContract : Prop :=
  ∀ (Move : Type) (pv : Bool) (search : ChildSearch Move) (value : Move → Int)
    (completed : ChildRelation Move) (moves : List Move) (alpha beta : Int),
    moves ≠ [] → Window alpha beta → (pv = false → beta = alpha + 1) →
    (∀ m ∈ moves, value m ≤ scoreMax) →
    ScanCalls completed search pv beta moves true alpha (-scoreInf) →
    (∀ m ∈ moves, ∀ childPv a b s, Window a b → (childPv = false → b = a + 1) →
      completed m childPv a b s → SoundResult (-(value m)) a b s) →
    SoundResult (bestValue value moves (-scoreInf)) alpha beta (scan pv search beta moves true alpha (-scoreInf))

theorem selectedContract_proven : SelectedContract := by
  intro Move pv search value completed moves alpha beta ne w null bounds calls sound
  obtain ⟨lo, hi, h⟩ := scan_sound pv search value completed moves true alpha alpha beta
    (-scoreInf) (-scoreInf) w null
    (by have := w.lower; unfold scoreInf; omega)
    (by have := w.lower; have := w.ordered; unfold scoreInf; omega)
    ⟨fun _ => Int.le_refl _, fun _ => Int.le_refl _⟩ (by decide)
    bounds calls sound (.inr ne)
  exact h.toSoundResult w lo hi

def Contract : Prop :=
  ∀ (Move : Type) (pv : Bool) (search : ChildSearch Move) (value : Move → Int)
    (moves : List Move) (alpha beta : Int),
    moves ≠ [] → Window alpha beta → (pv = false → beta = alpha + 1) →
    (∀ m ∈ moves, value m ≤ scoreMax) →
    (∀ m ∈ moves, ∀ childPv a b, Window a b → (childPv = false → b = a + 1) →
      SoundResult (-(value m)) a b (search m childPv a b)) →
    SoundResult (bestValue value moves (-scoreInf)) alpha beta
      (scan pv search beta moves true alpha (-scoreInf))

theorem contract_proven : Contract := by
  intro Move pv search value moves alpha beta ne w null bounds sound
  exact selectedContract_proven Move pv search value (fun m cp a b s => s = search m cp a b)
    moves alpha beta ne w null bounds (ScanCalls.ofAll (by intros; rfl))
    (fun m mem cp a b _ window narrow same => same ▸ sound m mem cp a b window narrow)

inductive Tree where
  | leaf (score : {s : Int // -scoreMax ≤ s ∧ s ≤ scoreMax})
  | pass (child : Tree)
  | turn (head : Tree) (tail : List Tree)

theorem Tree.earlier {head : Tree} {tail : List Tree}
    (child : {t : Tree // t ∈ head :: tail}) : sizeOf child.val < sizeOf (Tree.turn head tail) := by
  have mem := List.sizeOf_lt_of_mem child.property
  simp only [Tree.turn.sizeOf_spec, List.cons.sizeOf_spec] at mem ⊢
  omega

def minimax (tree : Tree) : Int :=
  match tree with
  | .leaf score => score.val
  | .pass child => -minimax child
  | .turn head tail =>
    bestValue (fun child : {t : Tree // t ∈ head :: tail} => -minimax child.val)
      (head :: tail).attach (-scoreInf)
termination_by sizeOf tree
decreasing_by
  · simp_wf
  · exact Tree.earlier child

theorem minimax_turn (head : Tree) (tail : List Tree) :
    minimax (.turn head tail) =
      bestValue (fun child => -minimax child) (head :: tail) (-scoreInf) := by
  rw [minimax.eq_def]
  let children := (head :: tail).attach
  change bestValue (fun child => -minimax child.val) children (-scoreInf) = _
  rw [← bestValue_map (fun child => -minimax child) Subtype.val]
  simp [children]

def run (tree : Tree) (pv : Bool) (alpha beta : Int) : Int :=
  match tree with
  | .leaf score => score.val
  | .pass child => -run child pv (-beta) (-alpha)
  | .turn head tail =>
    scan pv (fun child : {t : Tree // t ∈ head :: tail} => run child.val) beta
      (head :: tail).attach true alpha (-scoreInf)
termination_by sizeOf tree
decreasing_by
  · simp_wf
  · exact Tree.earlier child

private theorem attached_nonempty (head : Tree) (tail : List Tree) :
    (head :: tail).attach ≠ [] := by
  intro empty
  have n := congrArg List.length empty
  simp at n

theorem minimax_range (tree : Tree) : -scoreMax ≤ minimax tree ∧ minimax tree ≤ scoreMax := by
  rw [minimax.eq_def]
  split
  next score => exact score.property
  next child =>
    have := minimax_range child
    omega
  next head tail =>
    let moves := (head :: tail).attach
    let value := fun child : {t : Tree // t ∈ head :: tail} => -minimax child.val
    have range (c : {t : Tree // t ∈ head :: tail}) : -scoreMax ≤ value c ∧ value c ≤ scoreMax := by
      have := minimax_range c.val
      dsimp only [value]
      omega
    change -scoreMax ≤ bestValue value moves (-scoreInf) ∧
      bestValue value moves (-scoreInf) ≤ scoreMax
    refine ⟨?_, bestValue_le _ _ _ _ (by decide) (fun c _ => (range c).2)⟩
    cases hm : moves with
    | nil => exact False.elim (attached_nonempty head tail hm)
    | cons c cs =>
      rw [bestValue]
      exact Int.le_trans (range c).1 (Int.le_trans (Int.le_max_right _ _) (bestValue_ge_seed _ _ _))
termination_by sizeOf tree
decreasing_by
  · subst_vars
    simp_wf
  · subst_vars
    exact Tree.earlier c

def RecursiveContract : Prop :=
  ∀ (tree : Tree) (pv : Bool) (alpha beta : Int), Window alpha beta →
    (pv = false → beta = alpha + 1) →
    SoundResult (minimax tree) alpha beta (run tree pv alpha beta)

theorem run_sound (tree : Tree) (pv : Bool) (alpha beta : Int) (w : Window alpha beta)
    (null : pv = false → beta = alpha + 1) :
    SoundResult (minimax tree) alpha beta (run tree pv alpha beta) := by
  rw [minimax.eq_def, run.eq_def]
  split
  next score =>
    have := score.property.1
    have := score.property.2
    exact ⟨by unfold scoreMax at *; omega, by unfold scoreMax at *; omega,
      fun _ => Int.le_refl _, fun _ => Int.le_refl _, fun _ _ => rfl⟩
  next child =>
    have childResult := run_sound child pv (-beta) (-alpha)
      (reflectedWindow w) (by intro h; have := null h; omega)
    exact (Bound.ofSoundResult childResult).negate.toSoundResult w
      (by have := childResult.upper; unfold scoreInf; omega)
      (by have := childResult.lower; unfold scoreInf; omega)
  next head tail =>
    apply contract_proven
    · exact attached_nonempty head tail
    · exact w
    · exact null
    · intro child _
      have := (minimax_range child.val).1
      omega
    · intro child _ childPv a b cw cn
      simpa only [Int.neg_neg] using run_sound child.val childPv a b cw cn
termination_by sizeOf tree
decreasing_by
  · subst_vars
    simp_wf
  · subst_vars
    exact Tree.earlier child

theorem recursiveContract_proven : RecursiveContract := run_sound

theorem fullWindow_exact (tree : Tree) :
    run tree true (-scoreInf) scoreInf = minimax tree := by
  have result := recursiveContract_proven tree true (-scoreInf) scoreInf ⟨by decide, by decide, by decide⟩ (by simp)
  exact result.exact result.lower result.upper

private def researchExample (second pv : Bool) (alpha _beta : Int) : Int :=
  if !second then 0 else if pv then -10 else max (-10) alpha

example : scan true researchExample 20 [false, true] true (-20) (-scoreInf) = 10 ∧
    moveScore false false researchExample true 0 20 = 1 := by decide

end NeuralReversi.PvsMoveLoop
