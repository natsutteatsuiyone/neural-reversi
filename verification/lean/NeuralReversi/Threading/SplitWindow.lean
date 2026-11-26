import NeuralReversi.Threading.SplitPublication
import NeuralReversi.Threading.SplitResult

namespace NeuralReversi.Threading.SplitWindow

open Aspiration (SoundResult Window)
open AtomicWord (Event Read ReadWindow SequencedBefore Write signed32At)
open CutoffVisibility (HappensBefore History)
open PvsMoveLoop (Bound scoreMax)
open SplitResult (Merge Report)

structure State where
  alpha : Int
  best : Int

abbrev choosesMove (state : State) (score : Int) : Prop :=
  state.best < score ∧ state.alpha < score

abbrev raisesAlpha (pv : Bool) (beta : Int) (state : State) (score : Int) : Prop :=
  state.best < score ∧ state.alpha < score ∧ pv = true ∧ score < beta

def publish (pv : Bool) (beta : Int) (state : State) (score : Int) : State :=
  ⟨if raisesAlpha pv beta state score then score else state.alpha, max state.best score⟩

def scan (pv : Bool) (beta : Int) (initial : State) (reports : List Int) : State :=
  reports.foldl (publish pv beta) initial

private theorem publish_alpha_le (pv : Bool) (beta : Int) (state : State) (score : Int) :
    state.alpha ≤ (publish pv beta state score).alpha := by
  simp only [publish]
  split <;> omega

private theorem scan_alpha_le (pv : Bool) (beta : Int) (initial : State) (reports : List Int) :
    initial.alpha ≤ (scan pv beta initial reports).alpha := by
  induction reports generalizing initial with
  | nil => exact Int.le_refl _
  | cons score rest ih => exact Int.le_trans (publish_alpha_le _ _ _ _) (ih _)

def StateContract : Prop :=
  ∀ (pv : Bool) (original beta : Int) (initial : State) (reports : List Int),
    Window initial.alpha beta → initial.alpha ≤ max original initial.best →
    let locked := scan pv beta initial reports
    initial.alpha ≤ locked.alpha ∧ Window locked.alpha beta ∧
      locked.alpha ≤ max original locked.best ∧ (pv = false → locked.alpha = initial.alpha)

theorem stateContract_proven : StateContract := by
  intro pv original beta initial reports window capped
  induction reports generalizing initial with
  | nil => exact ⟨Int.le_refl _, window, capped, fun _ => rfl⟩
  | cons score rest ih =>
    have nextWindow : Window (publish pv beta initial score).alpha beta := by
      refine ⟨Int.le_trans window.lower (publish_alpha_le _ _ _ _), ?_, window.upper⟩
      simp only [publish]
      split <;> have := window.ordered <;> omega
    have nextCap : (publish pv beta initial score).alpha ≤
        max original (publish pv beta initial score).best := by
      simp only [publish]
      split <;> omega
    obtain ⟨raises, staysOpen, limit, fixed⟩ := ih _ nextWindow nextCap
    refine ⟨Int.le_trans (publish_alpha_le _ _ _ _) raises, staysOpen, limit, ?_⟩
    intro notPv
    exact (fixed notPv).trans (by simp [publish, notPv])

structure AlphaRead (history : History) {writes : List Write} (load : Read writes)
    (first last : Fin writes.length) where
  published : SplitPublication.PublishedThrough history (writes.get first).event load.event
  acquire : Event
  beforeAcquire : SequencedBefore load.event acquire
  afterAcquire : ∀ i : Fin writes.length, i.val = last.val + 1 →
    HappensBefore history acquire (writes.get i).event
  writeRead : ∀ i : Fin writes.length, HappensBefore history (writes.get i).event load.event →
    i.val ≤ load.source.val
  readWrite : ∀ i : Fin writes.length, HappensBefore history load.event (writes.get i).event →
    load.source.val < i.val

theorem AlphaRead.window (read : AlphaRead history load first last) :
    ReadWindow load (HappensBefore history) first last.val :=
  ⟨SplitPublication.initializationContract_proven _ _ _ read.published,
    fun i next => .trans (.sequenced read.beforeAcquire) (read.afterAcquire i next),
    read.writeRead, read.readWrite⟩

def Matches (writes : List Write) (first last : Fin writes.length) (pv : Bool) (beta : Int)
    (initial : State) (reports : List Int) : Prop :=
  ∀ i : Fin writes.length, first.val ≤ i.val → i.val ≤ last.val →
    ∃ earlier rest : List Int, reports = earlier ++ rest ∧
      signed32At writes i = (scan pv beta initial earlier).alpha

def AlphaReadContract : Prop :=
  ∀ (history : History) (writes : List Write) (load : Read writes)
    (first last : Fin writes.length) (pv : Bool) (original beta : Int)
    (initial : State) (reports : List Int),
    AlphaRead history load first last → Matches writes first last pv beta initial reports →
    Window initial.alpha beta → initial.alpha ≤ max original initial.best →
    let observed := signed32At writes load.source
    let locked := scan pv beta initial reports
    Window observed beta ∧ observed ≤ locked.alpha ∧ observed ≤ max original locked.best ∧
      (pv = false → observed = initial.alpha)

theorem alphaReadContract_proven : AlphaReadContract := by
  intro history writes load first last pv original beta initial reports read matched window capped
  obtain ⟨afterInit, beforeNext⟩ := read.window.bounds
  obtain ⟨earlier, rest, ordered, observed⟩ := matched load.source afterInit beforeNext
  obtain ⟨_, openWindow, _, fixed⟩ := stateContract_proven pv original beta initial earlier window capped
  have upper : signed32At writes load.source ≤ (scan pv beta initial reports).alpha := by
    simpa only [observed, ordered, scan, List.foldl_append] using
      scan_alpha_le pv beta (scan pv beta initial earlier) rest
  have lockCap := (stateContract_proven pv original beta initial reports window capped).2.2.1
  exact ⟨observed ▸ openWindow, upper, Int.le_trans upper lockCap, fun eq => observed.trans (fixed eq)⟩

def reSearch (pv : Bool) (alpha scout : Int) : Bool :=
  decide (pv = true ∧ alpha < -scout)

def selectReport (pv : Bool) (index : Nat) (alpha fresh scout wide : Int) : Report :=
  if reSearch pv alpha scout then ⟨index, fresh, -wide⟩ else ⟨index, alpha, -scout⟩

theorem childReport {value scout wide alpha fresh beta limit : Int} {pv : Bool}
    (index : Nat) (window : Window alpha beta) (ordered : alpha ≤ limit)
    (null : pv = false → beta = alpha + 1)
    (freshWindow : reSearch pv alpha scout = true → Window fresh beta ∧ fresh ≤ limit)
    (scoutSound : Window (-(alpha + 1)) (-alpha) → SoundResult (-value) (-(alpha + 1)) (-alpha) scout)
    (wideSound : reSearch pv alpha scout = true →
      Window (-beta) (-fresh) → SoundResult (-value) (-beta) (-fresh) wide) :
    let report := selectReport pv index alpha fresh scout wide
    Bound value report.searchedAlpha beta report.score ∧ report.searchedAlpha ≤ limit := by
  have narrow : Bound value alpha (alpha + 1) (-scout) := by
    simpa only [Int.neg_neg] using
      (Bound.ofSoundResult (scoutSound (PvsMoveLoop.scoutWindow window))).negate
  dsimp only [selectReport]
  split
  · rename_i retry
    obtain ⟨wideWindow, freshLe⟩ := freshWindow retry
    have bound : Bound value fresh beta (-wide) := by
      simpa only [Int.neg_neg] using
        (Bound.ofSoundResult (wideSound retry (PvsMoveLoop.reflectedWindow wideWindow))).negate
    exact ⟨bound, freshLe⟩
  · rename_i done
    change Bound value alpha beta (-scout) ∧ alpha ≤ limit
    refine ⟨?_, ordered⟩
    cases pv with
    | false => simpa only [null rfl] using narrow
    | true =>
      have failLow : -scout ≤ alpha := by
        simpa [reSearch, Int.not_lt] using done
      exact ⟨fun _ => narrow.1 (by omega), fun improved => False.elim (by omega)⟩

def PublishedReportContract : Prop :=
  ∀ (history : History) (writes : List Write) (scoutLoad wideLoad : Read writes)
    (first last : Fin writes.length) (pv : Bool) (original beta : Int)
    (initial : State) (reports : List Int) (value : Nat → Int) (index : Nat) (scout wide : Int),
    let alpha := signed32At writes scoutLoad.source
    let fresh := signed32At writes wideLoad.source
    AlphaRead history scoutLoad first last →
    (reSearch pv alpha scout = true → AlphaRead history wideLoad first last) →
    Matches writes first last pv beta initial reports →
    Window initial.alpha beta → initial.alpha ≤ max original initial.best →
    (pv = false → beta = initial.alpha + 1) →
    (Window (-(alpha + 1)) (-alpha) →
      SoundResult (-(value index)) (-(alpha + 1)) (-alpha) scout) →
    (reSearch pv alpha scout = true → Window (-beta) (-fresh) →
      SoundResult (-(value index)) (-beta) (-fresh) wide) →
    let locked := scan pv beta initial reports
    let report := selectReport pv index alpha fresh scout wide
    report.searchedAlpha ≤ locked.alpha ∧
      Merge value original beta locked.best [report] (publish pv beta locked report.score).best

theorem publishedReportContract_proven : PublishedReportContract := by
  intro history writes scoutLoad wideLoad first last pv original beta initial reports value index
    scout wide alpha fresh probe reread matched window capped narrow scoutSound wideSound locked report
  obtain ⟨probeWindow, probeLe, _, fixed⟩ :=
    alphaReadContract_proven history writes scoutLoad first last pv original beta initial reports
      probe matched window capped
  have latest (retry : reSearch pv alpha scout = true) : Window fresh beta ∧ fresh ≤ locked.alpha := by
    obtain ⟨openWindow, beforeReport, _, _⟩ :=
      alphaReadContract_proven history writes wideLoad first last pv original beta initial reports
        (reread retry) matched window capped
    exact ⟨openWindow, beforeReport⟩
  have ⟨bound, usable⟩ := childReport index probeWindow probeLe (fun noPv => by
    rw [fixed noPv]
    exact narrow noPv) latest scoutSound wideSound
  have same : report.index = index := by
    simp only [report, selectReport]
    split <;> rfl
  have lockCap := (stateContract_proven pv original beta initial reports window capped).2.2.1
  refine ⟨usable, .accept (Int.le_trans usable lockCap) ?_ .done⟩
  rw [same]
  exact bound

def marksCutoff (pv : Bool) (beta : Int) (state : State) (score : Int) : Bool :=
  decide (state.best < score ∧ state.alpha < score ∧
    (pv = false ∨ beta ≤ score ∨ scoreMax ≤ score))

private theorem scan_best_le (pv : Bool) (beta : Int) (initial : State) (reports : List Int) :
    initial.best ≤ (scan pv beta initial reports).best := by
  induction reports generalizing initial with
  | nil => exact Int.le_refl _
  | cons score rest ih => exact Int.le_trans (Int.le_max_left _ score) (ih _)

def CutoffWitnessContract : Prop :=
  ∀ (pv : Bool) (original beta : Int) (initial : State) (earlier later : List Int) (score : Int),
    Window initial.alpha beta → initial.alpha ≤ max original initial.best →
    (pv = false → beta = initial.alpha + 1) →
    marksCutoff pv beta (scan pv beta initial earlier) score = true →
    let final := scan pv beta initial (earlier ++ score :: later)
    beta ≤ final.best ∨ scoreMax ≤ final.best

theorem cutoffWitnessContract_proven : CutoffWitnessContract := by
  intro pv original beta initial earlier later score window capped narrow marked
  let locked := scan pv beta initial earlier
  obtain ⟨_, improved, reason⟩ := of_decide_eq_true marked
  have retained : score ≤ (scan pv beta initial (earlier ++ score :: later)).best := by
    simpa only [locked, scan, List.foldl_append, List.foldl_cons] using
      Int.le_trans (Int.le_max_right locked.best score)
        (scan_best_le pv beta (publish pv beta locked score) later)
  rcases reason with nonPv | high | maximum
  · have fixed := (stateContract_proven pv original beta initial earlier window capped).2.2.2 nonPv
    have width := narrow nonPv
    exact .inl (Int.le_trans (by omega) retained)
  · exact .inl (Int.le_trans high retained)
  · exact .inr (Int.le_trans maximum retained)

example : (selectReport true 2 0 6 (-1) (-4)).searchedAlpha = 6 ∧
    (selectReport true 2 0 6 (-1) (-4)).score = 4 := by decide

example : (selectReport true 2 0 6 1 (-4)).searchedAlpha = 0 ∧
    (selectReport true 2 0 6 1 (-4)).score = -1 := by decide

example : (scan true 10 ⟨-2, -3⟩ [3, -5, 11]).alpha = 3 ∧
    (scan true 10 ⟨-2, -3⟩ [3, -5, 11]).best = 11 := by decide

example : marksCutoff false 1 ⟨0, -2⟩ 1 = true ∧
    marksCutoff true 10 ⟨0, -2⟩ 1 = false ∧
    marksCutoff true 17000 ⟨0, -2⟩ scoreMax = true := by decide

example : signed32At [⟨⟨0, 0⟩, 0xFFFFFFEC#64, false, false⟩] ⟨0, by decide⟩ = -20 := by decide

end NeuralReversi.Threading.SplitWindow
