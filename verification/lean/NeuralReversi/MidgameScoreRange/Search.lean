module
public import NeuralReversi.MidgameScoreRange.Table
public import NeuralReversi.MidgameProbCut
public import NeuralReversi.MidgameAspiration.Termination

@[expose] public section

namespace NeuralReversi.MidgameScoreRange

open Aspiration (Window)
open MidgameAspiration (FiniteScore)
open PvsParallel.Cutoffs (History Lookup)

def guardedProbcut (rawBeta : BitVec 32) (sel : Nat) (evaluation : Int)
    (verify : MidgameProbCut.Call → Int) (plan : MidgameProbCut.Plan) : MidgameProbCut.Outcome :=
  if isFinite rawBeta then MidgameProbCut.run sel rawBeta.toInt evaluation verify plan
  else ⟨none, sel, none⟩

def ProbcutContract : Prop :=
  ∀ rawBeta sel evaluation verify plan,
    (¬ FiniteScore rawBeta.toInt → guardedProbcut rawBeta sel evaluation verify plan = ⟨none, sel, none⟩) ∧
    (∀ score, (guardedProbcut rawBeta sel evaluation verify plan).cutoff = some score →
      score = rawBeta.toInt ∧ FiniteScore score)

theorem probcutContract_proven : ProbcutContract := by
  intro rawBeta sel evaluation verify plan
  constructor
  · intro invalid
    unfold guardedProbcut
    split
    · rename_i admitted
      exact (invalid ((gateContract_proven _).mp admitted)).elim
    · rfl
  · intro score result
    unfold guardedProbcut at result
    split at result
    · rename_i admitted
      have finite := (gateContract_proven _).mp admitted
      have equal : score = rawBeta.toInt := by
        cases plan with
        | skip => cases result
        | useStatic requested =>
          simp only [MidgameProbCut.run] at result
          split at result <;> cases result <;> rfl
        | verify call =>
          simp only [MidgameProbCut.run] at result
          split at result <;> cases result <;> rfl
      exact ⟨equal, equal.symm ▸ finite⟩
    · cases result

structure ChildCall where
  alpha : Int
  beta : Int
  result : Int

def ChildCall.window (c : ChildCall) : Prop := Window c.alpha c.beta
def ChildCall.score (c : ChildCall) : Int := -c.result

structure ObservedMove where
  scout : Int
  retry : Int
  pvSearch : Int

def chosenChild (pv first reduced : Bool) (alpha beta : Int) (seen : ObservedMove) : ChildCall :=
  if pv && first then ⟨-beta, -alpha, seen.pvSearch⟩
  else
    let narrow := if reduced && decide (alpha < -seen.scout) then seen.retry else seen.scout
    if pv && decide (alpha < -narrow) then ⟨-beta, -alpha, seen.pvSearch⟩
    else ⟨-(alpha + 1), -alpha, narrow⟩

def ChildWindowContract : Prop :=
  ∀ pv first reduced alpha beta seen, Window alpha beta →
    (chosenChild pv first reduced alpha beta seen).window

theorem childWindowContract_proven : ChildWindowContract := by
  intro pv first reduced alpha beta seen valid
  by_cases principal : (pv && first) = true
  · simpa [chosenChild, principal, ChildCall.window] using reflected valid
  · rw [chosenChild, ite_eq_right principal]
    generalize (if (reduced && decide (alpha < -seen.scout)) = true then seen.retry else seen.scout) = narrow
    dsimp only
    split
    · exact reflected valid
    · exact scout_window valid

def bestReturned (children : List ChildCall) : Int :=
  children.foldl (fun best child => max best child.score) (-32767)

private theorem best_fold_finite (children : List ChildCall) (seed : Int)
    (finite : FiniteScore seed) (scores : ∀ c ∈ children, FiniteScore c.score) :
    FiniteScore (children.foldl (fun best child => max best child.score) seed) := by
  induction children generalizing seed with
  | nil => exact finite
  | cons c cs ih =>
    exact ih _ (finite_max finite (scores c (by simp)))
      (fun next present => scores next (List.mem_cons_of_mem _ present))

def LoopContract : Prop :=
  ∀ (first : ChildCall) (remaining : List ChildCall),
    (∀ c ∈ first :: remaining, FiniteScore c.result) →
    FiniteScore (bestReturned (first :: remaining)) ∧
      ∀ c ∈ first :: remaining, FiniteScore c.score

theorem loopContract_proven : LoopContract := by
  intro first remaining finished
  have children : ∀ c ∈ first :: remaining, FiniteScore c.score :=
    fun c h => finite_neg (finished c h)
  have played := children first (by simp)
  have firstScore : max (-32767 : Int) first.score = first.score := Int.max_eq_right (by have := played.1; omega)
  refine ⟨?_, children⟩
  rw [bestReturned, List.foldl_cons, firstScore]
  exact best_fold_finite remaining _ played
    (fun c h => children c (List.mem_cons_of_mem _ h))

inductive Exit (history : History) (alpha beta : Int) : Int → Prop
  | leaf (raw : Int) : Exit history alpha beta (evalClamp raw)
  | leafCache {key word raw} (cached : EvalPublished word) (hit : EvalCache.read key word = some raw) :
      Exit history alpha beta raw
  | disc (discs : Int) (lo : -127 ≤ discs) (hi : discs ≤ 127) : Exit history alpha beta (discs * 256)
  | wipeout : Exit history alpha beta 16384
  | stopped : Exit history alpha beta 0
  | tt {board entry} (matched : TTRead history board entry) : Exit history alpha beta entry.score.toInt
  | etc {board entry} (matched : Lookup history board entry) : Exit history alpha beta (-entry.score.toInt)
  | probcut {rawBeta sel evaluation verify plan score}
      (threshold : rawBeta.toInt = beta)
      (cut : (guardedProbcut rawBeta sel evaluation verify plan).cutoff = some score) :
      Exit history alpha beta score
  | ttMargin {board entry} (matched : TTRead history board entry)
      (eligible : beta + ttMargin ≤ entry.score.toInt) : Exit history alpha beta (beta + ttMargin)

private theorem exit_finite {history : History} {alpha beta score : Int}
    (finiteHistory : FiniteHistory history) (window : Window alpha beta)
    (exit : Exit history alpha beta score) : FiniteScore score := by
  cases exit with
  | leaf => exact eval_clamp_finite _
  | leafCache cached hit =>
    have := evalCacheContract_proven _ _ _ cached hit
    unfold FiniteScore
    omega
  | disc _ lo hi => exact disc_score_finite _ lo hi
  | wipeout => exact ⟨by decide, by decide⟩
  | stopped => exact finite_zero
  | tt matched => exact (readContract_proven _ _ _ finiteHistory matched).1
  | etc matched => exact finite_neg (readContract_proven _ _ _ finiteHistory (.lookup matched)).1
  | probcut _ cut => exact ((probcutContract_proven ..).2 _ cut).2
  | ttMargin matched eligible =>
    exact margin_finite window (readContract_proven _ _ _ finiteHistory matched).1 eligible

inductive Returns (history : History) : Int → Int → Int → Prop
  | direct {alpha beta score} (exit : Exit history alpha beta score) : Returns history alpha beta score
  | passed {alpha beta score} (child : Returns history (-beta) (-alpha) score) :
      Returns history alpha beta (-score)
  | played {alpha beta} (first : ChildCall) (remaining : List ChildCall)
      (windows : ∀ child ∈ first :: remaining, child.window)
      (calls : ∀ child ∈ first :: remaining, Returns history child.alpha child.beta child.result) :
      Returns history alpha beta (bestReturned (first :: remaining))

def ReturnContract : Prop :=
  ∀ history alpha beta score, Produced history → Window alpha beta →
    Returns history alpha beta score → FiniteScore score

theorem returnContract_proven : ReturnContract := by
  intro history alpha beta score produced window execution
  induction execution with
  | direct exit => exact exit_finite (historyContract_proven _ produced) window exit
  | passed _ ih => exact finite_neg (ih (reflected window))
  | played first remaining windows _ ih =>
    exact (loopContract_proven first remaining (fun child member => ih child member (windows child member))).1

def MatchedTrials (search : Aspiration.Trials Payload) : Prop :=
  ∀ i alpha beta, Window alpha beta → (search i alpha beta).aborted = false →
    ∃ history, Produced history ∧ Returns history alpha beta (search i alpha beta).score

def AspirationContract : Prop :=
  ∀ (Payload : Type) (search : Aspiration.Trials Payload) (start : MidgameAspiration.Attempt),
    MatchedTrials search → start.window → 512 ≤ start.delta →
    ∃ result, MidgameAspiration.run search 16 0 start = some result ∧
      MidgameAspiration.Witness search 0 16 start result

theorem aspirationContract_proven : AspirationContract := by
  intro Payload search start matched valid width
  apply MidgameAspiration.terminationContract_proven _ search start _ valid width
  intro i alpha beta window completed
  obtain ⟨history, produced, source⟩ := matched i alpha beta window completed
  exact returnContract_proven history alpha beta _ produced window source

example : (guardedProbcut (-32767) 0 0 (fun _ => 0) (.useStatic (-32766))).cutoff = none := by decide
example : (chosenChild true true false (-100) 201 ⟨9, 11, -22⟩).score = 22 := by decide
example : bestReturned [⟨-32767, 32767, 3⟩, ⟨-1, 0, 2⟩] = -2 := by decide

end NeuralReversi.MidgameScoreRange
