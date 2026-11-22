import NeuralReversi.EndgameSearch.Cancellation
import NeuralReversi.PvsParallel.Cutoffs
import NeuralReversi.PvsParallel.Prefix
import NeuralReversi.Threading.SplitReturn

namespace NeuralReversi.PvsParallel

open Aspiration (SoundResult Window)
open PvsMoveLoop (Bound ChildSearch Tree bestValue bestValue_map minimax scoreInf scoreMax)
open Threading
open AtomicWord (Read Write signed32At)
open Threading.CutoffVisibility (History)
open Threading.SplitCompletion (Activation Retirement)
open Threading.SplitResult (Report slots)
open Threading.SplitWindow (AlphaRead Matches State reSearch selectReport)
open Threading.SplitReturn (Dispatches LocalCutoff ReportSteps checked stateAfter)
open ScoreRefresh (Score fromDiscDiff toDiscDiff)

def childAt (moves : List Tree) (index : Nat) : Tree :=
  (moves[index]?).getD (.leaf ⟨0, by decide⟩)

private theorem childAt_mem {moves : List Tree} {i : Nat} (inside : i < moves.length) :
    childAt moves i ∈ moves := by
  simp [childAt, List.getElem?_eq_getElem inside]

private theorem range_children (moves : List Tree) :
    (List.range moves.length).map (childAt moves) = moves := by
  apply List.ext_getElem
  · simp
  · intro i _ inside
    simp [childAt, List.getElem?_eq_getElem inside]

private theorem bestValue_append (value : Move → Int) (first rest : List Move) (seed : Int) :
    bestValue value (first ++ rest) seed = bestValue value rest (bestValue value first seed) := by
  induction first generalizing seed with
  | nil => rfl
  | cons move tail ih => exact ih (max seed (value move))

private theorem partition_value (moves : List Tree) (start : Nat) (inside : start ≤ moves.length) :
    let value := fun i => -minimax (childAt moves i)
    SplitResult.nodeValue value start moves.length (bestValue value (List.range start) (-scoreInf)) =
      bestValue (fun child => -minimax child) moves (-scoreInf) := by
  intro value
  have order : List.range moves.length = List.range start ++ slots start moves.length := by
    rw [List.range_eq_range', List.range_eq_range', List.range'_eq_append_iff]
    exact ⟨start, inside, rfl, by simp [slots]⟩
  unfold SplitResult.nodeValue
  rw [← bestValue_append, ← order, ← bestValue_map (fun child => -minimax child) (childAt moves),
      range_children]

private theorem ordered_value (head : Tree) (tail moves : List Tree) (order : moves.Perm (head :: tail)) :
    bestValue (fun child => -minimax child) moves (-scoreInf) = minimax (.turn head tail) := by
  rw [PvsMoveLoop.minimax_turn]
  exact PvsMoveLoop.bestValue_eq_of_same_moves _ _ (fun _ => order.mem_iff)

structure ReportReads (history : History) (pv : Bool) (beta : Int) (initial : State)
    (earlier : List Report) (report : Report) where
  writes : List Write
  first : Fin writes.length
  last : Fin writes.length
  scoutLoad : Read writes
  wideLoad : Read writes
  scout : Int
  wide : Int
  probe : AlphaRead history scoutLoad first last
  matched : Matches writes first last pv beta initial (earlier.map Report.score)
  reread : reSearch pv (signed32At writes scoutLoad.source) scout = true →
    AlphaRead history wideLoad first last
  selected : report = selectReport pv report.index (signed32At writes scoutLoad.source)
    (signed32At writes wideLoad.source) scout wide

def ReportReads.alpha (reads : ReportReads history pv beta initial earlier report) : Int :=
  signed32At reads.writes reads.scoutLoad.source

def ReportReads.fresh (reads : ReportReads history pv beta initial earlier report) : Int :=
  signed32At reads.writes reads.wideLoad.source

structure SplitExecution (len : Nat) (pv : Bool) (original beta score : Int) where
  history : History
  mask : List Write
  activation : Activation mask
  wait : Retirement history mask activation
  check : SplitReturn.Observation.OwnerCheck history
  width : Nat
  start : BitVec width
  last : BitVec width
  issued : List (BitVec width)
  entry : State
  reports : List Report
  writes : List Write
  result : Read writes
  lastScore : Fin writes.length
  capacity : len < 2 ^ width
  startValid : start.toNat ≤ len
  claims : MoveList.Concurrent.Claims len start issued last
  valid : ∀ r ∈ reports, r.index ∈ slots start.toNat len
  dispatches : Dispatches history wait.read.event activation.actions activation.owner len last
    (issued.map BitVec.toNat) reports check (LocalCutoff pv beta entry reports)
  read : SplitReturn.Observation.ScoreRead history result lastScore wait.read.event
  stored : signed32At writes lastScore = (stateAfter pv beta entry reports).best
  accepted : checked check.stopped (signed32At writes result.source) = some score
  reportReads : ∀ earlier report later, reports = earlier ++ report :: later →
    ReportReads history pv beta entry earlier report

abbrev Search := Tree → Bool → Int → Int → Int → Prop

def ChildCalls (completed : Search) (child : Move → Tree) (moves : List Move)
    (observe : ChildSearch Move) : Prop :=
  ∀ move ∈ moves, ∀ pv a b, Window a b → (pv = false → b = a + 1) →
    completed (child move) pv a b (observe move pv a b)

inductive Execution (completed : Search) : Search
  | leaf : Execution completed (.leaf score) pv alpha beta score.val
  | pass : completed child pv (-beta) (-alpha) score →
      Execution completed (.pass child) pv alpha beta (-score)
  | cutoff (mc : Bitboard.Moves.X86Config) (fc : Flip.Dispatch.X86Config)
      (board : PvsBoardTree.Position) :
      Cutoffs.Execution mc fc board pv alpha beta score →
      Execution completed (PvsBoardTree.build mc fc board) pv alpha beta score
  | shallow (target : Flip.Dispatch.TargetConfig) (bmi2 : Bool)
      (mc : Bitboard.Moves.X86Config) (fc : Flip.Dispatch.X86Config) (address : EndgameTreeCache.Address)
      (board : PvsBoardTree.Position) (alpha beta raw : Score)
      (state : EndgameSearch.Cancellation.State) (overflowChecks aborted : Bool) :
      state.empties.matches board → EndgameTreeCache.soundHistory mc fc state.history →
      (let opts := EndgameSearch.orderedOptions target bmi2 mc fc address
       let result := EndgameSearch.Cancellation.run mc fc opts board (toDiscDiff alpha).raw.toInt state
       fromDiscDiff overflowChecks ⟨BitVec.ofInt 32 (result.score.getD 0)⟩ = .ok raw ∧
         EndgameSearch.Cancellation.publish result aborted raw = some raw) →
      Execution completed (PvsBoardTree.build mc fc board) false alpha.raw.toInt beta.raw.toInt raw.raw.toInt
  | sequential (moves : List Tree) (order : moves.Perm (head :: tail))
      (observe : ChildSearch Tree) :
      ChildCalls completed id moves observe →
      Execution completed (.turn head tail) pv alpha beta
        (PvsMoveLoop.scan pv observe beta moves true alpha (-scoreInf))
  | split (moves : List Tree) (order : moves.Perm (head :: tail))
      (observe : ChildSearch Nat) (split : SplitExecution moves.length pv alpha beta score) :
      Prefix.run pv observe beta (List.range split.start.toNat) true ⟨alpha, -scoreInf⟩ =
        some split.entry →
      ChildCalls completed (childAt moves) (List.range split.start.toNat) observe →
      (∀ earlier report later (same : split.reports = earlier ++ report :: later),
        let calls := split.reportReads earlier report later same
        let child := childAt moves report.index
        completed child false (-(calls.alpha + 1)) (-calls.alpha) calls.scout ∧
          (reSearch pv calls.alpha calls.scout = true →
            completed child true (-beta) (-calls.fresh) calls.wide)) →
      Execution completed (.turn head tail) pv alpha beta score

def Follows (completed : Search) : Prop :=
  ∀ tree pv alpha beta score, completed tree pv alpha beta score →
    Execution completed tree pv alpha beta score

theorem bound_sound (tree : Tree) (window : Window alpha beta)
    (bounds : Bound (minimax tree) alpha beta score) : SoundResult (minimax tree) alpha beta score := by
  obtain ⟨lower, upper⟩ := PvsMoveLoop.minimax_range tree
  have ⟨lo, ord, hi⟩ := window
  apply bounds.toSoundResult window
  · by_cases valid : -scoreInf < score
    · exact valid
    · have := bounds.1 (by unfold scoreInf at valid; omega)
      unfold scoreInf scoreMax at *
      omega
  · by_cases valid : score < scoreInf
    · exact valid
    · have := bounds.2 (by unfold scoreInf at valid; omega)
      unfold scoreInf scoreMax at *
      omega

theorem split_bounds (split : SplitExecution moves.length pv alpha beta score)
    (window : Window split.entry.alpha beta) (cap : split.entry.alpha ≤ max alpha split.entry.best)
    (null : pv = false → beta = split.entry.alpha + 1)
    (seed : Bound (bestValue (fun i => -minimax (childAt moves i)) (List.range split.start.toNat)
      (-scoreInf)) alpha beta split.entry.best)
    (calls : ∀ earlier report later (same : split.reports = earlier ++ report :: later),
      let read := split.reportReads earlier report later same
      let v := minimax (childAt moves report.index)
      (Window (-(read.alpha + 1)) (-read.alpha) →
        SoundResult v (-(read.alpha + 1)) (-read.alpha) read.scout) ∧
      (reSearch pv read.alpha read.scout = true → Window (-beta) (-read.fresh) →
        SoundResult v (-beta) (-read.fresh) read.wide)) :
    Bound (bestValue (fun child => -minimax child) moves (-scoreInf)) alpha beta score ∧
      ReportSteps pv beta split.entry (fun i => -minimax (childAt moves i)) split.reports := by
  let value := fun i => -minimax (childAt moves i)
  let prefixValue := bestValue value (List.range split.start.toNat) (-scoreInf)
  have valueCap (i : Nat) : value i ≤ scoreMax := by
    have := (PvsMoveLoop.minimax_range (childAt moves i)).1
    dsimp only [value]
    omega
  have prefixCap : prefixValue ≤ scoreMax :=
    PvsMoveLoop.bestValue_le _ _ _ _ (by decide) (fun i _ => valueCap i)
  have steps : ReportSteps pv beta split.entry value split.reports := by
    intro earlier report later same
    let read := split.reportReads earlier report later same
    obtain ⟨scout, wide⟩ := calls earlier report later same
    have matched := SplitWindow.publishedReportContract_proven split.history read.writes
      read.scoutLoad read.wideLoad read.first read.last pv alpha beta split.entry (earlier.map Report.score)
      value report.index read.scout read.wide read.probe read.reread read.matched window cap null
      (by simpa only [value, Int.neg_neg, ReportReads.alpha] using scout)
      (by simpa only [value, Int.neg_neg, ReportReads.alpha, ReportReads.fresh] using wide)
    rw [← read.selected] at matched
    obtain ⟨ordered, bound⟩ := matched
    cases bound with
    | accept _ valid _ => exact ⟨ordered, valid⟩
  have result := SplitReturn.returnedScoreContract_proven split.history split.mask split.activation split.wait
    split.width moves.length split.start split.last split.issued pv split.check alpha beta prefixValue
    split.entry value split.reports split.writes split.result split.lastScore score split.capacity split.claims
    window cap null seed prefixCap (fun i _ => valueCap i) steps split.valid split.dispatches split.read
    split.stored split.accepted
  rw [partition_value moves split.start.toNat split.startValid] at result
  exact ⟨result, steps⟩

private theorem execution_sound (completed : Search) (tree : Tree) (pv : Bool) (alpha beta score : Int)
    (window : Window alpha beta) (null : pv = false → beta = alpha + 1)
    (observed : Execution completed tree pv alpha beta score)
    (children : ∀ child, sizeOf child < sizeOf tree → ∀ cp a b s, Window a b →
      (cp = false → b = a + 1) → completed child cp a b s → SoundResult (minimax child) a b s) :
    SoundResult (minimax tree) alpha beta score := by
  cases observed with
  | leaf =>
    apply bound_sound _ window
    rw [PvsMoveLoop.minimax.eq_def]
    exact ⟨fun _ => Int.le_refl _, fun _ => Int.le_refl _⟩
  | pass child =>
    have proof := children _ (by simp) _ _ _ _ (PvsMoveLoop.reflectedWindow window)
      (by intro noPv; have := null noPv; omega) child
    apply bound_sound _ window
    rw [PvsMoveLoop.minimax.eq_def]
    exact (Bound.ofSoundResult proof).negate
  | cutoff mc fc board exit =>
    exact bound_sound (PvsBoardTree.build mc fc board) window
      (Cutoffs.contract_proven mc fc board pv alpha beta score null exit)
  | shallow target bmi2 mc fc address board alpha beta raw state overflowChecks aborted aligned cache accepted =>
    obtain ⟨_, _, _, score, converted, _, valid⟩ :=
      EndgameSearch.Cancellation.boundaryContract_proven target bmi2 mc fc address board alpha beta state
        overflowChecks aborted aligned cache (null rfl)
    have same := Except.ok.inj (accepted.1.symm.trans converted)
    cases same
    exact valid accepted.2
  | sequential moves order observe calls =>
    rename_i head tail
    rw [← ordered_value head tail moves order]
    apply PvsMoveLoop.contract_proven Tree pv observe (fun t => -minimax t) moves alpha beta
    · intro empty
      have len := order.length_eq
      simp [empty] at len
    · exact window
    · exact null
    · intro child _
      have := (PvsMoveLoop.minimax_range child).1
      omega
    · intro child member cp a b w narrow
      simpa only [Int.neg_neg] using
        children child (Tree.earlier ⟨child, order.mem_iff.mp member⟩) cp a b _ w narrow
          (calls child member cp a b w narrow)
  | split moves order observe split continued prefixCalls reportCalls =>
    rename_i head tail
    let value := fun i => -minimax (childAt moves i)
    have smaller (i : Nat) (inside : i < moves.length) :
        sizeOf (childAt moves i) < sizeOf (Tree.turn head tail) :=
      Tree.earlier ⟨_, order.mem_iff.mp (childAt_mem inside)⟩
    have headSound := Prefix.contract_proven Nat pv observe value (List.range split.start.toNat) alpha beta
      split.entry window null continued
      (fun index _ => by have := (PvsMoveLoop.minimax_range (childAt moves index)).1; dsimp only [value]; omega)
      (fun index member cp a b w narrow => by
        have inside := Nat.lt_of_lt_of_le (List.mem_range.mp member) split.startValid
        simpa only [value, Int.neg_neg] using
          children _ (smaller index inside) cp a b _ w narrow (prefixCalls index member cp a b w narrow))
    have bound := (split_bounds split headSound.1 headSound.2.1 headSound.2.2.1 headSound.2.2.2 (by
      intro earlier report later same
      obtain ⟨scout, wide⟩ := reportCalls earlier report later same
      have member : report ∈ split.reports := by simp [same]
      have inSuffix := (List.mem_range'_1.mp (split.valid report member)).2
      have inside : report.index < moves.length := by have := split.startValid; omega
      refine ⟨fun w => children _ (smaller _ inside) false _ _ _ w
        (by intro _; omega) scout, fun retry w =>
        children _ (smaller _ inside) true _ _ _ w (by simp) (wide retry)⟩)).1
    apply bound_sound _ window
    rw [← ordered_value head tail moves order]
    exact bound

def Contract : Prop :=
  ∀ (completed : Search), Follows completed →
    ∀ tree pv alpha beta score, Window alpha beta → (pv = false → beta = alpha + 1) →
      completed tree pv alpha beta score → SoundResult (minimax tree) alpha beta score

theorem sound (completed : Search) (follows : Follows completed) (tree : Tree) (pv : Bool)
    (alpha beta score : Int) (window : Window alpha beta) (null : pv = false → beta = alpha + 1)
    (trace : completed tree pv alpha beta score) : SoundResult (minimax tree) alpha beta score := by
  exact execution_sound completed tree pv alpha beta score window null (follows _ _ _ _ _ trace)
    (fun child smaller cp a b s cw narrow run =>
      sound completed follows child cp a b s cw narrow run)
termination_by sizeOf tree

theorem contract_proven : Contract := sound

def BoardContract : Prop :=
  ∀ (completed : Search), Follows completed →
    ∀ (mc : Bitboard.Moves.X86Config) (fc : Flip.Dispatch.X86Config)
      (board : PvsBoardTree.Position) (pv : Bool) (alpha beta score : Int),
      Window alpha beta → (pv = false → beta = alpha + 1) →
      completed (PvsBoardTree.build mc fc board) pv alpha beta score →
      SoundResult (PvsBoardTree.minimax mc fc board) alpha beta score

theorem boardContract_proven : BoardContract := by
  intro completed follows mc fc board pv alpha beta score window null trace
  exact contract_proven completed follows _ pv alpha beta score window null trace

theorem fullWindow_exact (completed : Search) (follows : Follows completed)
    (mc : Bitboard.Moves.X86Config) (fc : Flip.Dispatch.X86Config) (board : PvsBoardTree.Position)
    (score : Int) (trace : completed (PvsBoardTree.build mc fc board) true (-scoreInf) scoreInf score) :
    score = PvsBoardTree.minimax mc fc board := by
  have result := boardContract_proven completed follows mc fc board true (-scoreInf) scoreInf score
    ⟨by decide, by decide, by decide⟩ (by simp) trace
  exact result.exact result.lower result.upper

private theorem scan_map (child : Move → Tree) (moves : List Move) (pv : Bool)
    (beta : Int) (first : Bool) (alpha best : Int) :
    PvsMoveLoop.scan pv (fun m => PvsMoveLoop.run (child m)) beta moves first alpha best =
      PvsMoveLoop.scan pv PvsMoveLoop.run beta (moves.map child) first alpha best := by
  induction moves generalizing first alpha best with
  | nil => rfl
  | cons move moves ih => simp [PvsMoveLoop.scan, PvsMoveLoop.moveScore, ih]

theorem sequential_follows :
    Follows (fun tree pv alpha beta score => score = PvsMoveLoop.run tree pv alpha beta) := by
  intro tree pv alpha beta score computes
  subst score
  cases tree with
  | leaf score =>
    rw [PvsMoveLoop.run.eq_def]
    exact .leaf
  | pass child =>
    rw [PvsMoveLoop.run.eq_def]
    exact .pass rfl
  | turn head tail =>
    rw [PvsMoveLoop.run.eq_def]
    change Execution _ _ _ _ _ (PvsMoveLoop.scan pv (fun child => PvsMoveLoop.run child.val) beta
      (head :: tail).attach true alpha (-scoreInf))
    rw [scan_map Subtype.val, List.attach_map_subtype_val]
    exact .sequential (head :: tail) (List.Perm.refl _) PvsMoveLoop.run (fun _ _ _ _ _ _ _ => rfl)

end NeuralReversi.PvsParallel
