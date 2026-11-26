import NeuralReversi.EndgameSearch.Direct
import NeuralReversi.EndgameSearch.Cancellation
import NeuralReversi.PvsParallel.Cutoffs
import NeuralReversi.PvsParallel.Prefix
import NeuralReversi.Threading.SplitReturn
import NeuralReversi.Threading.SplitWindow.Stores

namespace NeuralReversi.PvsParallel

open Aspiration (SoundResult Window)
open PvsMoveLoop (Bound ChildSearch ScanCalls Tree bestValue bestValue_map minimax scoreInf scoreMax)
open Threading
open AtomicWord (Read Write signed32At)
open Threading.CutoffVisibility (History)
open Threading.SplitCompletion (Activation Retirement)
open Threading.SplitResult (Report slots)
open Threading.SplitWindow (AlphaRead BestStoreWindow State StoreWindow reSearch selectReport)
open Threading.SplitReturn (Dispatches LocalCutoff ReportSteps checked stateAfter)
open ScoreRefresh (Score fromDiscDiff toDiscDiff)

def childAt (moves : List Tree) (index : Nat) : Tree :=
  (moves[index]?).getD (.leaf ⟨0, by decide⟩)

private theorem childAt_mem {moves : List Tree} {i : Nat} (inside : i < moves.length) :
    childAt moves i ∈ moves := by
  simp [childAt, List.getElem?_eq_getElem inside]

theorem childAt_map (child : Move → Tree) (labels : List Move) (fallback : Move) {i : Nat}
    (inside : i < labels.length) :
    childAt (labels.map child) i = child (labels[i]?.getD fallback) := by
  simp [childAt, List.getElem?_map, List.getElem?_eq_getElem inside]

private theorem range_children (moves : List Tree) :
    (List.range moves.length).map (childAt moves) = moves := by
  apply List.ext_getElem
  · simp
  · intro i _ inside
    simp [childAt, List.getElem?_eq_getElem inside]

private theorem partition_value (moves : List Tree) (start : Nat) (inside : start ≤ moves.length) :
    let value := fun i => -minimax (childAt moves i)
    SplitResult.nodeValue value start moves.length (bestValue value (List.range start) (-scoreInf)) =
      bestValue (fun child => -minimax child) moves (-scoreInf) := by
  intro value
  have order : List.range moves.length = List.range start ++ slots start moves.length := by
    rw [List.range_eq_range', List.range_eq_range', List.range'_eq_append_iff]
    exact ⟨start, inside, rfl, by simp [slots]⟩
  unfold SplitResult.nodeValue
  rw [← SplitResult.bestValue_append, ← order, ← bestValue_map (fun child => -minimax child) (childAt moves),
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
  stores : StoreWindow writes first last pv beta initial (earlier.map Report.score)
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
  read : SplitReturn.Observation.ScoreRead history wait result lastScore
  bestStores : BestStoreWindow writes lastScore entry.best (reports.map Report.score)
  accepted : checked check.stopped (signed32At writes result.source) = some score
  reportReads : ∀ earlier report later, reports = earlier ++ report :: later →
    ReportReads history pv beta entry earlier report
  reportPublications : ∀ earlier report later (same : reports = earlier ++ report :: later),
    let reads := reportReads earlier report later same
    { publication : SplitReturn.Observation.ReportPublication activation //
      report ∈ (dispatches.work publication.worker).reports ∧
      publication.acquired = reads.probe.acquire ∧
      ∀ needed, publication.acquired = (reads.reread needed).acquire }

structure SplitExecution.MoveRead (split : SplitExecution len pv alpha beta score)
    (encode : Option Move → BitVec 8) (moveAt : Nat → Option Move) (seed picked : Option Move) where
  writes : List Write
  load : Read writes
  last : Fin writes.length
  order : SplitReturn.Observation.ScoreRead split.history split.wait load last
  afterScore : AtomicWord.SequencedBefore split.result.event load.event
  stores : SplitWindow.MoveStoreWindow encode moveAt pv beta split.entry seed split.reports picked writes last

theorem SplitExecution.MoveRead.value
    (read : SplitExecution.MoveRead split encode moveAt seed picked) :
    (read.writes.get read.load.source).value.setWidth 8 = encode picked := by
  have source := (SplitReturn.Observation.scoreReadContract_proven split.history read.writes read.load
    read.last split.wait read.order).1
  rw [source]
  exact read.stores.lastValue

def SplitMoveReadContract : Prop :=
  ∀ (Move : Type) len pv alpha beta score (split : SplitExecution len pv alpha beta score)
    (encode : Option Move → BitVec 8) (moveAt : Nat → Option Move) (seed picked : Option Move)
    (read : SplitExecution.MoveRead split encode moveAt seed picked),
    (read.writes.get read.load.source).value.setWidth 8 = encode picked ∧
      (picked = seed ∨ ∃ report ∈ split.reports, picked = moveAt report.index)

theorem splitMoveReadContract_proven : SplitMoveReadContract := by
  intro Move len pv alpha beta score split encode moveAt seed picked read
  exact ⟨read.value, read.stores.member⟩

abbrev Search := Tree → Bool → Int → Int → Int → Prop

structure ReportUse where
  history : History
  load : AtomicWord.Event
  next : AtomicWord.Event
  ownerRead : AtomicWord.Event
  handoff : CutoffVisibility.HappensBefore history next ownerRead

inductive ChildUse where
  | serial
  | report (use : ReportUse)

abbrev ChildResults := ChildUse → Search

def SplitExecution.reportUse (split : SplitExecution len pv alpha beta score)
    (earlier : List Report) (report : Report) (later : List Report)
    (same : split.reports = earlier ++ report :: later) (wide : Bool) : ChildUse := by
  let reads := split.reportReads earlier report later same
  let recorded := split.reportPublications earlier report later same
  let publication := recorded.val
  let loaded := if wide then reads.wideLoad.event else reads.scoutLoad.event
  have visible := publication.beforeOwnerRead split.wait split.read.beforeLoad
  by_cases retry : wide = false ∧ reSearch pv reads.alpha reads.scout = true
  · let reread := reads.reread retry.2
    have beforeLock := reread.beforeAcquire
    have sameLock : publication.acquired = reread.acquire := recorded.property.2.2 retry.2
    rw [← sameLock] at beforeLock
    exact .report ⟨split.history, loaded, reads.wideLoad.event, split.result.event,
      .trans (.sequenced beforeLock) visible⟩
  · exact .report ⟨split.history, loaded, publication.acquired, split.result.event, visible⟩

def ReportCalls (completed : ChildResults) (moves : List Tree)
    (split : SplitExecution moves.length pv alpha beta score) : Prop :=
  ∀ earlier report later (same : split.reports = earlier ++ report :: later),
    let reads := split.reportReads earlier report later same
    let child := childAt moves report.index
    completed (split.reportUse earlier report later same false) child false (-(reads.alpha + 1)) (-reads.alpha)
      reads.scout ∧
      (reSearch pv reads.alpha reads.scout = true →
        completed (split.reportUse earlier report later same true) child true (-beta) (-reads.fresh) reads.wide)

theorem ReportCalls.weaken (calls : ReportCalls source moves split)
    (lift : ∀ stage tree pv alpha beta score, source stage tree pv alpha beta score →
      target stage tree pv alpha beta score) : ReportCalls target moves split := by
  intro earlier report later same
  obtain ⟨scout, wide⟩ := calls earlier report later same
  exact ⟨lift _ _ _ _ _ _ scout, fun retry => lift _ _ _ _ _ _ (wide retry)⟩

abbrev Exit := Bitboard.Moves.X86Config → Flip.Dispatch.X86Config →
  PvsBoardTree.Position → Bool → Int → Int → Int → Prop

inductive ExecutionWithUses (exit : Exit) (completed : ChildResults) : Search
  | leaf : ExecutionWithUses exit completed (.leaf score) pv alpha beta score.val
  | pass : completed .serial child pv (-beta) (-alpha) score →
      ExecutionWithUses exit completed (.pass child) pv alpha beta (-score)
  | cutoff (mc : Bitboard.Moves.X86Config) (fc : Flip.Dispatch.X86Config)
      (board : PvsBoardTree.Position) :
      exit mc fc board pv alpha beta score →
      ExecutionWithUses exit completed (PvsBoardTree.build mc fc board) pv alpha beta score
  | direct (target : Flip.Dispatch.TargetConfig) (bmi2 : Bool)
      (mc : Bitboard.Moves.X86Config) (fc : Flip.Dispatch.X86Config) (board : PvsBoardTree.Position)
      (alpha beta raw : Score) (checked : Bool) :
      EndgameSearch.Direct.Returned target bmi2 mc fc board alpha checked raw →
      ExecutionWithUses exit completed (PvsBoardTree.build mc fc board) false
        alpha.raw.toInt beta.raw.toInt raw.raw.toInt
  | shallow (target : Flip.Dispatch.TargetConfig) (bmi2 : Bool)
      (mc : Bitboard.Moves.X86Config) (fc : Flip.Dispatch.X86Config) (address : EndgameTreeCache.Address)
      (board : PvsBoardTree.Position) (alpha beta raw : Score)
      (state : EndgameSearch.Cancellation.State) (overflowChecks aborted : Bool) :
      state.empties.matches board → EndgameTreeCache.soundHistory mc fc state.history →
      (let opts := EndgameSearch.orderedOptions target bmi2 mc fc address
       let result := EndgameSearch.Cancellation.run mc fc opts board (toDiscDiff alpha).raw.toInt state
       fromDiscDiff overflowChecks ⟨BitVec.ofInt 32 (result.score.getD 0)⟩ = .ok raw ∧
         EndgameSearch.Cancellation.publish result aborted raw = some raw) →
      ExecutionWithUses exit completed (PvsBoardTree.build mc fc board) false
        alpha.raw.toInt beta.raw.toInt raw.raw.toInt
  | sequential (moves : List Move) (child : Move → Tree) (order : (moves.map child).Perm (head :: tail))
      (observe : ChildSearch Move) :
      ScanCalls (fun move => completed .serial (child move)) observe pv beta moves true alpha (-scoreInf) →
      ExecutionWithUses exit completed (.turn head tail) pv alpha beta
        (PvsMoveLoop.scan pv observe beta moves true alpha (-scoreInf))
  | split (moves : List Tree) (order : moves.Perm (head :: tail))
      (observe : ChildSearch Nat) (split : SplitExecution moves.length pv alpha beta score) :
      Prefix.run pv observe beta (List.range split.start.toNat) true ⟨alpha, -scoreInf⟩ =
        some split.entry →
      ScanCalls (fun index => completed .serial (childAt moves index)) observe pv beta
        (List.range split.start.toNat) true alpha (-scoreInf) →
      ReportCalls completed moves split →
      ExecutionWithUses exit completed (.turn head tail) pv alpha beta score

abbrev ExecutionWith (exit : Exit) (completed : Search) := ExecutionWithUses exit (fun _ => completed)

abbrev Execution := ExecutionWith Cutoffs.Execution

theorem ExecutionWithUses.ofSerialRun (child : Move → Tree)
    (order : (q.pending.map child).Perm (head :: tail)) (first : q.idx = 0)
    (trace : Prefix.Serial.Run (fun move => completed .serial (child move)) observe pv beta q
      ⟨alpha, -scoreInf⟩ visited remaining (.returned score)) :
    ExecutionWithUses exit completed (.turn head tail) pv alpha beta score := by
  obtain ⟨permutation, _, reproduced⟩ := trace.reproduced
  have scanEvidence : PvsMoveLoop.scan pv observe beta (visited ++ remaining.pending) true alpha (-scoreInf) = score ∧
      ScanCalls (fun move => completed .serial (child move)) observe pv beta
        (visited ++ remaining.pending) true alpha (-scoreInf) := by
    simpa [Prefix.Serial.Reproduced, first] using reproduced
  rw [← scanEvidence.1]
  exact .sequential _ child ((permutation.map child).trans order) observe scanEvidence.2

theorem ExecutionWithUses.ofSerialHandoff (child : Move → Tree) (fallback : Move)
    (order : (q.pending.map child).Perm (head :: tail)) (first : q.idx = 0)
    (split : SplitExecution ((visited ++ remaining.pending).map child).length pv alpha beta score)
    (beginAt : split.start.toNat = remaining.idx)
    (trace : Prefix.Serial.Run (fun move => completed .serial (child move)) observe pv beta q
      ⟨alpha, -scoreInf⟩ visited remaining (.handoff split.entry))
    (reports : ReportCalls completed ((visited ++ remaining.pending).map child) split) :
    ExecutionWithUses exit completed (.turn head tail) pv alpha beta score := by
  let labels := visited ++ remaining.pending
  let atIndex := fun i : Nat => labels[i]?.getD fallback
  have indexed := trace.handoffOrdered (target := fun i => completed .serial (childAt (labels.map child) i))
    first labels fallback rfl (by
      intro i inside p a b s used
      rwa [childAt_map child labels fallback inside])
  rw [← beginAt] at indexed
  exact .split _ ((trace.reproduced.1.map child).trans order) (fun i => observe (atIndex i)) split
    indexed.1 indexed.2 reports

def SerialExecutionContract : Prop :=
  ∀ (Move : Type) (exit : Exit) (completed : ChildResults) (child : Move → Tree)
    (head : Tree) (tail : List Tree) (observe : ChildSearch Move) (pv : Bool) (alpha beta score : Int)
    (q remaining : PvsPostpone.Queue Move) (visited : List Move),
    (q.pending.map child).Perm (head :: tail) → q.idx = 0 →
    (Prefix.Serial.Run (fun move => completed .serial (child move)) observe pv beta q
      ⟨alpha, -scoreInf⟩ visited remaining (.returned score) →
      ExecutionWithUses exit completed (.turn head tail) pv alpha beta score) ∧
    (∀ split : SplitExecution ((visited ++ remaining.pending).map child).length pv alpha beta score,
      split.start.toNat = remaining.idx →
      Prefix.Serial.Run (fun move => completed .serial (child move)) observe pv beta q
        ⟨alpha, -scoreInf⟩ visited remaining (.handoff split.entry) →
      ReportCalls completed ((visited ++ remaining.pending).map child) split →
      ExecutionWithUses exit completed (.turn head tail) pv alpha beta score)

theorem serialExecutionContract_proven : SerialExecutionContract := by
  intro Move exit completed child head tail observe pv alpha beta score q remaining visited order first
  obtain ⟨fallback, _, _⟩ := List.mem_map.mp (order.mem_iff.mpr (by simp : head ∈ head :: tail))
  exact ⟨ExecutionWithUses.ofSerialRun child order first,
    fun split => ExecutionWithUses.ofSerialHandoff child fallback order first split⟩

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
    have storedAlpha := SplitWindow.storeContract_proven read.writes read.first read.last pv alpha beta
      split.entry (earlier.map Report.score) read.stores window cap
    have matched := SplitWindow.publishedReportContract_proven split.history read.writes
      read.scoutLoad read.wideLoad read.first read.last pv alpha beta split.entry (earlier.map Report.score)
      value report.index read.scout read.wide read.probe read.reread storedAlpha window cap null
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
    split.bestStores split.accepted
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
  | direct _ _ mc fc board _ beta _ _ call =>
    exact bound_sound (PvsBoardTree.build mc fc board) window (call.bound beta (null rfl))
  | shallow target bmi2 mc fc address board alpha beta raw state overflowChecks aborted aligned cache accepted =>
    obtain ⟨_, _, _, score, converted, _, valid⟩ :=
      EndgameSearch.Cancellation.boundaryContract_proven target bmi2 mc fc address board alpha beta state
        overflowChecks aborted aligned cache (null rfl)
    have same := Except.ok.inj (accepted.1.symm.trans converted)
    cases same
    exact valid accepted.2
  | sequential moves child order observe calls =>
    rename_i head tail
    have value := bestValue_map (fun tree => -minimax tree) child moves (-scoreInf)
    rw [ordered_value head tail _ order] at value
    rw [value]
    apply PvsMoveLoop.selectedContract_proven _ pv observe (fun m => -minimax (child m))
      (fun m => completed (child m)) moves alpha beta
    · intro empty
      have len := order.length_eq
      simp [empty] at len
    · exact window
    · exact null
    · intro m _
      have := (PvsMoveLoop.minimax_range (child m)).1
      omega
    · exact calls
    · intro m member cp a b s w narrow returned
      simpa only [Int.neg_neg] using
        children (child m) (Tree.earlier ⟨_, order.mem_iff.mp (List.mem_map_of_mem member)⟩)
          cp a b s w narrow returned
  | split moves order observe split continued prefixCalls reportCalls =>
    rename_i head tail
    let value := fun i => -minimax (childAt moves i)
    have smaller (i : Nat) (inside : i < moves.length) :
        sizeOf (childAt moves i) < sizeOf (Tree.turn head tail) :=
      Tree.earlier ⟨_, order.mem_iff.mp (childAt_mem inside)⟩
    have headSound := Prefix.selectedContract_proven Nat pv observe value (List.range split.start.toNat)
      (fun index => completed (childAt moves index)) alpha beta split.entry window null continued
      (fun index _ => by have := (PvsMoveLoop.minimax_range (childAt moves index)).1; dsimp only [value]; omega)
      prefixCalls (fun index member cp a b s w narrow returned => by
        have inside := Nat.lt_of_lt_of_le (List.mem_range.mp member) split.startValid
        simpa only [value, Int.neg_neg] using
          children _ (smaller index inside) cp a b s w narrow returned)
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
    exact .sequential (head :: tail) id (by simp) PvsMoveLoop.run
      (ScanCalls.ofAll (by intros; rfl))

end NeuralReversi.PvsParallel
