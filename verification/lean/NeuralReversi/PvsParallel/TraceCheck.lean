import NeuralReversi.Board.MovesProperties
import NeuralReversi.PvsBoardTree
import NeuralReversi.PvsParallel.Cutoffs
import NeuralReversi.PvsParallel.Prefix
import NeuralReversi.PvsParallel.TraceCheck.Frames
import NeuralReversi.TranspositionTable.LookupCut
import NeuralReversi.TranspositionTable.Snapshot
import NeuralReversi.TranspositionTable.StoredBound
import Lean.Data.Json.FromToJson

namespace NeuralReversi.PvsParallel.TraceCheck

open Lean
open Aspiration (Window SoundResult)
open PvsPostpone (Queue step)
open PvsMoveLoop (Bound ChildSearch MoveCalls bestValue moveScore scoreInf scoreMax)
open Prefix.Serial (Outcome Run)
open Threading.SplitResult (Merge slots)
open Threading.SplitWindow (State choosesMove marksCutoff publish reSearch)

structure ChildReturn where
  frame : Nat
  square : Nat
  pv : Bool
  alpha : Int
  beta : Int
  score : Int
  deriving FromJson

def findChild (calls : List ChildReturn) (move : Nat) (pv : Bool) (alpha beta : Int) : Option Int :=
  (calls.find? fun child =>
    child.square == move && child.pv == pv && child.alpha == alpha && child.beta == beta).map (·.score)

def observe (calls : List ChildReturn) : ChildSearch Nat :=
  fun move pv alpha beta => (findChild calls move pv alpha beta).getD 0

def Completed (calls : List ChildReturn) (move : Nat) (pv : Bool) (alpha beta score : Int) : Prop :=
  findChild calls move pv alpha beta = some score

private def checkedCalls (calls : List ChildReturn) (pv first : Bool) (move : Nat) (alpha beta : Int) : Bool :=
  let scout := findChild calls move false (-(alpha + 1)) (-alpha)
  let wide := (findChild calls move true (-beta) (-alpha)).isSome
  if pv && first then wide
  else scout.isSome && (if pv = true ∧ alpha < -scout.getD 0 then wide else true)

private theorem completed_of_some (present : (findChild calls move pv alpha beta).isSome = true) :
    Completed calls move pv alpha beta (observe calls move pv alpha beta) := by
  unfold Completed observe
  cases found : findChild calls move pv alpha beta <;> simp_all

private theorem checkedCalls_sound (checked : checkedCalls calls pv first move alpha beta = true) :
    MoveCalls (Completed calls) (observe calls) pv first move alpha beta := by
  unfold checkedCalls at checked
  unfold MoveCalls
  by_cases eldest : (pv && first) = true
  · rw [ite_eq_left eldest] at checked ⊢
    exact completed_of_some checked
  · rw [ite_eq_right eldest] at checked ⊢
    rw [Bool.and_eq_true_iff] at checked
    refine ⟨completed_of_some checked.1, ?_⟩
    intro retry
    apply completed_of_some
    have required : pv = true ∧ alpha < -(findChild calls move false (-(alpha + 1)) (-alpha)).getD 0 := retry
    have wide := checked.2
    rwa [ite_eq_left required] at wide

structure Selection where
  index : Nat
  freshEnd : Nat
  square : Nat
  score : Option Int
  deriving FromJson

structure SerialResult where
  visited : List Nat
  remaining : Queue Nat
  outcome : Outcome

def replay (calls : List ChildReturn) (pv : Bool) (beta : Int) (handoff : Bool) :
    List Selection → Queue Nat → State → Nat → Option SerialResult
  | [], queue, state, _ =>
    if handoff then some ⟨[], queue, .handoff state⟩
    else if queue.pending.isEmpty then some ⟨[], queue, .returned state.best⟩
    else none
  | choice :: rest, queue, state, freshEnd => do
    if choice.index != queue.idx || choice.freshEnd != freshEnd ||
        queue.fresh.length != freshEnd - queue.idx then none
    else match choice.score with
    | none =>
      if queue.fresh.head? != some choice.square then none
      else match step true queue with
        | some (none, next) => replay calls pv beta handoff rest next state (freshEnd - 1)
        | _ => none
    | some score =>
      let first := queue.idx == 0
      if score != moveScore pv first (observe calls) choice.square state.alpha beta ||
          !checkedCalls calls pv first choice.square state.alpha beta then none
      else match step false queue with
        | some (some move, next) =>
          if move != choice.square then none
          else if marksCutoff pv beta state score then
            if rest.isEmpty && !handoff then some ⟨[move], next, .returned score⟩ else none
          else (replay calls pv beta handoff rest next (publish pv beta state score) freshEnd).map fun after =>
            { after with visited := move :: after.visited }
        | _ => none

theorem replay_sound (checked : replay calls pv beta handoff steps queue state freshEnd = some result) :
    Run (Completed calls) (observe calls) pv beta queue state result.visited result.remaining result.outcome := by
  induction steps generalizing queue state freshEnd result with
  | nil =>
    rw [replay] at checked
    split at checked
    · cases Option.some.inj checked
      exact .handoff
    · split at checked
      · rename_i empty
        cases Option.some.inj checked
        exact .exhausted (by simpa using empty)
      · contradiction
  | cons choice rest ih =>
    rw [replay] at checked
    split at checked
    · contradiction
    · cases scoreField : choice.score with
      | none =>
        simp only [scoreField] at checked
        split at checked
        · contradiction
        · cases advance : step true queue with
          | none => simp [advance] at checked
          | some pair =>
            obtain ⟨taken, next⟩ := pair
            cases taken with
            | none =>
              rw [advance] at checked
              exact .postpone advance (ih checked)
            | some m => simp [advance] at checked
      | some score =>
        simp only [scoreField] at checked
        split at checked
        · contradiction
        · rename_i checkedChild
          have guards : score = moveScore pv (queue.idx == 0) (observe calls) choice.square state.alpha beta ∧
              checkedCalls calls pv (queue.idx == 0) choice.square state.alpha beta = true := by
            simpa using checkedChild
          cases advance : step false queue with
          | none => simp [advance] at checked
          | some pair =>
            obtain ⟨taken, next⟩ := pair
            cases taken with
            | none => simp [advance] at checked
            | some move =>
              simp only [advance] at checked
              by_cases selected : move = choice.square
              · rcases guards with ⟨rfl, checkedCall⟩
                subst move
                simp only [bne_self_eq_false, Bool.false_eq_true, ite_false] at checked
                have child := checkedCalls_sound checkedCall
                split at checked
                · rename_i stops
                  split at checked
                  · cases Option.some.inj checked
                    exact .cutoff advance child stops
                  · contradiction
                · rename_i continues
                  have goes : marksCutoff pv beta state
                      (moveScore pv (queue.idx == 0) (observe calls) choice.square state.alpha beta) = false := by
                    simpa using continues
                  cases later : replay calls pv beta handoff rest next
                      (publish pv beta state
                        (moveScore pv (queue.idx == 0) (observe calls) choice.square state.alpha beta)) freshEnd with
                  | none => simp [later] at checked
                  | some after =>
                    simp only [later, Option.map_some] at checked
                    cases Option.some.inj checked
                    exact .searched advance child goes (ih later)
              · simp [selected] at checked

structure WideReturn where
  frame : Nat
  alpha : Int
  score : Int
  deriving FromJson

structure ReportRow where
  index : Nat
  square : Nat
  worker : Nat
  scoutFrame : Nat
  scoutAlpha : Int
  scout : Int
  wide : Option WideReturn
  score : Int
  bestAfter : Int
  alphaAfter : Int
  moveAfter : Option Nat
  deriving FromJson

private def ReportRow.fresh (row : ReportRow) : Int := (row.wide.map (·.alpha)).getD row.scoutAlpha
private def ReportRow.wideScore (row : ReportRow) : Int := (row.wide.map (·.score)).getD 0

def ReportRow.selected (row : ReportRow) (pv : Bool) : Threading.SplitResult.Report :=
  Threading.SplitWindow.selectReport pv row.index row.scoutAlpha row.fresh row.scout row.wideScore

private theorem ReportRow.selected_index (row : ReportRow) (pv : Bool) : (row.selected pv).index = row.index := by
  simp only [selected, Threading.SplitWindow.selectReport]
  split <;> rfl

def ReportRow.Children (row : ReportRow) (value beta : Int) : Prop :=
  (Window (-(row.scoutAlpha + 1)) (-row.scoutAlpha) →
    SoundResult (-value) (-(row.scoutAlpha + 1)) (-row.scoutAlpha) row.scout) ∧
  (∀ wide, row.wide = some wide → Window (-beta) (-wide.alpha) →
    SoundResult (-value) (-beta) (-wide.alpha) wide.score)

private def reportAccepted (row : ReportRow) (moves : List Nat) (start : Nat) (seen : List Nat)
    (alphas : List Int) (pv : Bool) (beta : Int) (locked : State) (chosen : Option Nat) : Bool :=
  (((decide (start ≤ row.index) && !seen.contains row.index) &&
    (alphas.contains row.scoutAlpha && row.wide.isSome == reSearch pv row.scoutAlpha row.scout &&
      (row.wide.all fun wide => alphas.contains wide.alpha))) &&
    (row.score == (row.selected pv).score && row.alphaAfter == (publish pv beta locked row.score).alpha &&
      row.bestAfter == (publish pv beta locked row.score).best)) &&
  (moves[row.index]? == some row.square &&
    row.moveAfter == (if choosesMove locked row.score then some row.square else chosen))

def replayReports (moves : List Nat) (start : Nat) (pv : Bool) (beta : Int) :
    List ReportRow → List Nat → List Int → State → Option Nat → Option (State × Option Nat)
  | [], _, _, locked, chosen => some (locked, chosen)
  | row :: tail, seen, alphas, locked, chosen =>
    if reportAccepted row moves start seen alphas pv beta locked chosen then
      replayReports moves start pv beta tail (row.index :: seen) (row.alphaAfter :: alphas)
        (publish pv beta locked row.score) row.moveAfter
    else none

theorem replayReports_sound
    (checked : replayReports moves start pv beta reports seen alphas locked chosen = some (final, picked)) :
    final = Threading.SplitWindow.scan pv beta locked (reports.map ReportRow.score) ∧
      (picked = chosen ∨ ∃ row ∈ reports, picked = some row.square ∧ moves[row.index]? = picked) := by
  induction reports generalizing seen alphas locked chosen with
  | nil =>
    have h : locked = final ∧ chosen = picked := by simpa [replayReports] using checked
    exact ⟨h.1.symm, Or.inl h.2.symm⟩
  | cons row tail ih =>
    rw [replayReports] at checked
    split at checked
    · rename_i accepted
      obtain ⟨s, pickedFrom⟩ := ih checked
      refine ⟨s, ?_⟩
      rcases pickedFrom with kept | ⟨later, present, fromRow, indexed⟩
      · unfold reportAccepted at accepted
        simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at accepted
        have indexed := accepted.2.1
        have after := accepted.2.2
        rw [after] at kept
        split at kept
        · exact Or.inr ⟨row, by simp, kept, indexed.trans kept.symm⟩
        · exact Or.inl kept
      · exact Or.inr ⟨later, List.mem_cons_of_mem _ present, fromRow, indexed⟩
    · contradiction

private theorem replayReports_merge
    (checked : replayReports moves start pv beta reports seen alphas locked chosen = some (final, picked))
    (value : Nat → Int)
    (window : Window locked.alpha beta) (capped : locked.alpha ≤ max original locked.best)
    (narrow : pv = false → beta = locked.alpha + 1)
    (samples : ∀ a ∈ alphas, Window a beta ∧ a ≤ locked.alpha ∧ (pv = false → beta = a + 1))
    (children : ∀ row ∈ reports, row.Children (value row.square) beta) :
    Merge (fun i => value (moves[i]?.getD 0)) original beta locked.best
        (reports.map fun row => row.selected pv) final.best ∧
      (∀ row ∈ reports, row.index ∈ slots start moves.length ∧ moves[row.index]? = some row.square) := by
  induction reports generalizing seen alphas locked chosen with
  | nil =>
    cases Option.some.inj checked
    exact ⟨.done, by simp⟩
  | cons row tail ih =>
    rw [replayReports] at checked
    split at checked
    · rename_i accepted
      unfold reportAccepted at accepted
      simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at accepted
      obtain ⟨⟨⟨⟨beginAt, _⟩, ⟨⟨scoutPresent, wideGuard⟩, widePresent⟩⟩,
        ⟨⟨sameScore, sameAlpha⟩, _⟩⟩, ⟨indexed, _⟩⟩ := accepted
      let next := publish pv beta locked row.score
      obtain ⟨monotone, nextWindow, nextCap, stable⟩ :=
        Threading.SplitWindow.stateContract_proven pv original beta locked [row.score] window capped
      change locked.alpha ≤ next.alpha at monotone
      change Window next.alpha beta at nextWindow
      change next.alpha ≤ max original next.best at nextCap
      change (pv = false → next.alpha = locked.alpha) at stable
      have nextNull (nonPv : pv = false) : beta = next.alpha + 1 := by rw [stable nonPv, narrow nonPv]
      have retained : ∀ a ∈ row.alphaAfter :: alphas, Window a beta ∧ a ≤ next.alpha ∧
          (pv = false → beta = a + 1) := by
        intro a present
        rcases List.mem_cons.mp present with rfl | prior
        · rw [sameAlpha]
          exact ⟨nextWindow, Int.le_refl _, nextNull⟩
        · obtain ⟨valid, priorAlpha, null⟩ := samples a prior
          exact ⟨valid, Int.le_trans priorAlpha monotone, null⟩
      have ⟨merged, validTail⟩ := ih checked nextWindow nextCap nextNull retained
        (fun r mem => children r (List.mem_cons_of_mem _ mem))
      obtain ⟨scoutWindow, scoutLe, scoutNull⟩ := samples row.scoutAlpha (by simpa using scoutPresent)
      obtain ⟨scout, wide⟩ := children row (by simp)
      have wideUsed (retry : reSearch pv row.scoutAlpha row.scout = true) :
          ∃ w, row.wide = some w := by
        cases shape : row.wide with
        | none => simp [shape, retry] at wideGuard
        | some w => exact ⟨w, rfl⟩
      have freshWindow (retry : reSearch pv row.scoutAlpha row.scout = true) :
          Window row.fresh beta ∧ row.fresh ≤ locked.alpha := by
        obtain ⟨w, shape⟩ := wideUsed retry
        obtain ⟨hw, bounded, _⟩ := samples w.alpha (by simpa [shape] using widePresent)
        simpa [ReportRow.fresh, shape] using And.intro hw bounded
      have wideSound (retry : reSearch pv row.scoutAlpha row.scout = true)
          (test : Window (-beta) (-row.fresh)) :
          SoundResult (-(value row.square)) (-beta) (-row.fresh) row.wideScore := by
        obtain ⟨w, shape⟩ := wideUsed retry
        simpa [ReportRow.fresh, ReportRow.wideScore, shape] using
          wide w shape (by simpa [ReportRow.fresh, shape] using test)
      obtain ⟨bounds, canUse⟩ := Threading.SplitWindow.childReport row.index scoutWindow scoutLe
        scoutNull freshWindow scout wideSound
      have reportBound : Bound (value (moves[(row.selected pv).index]?.getD 0))
          (row.selected pv).searchedAlpha beta (row.selected pv).score := by
        rw [ReportRow.selected_index, indexed, Option.getD_some]
        exact bounds
      have rest : Merge (fun i => value (moves[i]?.getD 0)) original beta (max locked.best (row.selected pv).score)
          (tail.map fun r => r.selected pv) final.best := by
        simpa only [next, publish, sameScore] using merged
      refine ⟨.accept (Int.le_trans canUse capped) reportBound rest, ?_⟩
      intro r present
      rcases List.mem_cons.mp present with rfl | later
      · have inside := (List.getElem?_eq_some_iff.mp indexed).1
        exact ⟨Threading.SplitResult.mem_slots.mpr ⟨beginAt, inside⟩, indexed⟩
      · exact validTail _ later
    · contradiction

structure Handoff where
  start : Nat
  moves : List Nat
  alpha : Int
  best : Int
  seed : Option Nat
  reports : List ReportRow
  deriving FromJson

structure SlotStamp where
  scope : Nat
  slot : Nat
  sequence : Nat
  deriving FromJson

def SlotStamp.key (stamp : SlotStamp) : Nat × Nat × Nat :=
  (stamp.scope, stamp.slot, stamp.sequence)

structure Node where
  player : Nat
  opponent : Nat
  root : Bool
  pv : Bool
  alpha : Int
  beta : Int
  generated : List Nat
  moves : List Nat
  calls : List ChildReturn
  selections : List Selection
  handoff : Option Handoff
  result : Int
  move : Option Nat
  publication : Option SlotStamp
  deriving FromJson

def Node.initial (node : Node) : State := ⟨node.alpha, -scoreInf⟩
def Node.queue (node : Node) : Queue Nat := ⟨0, node.moves, []⟩
def Node.board (node : Node) : Board.Board :=
  ⟨BitVec.ofNat 64 node.player, BitVec.ofNat 64 node.opponent⟩

def Node.legalMask (node : Node) : Bitboard.Bits := Board.getMoves moveConfig node.board

def Node.coversAll (node : Node) : Bool :=
  node.moves.length == ((List.range 64).filter node.legalMask.getLsbD).length

private def candidatesAgree (node : Node) : Bool :=
  decide (node.player < 2 ^ 64 ∧ node.opponent < 2 ^ 64 ∧ node.moves.Nodup) &&
    (node.board.player &&& node.board.opponent) == 0 && !node.moves.isEmpty &&
    (node.root || node.coversAll) &&
    node.generated.mergeSort == node.moves.mergeSort

private def covered (split : Handoff) (beta best : Int) : Bool :=
  decide (beta ≤ best ∨ PvsMoveLoop.scoreMax ≤ best) ||
    (Threading.SplitResult.slots split.start split.moves.length).all fun index =>
      split.reports.any (·.index == index)

private theorem covered_sound {split : Handoff} {beta best : Int} (checked : covered split beta best = true) :
    beta ≤ best ∨ PvsMoveLoop.scoreMax ≤ best ∨
      ∀ index ∈ Threading.SplitResult.slots split.start split.moves.length,
        ∃ row ∈ split.reports, row.index = index := by
  rcases Bool.or_eq_true_iff.mp checked with cutoff | every
  · rcases of_decide_eq_true cutoff with beta | max
    · exact Or.inl beta
    · exact Or.inr (Or.inl max)
  · apply Or.inr ∘ Or.inr
    intro index member
    obtain ⟨row, present, same⟩ := List.any_eq_true.mp (List.all_eq_true.mp every index member)
    exact ⟨row, present, beq_iff_eq.mp same⟩

private def matchEnd (node : Node) (visited : List Nat) (remaining : Queue Nat)
    (picked : Option Nat) : Outcome → Option Handoff → Bool
  | .returned score, none => score == node.result && picked == node.move
  | .handoff state, some split =>
    ((split.start == remaining.idx && split.alpha == state.alpha && split.best == state.best &&
        decide (split.start + 2 ≤ split.moves.length) && !(node.pv && split.start == 0)) &&
        split.moves == visited ++ remaining.pending && split.seed == picked) &&
      match replayReports split.moves split.start node.pv node.beta split.reports [] [split.alpha] state split.seed with
      | some (final, chosen) => final.best == node.result && chosen == node.move && covered split node.beta final.best
      | none => false
  | _, _ => false

def Node.check (node : Node) : Bool :=
  decide (-scoreInf ≤ node.alpha ∧ node.alpha < node.beta ∧ node.beta ≤ scoreInf) &&
    (node.pv || node.beta == node.alpha + 1) &&
    candidatesAgree node && node.moves.all node.legalMask.getLsbD &&
    match replay node.calls node.pv node.beta node.handoff.isSome node.selections node.queue node.initial node.moves.length with
    | none => false
    | some serial =>
      let picked := Prefix.Serial.bestMove node.pv (observe node.calls) node.beta serial.visited true node.initial none
      matchEnd node serial.visited serial.remaining picked serial.outcome node.handoff

def Node.scoreAgrees (node : Node) : Outcome → Option Handoff → Prop
  | .returned score, none => score = node.result
  | .handoff state, some split =>
    (Threading.SplitWindow.scan node.pv node.beta state (split.reports.map ReportRow.score)).best = node.result ∧
      (node.beta ≤ node.result ∨ PvsMoveLoop.scoreMax ≤ node.result ∨
        ∀ index ∈ Threading.SplitResult.slots split.start split.moves.length,
          ∃ row ∈ split.reports, row.index = index)
  | _, _ => False

private theorem matchEnd_sound (node : Node) (visited : List Nat) (remaining : Queue Nat)
    (picked : Option Nat) (outcome : Outcome) (handoff : Option Handoff)
    (matched : matchEnd node visited remaining picked outcome handoff = true) :
    node.scoreAgrees outcome handoff ∧
      (node.move = picked ∨ ∃ m, node.move = some m ∧ m ∈ visited ++ remaining.pending) := by
  cases outcome with
  | returned score =>
    cases handoff with
    | none =>
      simp only [matchEnd, Bool.and_eq_true, beq_iff_eq] at matched
      exact ⟨matched.1, Or.inl matched.2.symm⟩
    | some _ => contradiction
  | handoff state =>
    cases handoff with
    | none => contradiction
    | some split =>
      simp only [matchEnd, Bool.and_eq_true, beq_iff_eq] at matched
      obtain ⟨⟨⟨_, moves⟩, seed⟩, reports⟩ := matched
      cases ran : replayReports split.moves split.start node.pv node.beta split.reports []
          [split.alpha] state split.seed with
      | none => simp [ran] at reports
      | some final =>
        obtain ⟨finalState, choice⟩ := final
        rw [ran] at reports
        obtain ⟨agrees, fromSeed⟩ := replayReports_sound ran
        obtain ⟨selected, coverage⟩ := Bool.and_eq_true_iff.mp reports
        obtain ⟨lastScore, pickedEq⟩ := Bool.and_eq_true_iff.mp selected
        have lastScore := beq_iff_eq.mp lastScore
        refine ⟨⟨agrees ▸ lastScore, lastScore ▸ covered_sound coverage⟩, ?_⟩
        have pickedEq := beq_iff_eq.mp pickedEq
        rcases fromSeed with keep | ⟨row, _, chosen, indexed⟩
        · exact Or.inl (pickedEq.symm.trans (keep.trans seed))
        · refine Or.inr ⟨row.square, pickedEq.symm.trans chosen, ?_⟩
          rw [← moves]
          exact List.mem_of_getElem? (indexed.trans chosen)

private theorem Node.checkedReplay {node : Node} (validated : node.check = true) :
    ∃ serial : SerialResult,
      Run (Completed node.calls) (observe node.calls) node.pv node.beta node.queue node.initial
        serial.visited serial.remaining serial.outcome ∧
      matchEnd node serial.visited serial.remaining
        (Prefix.Serial.bestMove node.pv (observe node.calls) node.beta serial.visited true node.initial none)
        serial.outcome node.handoff = true := by
  have endCheck := (Bool.and_eq_true_iff.mp validated).2
  cases found : replay node.calls node.pv node.beta node.handoff.isSome node.selections node.queue node.initial
      node.moves.length with
  | none => simp [found] at endCheck
  | some serial =>
    rw [found] at endCheck
    exact ⟨serial, replay_sound found, endCheck⟩

theorem Node.check_sound {node : Node} (validated : node.check = true) :
    ∃ serial : SerialResult,
      Run (Completed node.calls) (observe node.calls) node.pv node.beta node.queue node.initial
        serial.visited serial.remaining serial.outcome ∧
      node.scoreAgrees serial.outcome node.handoff ∧
      (∀ square, node.move = some square → square ∈ node.moves) := by
  obtain ⟨serial, trace, endCheck⟩ := node.checkedReplay validated
  obtain ⟨same, selected⟩ := matchEnd_sound node serial.visited serial.remaining _ _ _ endCheck
  refine ⟨serial, trace, same, ?_⟩
  intro square returned
  have order := trace.reproduced.1
  have inQueue : square ∈ node.queue.pending := by
    rcases selected with old | ⟨m, picked, present⟩
    · have chosen := trace.bestMoveMember (previous := none) (old.symm.trans returned)
      exact chosen.resolve_left (by simp)
    · have equal := Option.some.inj (returned.symm.trans picked)
      exact order.mem_iff.mp (equal ▸ present)
  simpa [Node.queue, Queue.pending] using inQueue

private theorem Node.candidate_legal {node : Node} (validated : node.check = true)
    (member : square ∈ node.moves) : Bitboard.Moves.LegalMove node.board.player node.board.opponent square := by
  have guards := ((Bool.and_eq_true_iff.mp validated).1)
  have candidates : node.moves.all node.legalMask.getLsbD = true := (Bool.and_eq_true_iff.mp guards).2
  exact (Board.getMoves_exact moveConfig node.board square).mp
    ((List.all_eq_true.mp candidates) square member)

theorem Node.checked_move_legal {node : Node} (validated : node.check = true)
    (selected : node.move = some square) : Bitboard.Moves.LegalMove node.board.player node.board.opponent square := by
  obtain ⟨_, _, _, owns⟩ := node.check_sound validated
  exact node.candidate_legal validated (owns square selected)

private theorem Node.all_candidates {node : Node} (validated : node.check = true) (coverage : node.coversAll = true)
    (legal : Bitboard.Moves.LegalMove node.board.player node.board.opponent square) : square ∈ node.moves := by
  have guards := (Bool.and_eq_true_iff.mp validated).1
  have matching := (Bool.and_eq_true_iff.mp (Bool.and_eq_true_iff.mp guards).1).2
  unfold candidatesAgree at matching
  simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at matching
  obtain ⟨⟨⟨⟨⟨_, _, nodup⟩, _⟩, _⟩, _⟩, _⟩ := matching
  have lengthEq := beq_iff_eq.mp coverage
  let available := (List.range 64).filter node.legalMask.getLsbD
  have present {m : Nat} (ok : Bitboard.Moves.LegalMove node.board.player node.board.opponent m) :
      m ∈ available :=
    List.mem_filter.mpr ⟨List.mem_range.mpr ok.1, (Board.getMoves_exact moveConfig node.board m).mpr ok⟩
  by_cases seen : square ∈ node.moves
  · exact seen
  · have longer : (square :: node.moves).length ≤ available.length :=
      (List.nodup_cons.mpr ⟨seen, nodup⟩).length_le_of_subset (by
        intro m member
        rcases List.mem_cons.mp member with rfl | already
        · exact present legal
        · exact present (node.candidate_legal validated already))
    change node.moves.length + 1 ≤ available.length at longer
    change node.moves.length = available.length at lengthEq
    omega

theorem Node.nonroot_candidates {node : Node} (validated : node.check = true) (internal : node.root = false) :
    square ∈ node.moves ↔ Bitboard.Moves.LegalMove node.board.player node.board.opponent square := by
  have guards := (Bool.and_eq_true_iff.mp validated).1
  have matching := (Bool.and_eq_true_iff.mp (Bool.and_eq_true_iff.mp guards).1).2
  simp only [candidatesAgree, Bool.and_eq_true, internal, Bool.false_or] at matching
  exact ⟨node.candidate_legal validated, node.all_candidates validated matching.1.2⟩

private theorem Node.check_window {node : Node} (validated : node.check = true) :
    Window node.alpha node.beta ∧ (node.pv = false → node.beta = node.alpha + 1) := by
  have guards := (Bool.and_eq_true_iff.mp validated).1
  have windowGuard := (Bool.and_eq_true_iff.mp (Bool.and_eq_true_iff.mp guards).1).1
  obtain ⟨range, branch⟩ := Bool.and_eq_true_iff.mp windowGuard
  obtain ⟨lo, ascending, hi⟩ := of_decide_eq_true range
  exact ⟨⟨lo, ascending, hi⟩, fun nonPv => by simpa [nonPv] using branch⟩

private theorem Node.check_nonempty {node : Node} (validated : node.check = true) : node.moves ≠ [] := by
  have guards := (Bool.and_eq_true_iff.mp validated).1
  have boardCovered := (Bool.and_eq_true_iff.mp (Bool.and_eq_true_iff.mp guards).1).2
  intro empty
  simp [candidatesAgree, empty] at boardCovered

theorem Node.valid_board {node : Node} (validated : node.check = true) : Board.wellFormed node.board := by
  have guards := (Bool.and_eq_true_iff.mp validated).1
  have shape := (Bool.and_eq_true_iff.mp (Bool.and_eq_true_iff.mp guards).1).2
  unfold candidatesAgree at shape
  simp only [Bool.and_eq_true, beq_iff_eq] at shape
  exact shape.1.1.1.2

def Node.SerialChildren (node : Node) (value : Nat → Int) : Prop :=
  ∀ m ∈ node.moves, ∀ pv a b s, Window a b → (pv = false → b = a + 1) →
    Completed node.calls m pv a b s → SoundResult (-(value m)) a b s

def Node.SplitChildren (node : Node) (value : Nat → Int) : Prop :=
  ∀ split, node.handoff = some split →
    ∀ row ∈ split.reports, row.Children (value row.square) node.beta

theorem Node.serial_score_sound {node : Node} (validated : node.check = true)
    (unsplit : node.handoff = none) (value : Nat → Int)
    (cap : ∀ m ∈ node.moves, value m ≤ PvsMoveLoop.scoreMax)
    (children : node.SerialChildren value) :
    Aspiration.SoundResult (PvsMoveLoop.bestValue value node.moves (-scoreInf))
      node.alpha node.beta node.result := by
  obtain ⟨serial, trace, same, _⟩ := node.check_sound validated
  rw [unsplit] at same
  cases outcome : serial.outcome with
  | handoff _ => simp [outcome, Node.scoreAgrees] at same
  | returned score =>
    have scoreEq : score = node.result := by simpa [outcome, Node.scoreAgrees] using same
    obtain ⟨ordered, _, reproduced⟩ := trace.reproduced
    let moves := serial.visited ++ serial.remaining.pending
    have order : moves.Perm node.moves := by simpa [moves, Node.queue, Queue.pending] using ordered
    have witnesses : PvsMoveLoop.scan node.pv (observe node.calls) node.beta moves true node.alpha (-scoreInf) =
        score ∧ PvsMoveLoop.ScanCalls (Completed node.calls) (observe node.calls) node.pv node.beta
          moves true node.alpha (-scoreInf) := by
      simpa [Prefix.Serial.Reproduced, outcome, moves, Node.queue, Node.initial] using reproduced
    obtain ⟨w, narrow⟩ := node.check_window validated
    have nonempty : moves ≠ [] := fun empty =>
      node.check_nonempty validated (List.perm_nil.mp (empty ▸ order).symm)
    have replayed := PvsMoveLoop.selectedContract_proven Nat node.pv (observe node.calls) value
      (Completed node.calls) moves node.alpha node.beta nonempty w narrow
      (fun m mem => cap m (order.mem_iff.mp mem)) witnesses.2
      (fun m mem => children m (order.mem_iff.mp mem))
    rw [witnesses.1, scoreEq, PvsMoveLoop.bestValue_eq_of_same_moves value _ (fun _ => order.mem_iff)] at replayed
    exact replayed

theorem Node.split_score_bound {node : Node} (validated : node.check = true)
    (taken : node.handoff = some split) (value : Nat → Int)
    (cap : ∀ m ∈ node.moves, value m ≤ scoreMax) (children : node.SerialChildren value)
    (reports : ∀ row ∈ split.reports, row.Children (value row.square) node.beta) :
    Bound (bestValue value node.moves (-scoreInf)) node.alpha node.beta node.result := by
  obtain ⟨serial, trace, matched⟩ := node.checkedReplay validated
  rw [taken] at matched
  cases outcome : serial.outcome with
  | returned _ => simp [outcome, matchEnd] at matched
  | handoff state =>
    simp only [outcome, matchEnd, Bool.and_eq_true, beq_iff_eq] at matched
    obtain ⟨⟨⟨entry, layout⟩, _⟩, reported⟩ := matched
    have beginAt := entry.1.1.1.1
    have initialAlpha := entry.1.1.1.2
    cases folds : replayReports split.moves split.start node.pv node.beta split.reports
        [] [split.alpha] state split.seed with
    | none => simp [folds] at reported
    | some final =>
      obtain ⟨finalState, picked⟩ := final
      rw [folds] at reported
      obtain ⟨data, coverage⟩ := Bool.and_eq_true_iff.mp reported
      have scoreEq := beq_iff_eq.mp (Bool.and_eq_true_iff.mp data).1
      obtain ⟨permutation, count, reproduced⟩ := trace.reproduced
      let moves := serial.visited ++ serial.remaining.pending
      have order : moves.Perm node.moves := by simpa [moves, Node.queue, Queue.pending] using permutation
      have splitOrder : split.moves.Perm node.moves := layout ▸ order
      have startEq : split.start = serial.visited.length := beginAt.trans (by simpa [Node.queue] using count)
      have witnesses :
          Prefix.run node.pv (observe node.calls) node.beta serial.visited true node.initial = some state ∧
            PvsMoveLoop.ScanCalls (Completed node.calls) (observe node.calls) node.pv node.beta
              serial.visited true node.alpha (-scoreInf) := by
        simpa [Prefix.Serial.Reproduced, outcome, Node.queue, Node.initial] using reproduced
      have member : ∀ m ∈ serial.visited, m ∈ node.moves :=
        fun m present => order.mem_iff.mp (List.mem_append_left _ present)
      obtain ⟨window, narrow⟩ := node.check_window validated
      obtain ⟨entryWindow, entryCap, entryNull, prefixBound⟩ :=
        Prefix.selectedContract_proven Nat node.pv (observe node.calls) value serial.visited
          (Completed node.calls) node.alpha node.beta state window narrow witnesses.1
          (fun m present => cap m (member m present)) witnesses.2
          (fun m present => children m (member m present))
      have sampled : ∀ a ∈ [split.alpha], Window a node.beta ∧ a ≤ state.alpha ∧
          (node.pv = false → node.beta = a + 1) := by
        intro a present
        rw [List.mem_singleton.mp present, initialAlpha]
        exact ⟨entryWindow, Int.le_refl _, entryNull⟩
      obtain ⟨merge, valid⟩ := replayReports_merge folds value entryWindow entryCap entryNull sampled reports
      have suffixCap : ∀ i ∈ slots split.start split.moves.length,
          value (split.moves[i]?.getD 0) ≤ scoreMax := by
        intro i present
        have inside := (Threading.SplitResult.mem_slots.mp present).2
        rw [List.getElem?_eq_getElem inside, Option.getD_some]
        exact cap _ (splitOrder.mem_iff.mp (List.getElem_mem inside))
      have prefixCap := PvsMoveLoop.bestValue_le value serial.visited (-scoreInf) scoreMax
        (by decide) (fun m present => cap m (member m present))
      have legalReport : ∀ report ∈ split.reports.map (fun row => row.selected node.pv),
          report.index ∈ slots split.start split.moves.length := by
        intro report present
        obtain ⟨row, seen, rfl⟩ := List.mem_map.mp present
        simpa only [ReportRow.selected_index] using (valid row seen).1
      have coveredReports : node.beta ≤ finalState.best ∨ scoreMax ≤ finalState.best ∨
          ∀ i ∈ slots split.start split.moves.length,
            ∃ report ∈ split.reports.map (fun row => row.selected node.pv), report.index = i := by
        rcases covered_sound coverage with high | maximum | all
        · exact Or.inl high
        · exact Or.inr (Or.inl maximum)
        · refine Or.inr (Or.inr fun i present => ?_)
          obtain ⟨row, occurs, atIndex⟩ := all i present
          exact ⟨row.selected node.pv, List.mem_map_of_mem occurs, by simpa only [ReportRow.selected_index]⟩
      have bound := merge.bound_of_coverage prefixBound suffixCap prefixCap legalReport coveredReports
      rw [startEq, layout, Threading.SplitResult.partitionLabels, scoreEq,
        PvsMoveLoop.bestValue_eq_of_same_moves value _ (fun _ => order.mem_iff)] at bound
      exact bound

theorem Node.score_sound {node : Node} (validated : node.check = true)
    (value : Nat → Int) (range : ∀ m ∈ node.moves, -scoreMax ≤ value m ∧ value m ≤ scoreMax)
    (children : node.SerialChildren value)
    (reports : node.SplitChildren value) :
    SoundResult (bestValue value node.moves (-scoreInf)) node.alpha node.beta node.result := by
  have cap : ∀ m ∈ node.moves, value m ≤ scoreMax := fun m present => (range m present).2
  cases taken : node.handoff with
  | none => exact node.serial_score_sound validated taken value cap children
  | some split =>
    have bound := node.split_score_bound validated taken value cap children (reports split taken)
    obtain ⟨window, _⟩ := node.check_window validated
    have high := PvsMoveLoop.bestValue_le value node.moves (-scoreInf) scoreMax (by decide) cap
    have low : -scoreMax ≤ bestValue value node.moves (-scoreInf) := by
      obtain ⟨first, rest, ordered⟩ := List.exists_cons_of_ne_nil (node.check_nonempty validated)
      have head : first ∈ node.moves := by simp [ordered]
      exact Int.le_trans (range first head).1 (PvsMoveLoop.bestValue_ge_move _ _ _ head)
    apply bound.toSoundResult window
    · by_cases inside : -scoreInf < node.result
      · exact inside
      · unfold scoreInf at inside
        have tooLow := bound.1 (by have := window.lower; have := window.ordered; omega)
        unfold scoreMax scoreInf at *
        omega
    · by_cases inside : node.result < scoreInf
      · exact inside
      · unfold scoreInf at inside
        have tooHigh := bound.2 (by have := window.upper; have := window.ordered; omega)
        unfold scoreMax scoreInf at *
        omega

theorem Node.fullWindow_exact {node : Node} (validated : node.check = true)
    (lower : node.alpha = -scoreInf) (upper : node.beta = scoreInf)
    (value : Nat → Int) (range : ∀ m ∈ node.moves, -scoreMax ≤ value m ∧ value m ≤ scoreMax)
    (children : node.SerialChildren value)
    (reports : node.SplitChildren value) :
    node.result = bestValue value node.moves (-scoreInf) := by
  have sound := node.score_sound validated value range children reports
  rw [lower, upper] at sound
  exact sound.exact sound.lower sound.upper

def childValue (fc : Flip.Dispatch.X86Config) (board : PvsBoardTree.Position) (square : Nat) : Int :=
  if inside : square < 64 then
    if legal : (Board.getMoves moveConfig board.val).getLsbD square = true then
      -PvsBoardTree.minimax moveConfig fc (PvsBoardTree.play moveConfig fc board ⟨⟨square, inside⟩, legal⟩).val
    else 0
  else 0

private theorem childValue_generated (fc : Flip.Dispatch.X86Config) (board : PvsBoardTree.Position)
    (move : PvsBoardTree.GeneratedSquare moveConfig board.val) :
    childValue fc board move.val.val = -PvsBoardTree.minimax moveConfig fc
      (PvsBoardTree.play moveConfig fc board move).val := by
  simp [childValue, move.val.isLt, move.property]

private theorem childValue_range (fc : Flip.Dispatch.X86Config) (board : PvsBoardTree.Position) (square : Nat) :
    -scoreMax ≤ childValue fc board square ∧ childValue fc board square ≤ scoreMax := by
  unfold childValue
  split
  · split
    · have truth := PvsMoveLoop.minimax_range
        (PvsBoardTree.build moveConfig fc (PvsBoardTree.play moveConfig fc board ⟨⟨square, ‹_›⟩, ‹_›⟩).val)
      unfold PvsBoardTree.minimax
      omega
    · decide
  · decide

theorem Node.board_value {node : Node} (validated : node.check = true)
    (fc : Flip.Dispatch.X86Config)
    (covers : ∀ sq, Bitboard.Moves.LegalMove node.board.player node.board.opponent sq → sq ∈ node.moves) :
    bestValue (childValue fc ⟨node.board, node.valid_board validated⟩) node.moves (-scoreInf) =
      PvsBoardTree.minimax moveConfig fc ⟨node.board, node.valid_board validated⟩ := by
  let board : PvsBoardTree.Position := ⟨node.board, node.valid_board validated⟩
  let legal := PvsBoardTree.legalSquares moveConfig board.val
  let value := childValue fc board
  let index := fun (m : PvsBoardTree.GeneratedSquare moveConfig board.val) => m.val.val
  have membership (sq : Nat) : sq ∈ node.moves ↔ sq ∈ legal.map index := by
    constructor
    · intro present
      have ok := node.candidate_legal validated present
      obtain ⟨generated, member, same⟩ :=
        (PvsBoardTree.legalSquares_complete moveConfig board.val ⟨sq, ok.1⟩).mpr ok
      exact List.mem_map.mpr ⟨generated, member, congrArg Fin.val same⟩
    · intro present
      obtain ⟨move, _, rfl⟩ := List.mem_map.mp present
      exact covers _ ((Board.getMoves_exact moveConfig board.val _).mp move.property)
  have available : legal ≠ [] := by
    intro empty
    apply node.check_nonempty validated
    cases moves : node.moves with
    | nil => rfl
    | cons first rest =>
      have found := (membership first).mp (by simp [moves])
      simp [empty] at found
  obtain ⟨first, rest, order⟩ := List.exists_cons_of_ne_nil available
  rw [PvsMoveLoop.bestValue_eq_of_same_moves value _ membership, PvsMoveLoop.bestValue_map, order]
  rw [PvsBoardTree.minimax_moves moveConfig fc board first rest order]
  exact congrArg (fun v => bestValue v (first :: rest) (-scoreInf))
    (funext (childValue_generated fc board))

theorem Node.board_score {node : Node} (validated : node.check = true)
    (fc : Flip.Dispatch.X86Config)
    (covers : ∀ sq, Bitboard.Moves.LegalMove node.board.player node.board.opponent sq → sq ∈ node.moves)
    (children : node.SerialChildren (childValue fc ⟨node.board, node.valid_board validated⟩))
    (reports : node.SplitChildren (childValue fc ⟨node.board, node.valid_board validated⟩)) :
    SoundResult (PvsBoardTree.minimax moveConfig fc ⟨node.board, node.valid_board validated⟩)
      node.alpha node.beta node.result := by
  rw [← node.board_value validated fc covers]
  exact node.score_sound validated _ (fun m _ => childValue_range fc _ m) children reports

theorem Node.internal_board_score {node : Node} (validated : node.check = true) (internal : node.root = false)
    (fc : Flip.Dispatch.X86Config)
    (children : node.SerialChildren (childValue fc ⟨node.board, node.valid_board validated⟩))
    (reports : node.SplitChildren (childValue fc ⟨node.board, node.valid_board validated⟩)) :
    SoundResult (PvsBoardTree.minimax moveConfig fc ⟨node.board, node.valid_board validated⟩)
      node.alpha node.beta node.result :=
  node.board_score validated fc (fun _ => (node.nonroot_candidates validated internal).mpr) children reports

private def ReportRow.scoutCall (row : ReportRow) : ChildReturn :=
  ⟨row.scoutFrame, row.square, false, -(row.scoutAlpha + 1), -row.scoutAlpha, row.scout⟩

private def ReportRow.wideCall (row : ReportRow) (beta : Int) (wide : WideReturn) : ChildReturn :=
  ⟨wide.frame, row.square, true, -beta, -wide.alpha, wide.score⟩

private def checkChild (known : Nat → Option FrameReturn) (fc : Flip.Dispatch.X86Config)
    (board : Board.Board) (depth worker : Nat) (requiredParent : Option Nat) (call : ChildReturn) : Bool :=
  match known call.frame with
  | none => false
  | some child =>
    (child.worker == worker && child.depth + 1 == depth && requiredParent.all (· == child.parent)) &&
      child.onMove fc board call.square && child.windowMatches call.pv call.alpha call.beta call.score

def Node.matchesReturn (node : Node) (parent : FrameReturn) : Bool :=
  parent.player == node.player && parent.opponent == node.opponent &&
    parent.windowMatches node.pv node.alpha node.beta node.result

theorem Node.return_agrees {node : Node} {parent : FrameReturn} (matched : node.matchesReturn parent = true) :
    parent.board = node.board ∧ parent.alpha = node.alpha ∧ parent.beta = node.beta ∧ parent.score = node.result := by
  simp only [Node.matchesReturn, Bool.and_eq_true, beq_iff_eq] at matched
  obtain ⟨⟨player, opponent⟩, window⟩ := matched
  exact ⟨by simp only [FrameReturn.board, Node.board, player, opponent], FrameReturn.windowMatches_sound window⟩

def Node.checkLinks (node : Node) (known : Nat → Option FrameReturn) (fc : Flip.Dispatch.X86Config)
    (frameId : Nat) (parent : FrameReturn) : Bool :=
  node.matchesReturn parent &&
  node.calls.all (checkChild known fc node.board parent.depth parent.worker (some frameId)) &&
  node.handoff.all (fun split => split.reports.all fun row =>
    let check := checkChild known fc node.board parent.depth row.worker none
    check row.scoutCall && row.wide.all (fun wide => check (row.wideCall node.beta wide)))

def SoundReturns (known : Nat → Option FrameReturn) (fc : Flip.Dispatch.X86Config) : Prop :=
  ∀ id frame, known id = some frame → frame.BoardSound fc

private theorem checkChild_sound {known : Nat → Option FrameReturn} {fc : Flip.Dispatch.X86Config}
    {board : PvsBoardTree.Position} {call : ChildReturn}
    (checked : checkChild known fc board.val depth worker requiredParent call = true)
    (sound : SoundReturns known fc) :
    SoundResult (-(childValue fc board call.square)) call.alpha call.beta call.score := by
  unfold checkChild at checked
  cases read : known call.frame with
  | none => simp [read] at checked
  | some child =>
    simp only [read, Bool.and_eq_true] at checked
    obtain ⟨⟨_, onMove⟩, window⟩ := checked
    obtain ⟨move, chosen, nextBoard⟩ := FrameReturn.onMove_sound (board := board) onMove
    let next := (PvsBoardTree.play moveConfig fc board move).val
    have wellFormed : Board.wellFormed child.board := nextBoard.symm ▸ next.property
    have same : (⟨child.board, wellFormed⟩ : PvsBoardTree.Position) = next := Subtype.ext nextBoard
    obtain ⟨alpha, beta, score⟩ := FrameReturn.windowMatches_sound window
    have returned := sound _ _ read wellFormed
    rw [same, alpha, beta, score] at returned
    rw [← chosen, childValue_generated, Int.neg_neg]
    exact returned

private theorem Completed.call {calls : List ChildReturn} (completed : Completed calls move pv alpha beta score) :
    ∃ call ∈ calls, call.square = move ∧ call.pv = pv ∧ call.alpha = alpha ∧
      call.beta = beta ∧ call.score = score := by
  let requested := fun (child : ChildReturn) =>
    child.square == move && child.pv == pv && child.alpha == alpha && child.beta == beta
  change ((calls.find? requested).map (·.score)) = some score at completed
  cases found : calls.find? requested with
  | none => simp [found] at completed
  | some call =>
    have params := List.find?_some found
    simp only [requested, Bool.and_eq_true, beq_iff_eq] at params
    refine ⟨call, List.mem_of_find?_eq_some found, params.1.1.1, params.1.1.2, params.1.2, params.2, ?_⟩
    simpa [found] using completed

theorem Node.checked_children {node : Node} {known : Nat → Option FrameReturn} {fc : Flip.Dispatch.X86Config}
    (validated : node.check = true) (links : node.checkLinks known fc frameId parent = true)
    (sound : SoundReturns known fc) :
    let value := childValue fc ⟨node.board, node.valid_board validated⟩
    node.SerialChildren value ∧ node.SplitChildren value := by
  let board : PvsBoardTree.Position := ⟨node.board, node.valid_board validated⟩
  simp only [Node.checkLinks, Bool.and_eq_true] at links
  have serial := links.1.2
  have split := links.2
  refine ⟨?_, ?_⟩
  · intro m _ p a b s _ _ returns
    obtain ⟨call, present, move, _, alpha, beta, score⟩ := returns.call
    have linked := List.all_eq_true.mp serial call present
    have checked := checkChild_sound (board := board) linked sound
    rw [move, alpha, beta, score] at checked
    exact checked
  · intro handoff placed row present
    rw [placed] at split
    have checked := List.all_eq_true.mp split row present
    obtain ⟨scout, wide⟩ := Bool.and_eq_true_iff.mp checked
    refine ⟨fun _ => checkChild_sound (board := board) scout sound, ?_⟩
    intro retry returned _
    have found : checkChild known fc node.board parent.depth row.worker none
        (row.wideCall node.beta retry) = true := by simpa [returned] using wide
    exact checkChild_sound (board := board) found sound

theorem Node.board_score_from_frames {node : Node} {known : Nat → Option FrameReturn} {fc : Flip.Dispatch.X86Config}
    (validated : node.check = true) (links : node.checkLinks known fc frameId parent = true)
    (covers : ∀ sq, Bitboard.Moves.LegalMove node.board.player node.board.opponent sq → sq ∈ node.moves)
    (sound : SoundReturns known fc) :
    SoundResult (PvsBoardTree.minimax moveConfig fc ⟨node.board, node.valid_board validated⟩)
      node.alpha node.beta node.result := by
  obtain ⟨serial, split⟩ := node.checked_children validated links sound
  exact node.board_score validated fc covers serial split

theorem Node.frame_sound {node : Node} {known : Nat → Option FrameReturn} {fc : Flip.Dispatch.X86Config}
    {parent : FrameReturn} (validated : node.check = true) (links : node.checkLinks known fc frameId parent = true)
    (covers : ∀ sq, Bitboard.Moves.LegalMove node.board.player node.board.opponent sq → sq ∈ node.moves)
    (sound : SoundReturns known fc) : parent.BoardSound fc := by
  have result := node.board_score_from_frames validated links covers sound
  simp only [Node.checkLinks, Bool.and_eq_true] at links
  obtain ⟨board, alpha, beta, score⟩ := node.return_agrees links.1.1
  intro valid
  have equal : (⟨parent.board, valid⟩ : PvsBoardTree.Position) = ⟨node.board, node.valid_board validated⟩ := Subtype.ext board
  rw [equal, alpha, beta, score]
  exact result

structure Sample where
  worker : Nat
  scope : Nat
  slot : Nat
  first : Nat
  last : Nat
  player : Nat
  opponent : Nat
  metadata : Nat
  deriving FromJson

namespace Sample

open TranspositionTable

def payload (sample : Sample) : SeqLock.Payload :=
  ⟨BitVec.ofNat 64 sample.player, BitVec.ofNat 64 sample.opponent, BitVec.ofNat 64 sample.metadata⟩

def sane (sample : Sample) : Bool :=
  decide (sample.first < 2 ^ 64 ∧ sample.last < 2 ^ 64 ∧
    sample.player < 2 ^ 64 ∧ sample.opponent < 2 ^ 64 ∧ sample.metadata < 2 ^ 64)

def accepted (lookup : Bool) (sample : Sample) : Bool :=
  let first := BitVec.ofNat 64 sample.first
  let last := BitVec.ofNat 64 sample.last
  if lookup then
    (LookupCut.readOccupied false first last sample.payload sample.payload.player sample.payload.opponent).isSome
  else (Snapshot.read false first last sample.payload).isSome &&
    (Packing.unpack false sample.payload.metadata).bound != 0

def samePublication (sample publication : Sample) : Bool :=
  sample.sane && publication.sane &&
    sample.scope == publication.scope && sample.slot == publication.slot && sample.first == publication.first &&
    sample.last == sample.first && publication.last == publication.first &&
    decide (publication.first % 2 = 0) &&
    (sample.player == publication.player && sample.opponent == publication.opponent &&
      sample.metadata == publication.metadata)

theorem matches_payload {sample publication : Sample} (checked : sample.samePublication publication = true) :
    sample.payload = publication.payload := by
  unfold samePublication at checked
  simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at checked
  obtain ⟨⟨player, opponent⟩, metadata⟩ := checked.2
  simp [payload, player, opponent, metadata]

def ofNode (sample : Sample) (node : Node) : Bool :=
  let data := Packing.unpack false sample.payload.metadata
  let score := BitVec.ofInt 32 node.result
  let alpha := BitVec.ofInt 32 node.alpha
  let beta := BitVec.ofInt 32 node.beta
  let hint := (Packing.decodeBestMove data.bestMove).map Fin.val
  sample.player == node.player && sample.opponent == node.opponent &&
    data.score.toInt == node.result &&
    data.bound == Packing.boundByte (TranspositionTable.classify node.pv ⟨score⟩ ⟨alpha⟩ ⟨beta⟩) &&
    (match node.move with
      | some square => hint == some square
      | none => hint.all node.legalMask.getLsbD)

theorem ofNode_score {sample : Sample} {node : Node} (checked : sample.ofNode node = true) :
    (Packing.unpack false sample.payload.metadata).score.toInt = node.result := by
  unfold ofNode at checked
  simp only [Bool.and_eq_true, beq_iff_eq] at checked
  exact checked.1.1.2

theorem sampled_score {sample publication : Sample} {node : Node}
    (source : sample.samePublication publication = true) (saved : publication.ofNode node = true) :
    (Packing.unpack false sample.payload.metadata).score.toInt = node.result := by
  rw [matches_payload source]
  exact ofNode_score saved

private theorem score_word_fits (n : Int) (lower : -32767 ≤ n) (upper : n ≤ 32767) :
    (BitVec.ofInt 32 n).toInt = n :=
  BitVec.toInt_ofInt_eq_self (by decide) (by omega) (by omega)

theorem ofNode_bound {publication : Sample} {node : Node} {truth : Int}
    (saved : publication.ofNode node = true) (window : Window node.alpha node.beta)
    (proof : SoundResult truth node.alpha node.beta node.result) :
    BoundSound (Packing.unpack false publication.payload.metadata) truth := by
  let score : Score := ⟨BitVec.ofInt 32 node.result⟩
  let alpha : Score := ⟨BitVec.ofInt 32 node.alpha⟩
  let beta : Score := ⟨BitVec.ofInt 32 node.beta⟩
  let stored := Packing.newData score none (TranspositionTable.classify node.pv score alpha beta)
    0 ⟨0, by decide⟩ 0 true node.pv
  obtain ⟨lo, order, hi⟩ := window
  have a : alpha.raw.toInt = node.alpha := score_word_fits _ lo (by omega)
  have b : beta.raw.toInt = node.beta := score_word_fits _ (by omega) hi
  have r : score.raw.toInt = node.result := score_word_fits _ (Int.le_of_lt proof.lower) (Int.le_of_lt proof.upper)
  have wordWindow : Window alpha.raw.toInt beta.raw.toInt := by rw [a, b]; exact ⟨lo, order, hi⟩
  have wordSound : SoundResult truth alpha.raw.toInt beta.raw.toInt
      score.raw.toInt := by rwa [a, b, r]
  have data := StoredBound.contract_proven node.pv score alpha beta _ wordWindow wordSound
    none 0 ⟨0, by decide⟩ 0 true node.pv false
  simp only [Packing.unpack_pack] at data
  have sameScore : (Packing.unpack false publication.payload.metadata).score.toInt = stored.score.toInt :=
    (ofNode_score saved).trans (data.1.trans r).symm
  have sameBound : (Packing.unpack false publication.payload.metadata).bound = stored.bound := by
    unfold ofNode at saved
    simp only [Bool.and_eq_true, beq_iff_eq] at saved
    exact saved.1.2
  exact ⟨fun lower => sameScore ▸ data.2.1.1 (sameBound ▸ lower),
    fun upper => sameScore ▸ data.2.1.2 (sameBound ▸ upper)⟩

theorem sampled_bound {sample publication : Sample} {node : Node}
    (source : sample.samePublication publication = true) (saved : publication.ofNode node = true)
    (validated : node.check = true)
    (value : Nat → Int) (range : ∀ m ∈ node.moves, -scoreMax ≤ value m ∧ value m ≤ scoreMax)
    (children : node.SerialChildren value) (reports : node.SplitChildren value) :
    BoundSound (Packing.unpack false sample.payload.metadata) (bestValue value node.moves (-scoreInf)) := by
  rw [matches_payload source]
  exact ofNode_bound saved (node.check_window validated).1 (node.score_sound validated value range children reports)

theorem sampled_board_bound {sample publication : Sample} {node : Node}
    (source : sample.samePublication publication = true) (saved : publication.ofNode node = true)
    (validated : node.check = true) (fc : Flip.Dispatch.X86Config)
    (covers : ∀ sq, Bitboard.Moves.LegalMove node.board.player node.board.opponent sq → sq ∈ node.moves)
    (children : node.SerialChildren (childValue fc ⟨node.board, node.valid_board validated⟩))
    (reports : node.SplitChildren (childValue fc ⟨node.board, node.valid_board validated⟩)) :
    BoundSound (Packing.unpack false sample.payload.metadata)
      (PvsBoardTree.minimax moveConfig fc ⟨node.board, node.valid_board validated⟩) := by
  rw [← node.board_value validated fc covers]
  exact sampled_bound source saved validated _ (fun m _ => childValue_range fc _ m) children reports

def board (sample : Sample) : Board.Board := ⟨sample.payload.player, sample.payload.opponent⟩

def BoardBound (sample : Sample) (fc : Flip.Dispatch.X86Config) : Prop :=
  ∀ valid : Board.wellFormed sample.board,
    BoundSound (Packing.unpack false sample.payload.metadata)
      (PvsBoardTree.minimax moveConfig fc ⟨sample.board, valid⟩)

theorem saved_frame_bound {publication : Sample} {node : Node} {frame : FrameReturn}
    {fc : Flip.Dispatch.X86Config} (saved : publication.ofNode node = true) (validated : node.check = true)
    (matched : node.matchesReturn frame = true) (sound : frame.BoardSound fc) : publication.BoardBound fc := by
  obtain ⟨boardEq, alpha, beta, score⟩ := node.return_agrees matched
  have key := saved
  simp only [ofNode, Bool.and_eq_true, beq_iff_eq] at key
  have sameBoard : publication.board = node.board := by
    simp only [board, payload, Node.board, key.1.1.1.1, key.1.1.1.2]
  intro valid
  have equal : frame.board = publication.board := boardEq.trans sameBoard.symm
  have h : Board.wellFormed frame.board := equal.symm ▸ valid
  have child := sound h
  have same : (⟨frame.board, h⟩ : PvsBoardTree.Position) = ⟨publication.board, valid⟩ := Subtype.ext equal
  rw [same, alpha, beta, score] at child
  exact ofNode_bound saved (node.check_window validated).1 child

theorem inherits_bound {sample publication : Sample} {fc : Flip.Dispatch.X86Config}
    (matched : sample.samePublication publication = true) (sound : publication.BoardBound fc) : sample.BoardBound fc := by
  have equal := matches_payload matched
  unfold BoardBound board at sound ⊢
  simpa only [equal] using sound

def cutMatches (sample : Sample) (frame : FrameReturn) : Bool :=
  let data := Packing.unpack false sample.payload.metadata
  (sample.worker == frame.worker && sample.player == frame.player && sample.opponent == frame.opponent) &&
    (frame.pv == false && data.score.toInt == frame.score) &&
    (match canCut false data ⟨BitVec.ofInt 32 frame.beta⟩ (BitVec.ofNat 32 frame.depth)
        Cutoffs.exactSelectivity true with
      | .ok true => true
      | _ => false)

theorem cut_sound {sample : Sample} {frame : FrameReturn} {fc : Flip.Dispatch.X86Config}
    (matched : sample.cutMatches frame = true) (sane : frame.sane = true) (bound : sample.BoardBound fc) :
    frame.BoardSound fc := by
  let data := Packing.unpack false sample.payload.metadata
  let alpha : Score := ⟨BitVec.ofInt 32 frame.alpha⟩
  let beta : Score := ⟨BitVec.ofInt 32 frame.beta⟩
  let depth := BitVec.ofNat 32 frame.depth
  simp only [cutMatches, Bool.and_eq_true, beq_iff_eq] at matched
  obtain ⟨⟨⟨_, player⟩, opponent⟩, ⟨nonpv, score⟩⟩ := matched.1
  have cut : canCut false data beta depth Cutoffs.exactSelectivity true = .ok true := by
    have accepted := matched.2
    change (match canCut false data beta depth Cutoffs.exactSelectivity true with
      | .ok true => true | _ => false) = true at accepted
    split at accepted <;> first | assumption | contradiction
  obtain ⟨window, nullWindow⟩ := frame.sane_window sane
  obtain ⟨lo, order, hi⟩ := window
  have a : alpha.raw.toInt = frame.alpha := score_word_fits _ lo (by omega)
  have b : beta.raw.toInt = frame.beta := score_word_fits _ (by omega) hi
  have narrow : beta.raw.toInt = alpha.raw.toInt + 1 := by rw [a, b]; exact nullWindow nonpv
  intro valid
  have boardEq : sample.board = frame.board := by simp only [board, payload, FrameReturn.board, player, opponent]
  have ready : Board.wellFormed sample.board := boardEq.symm ▸ valid
  have equal : (⟨sample.board, ready⟩ : PvsBoardTree.Position) = ⟨frame.board, valid⟩ := Subtype.ext boardEq
  have sound := Cutoffs.cutBound data alpha beta depth false _ narrow (bound ready) cut
  rw [a, b, score, equal] at sound
  exact sound.toSoundResult ⟨lo, order, hi⟩ (frame.sane_score sane).1 (frame.sane_score sane).2

end Sample

private def parseRow [FromJson α] (json : String) : Except String α := do
  fromJson? (← Json.parse json)

private def frameBefore {α : Type} (frames : Std.HashMap Nat (Nat × α)) (limit id : Nat) : Option α := do
  let (line, frame) ← frames[id]?
  if line < limit then some frame else none

private abbrev CheckedReturn := {frame : FrameReturn // frame.BoardSound flipConfig}

private abbrev CheckedSample := {sample : Sample // sample.BoardBound flipConfig}

private def certifySample (sample : Sample) (publication : CheckedSample) :
    Option (PLift (sample.BoardBound flipConfig)) :=
  if h : sample.samePublication publication.val = true then
    some ⟨Sample.inherits_bound h publication.property⟩
  else none

private def knownBefore (checked : Std.HashMap Nat (Nat × CheckedReturn)) (limit id : Nat) : Option FrameReturn :=
  (frameBefore checked limit id).map Subtype.val

private theorem knownBefore_sound (checked : Std.HashMap Nat (Nat × CheckedReturn)) (limit : Nat) :
    SoundReturns (knownBefore checked limit) flipConfig := by
  intro id frame read
  cases result : frameBefore checked limit id with
  | none => simp [knownBefore, result] at read
  | some found =>
    have same : found.val = frame := by simpa [knownBefore, result] using read
    exact same ▸ found.property

def runFile (path : String) (thresholdDepth : Nat := 8) : IO UInt32 := do
  let input ← IO.FS.Handle.mk path .read
  let mut publications : Std.HashMap (Nat × Nat × Nat) (Nat × Sample) := {}
  let mut line := 0
  repeat
    let row ← input.getLine
    if row.isEmpty then break
    line := line + 1
    if let ["publish", fields] := row.trimAscii.toString.splitOn "|" then
      match parseRow fields with
      | .error err => IO.eprintln s!"line {line}: {err}"; return 1
      | .ok (sample : Sample) =>
        let key := (sample.scope, sample.slot, sample.first)
        if !sample.samePublication sample || publications.contains key then
          IO.eprintln s!"line {line}: invalid or duplicate slot publication {key}"
          return 1
        publications := publications.insert key (line, sample)
  let input ← IO.FS.Handle.mk path .read
  let mut nodes := 0
  let mut splits := 0
  let mut reads := 0
  let mut external := 0
  let mut stores := 0
  let mut frames : Std.HashMap Nat (Nat × FrameReturn) := {}
  let mut checked : Std.HashMap Nat (Nat × CheckedReturn) := {}
  let mut terminals := 0
  let mut thresholds := 0
  let mut provedPasses := 0
  let mut provedNodes := 0
  let mut passes := 0
  let mut childLinks := 0
  let mut checkedPublications : Std.HashSet (Nat × Nat × Nat) := {}
  let mut provedPublications : Std.HashMap (Nat × Nat × Nat) CheckedSample := {}
  let mut lastProbe : Std.HashMap Nat (Nat × Sample) := {}
  let mut cutoffs := 0
  let mut provedCuts := 0
  let mut provedSamples := 0
  let mut readCounts : Std.HashMap (Nat × Nat × Nat) Nat := {}
  line := 0
  repeat
    let inputRow ← input.getLine
    if inputRow.isEmpty then break
    line := line + 1
    let columns := inputRow.trimAscii.toString.splitOn "|"
    match columns with
    | ["frame", id, fields] =>
      let some id := id.toNat?
        | IO.eprintln s!"line {line}: malformed returned frame number"; return 1
      match parseRow fields with
      | .ok (frame : FrameReturn) =>
        if id == 0 || frames.contains id then
          IO.eprintln s!"line {line}, frame {id}: invalid or duplicate return"
          return 1
        if sane : frame.sane = true then
          if terminal : frame.isTerminal = true then
            if valid : frame.checkBound frame.terminalValue = true then
              checked := checked.insert id (line, ⟨frame, frame.terminal_sound terminal valid⟩)
              terminals := terminals + 1
            else
              IO.eprintln s!"line {line}, frame {id}: returned score disagrees with terminal value"
              return 1
          else if frame.depth ≤ thresholdDepth then
            if valid : frame.checkThreshold flipConfig (frame.sane_board sane) = true then
              checked := checked.insert id (line, ⟨frame, frame.threshold_sound valid⟩)
              thresholds := thresholds + 1
            else
              IO.eprintln s!"line {line}, frame {id}: returned score disagrees with the minimax threshold"
              return 1
          frames := frames.insert id (line, frame)
        else
          IO.eprintln s!"line {line}, frame {id}: invalid or duplicate return"
          return 1
      | .error err => IO.eprintln s!"line {line}, frame {id}: {err}"; return 1
    | ["pass", frame, childId] =>
      let some id := frame.toNat?
        | IO.eprintln s!"line {line}: malformed pass frame number"; return 1
      let some childId := childId.toNat?
        | IO.eprintln s!"line {line}, frame {frame}: malformed pass child number"; return 1
      let some (atReturn, parent) := frames[id]?
        | IO.eprintln s!"line {line}, frame {id}: pass return missing"; return 1
      let some child := frameBefore frames atReturn childId
        | IO.eprintln s!"line {line}, frame {id}: earlier pass reply missing"; return 1
      if !child.followsPass parent id then
        IO.eprintln s!"line {line}, frame {id}: pass child disagrees with reflected position or result"
        return 1
      if let some known := frameBefore checked atReturn childId then
        if link : known.val.followsPass parent id = true then
          if sane : parent.sane = true then
            if !checked.contains id then
              checked := checked.insert id (atReturn, ⟨parent, FrameReturn.pass_sound sane link known.property⟩)
              provedPasses := provedPasses + 1
      passes := passes + 1
    | ["cut", frame, fields] =>
      let some id := frame.toNat?
        | IO.eprintln s!"line {line}: malformed cutoff frame number"; return 1
      let some (atReturn, parent) := frames[id]?
        | IO.eprintln s!"line {line}, frame {id}: cutoff return missing"; return 1
      match parseRow fields with
      | .ok (stamp : SlotStamp) =>
        let some (observedAt, sample) := lastProbe[parent.worker]?
          | IO.eprintln s!"line {line}, frame {id}: cutoff probe missing"; return 1
        if !(observedAt < atReturn && stamp.key == (sample.scope, sample.slot, sample.first)) then
          IO.eprintln s!"line {line}, frame {id}: cutoff names a different probe"
          return 1
        if matched : sample.cutMatches parent = true then
          if let some publication := provedPublications[stamp.key]? then
            if let some checkedRead := certifySample sample publication then
              if sane : parent.sane = true then
                if !checked.contains id then
                  let proof := Sample.cut_sound matched sane checkedRead.down
                  checked := checked.insert id (atReturn, ⟨parent, proof⟩)
                  provedCuts := provedCuts + 1
          cutoffs := cutoffs + 1
        else
          IO.eprintln s!"line {line}, frame {id}: probe cannot justify the recorded cutoff"
          return 1
      | .error err => IO.eprintln s!"line {line}, frame {frame}: {err}"; return 1
    | ["node", frame, fields] =>
      let some id := frame.toNat?
        | IO.eprintln s!"line {line}: malformed search frame number"; return 1
      let some (atReturn, parent) := frames[id]?
        | IO.eprintln s!"line {line}, frame {id}: search return missing"; return 1
      match parseRow fields with
      | .ok (node : Node) =>
        let some (valid : PLift (node.check = true)) := if h : node.check = true then some ⟨h⟩ else none
          | IO.eprintln s!"line {line}, frame {frame}: search replay disagrees"; return 1
        if !node.checkLinks (frameBefore frames atReturn) flipConfig id parent then
          IO.eprintln s!"line {line}, frame {frame}: returned child or parent position disagrees with search"
          return 1
        if coverage : node.coversAll = true then
          if !checked.contains id then
            let known := knownBefore checked atReturn
            if links : node.checkLinks known flipConfig id parent = true then
              let proof := node.frame_sound valid.down links (fun _ => node.all_candidates valid.down coverage)
                (knownBefore_sound checked atReturn)
              checked := checked.insert id (atReturn, ⟨parent, proof⟩)
              provedNodes := provedNodes + 1
        if let some stamp := node.publication then
          let some (prior, publication) := publications[stamp.key]?
            | IO.eprintln s!"line {line}, frame {frame}: publication omitted"; return 1
          if !(decide (prior < line)) then
            IO.eprintln s!"line {line}, frame {frame}: saved table entry disagrees with search"
            return 1
          let some (saved : PLift (publication.ofNode node = true)) :=
            if h : publication.ofNode node = true then some ⟨h⟩ else none
            | IO.eprintln s!"line {line}, frame {frame}: saved table entry disagrees with search"; return 1
          if let some (_, certified) := checked[id]? then
            if matched : node.matchesReturn certified.val = true then
              let proof := Sample.saved_frame_bound saved.down valid.down matched certified.property
              provedPublications := provedPublications.insert stamp.key ⟨publication, proof⟩
          checkedPublications := checkedPublications.insert stamp.key
          stores := stores + 1
        nodes := nodes + 1
        splits := splits + if node.handoff.isSome then 1 else 0
        childLinks := childLinks + node.calls.length + (node.handoff.map fun split =>
          split.reports.foldl (fun count row => count + 1 + row.wide.toList.length) 0).getD 0
      | .error err => IO.eprintln s!"line {line}, frame {frame}: {err}"; return 1
    | [kind, fields] =>
      if kind == "publish" then continue
      if kind != "probe" && kind != "lookup" then
        IO.eprintln s!"line {line}: unknown record {kind}"
        return 1
      match parseRow fields with
      | .ok (sample : Sample) =>
        let key := (sample.scope, sample.slot, sample.first)
        let some (prior, publication) := publications[key]?
          | IO.eprintln s!"line {line}: missing slot publication {key}"; return 1
        if !sample.samePublication publication || !sample.accepted (kind == "lookup") ||
            (sample.worker == publication.worker && !(decide (prior < line))) then
          IO.eprintln s!"line {line}: slot sample disagrees with publication {key}"
          return 1
        reads := reads + 1
        external := external + if sample.worker != publication.worker then 1 else 0
        readCounts := readCounts.insert key (readCounts[key]?.getD 0 + 1)
        if kind == "probe" then
          lastProbe := lastProbe.insert sample.worker (line, sample)
        if let some origin := provedPublications[key]? then
          if (certifySample sample origin).isSome then provedSamples := provedSamples + 1
      | .error err => IO.eprintln s!"line {line}: {err}"; return 1
    | _ =>
      IO.eprintln s!"line {line}: malformed search trace record"
      return 1
  let linked := checkedPublications.fold (fun acc stamp => acc + readCounts[stamp]?.getD 0) 0
  if nodes == 0 || splits == 0 || reads == 0 || linked == 0 || childLinks == 0 then
    IO.eprintln "trace lacks a completed serial frame, split handoff or sampled search publication"
    return 1
  IO.println s!"{nodes} frames, {splits} split handoffs, {stores} search publications, {reads} occupied table samples ({external} cross-thread)"
  IO.println s!"{childLinks} move children and {passes} forced pass children matched among {frames.size} returned frames"
  IO.println s!"{checked.size} of {frames.size} returned bounds certified ({terminals} terminal, {thresholds} threshold up to {thresholdDepth} empties, {provedPasses} through pass, {provedNodes} through replay)"
  IO.println s!"{linked} table samples agree with a checked search publication"
  IO.println s!"{provedPublications.size} table publications and {provedSamples} reads have certified bounds"
  IO.println s!"{cutoffs} probe cutoffs matched, {provedCuts} new returns certified through the table"
  return 0

end NeuralReversi.PvsParallel.TraceCheck
