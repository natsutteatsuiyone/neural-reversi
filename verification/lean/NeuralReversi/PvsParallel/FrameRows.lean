import NeuralReversi.PvsParallel.FrameReturns
import NeuralReversi.PvsParallel.SplitWorkerMir.Correct

namespace NeuralReversi.PvsParallel.FrameReturns

open RootMoveStorage (Row)
open RootReplay (Snapshot)
open Bitboard (Square)
open PvsBoardTree (GeneratedSquare)

def RootOwner.rows {log : Log mc fc} {initial : List Row} (owner : RootOwner log f d (initial.map Row.square) idx a z)
    (pv : Square → List Square) : List Row :=
  RootSerialMir.rowsAfter Subtype.val (fun m => pv m.val) (RootOwnerMir.observed (log.pvReplies f d)) z
    owner.receipt.replayMoves (RootReplay.start a) (RootMoveStorage.resetFrom initial idx)

theorem RootOwner.writes {log : Log mc fc} {initial : List Row} (owner : RootOwner log f d (initial.map Row.square) idx a z)
    (covers : Root.Covers mc f.position (initial.map Row.square)) (pv : Square → List Square) :
    RootSuffix.Writes initial idx a z (owner.rows pv) owner.receipt.snapshot := by
  have uses : ∀ m cp low high v, log.pvReplies f d m cp low high = some v → True := fun _ _ _ _ _ _ => trivial
  obtain ⟨trace, snapshot, _⟩ := RootOwnerMir.refines Subtype.val (log.pvReplies f d) (fun _ _ _ _ _ => True)
    uses _ _ _ _ _ _ _ _ _ _ (by simp) rfl owner.normal
  have before : (owner.receipt.visited ++ owner.receipt.remaining.pending).Perm owner.order := by
    simpa [RootOwner.order,PvsPostpone.Queue.pending] using trace.reproduced.1
  have order := before.trans owner.candidates
  rw [snapshot]
  apply RootSerialMir.rowsAfter_writes _ _ _ _ _ .reset (Root.roots_nodup covers)
  intro m mem
  have member : m ∈ owner.receipt.visited ++ owner.receipt.remaining.pending := by
    cases h : owner.receipt.handoff <;> simp [RootOwnerMir.Result.replayMoves, h] at mem <;> simp_all
  have kept := (Root.choices_squares_mem (idx := idx) covers m.val).mp (List.mem_map_of_mem (order.mem_iff.mp member))
  simpa [List.map_drop] using kept

def reportRows (square : Nat → Square) (pv : Square → List Square) (z : Int)
    (rows : List Row) (s : Snapshot) : List Threading.SplitResult.Report → List Row
  | [] => rows
  | report :: rest =>
    let sq := square report.index
    let shown := decide (s.state.alpha < report.score)
    let updated := RootSuffix.writeFirst sq (fun row => RootSuffix.updateRow row report.score shown (pv sq)) rows
    reportRows square pv z updated (s.record z ⟨sq,report.searchedAlpha,report.score⟩) rest

theorem reportWrites (square : Nat → Square) (pv : Square → List Square) (z : Int)
    (reports : List Threading.SplitResult.Report) (s : Snapshot)
    (prior : RootSuffix.Writes initial idx a z rows s) (begun : s.seen ≠ [])
    (unique : (initial.map Row.square).Nodup)
    (members : ∀ r ∈ reports, square r.index ∈ (initial.drop idx).map Row.square) :
    RootSuffix.Writes initial idx a z (reportRows square pv z rows s reports) (RootReplay.Split.replay square z reports s) := by
  induction reports generalizing rows s with
  | nil => exact prior
  | cons r rest ih =>
    let report : RootOrder.Report := ⟨square r.index, r.searchedAlpha, r.score⟩
    have member := members r (by simp)
    rw [List.map_drop, ← (prior.scores unique).1, ← List.map_drop] at member
    have written := RootSuffix.Writes.record prior report (decide (s.state.alpha < r.score)) (pv (square r.index)) member
      (by simp [List.isEmpty_eq_false_iff.mpr begun, report])
    exact ih _ written (by simp [RootReplay.Snapshot.record]) (fun t mem => members t (List.mem_cons_of_mem r mem))

private theorem seenAfter (sq : M → Square) (search : PvsMoveLoop.ChildSearch M) (z : Int) (ms : List M) (s : Snapshot)
    (begun : s.seen ≠ [] ∨ ms ≠ []) : (RootReplay.serial sq search z ms s).seen ≠ [] := by
  induction ms generalizing s with
  | nil => simpa [RootReplay.serial] using begun
  | cons m rest ih =>
    rw [RootReplay.serial]
    split
    · simp [RootReplay.Snapshot.record]
    · exact ih _ (.inl (by simp [RootReplay.Snapshot.record]))

theorem RootOwner.handoffWrites {log : Log mc fc} {initial : List Row}
    (owner : RootOwner log f d (initial.map Row.square) idx a z)
    (first : GeneratedSquare mc f.position.val) (rest : List (GeneratedSquare mc f.position.val))
    (division : SplitExecution ((first :: rest).map (fun m => PvsBoardTree.build mc fc
      (PvsBoardTree.play mc fc f.position m).val)).length true a z score)
    (covers : Root.Covers mc f.position (initial.map Row.square)) (pv : Square → List Square)
    (handoff : owner.receipt.handoff = true)
    (labels : first :: rest = owner.receipt.visited ++ owner.receipt.remaining.pending)
    (entry : division.entry = owner.receipt.snapshot.state) :
    let sq := fun i : Nat => (((first :: rest)[i]?).getD first).val
    let final := RootReplay.Split.replay sq z division.reports owner.receipt.snapshot
    RootSuffix.Writes initial idx a z (reportRows sq pv z (owner.rows pv) owner.receipt.snapshot division.reports) final ∧
      final.state.best = score := by
  intro sq final
  have uses : ∀ m cp x y v, log.pvReplies f d m cp x y = some v → True := fun _ _ _ _ _ _ => trivial
  obtain ⟨trace, snapshot, positive⟩ := RootOwnerMir.refines Subtype.val (log.pvReplies f d) (fun _ _ _ _ _ => True)
    uses _ _ _ _ _ _ _ _ _ _ (by simp) rfl owner.normal
  have before : (owner.receipt.visited ++ owner.receipt.remaining.pending).Perm owner.order := by
    simpa [RootOwner.order,PvsPostpone.Queue.pending] using trace.reproduced.1
  have length : owner.receipt.remaining.idx = owner.receipt.visited.length := by simpa using trace.reproduced.2.1
  have nonempty : owner.receipt.visited ≠ [] := by
    have n := positive handoff
    rintro h
    simp [length,h] at n
  have started : owner.receipt.snapshot.seen ≠ [] := by
    rw [snapshot]
    exact seenAfter _ _ _ _ _ (.inr (by simpa [RootOwnerMir.Result.replayMoves,handoff] using nonempty))
  constructor
  · apply reportWrites _ _ _ _ _ (owner.writes covers pv) started (Root.roots_nodup covers)
    intro r _
    have belongs : ((first :: rest)[r.index]?).getD first ∈ first :: rest := by
      cases hit : (first :: rest)[r.index]? with
      | none => simp
      | some m => exact List.mem_of_getElem? hit
    have perm := labels.symm ▸ before.trans owner.candidates
    have kept := (Root.choices_squares_mem (idx := idx) covers _).mp
      (List.mem_map_of_mem (perm.mem_iff.mp belongs))
    simpa [List.map_drop] using kept
  · have observed := Threading.SplitReturn.ownerBestContract_proven division.history division.writes division.result
      division.lastScore division.wait true z division.entry division.reports division.read division.bestStores
    have raw := division.accepted
    rw [observed,entry] at raw
    change (RootReplay.Split.replay sq z division.reports owner.receipt.snapshot).state.best = score
    rw [RootReplay.Split.replay_state]
    cases stopped : division.check.stopped <;> simp [Threading.SplitReturn.checked,stopped] at raw
    exact raw

def wipeRows (rows : List Row) (idx : Nat) (sq : Square) (pv : Square → List Square) : List Row :=
  RootSuffix.writeFirst sq (fun row => RootSuffix.updateRow row PvsMoveLoop.scoreMax true (pv sq))
    (RootMoveStorage.resetFrom rows idx)

private theorem wipeWrites {f : Frame} {initial : List Row}
    (covers : Root.Covers mc f.position (initial.map Row.square)) (pv : Square → List Square)
    (cache : (MoveList.Generation.fill false paired fc f.position.val
      (Root.remainingMask mc f.position.val (initial.map Row.square) idx) mem).wipeout = some (some sq)) :
    RootSuffix.Writes initial idx a z (wipeRows initial idx sq pv)
      ((RootReplay.start a).record z ⟨sq,a,PvsMoveLoop.scoreMax⟩) := by
    obtain ⟨move, allowed, square, _⟩ := ConcurrentReplay.generatedWipeout mc fc f.position
      (initial.map Row.square) idx paired mem sq covers cache
    have kept := (Root.choices_squares_mem (idx := idx) covers move.val).mp (List.mem_map_of_mem allowed)
    have prior := RootSuffix.Writes.reset (initial := initial) (idx := idx) (alpha := a) (beta := z)
    have member : sq ∈ ((RootMoveStorage.resetFrom initial idx).drop idx).map Row.square := by
      rw [List.map_drop,(prior.scores (Root.roots_nodup covers)).1,← square]
      exact kept
    exact RootSuffix.Writes.record prior (⟨sq,a,PvsMoveLoop.scoreMax⟩ : RootOrder.Report) true (pv sq)
      member (by simp [RootReplay.start])

/-- The row array resulting from the owner's actual return branch, before the suffix sort. -/
inductive RootRows (log : Log mc fc) (f : Frame) (d : TranspositionTable.Depth)
    (initial : List Row) (idx : Nat) (a z : Int) (pv : Square → List Square) : Int → List Row → Prop
  | serial (owner : RootOwner log f d (initial.map Row.square) idx a z) :
      owner.receipt.handoff = false → RootRows log f d initial idx a z pv owner.receipt.snapshot.state.best (owner.rows pv)
  | split (owner : RootOwner log f d (initial.map Row.square) idx a z)
      (first : GeneratedSquare mc f.position.val) (rest : List (GeneratedSquare mc f.position.val))
      (division : SplitExecution ((first :: rest).map (fun m => PvsBoardTree.build mc fc
        (PvsBoardTree.play mc fc f.position m).val)).length true a z score) :
      owner.receipt.handoff = true → first :: rest = owner.receipt.visited ++ owner.receipt.remaining.pending →
      division.start.toNat = owner.receipt.remaining.idx → division.entry = owner.receipt.snapshot.state →
      ReportCalls (Children log f) ((first :: rest).map (fun m => PvsBoardTree.build mc fc
        (PvsBoardTree.play mc fc f.position m).val)) division →
      RootRows log f d initial idx a z pv score (reportRows (fun i => (((first :: rest)[i]?).getD first).val)
        pv z (owner.rows pv) owner.receipt.snapshot division.reports)
  | wipe (paired : Bool) (mem : MoveList.Initialization.Memory Move.Move) (sq : Square) :
      (MoveList.Generation.fill false paired fc f.position.val
        (Root.remainingMask mc f.position.val (initial.map Row.square) idx) mem).wipeout = some (some sq) →
      RootRows log f d initial idx a z pv PvsMoveLoop.scoreMax (wipeRows initial idx sq pv)

theorem RootRows.checked {log : Log mc fc} (rows : RootRows log f d initial idx a z pv score physical)
    (covers : Root.Covers mc f.position (initial.map Row.square)) :
    ∃ s, RootStep log f d (initial.map Row.square) idx a z score s.scores ∧
      RootSuffix.Writes initial idx a z physical s ∧ s.state.best = score := by
  cases rows with
  | serial owner normal => exact ⟨_, .serial owner normal,owner.writes covers pv,rfl⟩
  | split owner first rest division handed labels start entry reports =>
    obtain ⟨written,value⟩ := owner.handoffWrites first rest division covers pv handed labels entry
    exact ⟨_,.split owner first rest division handed labels start entry reports,written,value⟩
  | wipe paired mem sq cache =>
    exact ⟨_,.wipe paired mem sq cache,wipeWrites covers pv cache,
      by change max (-PvsMoveLoop.scoreInf) PvsMoveLoop.scoreMax = PvsMoveLoop.scoreMax; decide⟩

theorem RootStep.materialize {log : Log mc fc} {initial : List Row}
    (step : RootStep log f d (initial.map Row.square) idx a z score lanes)
    (covers : Root.Covers mc f.position (initial.map Row.square)) (pv : Square → List Square) :
    ∃ rows s, RootSuffix.Writes initial idx a z rows s ∧ s.state.best = score ∧ s.scores = lanes := by
  cases step with
  | serial owner _ => exact ⟨_,_,owner.writes covers pv,rfl,rfl⟩
  | split owner first rest division handed labels _ entry _ =>
    obtain ⟨written,value⟩ := owner.handoffWrites first rest division covers pv handed labels entry
    exact ⟨_,_,written,value,rfl⟩
  | wipe paired mem sq cache =>
    exact ⟨_,_,wipeWrites covers pv cache,
      by change max (-PvsMoveLoop.scoreInf) PvsMoveLoop.scoreMax = PvsMoveLoop.scoreMax; decide,rfl⟩

end NeuralReversi.PvsParallel.FrameReturns
