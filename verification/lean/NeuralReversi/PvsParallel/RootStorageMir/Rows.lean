module
public import NeuralReversi.PvsParallel.RootStorageMir.Program
public import NeuralReversi.PvsParallel.FrameRows

@[expose] public section

namespace NeuralReversi.PvsParallel.RootStorageMir

open RootMoveStorage (Row)
open Bitboard (Square)
open RootReplay (Snapshot)
open PvsBoardTree (GeneratedSquare build play)
open FrameReturns (Frame Log RootOwner RootRows reportRows)
open Threading.SplitResult (Report)

theorem reset_eq (idx : Nat) (rows : List Row) : reset idx rows = RootMoveStorage.resetFrom rows idx := rfl

theorem sort_eq (idx : Nat) (rows : List Row) : sort idx rows = RootMoveStorage.sortRows rows idx := by
  by_cases h : idx < rows.length
  · simp [sort,h,RootMoveStorage.sortRows]
  · simp [sort,h,(RootMoveStorage.sortRowsContract_proven rows idx).2.2.2.2 (Nat.le_of_not_gt h)]

theorem changed_eq (old : Row) (value : Int) (shown : Bool) (slots : Slots) :
    changed old value shown slots = RootSuffix.updateRow old value shown (pvPrefix slots.toList) := by
  by_cases seed : old.average = negInf <;>
    simp [changed,RootSuffix.updateRow,MidgameAspiration.History.update,MidgameAspiration.signedSquare,
      negInf,PvsMoveLoop.scoreInf] at seed ⊢ <;> simp [seed]

theorem valid_numeric (old : Row) (value : Int)
    (valid : (⟨old.average,old.meanSquared⟩ : MidgameAspiration.History).valid)
    (finite : MidgameAspiration.FiniteScore value) : numeric old value = true := by
  obtain ⟨a,b,c,d⟩ := MidgameAspiration.history_storage valid finite
  exact decide_eq_true ⟨a,b,fun _ => ⟨c,d⟩⟩

theorem updated (got : update sq value shown slots rows = some after) :
    after = RootSuffix.writeFirst sq (fun old => RootSuffix.updateRow old value shown (pvPrefix slots.toList)) rows := by
  induction rows generalizing after with
  | nil => contradiction
  | cons old rest ih =>
    by_cases pos : old.square = sq
    · simp only [update,pos,ite_true] at got
      cases h : numeric old value <;> simp only [h,Bool.false_eq_true,ite_false,ite_true,reduceCtorEq] at got
      cases got
      simp [RootSuffix.writeFirst,pos,changed_eq]
    · simp only [update,pos,ite_false] at got
      obtain ⟨tail,write,eq⟩ := Option.map_eq_some_iff.mp got
      exact eq.symm.trans (by simp [RootSuffix.writeFirst,pos,ih write])

theorem update_succeeds {rows : List Row} (member : sq ∈ rows.map Row.square)
    (valid : ∀ row ∈ rows, (⟨row.average,row.meanSquared⟩ : MidgameAspiration.History).valid)
    (finite : MidgameAspiration.FiniteScore value) : ∃ after, update sq value shown slots rows = some after := by
  induction rows with
  | nil => contradiction
  | cons first tail ih =>
    by_cases same : first.square = sq
    · simp [update,same,valid_numeric first value (valid first (by simp)) finite]
    · have rest : sq ∈ tail.map Row.square := by simpa [Ne.symm same] using member
      obtain ⟨after,wrote⟩ := ih rest (fun row h => valid row (by simp [h]))
      simp [update,same,wrote]

structure Operand where
  sq : Square
  score : Int
  shown : Bool
  deriving DecidableEq

structure Packet where
  worker : Nat
  tick : Nat
  sq : Square
  value : BitVec 32
  shown : Bool
  slots : Slots

def Packet.operand (r : Packet) : Operand := ⟨r.sq,r.value.toInt,r.shown⟩

def line (packets : List Packet) (sq : Square) : List Square :=
  ((packets.find? (fun p => p.shown && p.sq == sq)).map (fun p => pvPrefix p.slots.toList)).getD []

def applyOperand (pv : Square → List Square) (rows : List Row) (r : Operand) : List Row :=
  RootSuffix.writeFirst r.sq (fun old => RootSuffix.updateRow old r.score r.shown (pv r.sq)) rows

def foldUpdates (pvs : Square → List Square) (rows : List Row) : List Packet → Option (List Row)
  | [] => some rows
  | p :: rest => do
    if p.shown && decide (pvPrefix p.slots.toList ≠ pvs p.sq) then none else
      let changed ← update p.sq p.value.toInt p.shown p.slots rows
      foldUpdates pvs changed rest

private theorem fold_eq (ps : List Packet) (got : foldUpdates pv rows ps = some after) :
    after = (ps.map Packet.operand).foldl (applyOperand pv) rows := by
  induction ps generalizing rows with
  | nil => cases got; rfl
  | cons p ps ih =>
    simp only [foldUpdates] at got
    split at got <;> try contradiction
    rename_i consistent
    obtain ⟨head,done,more⟩ := Option.bind_eq_some_iff.mp got
    rw [ih more,updated done]
    simp only [List.map_cons,List.foldl_cons,applyOperand,Packet.operand]
    congr 1
    cases v : p.shown with
    | false => simp [RootSuffix.updateRow]
    | true =>
      have same : pvPrefix p.slots.toList = pv p.sq := by simpa [v] using consistent
      rw [same]

def ownerOperands (sq : M → Square) (child : PvsMoveLoop.ChildSearch M) (z : Int) :
    List M → Snapshot → List Operand
  | [],_ => []
  | m :: rest,s =>
    let score := PvsMoveLoop.moveScore true s.seen.isEmpty child m s.state.alpha z
    let decision := RootUpdateMir.decision s.seen.isEmpty s.state.alpha z s.state.best score
    ⟨sq m,score,decision.visible⟩ :: (if decision.cuts then [] else
      ownerOperands sq child z rest (RootUpdateMir.after s (sq m) score decision))

private theorem owner_eq (sq : M → Square) (child : PvsMoveLoop.ChildSearch M) (pv : Square → List Square) (z : Int)
    (moves : List M) (s : Snapshot) (rows : List Row) :
    (ownerOperands sq child z moves s).foldl (applyOperand pv) rows =
      RootSerialMir.rowsAfter sq (fun m => pv (sq m)) child z moves s rows := by
  induction moves generalizing s rows with
  | nil => rfl
  | cons m rest ih =>
    simp only [ownerOperands,RootSerialMir.rowsAfter,List.foldl_cons]
    split <;> simp [applyOperand,ih]

def sharedOperands (square : Nat → Square) (z : Int) : List Report → Snapshot → List Operand
  | [],_ => []
  | r :: rest,s =>
    let sq := square r.index
    ⟨sq,r.score,decide (s.state.alpha < r.score)⟩ ::
      sharedOperands square z rest (s.record z ⟨sq,r.searchedAlpha,r.score⟩)

private theorem shared_eq (sq : Nat → Square) (pv : Square → List Square) (z : Int)
    (reports : List Report) (s : Snapshot) (rows : List Row) :
    (sharedOperands sq z reports s).foldl (applyOperand pv) rows = reportRows sq pv z rows s reports := by
  induction reports generalizing s rows with
  | nil => rfl
  | cons r rest ih => simp [sharedOperands,reportRows,applyOperand,ih]

structure Writer where
  worker : Nat
  locked : Option (Nat × Nat)

def Writer.accepts (w : Writer) (p : Packet) : Bool :=
  decide (p.worker = w.worker) && w.locked.all (fun bound => decide (bound.1 < p.tick ∧ p.tick < bound.2))

def publications (split : SplitExecution len pv alpha beta score) : List Writer :=
  List.ofFn fun n : Fin split.reports.length =>
    let pre := split.reports.take n.val
    let post := split.reports.drop (n.val+1)
    have same : split.reports = pre ++ split.reports[n] :: post := by
      simp [pre,post,List.getElem_cons_drop]
    let pub := (split.reportPublications pre split.reports[n] post same).val
    ⟨pub.worker.val,some (pub.checked.step,pub.published.step)⟩

structure Serial (log : Log mc fc) (f : Frame) (depth : TranspositionTable.Depth) (rows : List Row) (idx : Nat) where
  owner : RootOwner log f depth (rows.map Row.square) idx f.alpha f.beta
  done : owner.receipt.handoff = false

structure Handoff (log : Log mc fc) (f : Frame) (depth : TranspositionTable.Depth) (rows : List Row) (idx : Nat) where
  owner : RootOwner log f depth (rows.map Row.square) idx f.alpha f.beta
  first : GeneratedSquare mc f.position.val
  rest : List (GeneratedSquare mc f.position.val)
  score : Int
  children : List PvsMoveLoop.Tree
  paired : children = (first :: rest).map (fun m => build mc fc (play mc fc f.position m).val)
  division : SplitExecution children.length true f.alpha f.beta score
  gave : owner.receipt.handoff = true
  frozen : first :: rest = owner.receipt.visited ++ owner.receipt.remaining.pending
  start : division.start.toNat = owner.receipt.remaining.idx
  entry : division.entry = owner.receipt.snapshot.state
  attended : ReportCalls (FrameReturns.Children log f) children division

structure Wipe (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (f : Frame) (rows : List Row) (idx : Nat) where
  paired : Bool
  mem : MoveList.Initialization.Memory Move.Move
  sq : Square
  found : (MoveList.Generation.fill false paired fc f.position.val
      (Root.remainingMask mc f.position.val (rows.map Row.square) idx) mem).wipeout = some (some sq)

inductive Case (S D W : Type) where | serial (e : S) | split (e : D) | wipe (e : W)

abbrev Branch (log : Log mc fc) (f : Frame) (depth : TranspositionTable.Depth) (rows : List Row) (idx : Nat) :=
  Case (Serial log f depth rows idx) (Handoff log f depth rows idx) (Wipe mc fc f rows idx)

def ownerCalls {log : Log mc fc} {rows : List Row}
    (o : RootOwner log f d (rows.map Row.square) idx f.alpha f.beta) : List Operand :=
  ownerOperands Subtype.val (RootOwnerMir.observed (log.pvReplies f d)) f.beta o.receipt.replayMoves (RootReplay.start f.alpha)

def Branch.output {log : Log mc fc} : Branch log f d rows idx → Int
  | .serial s => s.owner.receipt.snapshot.state.best
  | .split h => h.score
  | .wipe _ => PvsMoveLoop.scoreMax

def Branch.calls {log : Log mc fc} : Branch log f d rows idx → List Operand
  | .serial s => ownerCalls s.owner
  | .split h => ownerCalls h.owner ++ sharedOperands (fun i => (((h.first :: h.rest)[i]?).getD h.first).val)
      f.beta h.division.reports h.owner.receipt.snapshot
  | .wipe w => [⟨w.sq,PvsMoveLoop.scoreMax,true⟩]

def Branch.writers {log : Log mc fc} : Branch log f d rows idx → List Writer
  | .serial s => (ownerCalls s.owner).map (fun _ => ⟨f.worker,none⟩)
  | .split h => (ownerCalls h.owner).map (fun _ => ⟨f.worker,none⟩) ++ publications h.division
  | .wipe _ => [⟨f.worker,none⟩]

theorem Branch.applies {log : Log mc fc} (branch : Branch log f d rows idx) (pv : Square → List Square) :
    RootRows log f d rows idx f.alpha f.beta pv (Branch.output branch)
      ((Branch.calls branch).foldl (applyOperand pv) (reset idx rows)) := by
  cases branch with
  | serial s =>
    simp only [Branch.calls,Branch.output,ownerCalls,owner_eq,reset_eq]
    exact .serial s.owner s.done
  | split h =>
    obtain ⟨owner,first,rest,score,_,rfl,division,gave,frozen,start,entry,attended⟩ := h
    simp only [Branch.calls,Branch.output,List.foldl_append,ownerCalls,owner_eq,shared_eq,reset_eq]
    exact .split owner first rest division gave frozen start entry attended
  | wipe w => exact .wipe w.paired w.mem w.sq w.found

structure Stored where
  packets : List Packet
  rows : List Row

/-- The index is the value of the relaxed pv_idx load in reset_scores_from_pv_idx. The packets
    follow acquisition of the root-vector mutex and include the PV array read by update. -/
def replay {log : Log mc fc} (branch : Branch log f d initial idx) (observedIndex : Nat) (packets : List Packet) :
    Option Stored := do
  if observedIndex = idx ∧ packets.map Packet.operand = branch.calls ∧ f.score = branch.output ∧
      packets.length = branch.writers.length ∧ (branch.writers.zip packets).all (fun pair => pair.1.accepts pair.2) then
    let rows ← foldUpdates (line packets) (reset observedIndex initial) packets
    return ⟨packets,rows⟩
  else none

theorem rows_of_replay {log : Log mc fc} (branch : Branch log f d initial idx)
    (ok : replay branch observedIndex packets = some stored) :
    RootRows log f d initial idx f.alpha f.beta (line stored.packets) f.score stored.rows := by
  unfold replay at ok
  split at ok <;> try contradiction
  rename_i agree
  obtain ⟨rows,changes,eq⟩ := Option.bind_eq_some_iff.mp ok
  cases eq
  obtain ⟨rfl,sequence,score,_⟩ := agree
  have actual := fold_eq packets changes
  rw [sequence] at actual
  exact score.symm ▸ actual.symm ▸ branch.applies (line packets)

end NeuralReversi.PvsParallel.RootStorageMir
