import NeuralReversi.AspirationMir.NativeRootBirth
import NeuralReversi.PvsParallel.NativeOwners
import NeuralReversi.PvsParallel.NativeRootOwner
import NeuralReversi.PvsParallel.NativeTableWrites
import NeuralReversi.PvsParallel.NativeTableWrites.Allocations
import NeuralReversi.PvsParallel.NativeStopReturns.Check

namespace NeuralReversi.AspirationMir.NativeExecution

open PvsParallel
open NativeFrames (Row logOf)

abbrev trace (out : NativeFrameCache.Checked pool) (mc fc identify before seq) :=
  logOf mc fc out.physical.rows out.physical.table identify before seq

/-- Construct the table used by the public-score proof directly from addressed
    writer instructions interleaved with the native search frames. -/
def tableOfWrites (x : NativeTableWrites.Journal.Execution calls endian records)
    {out : NativeFrameCache.Checked pool}
    (captured : NativeFrameCache.decode pool mc fc inputs values
      (records.filterMap (NativeTableWrites.Journal.frameRecord calls endian)) = some out)
    (identify before seq) (ready : FrameTableReads.Address → Nat)
    (starts : ∀ a, NativeTableWrites.Journal.Start x before ready a)
    (agrees : ∀ a i j, (x.memory a).hb i j → before i j)
    (trans : ∀ {i j k}, before i j → before j k → before i k) :
    FrameTableReads.Table (trace out mc fc identify before seq) :=
  x.table _ ready starts (NativeFrames.decode_keeps (NativeFrameCache.decode_physical captured)).1 agrees trans

structure Evidence (out : NativeFrameCache.Checked pool) (mc fc identify before seq) (endian : Bool) where
  table : FrameTableReads.Table (trace out mc fc identify before seq)
  leaves : NativeLeafBranches.Registers
  probes : (r : Row) → Option (NativeLeafBranches.Probe table r endian)
  stops : NativeStopReturns.Memory
  owners : NativeOwners.Operands table endian stops.history
  root : (r : Row) → Option (NativeRootOwner.Input mc fc r out.physical identify before stops.history)
  receiver : Row → Nat
  writes : List (NativeTableWrites.Packet table endian)
  exits : NativeStopReturns.Evidence stops
  empties : Nat → Nat → Option (BitVec 32)
  overflowChecks : Bool

variable {out : NativeFrameCache.Checked pool}

noncomputable def Evidence.branch (e : Evidence out mc fc identify before seq endian) (r : Row)
    (rows : List RootMoveStorage.Row) (idx : Nat) :=
  (e.root r).bind fun p => p.branch seq rows idx

noncomputable def Evidence.roots (e : Evidence out mc fc identify before seq endian) :
    NativeRows.RootBranches (trace out mc fc identify before seq) := fun r rows => e.branch r rows 0

noncomputable def Evidence.lines (e : Evidence out mc fc identify before seq endian) :
    NativeRowLines.Supply out.physical (trace out mc fc identify before seq) b worker :=
  fun idx line rows => e.branch line.result.row rows idx

def Evidence.stopInputs (e : Evidence out mc fc identify before seq endian) (inputs : Nat → Nat → Option NativeFrames.Args)
    (samples : List NativeRows.Sample) :
    NativeStopReturns.Inputs := ⟨inputs,e.empties,NativeRows.engine samples,e.overflowChecks⟩

noncomputable def Evidence.check (e : Evidence out mc fc identify before seq endian) (inputs : Nat → Nat → Option NativeFrames.Args)
    (samples : List NativeRows.Sample) : Bool :=
  let rows := out.physical.rows
  let stream := NativeRows.engine samples
  NativeStopReturns.check (e.stopInputs inputs samples) e.exits &&
    NativeLeafBranches.checkRows e.table out stream e.leaves e.probes &&
    (rows.filter NativeOwners.larger).all e.owners.check &&
    rows.all (NativeRows.rootCheck e.roots e.receiver samples) &&
    NativeTableWrites.check e.table ⟨out,stream,rfl⟩ e.writes

theorem Evidence.discarded (e : Evidence out mc fc identify before seq endian)
    (checked : e.check inputs samples = true)
    (present : NativeFrames.Record.localCall w (.discarded last first raw) ∈ NativeRows.engine samples) :
    ∃ tr, e.exits w first last = some tr ∧ tr.frame.worker = w ∧ tr.frame.opening = first ∧
      tr.frame.finish = last ∧ tr.decode (e.stopInputs inputs samples) = some ⟨raw,false⟩ ∧
      ∃ t chain, tr.CutFrom t chain ∧ NativeStopReturns.Cause e.stops t chain := by
  simp only [Evidence.check,Bool.and_eq_true] at checked
  apply NativeStopReturns.discarded_source (supply := e.exits) _ present
  unfold NativeStopReturns.normalize
  rw [ite_eq_left checked.1.1.1.1]
  rfl

theorem Evidence.steps (e : Evidence out mc fc identify before seq endian)
    (captured : NativeFrameCache.decode pool mc fc inputs values (NativeRows.engine samples) = some out)
    (checked : e.check inputs samples = true) :
    (∀ r ∈ out.physical.rows, NativeFrames.whole r = true →
      NativeFrames.Branch out.physical (trace out mc fc identify before seq) endian r) ∧
    (∀ w ∈ out.physical.table, FrameReturns.WriteStep (trace out mc fc identify before seq) endian w) := by
  simp only [Evidence.check,Bool.and_eq_true] at checked
  obtain ⟨⟨⟨⟨_,leaf⟩,owners⟩,roots⟩,writes⟩ := checked
  exact ⟨NativeOwners.branches captured e.table e.leaves e.probes leaf e.owners
      (NativeRows.rootInputs e.roots e.receiver roots) owners,
    NativeTableWrites.stores e.table ⟨out,NativeRows.engine samples,rfl⟩ e.writes writes⟩

noncomputable def finalRows (e : Evidence out mc fc identify before seq endian) (samples : List NativeRows.Sample) (r : Row) :
    Option RootStorageMir.Stored :=
  do
    let locks ← NativeRows.collect r (e.receiver r) samples
    let branch ← e.roots r locks.original
    RootStorageMir.replay branch locks.index locks.changes

/-- The reported score is the actual last PVS return word, interpreted in the
    engine's fixed-point units. The root vector and table share the event stream. -/
theorem publicScore (e : Evidence out mc fc identify before seq endian)
    (cache : NativeFrameCache.decode pool mc fc inputs values (NativeRows.engine samples) = some out)
    (checked : e.check inputs samples = true) (irrefl : ∀ i, ¬ before i i)
    (b : PvsBoardTree.Position) (worker solve opening : Nat) {trial : NativeTrials.Line out.physical b worker 0}
    (ran : NativeTrials.decode out.physical inputs (NativeRows.engine samples) b worker 0 solve opening = some trial)
    (sort : PublicationMir.SliceSort) (minimum nodes : Nat) (reportable : minimum ≤ 5)
    (old : Option PublicationMir.Iteration) (fallback : Unit → PublicationMir.SearchResult) :
    ∃ t physical result,
      (NativeTrials.collect worker solve opening (NativeRows.engine samples)).bind (fun ts => ts[trial.result.last.index]?) = some t ∧
      t.opening = trial.result.row.opening ∧ finalRows e samples trial.result.row = some physical ∧
      OuterMir.finish minimum 5 sort physical.rows trial.result.row.depth.toNat nodes old fallback = some result ∧
      PublicationMir.score result = some t.score.toInt ∧ t.score.toInt = PvsBoardTree.minimax mc fc b := by
  obtain ⟨scores,writes⟩ := e.steps cache checked
  obtain ⟨member,_,_,zero,_,_,pos,_,_⟩ := rootReturn_fields trial.result.found
  have rootOK := (Bool.and_eq_true_iff.mp (Bool.and_eq_true_iff.mp checked).1).2
  have accepted := List.all_eq_true.mp rootOK trial.result.row member
  cases locks : NativeRows.collect trial.result.row (e.receiver trial.result.row) samples with
  | none => simp [NativeRows.rootCheck,zero,locks] at accepted
  | some l =>
    cases supplied : e.roots trial.result.row l.original with
    | none => simp [NativeRows.rootCheck,zero,locks,supplied] at accepted
    | some branch =>
      obtain ⟨⟨len,covers⟩,physical,replayed⟩ := (by simpa [NativeRows.rootCheck,zero,locks,supplied,
        Option.isSome_iff_exists] using accepted :
        (0 < l.original.length ∧ Root.Covers mc trial.result.row.frame.position (l.original.map RootMoveStorage.Row.square)) ∧
        ∃ result, RootStorageMir.replay branch l.index l.changes = some result)
      obtain ⟨t,result,actual,atRoot,published,score,truth⟩ := NativeTrials.publicScore cache identify before seq
        irrefl e.table.trans scores writes b worker solve opening ran l.original physical.rows
        (RootStorageMir.line physical.packets) (pos ▸ covers) len (RootStorageMir.rows_of_replay branch replayed)
        sort minimum nodes reportable old fallback
      exact ⟨t,physical,result,actual,atRoot,by simp [finalRows,locks,supplied,replayed],published,score,truth⟩

theorem publicLines (e : Evidence out mc fc identify before seq endian)
    (cache : NativeFrameCache.decode pool mc fc inputs values (NativeRows.engine samples) = some out)
    (checked : e.check inputs samples = true) (irrefl : ∀ i, ¬ before i i)
    (b : PvsBoardTree.Position) (worker solve object : Nat) (stages : List NativeRowLines.Stage)
    (ran : NativeRootBirth.run out.physical _ inputs samples b worker solve object e.lines paired mem stages = some final)
    (minimum : Nat) (reportable : minimum ≤ 5) (sort : PublicationMir.SliceSort) (nodes : Nat)
    (old : Option PublicationMir.Iteration) (fallback : Unit → PublicationMir.SearchResult) :
    ∃ result, OuterMir.finish minimum 5 sort final (Board.emptyCount b.val).toNat nodes old fallback = some result ∧
      PublicationMir.score result = some (PvsBoardTree.minimax mc fc b) ∧
      ∀ row ∈ final, row.score = Root.squareValue mc fc b row.square := by
  obtain ⟨scores,writes⟩ := e.steps cache checked
  obtain ⟨rows,played,covers,nonempty⟩ := NativeRootBirth.lines ran
  exact NativeRowLines.publicLines cache identify before seq irrefl e.table.trans scores writes b worker solve object
    stages e.lines played covers nonempty minimum reportable sort nodes old fallback

end NeuralReversi.AspirationMir.NativeExecution
