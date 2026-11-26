import NeuralReversi.AspirationMir.NativeRows

namespace NeuralReversi.AspirationMir.NativeRowLines

open PvsParallel NativeFrames
open NativeRows (Sample Mutation CallSite Locks)
open RootStorageMir (Packet Stored)
private structure Prepared (idx : Nat) (initial : List RootMoveStorage.Row) where
  rows : List RootMoveStorage.Row
  prior : Prepare idx initial rows
  steps : List (Nat × Nat)

private def applyPackets (idx : Nat) (a z : Int) (initial : List RootMoveStorage.Row) :
    (rows : List RootMoveStorage.Row) → (snap : RootReplay.Snapshot) →
    RootSuffix.Writes initial idx a z rows snap → List Packet →
    Option { after : List RootMoveStorage.Row // ∃ s, RootSuffix.Writes initial idx a z after s }
  | rows,snap,prior,[] => some ⟨rows,snap,prior⟩
  | rows,snap,prior,p :: rest => do
    if hv : p.sq ∈ (rows.drop idx).map RootMoveStorage.Row.square ∧
        p.shown = (snap.seen.isEmpty || decide (snap.state.alpha < p.value.toInt)) then
      let changed ← (RootStorageMir.update p.sq p.value.toInt p.shown p.slots rows).attach
      have writes := RootSuffix.Writes.record prior ⟨p.sq,snap.state.alpha,p.value.toInt⟩ p.shown
        (RootStorageMir.pvPrefix p.slots.toList) hv.1 hv.2
      applyPackets idx a z initial changed.val _ ((RootStorageMir.updated changed.property).symm ▸ writes) rest
    else none

inductive Prior where
  | trial (call : CallSite)
  | sort (step : Nat)

private def sorted (samples : List Sample) (w tick object idx : Nat) : Bool :=
  samples.any fun
    | .locked worker t receiver (.sort index) => worker == w && t == tick && receiver == object && index == idx
    | _ => false

private def prepare (samples : List Sample) (inputs : Nat → Nat → Option Args) (b : PvsBoardTree.Position)
    (w object idx : Nat) (initial : List RootMoveStorage.Row) :
    Prepared idx initial → List Prior → Option (Prepared idx initial)
  | s,[] => some s
  | s,.sort tick :: rest => do
    if sorted samples w tick object idx then
      prepare samples inputs b w object idx initial
        ⟨RootStorageMir.sort idx s.rows,(RootStorageMir.sort_eq idx s.rows).symm ▸ .sort s.prior,s.steps ++ [(w,tick)]⟩ rest
    else none
  | s,.trial call :: rest => do
    let args ← inputs w call.opening
    let locks ← NativeRows.collectAt call object samples
    if call.worker = w ∧ args.board = b.val ∧ args.pv = true ∧ args.cut = false ∧
        args.rootLine = some idx ∧ locks.index = idx ∧ locks.original.map NativeRows.fields = s.rows.map NativeRows.fields then
      let a := args.alpha.raw.toInt
      let z := args.beta.raw.toInt
      have init : RootSuffix.Writes s.rows idx a z (RootStorageMir.reset idx s.rows) (RootReplay.start a) := .reset
      let after ← applyPackets idx a z s.rows _ _ init locks.changes
      prepare samples inputs b w object idx initial
        ⟨after.val,.trial s.prior after.property.choose_spec,s.steps ++ locks.steps w⟩ rest
    else none

private def stamp : Sample → Option (Nat × Nat)
  | .locked w t _ _ => some (w,t)
  | .engine (.table _) => none
  | .engine (.localCall w ev) => some (w,match ev with
      | .entry t .. | .leave t .. | .scored t .. | .discarded t .. => t)

private def writesBetween (samples : List Sample) (w start stop object : Nat) : Option (List (Nat × Nat)) := do
  let _ :: after := samples.dropWhile (fun s => stamp s != some (w,start)) | none
  let pre := after.takeWhile (fun s => stamp s != some (w,stop))
  let release ← after[pre.length]?
  let lockId : Sample → Option (Nat × Nat) := fun
    | .locked who t obj _ => if obj == object then some (who,t) else none
    | _ => none
  return (pre ++ [release]).filterMap lockId

structure Stage where
  opening : Nat
  sortedAt : Nat
  earlier : List Prior

abbrev Supply (c : Capture pool) (log : FrameReturns.Log mc fc) (b : PvsBoardTree.Position) (w : Nat) :=
  (idx : Nat) → (line : NativeTrials.Line c b w idx) → (rows : List RootMoveStorage.Row) →
  Option (RootStorageMir.Branch log line.result.row.frame line.result.row.depth rows idx)

structure Batch (c : Capture pool) (log : FrameReturns.Log mc fc) (b : PvsBoardTree.Position)
    (idx : Nat) (rows : List RootMoveStorage.Row) where
  final : List RootMoveStorage.Row
  lines : NativeLines c log b (Board.emptyCount b.val) idx rows final

/-- Replay every mutation of the selected root vector between two successive line sorts.
    The complete lock sequence includes earlier selectivities and failed aspiration trials. -/
def run (c : Capture pool) (log : FrameReturns.Log mc fc) (inputs : Nat → Nat → Option Args)
    (samples : List Sample) (b : PvsBoardTree.Position) (w solve object : Nat) (supply : Supply c log b w) :
    (prior idx : Nat) → (rows : List RootMoveStorage.Row) → List Stage → Option (Batch c log b idx rows)
  | _,idx,rows,[] => if h : idx = rows.length then some ⟨rows,h ▸ .done rows⟩ else none
  | prior,idx,rows,stage :: rest => do
    let line ← NativeTrials.decode c inputs (NativeRows.engine samples) b w idx solve stage.opening
    let prepared ← prepare samples inputs b w object idx rows ⟨rows,.start,[]⟩ stage.earlier
    if inside : idx < prepared.rows.length then
      let branch ← supply idx line prepared.rows
      let row := line.result.row
      let locks ← NativeRows.collect row object samples
      let store ← (RootStorageMir.replay branch locks.index locks.changes).attach
      let returned ← row.frame.returnStep
      let observed ← writesBetween samples w prior stage.sortedAt object
      if prior < stage.opening ∧ returned < stage.sortedAt ∧ sorted samples w stage.sortedAt object idx ∧
          locks.original.map NativeRows.fields = prepared.rows.map NativeRows.fields ∧
          observed = prepared.steps ++ locks.steps w ++ [(w,stage.sortedAt)] then
        let batch ← run c log inputs samples b w solve object supply stage.sortedAt (idx+1)
          (RootStorageMir.sort idx store.val.rows) rest
        return ⟨batch.final,.next w (NativeTrials.stamp line.trials) _ _ _ row (RootStorageMir.line store.val.packets) _
          rfl line.result.path.ran line.result.found prepared.prior inside
          (RootStorageMir.rows_of_replay branch store.property) (RootStorageMir.sort_eq idx store.val.rows ▸ batch.lines)⟩
      else none
    else none

theorem publicLines {out : NativeFrameCache.Checked pool}
    (cache : NativeFrameCache.decode pool mc fc inputs values (NativeRows.engine samples) = some out)
    (identify before sequenced) (irrefl : ∀ i, ¬before i i)
    (trans : ∀ {i j k}, before i j → before j k → before i k)
    (scores : let c := out.physical
      let log := NativeFrames.logOf mc fc c.rows c.table identify before sequenced
      ∀ row ∈ c.rows, NativeFrames.whole row = true → NativeFrames.Branch c log endian row)
    (stores : let c := out.physical
      let log := NativeFrames.logOf mc fc c.rows c.table identify before sequenced
      ∀ e ∈ c.table, FrameReturns.WriteStep log endian e)
    (b : PvsBoardTree.Position) (worker solve object : Nat) (stages : List Stage)
    (supply : Supply out.physical
      (NativeFrames.logOf mc fc out.physical.rows out.physical.table identify before sequenced) b worker)
    (done : (run out.physical _ inputs samples b worker solve object supply solve 0 rows stages).map Batch.final = some final)
    (covers : Root.Covers mc b (rows.map RootMoveStorage.Row.square)) (nonempty : rows ≠ [])
    (minimum : Nat) (reportable : minimum ≤ 5) (sort : PublicationMir.SliceSort) (nodes : Nat)
    (old : Option PublicationMir.Iteration) (fallback : Unit → PublicationMir.SearchResult) :
    ∃ result, OuterMir.finish minimum 5 sort final (Board.emptyCount b.val).toNat nodes old fallback = some result ∧
      PublicationMir.score result = some (PvsBoardTree.minimax mc fc b) ∧
      ∀ row ∈ final, row.score = Root.squareValue mc fc b row.square := by
  obtain ⟨batch,_,rfl⟩ := Option.map_eq_some_iff.mp done
  exact publicCacheLines cache identify before sequenced irrefl trans scores stores batch.lines
    covers nonempty minimum reportable sort nodes old fallback

end NeuralReversi.AspirationMir.NativeRowLines

