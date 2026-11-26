import NeuralReversi.PvsParallel.IteratorFrames.EntryRegisters
import NeuralReversi.PvsParallel.SplitSchedulerMir.WorkerFields

namespace NeuralReversi.PvsParallel.IteratorFrames.ChildSitesRegression

open NativeFrames (Record Args)
open NativeFrames.EntryStream (Entry Sample capture)
open EndgameSearch.CacheHistory (Pool)
open Threading.AtomicWord (Event)

/- Position and both child return words from a nine-empty Rust split.
   The helper's scout returned -512 and its wide search returned -12288. -/
private def parent : Board.Board := ⟨17361623510135930879,3408925244719104⟩
private def moves := MoveList.Generation.records ⟨false,false,false⟩ parent (Board.getMoves ⟨false,false⟩ parent)
private def pool : Pool := ⟨2,64,by decide,fun _ => 0,0,⟨100,101,102⟩,⟨110,111,112⟩⟩
private def worker : EndgameSearch.CacheHistory.NativeTrace.Worker pool := ⟨1,.worker,by decide⟩
private def task : NativeWorkerFeed.Task := ⟨9,true,false,false,32767⟩
private def op : MoveList.ConcurrentTrace.Write 64 := ⟨1,3,1,2⟩
private def t (i : Nat) : Event := ⟨1,i⟩
private def obs : Observation :=
  ⟨1,3,2,some 20,5,10,t 8,t 13,t 4,t 9,t 14,t 15,t 16,task,-32767,-32767,⟨false,-32767,-32767⟩⟩

private def stream (wrongSite discard : Bool) : List Record :=
  let enter := fun step site node => Record.localCall 1 (.entry step site node worker.root none)
  [enter 0 0 .active,enter 1 0 .dispatch,enter 2 0 .splitSearch,enter 5 0 .pvs,
   .localCall 1 (.scored 7 ⟨-512⟩),.localCall 1 (.leave 8 5 .pvs none (some ⟨-512⟩)),
   enter 10 (if wrongSite then 0 else 1) .pvs] ++
    (if discard then [.localCall 1 (.discarded 13 10 ⟨-12288⟩)] else
      [.localCall 1 (.scored 12 ⟨-12288⟩),.localCall 1 (.leave 13 10 .pvs none (some ⟨-12288⟩))])

private def sampled (o : Observation) (rs : List Record) := entries parent moves [o] rs [op]
private def values (es : List Entry) := es.map fun ((worker,opening),a) =>
  (worker,opening,a.board.player.toNat,a.board.opponent.toNat,a.depth.toNat,a.pv,a.cut,a.alpha.raw.toInt,a.beta.raw.toInt)

#guard values (sampled obs (stream false false)) ==
  [(1,5,3408916654784512,17361624618237493247,8,false,true,32766,32767),
   (1,10,3408916654784512,17361624618237493247,8,true,false,-32767,32767)]

/- The callee's sampled registers are supplied independently of ticketEntries. -/
private def sample (r : Record) : Sample :=
  let args (pv cut : Bool) (alpha beta : BitVec 32) : Args :=
    ⟨⟨3408916654784512,17361624618237493247⟩,8,⟨alpha⟩,⟨beta⟩,true,Cutoffs.exactSelectivity,pv,cut,none⟩
  ⟨r,match r with
    | .localCall _ (.entry 5 _ .pvs ..) => some (args false true 32766 32767)
    | .localCall _ (.entry 10 _ .pvs ..) => some (args true false (-32767) 32767)
    | _ => none⟩

private def leaves (rs : List Record) : Option (List (Nat × Option Nat × Int)) := do
  let cap ← capture pool (rs.map sample)
  return cap.rows.map fun (r : NativeFrames.Row) => (r.opening,r.frame.returnStep,r.frame.score)

#guard values ((stream false false).filterMap (fun r => (sample r).entry)) == values (sampled obs (stream false false))
#guard leaves (stream false false) == some [(10,some 13,-12288),(5,some 8,-512)]
#guard (sampled obs ((stream false false).take 5)).isEmpty
#guard (sampled obs (stream true false)).isEmpty
#guard (sampled {obs with worker := 0} (stream false false)).isEmpty
#guard (sampled {obs with wideOpening := 5} (stream false false)).isEmpty
#guard (sampled obs (stream false true)).isEmpty
#guard (sampled {obs with locked := {obs.locked with stopped := true}} (stream false true)).length == 2

/- Reuse the captured child words to select the stamped field section for this
   ticket. The wide return, spinlock publication, and caller must all match. -/
private def fields : SplitSchedulerMir.WorkerFields.Dispatch := ⟨1,2,[
  ⟨3,[.load false,.returned ⟨1,8,false,32766,32767,true,-512⟩,
      .load true,.returned ⟨1,8,true,-32767,32767,false,-12288⟩,.lock],false,
    [.readBest (-32767),.writeBest 12288,.readAlpha (-32767),.writeMove 40,.writeAlpha 12288]⟩],
  [4,8,9,13,14,15,16,17,18,19,20,21],some 24⟩
private def reportLock := {obs with published := t 21}
#guard SplitSchedulerMir.WorkerFields.matchesTicket [fields] [reportLock] (stream false false) op
#guard !(SplitSchedulerMir.WorkerFields.matchesTicket [fields] [obs] (stream false false) op)
#guard !(SplitSchedulerMir.WorkerFields.matchesTicket [fields] [reportLock] (stream true false) op)
#guard !(SplitSchedulerMir.WorkerFields.matchesTicket [fields] [reportLock] (stream false true) op)

/- Two helper threads may have the same local timestamps. A repeated timestamp
   on a single worker fails the frame clock check. -/
private def twoWorkers : Pool := {pool with size := 3}
private def onWorker (w : EndgameSearch.CacheHistory.NativeTrace.Worker twoWorkers) : Record → Record
  | .localCall _ (.entry opening site node _ args) => .localCall w.ordinal (.entry opening site node w.root args)
  | .localCall _ ev => .localCall w.ordinal ev
  | .table e => .table e
private def pairs (other : EndgameSearch.CacheHistory.NativeTrace.Worker twoWorkers) := (stream false false).flatMap fun r => [sample r,sample (onWorker other r)]

#guard ((capture twoWorkers (pairs ⟨2,.worker,by decide⟩)).map fun cap => cap.rows.length) == some 4
#guard (capture twoWorkers (pairs ⟨1,.worker,by decide⟩)).isNone

end NeuralReversi.PvsParallel.IteratorFrames.ChildSitesRegression
