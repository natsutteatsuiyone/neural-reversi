import NeuralReversi.PvsParallel.SplitSchedulerMir.WorkerReturn

namespace NeuralReversi.PvsParallel.SplitSchedulerMir.WorkerFieldsRegression

open WorkerFields
open SplitWorkerMir (Query Step)
open IteratorFrames (Observation)
open Threading.AtomicWord (Event)

/- Actual worker-local stamps from the nine-empty split. The first ticket did
   a wide re-search and wrote all three fields. The next five left them alone. -/
private def wide : Section := ⟨4, [.load false, .returned ⟨1,8,false,32766,32767,true,-512⟩,
  .load true, .returned ⟨1,8,true,-32767,32767,false,-12288⟩, .lock], false,
  [.readBest (-32767), .writeBest 12288, .readAlpha (-32767), .writeMove 40, .writeAlpha 12288]⟩
private def scout (start idx : Nat) (out : BitVec 32) : Section :=
  ⟨start,[.load false, .returned ⟨idx,8,false,-12289,-12288,true,out⟩,.lock],false,[.readBest 12288]⟩
private def run : Dispatch := ⟨0,3, [wide, scout 17 2 (-11264), scout 24 3 (-11264),
    scout 31 4 (-12288), scout 38 5 (-12288), scout 45 6 (-12288)],
  List.range' 5 12 ++ ([18,25,32,39,46].flatMap fun start => List.range' start 6), some 54⟩
private def t (n : Nat) : Event := ⟨0,n⟩
private def lock : Observation := ⟨0,4,3,some 17,0,0,t 6,t 8,t 5,t 7,t 9,t 10,t 16,
  ⟨9,true,false,false,32767⟩,-32767,-32767,⟨false,-32767,-32767⟩⟩

private def origin := locate [run] lock
private def stores (ops : List SplitWorkerMir.FieldStores.Physical) := ops.filterMap fun op => match op with
  | .writeAlpha w | .writeBest w | .writeMove w => some (w.event.step, w.value.toNat)
  | _ => none
#guard run.valid && run.reset == t 54
#guard origin.any fun a => a.aligned lock wide.body && a.unlock == t 16
#guard stores (traffic [run] [lock] 0) == [(12,12288),(14,40),(15,12288)]
#guard !(origin.any fun a => a.aligned {lock with published := t 12} wide.body)
#guard !(origin.any fun a => a.aligned lock (wide.body.drop 2))
#guard (locate [run] {lock with worker := 1}).isNone
#guard (locate [run] {lock with dispatch := 4}).isNone
#guard (locate [{run with stamps := run.stamps.reverse}] lock).isNone
#guard (locate [{run with stamps := run.stamps.tail}] lock).isNone

/- A stopped ticket from a four-thread exact solve of the sixteen-empty board.
   It acquired the lock, polled true, and dropped the guard without field accesses. -/
private def stopRow : Section := ⟨725,[.load false,.returned ⟨3,12,false,2048,2049,false,0⟩,.lock],true,[]⟩
private def stopped : Dispatch := ⟨1,724,[stopRow],List.range' 726 5,some 733⟩
private def returns (w start : Nat) := ([.leave (start+7) start .splitSearch none none,
  .leave (start+8) (start-1) .dispatch none none, .leave (start+10) (start-2) .active none none] :
  List NativeFrames.Event).map (NativeFrames.Record.localCall w)
#guard stopped.valid && stopped.retired (returns 1 724) && !stopRow.needsReset
#guard !(stopped.retired (returns 0 724))
#guard !(stopped.retired (returns 1 723))
#guard !({stopped with maskReset := some 732}.retired (returns 1 724))
#guard !({stopped with maskReset := some 734}.retired (returns 1 724))
#guard !({stopped with maskReset := none}.retired (returns 1 724))
#guard stopped.retired ((returns 1 724).take 2)
private def lateUnlock : Dispatch := {stopped with stamps := [726,727,728,729,732]}
#guard lateUnlock.valid && !(lateUnlock.retired (returns 1 724))

end NeuralReversi.PvsParallel.SplitSchedulerMir.WorkerFieldsRegression
