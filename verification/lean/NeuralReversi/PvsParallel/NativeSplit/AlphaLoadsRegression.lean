module
public import NeuralReversi.PvsParallel.NativeSplit.AlphaLoads
public import NeuralReversi.PvsParallel.SplitSchedulerMir.Startup
meta import NeuralReversi.PvsParallel.NativeSplit.AlphaLoads
meta import NeuralReversi.Threading.AtomicWord
meta import NeuralReversi.PvsParallel.SplitSchedulerMir.Startup

@[expose] public section

namespace NeuralReversi.PvsParallel.NativeSplit.AlphaLoadsRegression

open Threading
open AtomicWord (Event Read Write)
open SplitSchedulerMir.Startup (ready)
open AlphaLoads (operand operands)

/- Actual assignment and entry stamps from an initial booking and a late join.
   Each list records consecutive guards of one physical mutex. -/
private def state : List SpinLock.Operation :=
  [.acquire ⟨2,10⟩, .release ⟨2,13⟩, .acquire ⟨4,31⟩, .release ⟨4,33⟩]
private def slot : List SpinLock.Operation :=
  [.acquire ⟨3,8692⟩, .release ⟨3,8695⟩, .acquire ⟨6,8720⟩, .release ⟨6,8725⟩]
#guard ready [] [] ⟨2,9⟩ ⟨2,28⟩
#guard ready [] [(4,state)] ⟨2,9⟩ ⟨4,36⟩
#guard !(ready [] [(2,state)] ⟨2,9⟩ ⟨4,36⟩)
#guard !(ready [] [(4,state)] ⟨2,13⟩ ⟨4,36⟩)
#guard ready slot [] ⟨3,8694⟩ ⟨6,8732⟩
#guard !(ready slot [] ⟨3,8694⟩ ⟨6,8720⟩)
#guard !(ready (slot.eraseIdx 1) [] ⟨3,8694⟩ ⟨6,8732⟩)

private def e (step : Nat) : Event := ⟨0,step⟩
private def words : List Write :=
  [⟨e 1,-32767,false,false⟩, ⟨e 2,-32767,false,false⟩, ⟨e 6,12288,false,false⟩]
private def first : Read words := ⟨e 5,⟨1,by decide⟩,false⟩
private def second : Read words := ⟨e 7,⟨2,by decide⟩,false⟩
private def observation : IteratorFrames.Observation :=
  ⟨0,4,3,none,0,0,e 6,e 8,e 5,e 7,e 9,e 10,e 16,
    ⟨9,true,false,false,32767⟩,-32767,12288,⟨false,12288,12288⟩⟩
private def input : SplitWorkerMir.Input := ⟨1,8,true,false,false,32767,-32767,12288,-512,-12288⟩

#guard operand [first,second] (e 5) (-32767)
#guard !(operand [first] ⟨1,5⟩ (-32767))
#guard !(operand [{first with acquire := true}] (e 5) (-32767))
#guard !(operand [first,{first with source := ⟨0,by decide⟩}] (e 5) (-32767))
#guard !(operand [{first with source := second.source}] (e 5) (-32767))
#guard operands [first,second] observation input
#guard !(operands [first] observation input)
#guard operands [first] observation {input with pv := false}
#guard !(operands [first,second] observation {input with wide := -2147483648})
#guard !(operands [first,second] observation {input with alpha := 2147483647})

end NeuralReversi.PvsParallel.NativeSplit.AlphaLoadsRegression
