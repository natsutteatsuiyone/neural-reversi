import NeuralReversi.PvsParallel.SplitWorkerMir.Dispatch

namespace NeuralReversi.PvsParallel.SplitSchedulerMir

open Threading
open AtomicWord (Write)

inductive Phase
  | init | idle | finalize | best | picked | returned | clear | mark | seed | seedMove | seedAlpha | assign | dispatch | release
  deriving DecidableEq

inductive Mode
  | NonPV | PV | Root
  deriving DecidableEq

inductive Op (n : Nat)
  | any (targets : List (Fin n))
  | shutdown (running closed : Fin n)
  | mask (busy empty : Fin n)
  | done
  deriving Inhabited

inductive Guard (writes : List Write)
  | shutdown (closed : Bool)
  | mask (source : Fin writes.length)

inductive Returned (code : Fin n → Op n) (writes : List Write) : Fin n → List (Guard writes) → Prop
  | done : code pc = .done → Returned code writes pc []
  | next : code pc = .any targets → new ∈ targets → Returned code writes new more →
      Returned code writes pc more
  | shutdown : code pc = .shutdown running closed → Returned code writes (if b then closed else running) more →
      Returned code writes pc (.shutdown b :: more)
  | mask (src : Fin writes.length) : code pc = .mask busy empty →
      Returned code writes (if (writes.get src).value = 0 then empty else busy) more →
      Returned code writes pc (.mask src :: more)

def preserves (waiting : Fin n → Bool) : Op n → Bool
  | .done => false
  | .any targets => targets.all waiting
  | .shutdown running _ | .mask running _ => waiting running

end NeuralReversi.PvsParallel.SplitSchedulerMir
