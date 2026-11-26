import NeuralReversi.PvsParallel.WorkerPollMir.Atomic
import NeuralReversi.PvsParallel.SplitWorkerMir.Iterator

namespace NeuralReversi.PvsParallel.WorkerPollMir

open SplitWorkerMir (Exit)

inductive Mark (w : Nat)
  | poll (result : Receipt w)
  | lock (ordinal : Nat)
  | leave (exit : SplitWorkerMir.Exit)

inductive Phase | poll | issued | locked | done
  deriving DecidableEq

inductive Op (n : Nat)
  | jump (dest : List (Fin n))
  | next (issued empty : Fin n)
  | lock (dest : Fin n)
  | stop (running stopped : Fin n)
  | cutoff (again cutoff : Fin n)
  | done

def delta : Phase → Mark w → Option Phase
  | .poll,.poll (.some _) => some .issued
  | .poll,.poll (.none ..) => some .done
  | .issued,.lock _ => some .locked
  | .locked,.leave .again => some .poll
  | .locked,.leave .cutoff | .locked,.leave .stopped => some .done
  | _,_ => none

def play (state : Phase) : List (Mark w) → Option Phase
  | [] => some state
  | mark::rest => (delta state mark).bind (fun next => play next rest)

def conforms (phase : Fin n → Phase) (pc : Fin n) : Op n → Bool
  | .jump dest => dest.all (fun next => decide (phase next=phase pc))
  | .next hit empty => decide (phase pc=.poll ∧ phase hit=.issued ∧ phase empty=.done)
  | .lock next => decide (phase pc=.issued ∧ phase next=.locked)
  | .stop running stopped => decide (phase pc=.locked ∧ phase running=.locked ∧ phase stopped=.done)
  | .cutoff again cut => decide (phase pc=.locked ∧ phase again=.poll ∧ phase cut=.done)
  | .done => decide (phase pc=.done)

inductive Returned (code : Fin n → Op n) : Fin n → List (Mark w) → Prop
  | done : code pc=.done → Returned code pc []
  | jump : code pc=.jump dest → next ∈ dest → Returned code next rest → Returned code pc rest
  | issued : code pc=.next hit empty → Returned code hit rest → Returned code pc (.poll (.some op)::rest)
  | empty : code pc=.next hit empty → Returned code empty rest → Returned code pc (.poll (.none i opening seen)::rest)
  | lock : code pc=.lock next → Returned code next rest → Returned code pc (.lock atLock::rest)
  | running : code pc=.stop running stopped → Returned code running rest → Returned code pc rest
  | stopped : code pc=.stop running stopped → Returned code stopped rest → Returned code pc (.leave .stopped::rest)
  | again : code pc=.cutoff again cut → Returned code again rest → Returned code pc (.leave .again::rest)
  | cutoff : code pc=.cutoff again cut → Returned code cut rest → Returned code pc (.leave .cutoff::rest)

theorem Returned.shape (phase : Fin n → Phase) (valid : ∀ pc, conforms phase pc (code pc)=true)
    (h : Returned code pc log) : play (phase pc) log=some .done := by
  have rule {i op} (eq : code i=op) : conforms phase i op=true := eq ▸ valid i
  induction h with
  | done eq =>
    have ok:=rule eq
    exact congrArg some (of_decide_eq_true ok)
  | jump eq mem _ ih =>
    have ok:=rule eq
    have same:=of_decide_eq_true (List.all_eq_true.mp ok _ mem)
    rwa [←same]
  | running eq _ ih =>
    have ok:=rule eq
    have ⟨a,b,_⟩:=of_decide_eq_true ok
    rwa [a,←b]
  | issued eq _ ih | empty eq _ ih | lock eq _ ih | stopped eq _ ih | again eq _ ih | cutoff eq _ ih =>
    have ok:=rule eq
    simp only [conforms,decide_eq_true_eq] at ok
    rcases ok with ⟨a,b⟩
    simp_all [play,delta]

end NeuralReversi.PvsParallel.WorkerPollMir
