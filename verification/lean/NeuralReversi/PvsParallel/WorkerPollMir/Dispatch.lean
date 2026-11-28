module
public import NeuralReversi.PvsParallel.WorkerPollMir.Program

@[expose] public section

namespace NeuralReversi.PvsParallel.WorkerPollMir
open MoveList.ConcurrentTrace (Write State forWorker)
open SplitWorkerMir (search commit selected emptyExit)
open SplitWorkerMir.Iterator (LockedCall WorkerResult acquired retire worker Drained Feed)

set_option maxRecDepth 16000 in
set_option maxHeartbeats 3000000 in
theorem control_sites : ∀ pc, conforms phase pc (code pc)=true := by decide

def polls (marks : List (Mark w)) : List (Receipt w) :=
  marks.filterMap fun | .poll r => some r | _ => none

def locks (marks : List (Mark w)) : List Nat :=
  marks.filterMap fun | .lock n => some n | _ => none

def published : List (Mark w) → List Nat
  | .lock rank::.leave route::tail => if route=.stopped then published tail else rank::published tail
  | _::tail => published tail
  | [] => []

/-- Per-ticket live score/stop words and lock order are read by the existing
    extracted search and commit programs. This imposes no final departure. -/
def Fields (feed : Feed) (marks : List (Mark w)) : Prop :=
  ∀ left (op : Write w) ordinal exit right,
    marks=left++[.poll (.some op),.lock ordinal,.leave exit]++right →
    ∃ call body, acquired feed op=some call ∧ call.order=ordinal ∧
      search call.ticket.input=some body ∧ (commit call.ticket.input call.ticket.locked body.chosen).exit=exit

private theorem Fields.tail {head tail : List (Mark w)} (f : Fields feed (head++tail)) : Fields feed tail := by
  intro left op atLock exit right eq
  apply f (head++left) op atLock exit right
  simp [eq,List.append_assoc]

def Summary (feed : Feed) (marks : List (Mark w)) : Prop :=
  ∃ calls result, ((polls marks).filterMap Receipt.ticket).mapM (acquired feed)=some calls ∧
    retire calls=some result ∧ calls.map LockedCall.order=locks marks ∧ result.published.map Prod.fst=published marks ∧
    (result.completed.departure=.exhausted → ∃ i opening seen, Receipt.none i opening seen∈polls marks)

private theorem done_empty {marks : List (Mark w)} (h : play .done marks=some .done) : marks=[] := by
  cases marks with
  | nil => rfl
  | cons x xs => cases x <;> cases h

theorem summarize (marks : List (Mark w)) (ran : play .poll marks=some .done) (f : Fields feed marks) :
    Summary feed marks := by
  cases marks with
  | nil => cases ran
  | cons event tail =>
    cases event with
    | lock | leave => cases ran
    | poll next =>
      cases next with
      | none i opening seen =>
        have endOfRun:=done_empty (show play .done tail=some .done from ran)
        subst tail
        exact ⟨[],⟨⟨[],[],emptyExit⟩,[]⟩,rfl,rfl,rfl,rfl,fun _ => ⟨i,opening,seen,by simp [polls]⟩⟩
      | some op =>
        cases tail with
        | nil => cases ran
        | cons event rest =>
          cases event with
          | poll | leave => cases ran
          | lock ordinal =>
            cases rest with
            | nil => cases ran
            | cons event more =>
              cases event with
              | poll | lock => cases ran
              | leave route =>
                obtain ⟨call,body,claimed,order,searched,ending⟩:=f [] op ordinal route more rfl
                cases route with
                | cutoff =>
                  have atEnd:=done_empty (show play .done more=some .done from ran)
                  subst more
                  refine ⟨[call],⟨⟨[call.ticket.input.index],[selected call.ticket.input body],.cutoff⟩,[(call.order,selected call.ticket.input body)]⟩,?_,?_,?_,?_,?_⟩
                  · change [op].mapM (acquired feed)=some [call]
                    simp [List.mapM_cons,claimed]
                  · simp [retire,searched,ending]
                  · simp [locks,order]
                  · simp [published,order]
                  · simp
                | stopped =>
                  have atEnd:=done_empty (show play .done more=some .done from ran)
                  subst more
                  refine ⟨[call],⟨⟨[call.ticket.input.index],[],.cancelled⟩,[]⟩,?_,?_,?_,?_,?_⟩
                  · change [op].mapM (acquired feed)=some [call]
                    simp [List.mapM_cons,claimed]
                  · simp [retire,searched,ending]
                  · simp [locks,order]
                  · simp [published]
                  · simp
                | again =>
                  have next:=show play .poll more=some .done from ran
                  obtain ⟨cs,res,items,completed,stamps,reported,exhausted⟩:=summarize more next
                    (Fields.tail (head:=[.poll (.some op),.lock ordinal,.leave .again]) f)
                  refine ⟨call::cs,⟨⟨call.ticket.input.index::res.completed.tickets,selected call.ticket.input body::res.completed.reports,res.completed.departure⟩,(call.order,selected call.ticket.input body)::res.published⟩,?_,?_,?_,?_,?_⟩
                  · change (op::(polls more).filterMap Receipt.ticket).mapM (acquired feed)=_
                    simp [List.mapM_cons,claimed,items]
                  · simp [retire,searched,ending,completed]
                  · change call.order::cs.map LockedCall.order=ordinal::locks more
                    rw [order,stamps]
                  · change call.order::res.published.map Prod.fst=ordinal::published more
                    rw [order,reported]
                  · intro why
                    obtain ⟨i,o,v,here⟩:=exhausted why
                    exact ⟨i,o,v,List.mem_cons_of_mem _ here⟩
termination_by marks.length

theorem dispatch {s : State w} (trace : Trace len start s receipts)
    (run : Returned code 0 marks) (matched : polls marks=receipts.filter (fun r => r.worker==i))
    (released : pendingTicket (s.localAt i)=[]) (fields : Fields feed marks) :
    ∃ result, worker feed s.writes i=some result ∧
      (∃ calls, ((forWorker s.writes i).mapM (acquired feed))=some calls ∧ calls.map LockedCall.order=locks marks) ∧ result.published.map Prod.fst=published marks ∧
      (result.completed.departure=.exhausted → Nonempty (Drained len start s i)) := by
  have language : play .poll marks=some .done := run.shape phase control_sites
  obtain ⟨cs,result,items,done,stamps,reported,exhausted⟩:=summarize marks language fields
  have ops:=trace.account i
  rw [released,List.append_nil] at ops
  have words : forWorker s.writes i=(polls marks).filterMap Receipt.ticket := by rw [matched];exact ops
  rw [←words] at items
  refine ⟨result,by simp [worker,items,done],⟨cs,items,stamps⟩,reported,?_⟩
  intro terminal
  obtain ⟨worker,opening,seen,here⟩:=exhausted terminal
  rw [matched,List.mem_filter] at here
  have eq:worker=i:=beq_iff_eq.mp here.2
  subst worker
  exact trace.drained here.1

end NeuralReversi.PvsParallel.WorkerPollMir
