module
public import NeuralReversi.EndgameSearch.CacheHistory.Payloads.Windows

@[expose] public section

namespace NeuralReversi.EndgameSearch.CacheHistory.Payloads

open CacheOwner (Node Edge)
open NativeTrace (Worker Cursor Activation Event Raw Ordered)

variable {p : Pool} {w : Worker p}

/-- All cache stores and shallow-cache operations can only acquire the private arrays
    in the boundary frame. Deep read-only probes may also run under `ecCutoff`. -/
private def boundaryOwned (n : Node) : Bool :=
  match n with
  | .deep | .deepChild | .shallow | .shallowMoves | .shallowChild
  | .deepStore | .shallowSlot | .shallowProbe | .shallowStore => true
  | _ => false

private theorem caller_loan (e : Edge) (member : e ∈ CacheOwner.transfers) (callee : boundaryOwned e.callee) :
    e.caller = .boundary ∨ boundaryOwned e.caller = true := by
  have all : CacheOwner.transfers.all (fun edge => !boundaryOwned edge.callee || edge.caller == .boundary || boundaryOwned edge.caller) := by decide
  simpa [callee] using List.all_eq_true.mp all e member

private theorem entered_from (s out : Cursor w) (ok : s.enter tick site node receiver args = some out) :
    ∃ e ∈ CacheOwner.transfers, e.caller = s.parent.node ∧ e.callee = node := by
  unfold Cursor.enter at ok
  dsimp only at ok
  split at ok <;> try contradiction
  rename_i edge found
  split at ok <;> try contradiction
  split at ok <;> try contradiction
  exact ⟨edge,List.mem_of_find?_eq_some found,by assumption,by assumption⟩

private def Loans : List (Activation w) → Prop
  | [] => True
  | a::rest =>
    (boundaryOwned a.node → ∃ f ∈ rest, f.node = .boundary ∧ f.first ≤ a.first) ∧ Loans rest

private theorem push_loan (s : Cursor w) (prior : Loans s.stack)
    (old : ∀ f ∈ s.stack, f.first < s.clock) (time : s.clock ≤ tick)
    (site : Edge) (member : site ∈ CacheOwner.transfers) (caller : site.caller = s.parent.node)
    (top : Activation w) (first : top.first = tick) (callee : site.callee = top.node) :
    Loans (top :: s.stack) := by
  refine ⟨fun loan => ?_,prior⟩
  have fromParent := caller_loan site member (callee ▸ loan)
  rw [caller] at fromParent
  cases stack : s.stack with
  | nil =>
    have impossible (k : CacheOwner.EntryPoint) : CacheOwner.entry k ≠ .boundary ∧ boundaryOwned (CacheOwner.entry k) = false := by
      cases k <;> decide
    simp [Cursor.parent,stack,(impossible w.kind).1,(impossible w.kind).2] at fromParent
  | cons parent rest =>
    have current : s.parent = parent := by simp [Cursor.parent,stack]
    rw [current] at fromParent
    have clock := old parent (stack ▸ List.mem_cons_self)
    rcases fromParent with boundary | inherited
    · exact ⟨parent,List.mem_cons_self,boundary,by omega⟩
    · obtain ⟨f,mem,tag,lo⟩ := (show Loans (parent::rest) from stack ▸ prior).1 inherited
      exact ⟨f,List.mem_cons_of_mem _ mem,tag,by omega⟩

private def Enclosed (s : Cursor w) (opening returned : Nat) : Prop :=
  (∃ v ∈ s.finished, v.first ≤ opening ∧ returned ≤ v.last) ∨
  (∃ f ∈ s.stack, f.node = .boundary ∧ f.first ≤ opening ∧ returned < s.clock)

private structure History (s : Cursor w) (events : List Raw) : Prop where
  ordered : Ordered s
  clocks : ∀ f ∈ s.stack, f.first < s.clock
  loans : Loans s.stack
  seen : ∀ {t o node score}, Raw.leave t o node score ∈ events → (target node).isSome →
    o ≤ t ∧ (boundaryOwned node = true → Enclosed s o t)

private theorem initial : History (.initial w) [] := by
  exact ⟨Ordered.initial,by simp [Cursor.initial],True.intro,by simp⟩

private theorem append_enter (s out : Cursor w) (h : History s events)
    (ok : s.advance (.enter t site node receiver args) = some out) :
    History out (events ++ [.enter t site node receiver args]) := by
  obtain ⟨time,tick⟩ := NativeTrace.advance_clock s _ ok
  change s.clock ≤ t at time
  have entered : s.enter t site node receiver args = some out := by simpa [Cursor.advance,Raw.step,time] using ok
  obtain ⟨f,stack,finish,first,kind,_,_,_,_⟩ := NativeTrace.checked_entry s out entered
  obtain ⟨e,member,caller,callee⟩ := entered_from s out entered
  change out.clock = t+1 at tick
  refine ⟨Ordered.advance s h.ordered _ ok,?_,stack ▸ push_loan s h.loans h.clocks time e member caller f first (callee.trans kind.symm),?_⟩
  · intro a mem
    rcases List.mem_cons.mp (stack ▸ mem) with rfl | old
    · omega
    · have := h.clocks a old; omega
  · intro end_ start callee_ score mem primitive
    have prev : Raw.leave end_ start callee_ score ∈ events := by simpa using mem
    obtain ⟨ordered,seen⟩ := h.seen prev primitive
    refine ⟨ordered,fun owned => ?_⟩
    rcases seen owned with past | ⟨b,hb,tag,lo,hi⟩
    · exact .inl (finish ▸ past)
    · exact .inr ⟨b,stack ▸ List.mem_cons_of_mem _ hb,tag,lo,by omega⟩

private theorem pop_frame (s out : Cursor w) (ordered : Ordered s) (top : Activation w) (stack : s.stack = top::rest)
    (ok : s.leave t o node score = some out) :
    top.first = o ∧ top.node = node ∧ out.stack = rest ∧ (∀ v ∈ s.finished, v ∈ out.finished) ∧
      (top.node = .boundary → ∃ v ∈ out.finished, v.first = o ∧ v.last = t) := by
  have matched : ¬(top.first ≠ o ∨ top.node ≠ node) := fun mismatch => by
    rw [NativeTrace.wrong_return s top stack mismatch] at ok
    contradiction
  obtain ⟨first,kind⟩ : top.first = o ∧ top.node = node := by simpa using matched
  have tag := ordered.tagged top (stack ▸ List.mem_cons_self)
  cases args : top.operands with
  | none =>
    obtain ⟨done_,pop⟩ := NativeTrace.propagation s out top stack args ok
    exact ⟨first,kind,pop,fun v hv => done_ ▸ hv,fun atBoundary => by simp [args,atBoundary] at tag⟩
  | some arg =>
    obtain ⟨v,done_,pop,_,opened,closed,_,_,_,_⟩ := NativeTrace.matched_boundary s out top stack args ok
    exact ⟨first,kind,pop,fun a ha => done_ ▸ List.mem_cons_of_mem _ ha,fun _ => ⟨v,done_ ▸ List.mem_cons_self,opened,closed⟩⟩

private theorem append_leave (s out : Cursor w) (h : History s events)
    (ok : s.advance (.leave t o node score) = some out) : History out (events ++ [.leave t o node score]) := by
  obtain ⟨time,tick⟩ := NativeTrace.advance_clock s _ ok
  change s.clock ≤ t at time
  change out.clock = t+1 at tick
  have left : s.leave t o node score = some out := by simpa [Cursor.advance,Raw.step,time] using ok
  cases stack : s.stack with
  | nil => simp [Cursor.leave,stack] at left
  | cons top rest =>
    obtain ⟨first,kind,pop,past,closed⟩ := pop_frame s out h.ordered top stack left
    have loan : Loans (top::rest) := stack ▸ h.loans
    refine ⟨Ordered.advance s h.ordered _ ok,?_,pop ▸ loan.2,?_⟩
    · intro a mem
      have := h.clocks a (stack ▸ List.mem_cons_of_mem _ (pop ▸ mem))
      omega
    · intro end_ start callee tag mem primitive
      rcases List.mem_append.mp mem with earlier | current
      · obtain ⟨before,cert⟩ := h.seen earlier primitive
        refine ⟨before,fun owned => ?_⟩
        rcases cert owned with ⟨v,done_,lo,hi⟩ | ⟨f,live,boundary,lo,hi⟩
        · exact .inl ⟨v,past v done_,lo,hi⟩
        · rcases List.mem_cons.mp (stack ▸ live) with rfl | tail
          · obtain ⟨v,hv,atOpen,atClose⟩ := closed boundary
            exact .inl ⟨v,hv,by omega,by omega⟩
          · exact .inr ⟨f,pop ▸ tail,boundary,lo,by omega⟩
      · cases List.mem_singleton.mp current
        have was := h.clocks top (stack ▸ List.mem_cons_self)
        refine ⟨by omega,fun owned => ?_⟩
        obtain ⟨f,hf,boundary,lo⟩ := loan.1 (kind ▸ owned)
        exact .inr ⟨f,pop ▸ hf,boundary,by omega,by omega⟩

private theorem history_run (w : Worker p) (stream : List (Event S)) (s : Cursor w) (ok : NativeTrace.run w stream = some s) :
    History s (NativeTrace.workerEvents w.ordinal stream) := by
  have loop (rest : List Raw) (past : List Raw) (at_ : Cursor w) (prior : History at_ past)
      (ran : rest.foldlM Cursor.advance at_ = some s) : History s (past ++ rest) := by
    induction rest generalizing at_ past with
    | nil =>
      cases Option.some.inj ran
      simpa using prior
    | cons ev tail ih =>
      rw [List.foldlM_cons] at ran
      obtain ⟨next,act,more⟩ := Option.bind_eq_some_iff.mp ran
      have full : History next (past ++ [ev]) := by
        cases ev with
        | enter => exact append_enter at_ next prior act
        | leave => exact append_leave at_ next prior act
      simpa using ih (past ++ [ev]) next full more
  exact loop _ [] (.initial w) initial ok

private theorem find_boundary (stack : List (Activation w))
    (unique : stack.Pairwise fun a b => a.node = .boundary → b.node ≠ .boundary)
    (f : Activation w) (mem : f ∈ stack) (tag : f.node = .boundary) :
    stack.find? (fun a => a.node == .boundary) = some f := by
  induction stack with
  | nil => contradiction
  | cons top rest ih =>
    obtain ⟨head,tail⟩ := List.pairwise_cons.mp unique
    rcases List.mem_cons.mp mem with rfl | below
    · simp [tag]
    · have ne : top.node ≠ .boundary := fun h => head f below h tag
      simpa [ne] using ih tail below

/-- Every returned cache store or shallow-cache primitive, including one in an
    unfinished outer call, has a concrete enclosing boundary on its own Rust worker.
    All returned primitives obey opening <= return time. -/
theorem worker_site (w : Worker p) (stream : List (Event S)) (s : Cursor w) (parsed : NativeTrace.run w stream = some s)
    (returned : Raw.leave t o node score ∈ NativeTrace.workerEvents w.ordinal stream) (primitive : (target node).isSome) :
    o ≤ t ∧ ((node ≠ .deepSlot ∧ node ≠ .deepProbe) →
      ((∃ v ∈ s.completed, v.worker = w.ordinal ∧ v.first ≤ o ∧ t ≤ v.last) ∨
        ∃ f, s.stack.find? (fun a => a.node == .boundary) = some f ∧ f.first ≤ o ∧ t ≤ s.clock)) := by
  have h := history_run w stream s parsed
  obtain ⟨order,covered_⟩ := h.seen returned primitive
  refine ⟨order,fun notDeepRead => ?_⟩
  have owned : boundaryOwned node = true := by
    cases node <;> simp [target, boundaryOwned] at primitive notDeepRead ⊢
  rcases covered_ owned with ⟨v,mem,lo,hi⟩ | ⟨f,mem,tag,lo,hi⟩
  · exact .inl ⟨v,List.mem_reverse.mpr mem,(h.ordered.past v mem).1,lo,hi⟩
  · exact .inr ⟨f,find_boundary _ h.ordered.unique f mem tag,lo,Nat.le_of_lt hi⟩

end NeuralReversi.EndgameSearch.CacheHistory.Payloads
