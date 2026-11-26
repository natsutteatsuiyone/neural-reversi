import NeuralReversi.PvsParallel.IteratorFrames
import NeuralReversi.PvsParallel.SplitWorkerMir.OwnerFields

namespace NeuralReversi.PvsParallel.SplitSchedulerMir.WorkerFields

open Threading.AtomicWord (Event Write SequencedBefore)
open SplitWorkerMir (Step lockedSteps)
open SplitWorkerMir.FieldStores (Value Physical)
open IteratorFrames (Observation)

/-- One successful iterator ticket, through dropping its spinlock guard. The
    field operands are in the order selected by the compiled commit branch. -/
structure Section where
  opening : Nat
  body : List Step
  stopped : Bool
  fields : List Value

def Section.needsReset (s : Section) := !s.stopped || !s.fields.isEmpty

def Section.steps (s : Section) := s.body ++ lockedSteps
def Section.gaps (s : Section) := List.replicate (s.body.length + 1) 0 ++ [s.fields.length]

private theorem gaps_length (ss : List Section) :
    (ss.flatMap Section.gaps).length = (ss.flatMap Section.steps).length := by
  induction ss with
  | nil => rfl
  | cons s ss ih => simp [Section.steps, Section.gaps, lockedSteps, ih, Nat.add_assoc]; omega

/-- Number of the full instruction at a projected index. A gap counts field
    accesses immediately before that instruction. Beyond the list, gaps are zero. -/
private def offset : List Nat → Nat → Nat
  | [], i => i
  | g :: _, 0 => g
  | g :: gs, i+1 => g + 1 + offset gs i

private theorem offset_strict (gaps : List Nat) (i j : Nat) (lt : i < j) :
    offset gaps i < offset gaps j := by
  induction gaps generalizing i j with
  | nil => exact lt
  | cons g gs ih =>
    cases j with
    | zero => omega
    | succ j =>
      cases i with
      | zero => simp only [offset]; omega
      | succ i => exact Nat.add_lt_add_left (ih i j (by omega)) _

private theorem offset_append (a b : List Nat) (i : Nat) :
    offset (a ++ b) (a.length + i) = a.sum + a.length + offset b i := by
  induction a with
  | nil => simp
  | cons x xs ih =>
    simp only [List.cons_append, List.length_cons, List.sum_cons]
    rw [show xs.length + 1 + i = (xs.length + i) + 1 by omega, offset, ih]
    omega

private def withGaps (clock : Clock) (gaps : List Nat) : Clock where
  worker := clock.worker
  tick i := clock.tick (offset gaps i)
  advances i j h := clock.advances _ _ (offset_strict gaps i j h)

private def gapsFrom : Nat → List Nat → List Nat
  | _, [] => []
  | lo, t :: rest => (t - lo) :: gapsFrom (t+1) rest

private def orderedFrom : Nat → List Nat → Bool
  | _, [] => true
  | lo, t :: rest => decide (lo ≤ t) && orderedFrom (t+1) rest

private theorem at_sample (ts : List Nat) (lo i : Nat) (ordered : orderedFrom lo ts)
    (seen : ts[i]? = some t) : lo + offset (gapsFrom lo ts) i = t := by
  induction ts generalizing lo i with
  | nil => simp at seen
  | cons x xs ih =>
    obtain ⟨lower, next⟩ := Bool.and_eq_true_iff.mp ordered
    have le : lo ≤ x := of_decide_eq_true lower
    cases i with
    | zero => cases seen; simp only [gapsFrom, offset]; omega
    | succ i =>
      have later := ih _ i next seen
      simp only [gapsFrom, offset]; omega

structure Dispatch where
  worker : Nat
  opening : Nat
  sections : List Section
  stamps : List Nat
  maskReset : Option Nat

def Dispatch.entry (d : Dispatch) : Event := ⟨d.worker,d.opening⟩

def Dispatch.steps (d : Dispatch) := d.sections.flatMap Section.steps

def Dispatch.count (d : Dispatch) := d.steps.length + (d.sections.flatMap Section.gaps).sum

def Dispatch.valid (d : Dispatch) :=
  d.stamps.length == d.count && orderedFrom (d.opening+1) (d.stamps ++ d.maskReset.toList)

def Dispatch.clock (d : Dispatch) : Clock :=
  withGaps ⟨d.worker, fun i => d.opening + 1 + i, fun _ _ lt => Nat.add_lt_add_left lt _⟩
    (gapsFrom (d.opening+1) (d.stamps ++ d.maskReset.toList))

theorem Dispatch.at_sample (d : Dispatch) (good : d.valid) (seen : (d.stamps ++ d.maskReset.toList)[i]? = some t) :
    d.clock.atStep i = ⟨d.worker, t⟩ := by
  apply congrArg (fun n => (⟨d.worker, n⟩ : Event))
  exact WorkerFields.at_sample _ _ _ (Bool.and_eq_true_iff.mp good).2 seen

def Dispatch.projected (d : Dispatch) := withGaps d.clock (d.sections.flatMap Section.gaps)

def Dispatch.reset (d : Dispatch) := d.projected.atStep (releaseIndex d.steps)

private theorem Dispatch.reset_at_count (d : Dispatch) : d.reset = d.clock.atStep d.count := by
  apply congrArg d.clock.atStep
  change offset (d.sections.flatMap Section.gaps) (releaseIndex d.steps) = d.count
  rw [releaseIndex_eq]
  have endAt := offset_append (d.sections.flatMap Section.gaps) [] 0
  simp only [List.append_nil, Nat.add_zero, offset] at endAt
  rw [gaps_length] at endAt
  exact endAt.trans (Nat.add_comm _ _)

theorem Dispatch.at_reset (d : Dispatch) (good : d.valid) (seen : d.maskReset = some t) :
    d.reset = ⟨d.worker, t⟩ := by
  have size := beq_iff_eq.mp (Bool.and_eq_true_iff.mp good).1
  have sample : (d.stamps ++ d.maskReset.toList)[d.count]? = some t := by rw [← size, seen]; simp
  exact d.reset_at_count.trans (d.at_sample good sample)

structure Address where
  dispatch : Dispatch
  index : Fin dispatch.sections.length

namespace Address

variable (a : Address)
abbrev row := a.dispatch.sections.get a.index
def pre := a.dispatch.sections.take a.index.val
def post := a.dispatch.sections.drop (a.index.val+1)
def prior := a.pre.flatMap Section.steps
def later := a.post.flatMap Section.steps

theorem split : a.dispatch.sections = a.pre ++ a.row :: a.post := by
  unfold pre post row
  simp only [List.get_eq_getElem]
  rw [List.getElem_cons_drop a.index.isLt, List.take_append_drop]

theorem steps_eq : a.dispatch.steps = a.prior ++ a.row.body ++ lockedSteps ++ a.later := by
  unfold Dispatch.steps
  conv => lhs; rw [a.split]
  simp [prior, later, Section.steps, List.append_assoc]

private def start := (a.pre.flatMap Section.gaps).sum + a.prior.length

private theorem at_unlock :
    a.dispatch.projected.atStep (a.prior.length + (a.row.body.length+1)) =
      a.dispatch.clock.atStep (a.start + a.row.body.length + 1 + a.row.fields.length) := by
  apply congrArg (fun n => a.dispatch.clock.atStep n)
  change offset (a.dispatch.sections.flatMap Section.gaps) _ = _
  conv => lhs; arg 1; rw [a.split, List.flatMap_append, List.flatMap_cons]
  unfold prior
  rw [←gaps_length a.pre, offset_append]
  simp only [Section.gaps, List.append_assoc]
  have zeros (n : Nat) (rest : List Nat) :
      offset (List.replicate n 0 ++ rest) n = n + offset rest 0 := by
    induction n with
    | zero => simp
    | succ n ih => simp [List.replicate_succ, offset, ih]; omega
  rw [zeros]
  simp only [start, prior, gaps_length, List.singleton_append, offset]; omega

private def physical (event : Event) : Value → Physical
  | .readAlpha val => .readAlpha val
  | .readBest val => .readBest val
  | .writeAlpha val => .writeAlpha ⟨event, val.zeroExtend 64, false, false⟩
  | .writeBest val => .writeBest ⟨event, val.zeroExtend 64, false, false⟩
  | .writeMove val => .writeMove ⟨event, val.zeroExtend 64, false, false⟩

def traffic := a.row.fields.zipIdx.map fun (v,i) =>
  physical (a.dispatch.clock.atStep (a.start + a.row.body.length + 1 + i)) v

def unlock := a.dispatch.projected.atStep (a.prior.length + (a.row.body.length+1))

private theorem store_index {op : Physical} (mem : op ∈ a.traffic) {w : Write}
    (store : op = .writeAlpha w ∨ op = .writeBest w ∨ op = .writeMove w) :
    ∃ i < a.row.fields.length, w.event = a.dispatch.clock.atStep (a.start + a.row.body.length + 1 + i) := by
  obtain ⟨⟨v,i⟩, issued, eq⟩ := List.mem_map.mp mem
  have bound : i < a.row.fields.length :=
    (List.getElem?_eq_some_iff.mp (List.mk_mem_zipIdx_iff_getElem?.mp issued)).1
  refine ⟨i, bound, ?_⟩
  subst op
  cases v <;> simp_all [physical] <;> subst w <;> rfl

theorem store_before_unlock {op : Physical} (mem : op ∈ a.traffic) {w : Write}
    (store : op = .writeAlpha w ∨ op = .writeBest w ∨ op = .writeMove w) :
    SequencedBefore w.event a.unlock := by
  obtain ⟨i, bound, same⟩ := a.store_index mem store
  rw [same, unlock, at_unlock]
  exact ⟨rfl, a.dispatch.clock.advances _ _ (by omega)⟩

theorem unlock_before_reset : SequencedBefore a.unlock a.dispatch.reset := by
  refine ⟨rfl, a.dispatch.projected.advances _ _ ?_⟩
  rw [releaseIndex_eq, a.steps_eq]
  simp [lockedSteps, Nat.add_assoc]

/-- Every protected write in a valid dispatch carries a timestamp from the captured
    instructions of that worker. This also applies to captures without a mask reset. -/
theorem recorded_store (good : a.dispatch.valid) {op : Physical} (mem : op ∈ a.traffic) {w : Write}
    (store : op = .writeAlpha w ∨ op = .writeBest w ∨ op = .writeMove w) :
    w.event.worker = a.dispatch.worker ∧ w.event.step ∈ a.dispatch.stamps := by
  obtain ⟨i, _, issued⟩ := a.store_index mem store
  let k := a.start + a.row.body.length + 1 + i
  have before := Nat.lt_trans (a.store_before_unlock mem store).2 a.unlock_before_reset.2
  rw [issued, a.dispatch.reset_at_count] at before
  change a.dispatch.clock.tick k < a.dispatch.clock.tick a.dispatch.count at before
  have limit : k < a.dispatch.count := by
    rcases Nat.lt_trichotomy k a.dispatch.count with h | h | h
    · exact h
    · rw [h] at before; omega
    · exact False.elim (Nat.lt_asymm before (a.dispatch.clock.advances _ _ h))
  have size := beq_iff_eq.mp (Bool.and_eq_true_iff.mp good).1
  have within : k < a.dispatch.stamps.length := size.symm ▸ limit
  have sample : (a.dispatch.stamps ++ a.dispatch.maskReset.toList)[k]? = some a.dispatch.stamps[k] := by
    rw [List.getElem?_append_left within, List.getElem?_eq_getElem within]
  rw [issued, a.dispatch.at_sample good sample]
  exact ⟨rfl, List.getElem_mem within⟩

def aligned (obs : Observation) (steps : List Step) : Bool :=
  (a.row.stopped == obs.locked.stopped && a.row.body == steps) && a.row.steps.zipIdx.all fun (step,n) =>
    obs.stamp step == a.dispatch.projected.atStep (a.prior.length+n)

theorem observed {obs : Observation} (good : a.aligned obs steps) :
    a.row.stopped = obs.locked.stopped ∧ a.row.body = steps ∧ ∀ n step, (steps ++ lockedSteps)[n]? = some step →
      obs.stamp step = a.dispatch.projected.atStep (a.prior.length+n) := by
  obtain ⟨body, stamps⟩ := Bool.and_eq_true_iff.mp good
  obtain ⟨stop, matched⟩ := Bool.and_eq_true_iff.mp body
  have same := beq_iff_eq.mp matched
  refine ⟨beq_iff_eq.mp stop, same, fun n step found => beq_iff_eq.mp (List.all_eq_true.mp stamps (step,n) ?_)⟩
  apply List.mk_mem_zipIdx_iff_getElem?.mpr
  change (a.row.body ++ lockedSteps)[n]? = some step
  rw [same]; exact found

theorem entry_before_load (searched : SplitWorkerMir.search input = some b) (matched : a.aligned obs b.steps)
    (wide : Bool) (used : wide = true → SplitWorkerMir.retry input = true) :
    SequencedBefore a.dispatch.entry (obs.stamp (.load wide)) := by
  let n := if wide then 2 else 0
  have position : (b.steps ++ lockedSteps)[n]? = some (.load wide) := by
    rw [SplitWorkerMir.issued_of_search searched]
    cases wide <;> cases again : SplitWorkerMir.retry input <;> simp_all [n,SplitWorkerMir.issued]
  rw [(a.observed matched).2.2 _ _ position]
  refine ⟨rfl, ?_⟩
  change a.dispatch.opening < a.dispatch.opening + 1 + _
  omega

/-- The acquired guard covers every physical store, including the first store
    after the stop poll when the preceding ticket issued several field accesses. -/
theorem acquire_before_store {b : SplitWorkerMir.Searched} (searched : SplitWorkerMir.search input = some b)
    (matched : a.aligned obs b.steps) {op : Physical} (mem : op ∈ a.traffic) {w : Write}
    (store : op = .writeAlpha w ∨ op = .writeBest w ∨ op = .writeMove w) :
    SequencedBefore obs.acquired w.event := by
  obtain ⟨_, sameBody, observed⟩ := a.observed matched
  have shape : ∃ front, b.steps = front ++ [.lock] := by
    rw [SplitWorkerMir.issued_of_search searched]
    cases again : SplitWorkerMir.retry input
    · exact ⟨[.load false, .returned (SplitWorkerMir.scoutQuery input)], by simp [SplitWorkerMir.issued, again]⟩
    · exact ⟨[.load false, .returned (SplitWorkerMir.scoutQuery input), .load true,
        .returned (SplitWorkerMir.wideQuery input)], by simp [SplitWorkerMir.issued, again]⟩
  obtain ⟨front, last⟩ := shape
  have atLock : (b.steps ++ lockedSteps)[front.length]? = some .lock := by simp [last]
  obtain ⟨i, _, stored⟩ := a.store_index mem store
  change SequencedBefore (obs.stamp .lock) _
  rw [observed _ _ atLock, stored]
  refine ⟨rfl, a.dispatch.clock.advances _ _ ?_⟩
  change offset (a.dispatch.sections.flatMap Section.gaps) (a.prior.length + front.length) < _
  conv => lhs; arg 1; rw [a.split, List.flatMap_append, List.flatMap_cons]
  unfold prior
  rw [← gaps_length a.pre, offset_append]
  have zeros (n k : Nat) (rest : List Nat) (lt : k < n) :
      offset (List.replicate n 0 ++ rest) k = k := by
    induction n generalizing k with
    | zero => omega
    | succ n ih =>
      cases k with
      | zero => simp [List.replicate_succ, offset]
      | succ k => simp [List.replicate_succ, offset, ih k (by omega)]; omega
  have len : a.row.body.length = front.length+1 := by rw [sameBody,last]; simp
  simp only [Section.gaps, List.append_assoc, List.singleton_append]
  rw [zeros _ _ _ (by omega), start, prior, gaps_length]
  omega

theorem needs_reset_of_access {op : Physical} (mem : op ∈ a.traffic) : a.row.needsReset := by
  cases field : a.row.fields with
  | nil => unfold traffic at mem; rw [field] at mem; contradiction
  | cons => unfold Section.needsReset; rw [field]; simp

theorem in_run : a.row ∈ a.dispatch.sections := List.get_mem _ _

end Address

def locate (runs : List Dispatch) (obs : Observation) : Option Address := do
  let d ← runs.find? (fun d => d.clock.worker == obs.worker && d.opening == obs.dispatch)
  if d.valid then
    let i ← d.sections.findFinIdx? (fun s => s.opening == obs.opening)
    return ⟨d,i⟩
  else none

theorem located (found : locate runs obs = some a) :
    a.dispatch ∈ runs ∧ a.dispatch.clock.worker = obs.worker ∧ a.dispatch.valid = true := by
  obtain ⟨d, dispatch, tail⟩ := Option.bind_eq_some_iff.mp found
  split at tail <;> try contradiction
  obtain ⟨_, _, eq⟩ := Option.bind_eq_some_iff.mp tail
  cases eq
  have binds := List.find?_some dispatch
  exact ⟨List.mem_of_find?_eq_some dispatch, by simpa using (Bool.and_eq_true_iff.mp binds).1, by assumption⟩

def traffic (runs : List Dispatch) (journal : List Observation) (rank : Nat) : List Physical :=
  ((journal[rank]?.bind (locate runs)).map Address.traffic).getD []

def matchesTicket (runs : List Dispatch) (journal : List Observation) (stream : List NativeFrames.Record)
    (op : MoveList.ConcurrentTrace.Write w) : Bool := (do
  let (_,obs) ← IteratorFrames.findLock journal op.worker op.opening
  let a ← locate runs obs
  let c ← SplitWorkerMir.Iterator.acquired (IteratorFrames.source journal stream) op
  let b ← SplitWorkerMir.search c.ticket.input
  return a.aligned obs b.steps).getD false

theorem for_posted {s : MoveList.ConcurrentTrace.State w}
    (x : IteratorFrames.Posted rc s journal stream earlier r seq)
    (read : matchesTicket runs journal stream x.op) :
    ∃ a, locate runs x.lock = some a ∧ a.aligned x.lock x.body.steps = true := by
  cases found : locate runs x.lock with
  | none => simp [matchesTicket, x.foundLock, found] at read
  | some a =>
    refine ⟨a, rfl, ?_⟩
    simpa [matchesTicket, x.foundLock, found, x.acquired, x.searched] using read

theorem traffic_source {op : Physical} (mem : op ∈ traffic runs journal rank) :
    ∃ obs a, locate runs obs = some a ∧ traffic runs journal rank = a.traffic := by
  cases obs : journal[rank]? with
  | none => simp [traffic, obs] at mem
  | some row =>
    cases found : locate runs row with
    | none => simp [traffic, obs, found] at mem
    | some a => exact ⟨row, a, found, by simp [traffic, obs, found]⟩

end NeuralReversi.PvsParallel.SplitSchedulerMir.WorkerFields
