import NeuralReversi.MoveList.ConcurrentTrace
import NeuralReversi.Threading.SplitResult

namespace NeuralReversi.PvsParallel.Interleaving.Tickets

open MoveList.Concurrent (advance)
open Threading.SplitResult (Report)

/-- One publication under the split-point lock; the cut bit is the worker commit's output. -/
structure Post where
  report : Report
  cut : Bool

structure State (w : Nat) where
  current : BitVec w
  pollers : List Nat
  pending : List (Nat × Nat)
  departing : List Nat
  booking : Bool
  reports : List Post
  discarded : List Nat
  outside : Bool

def initial (start : BitVec w) (owner : Nat) : State w :=
  ⟨start,[owner],[],[],true,[],[],false⟩

def State.retired (s : State w) : Prop :=
  s.pollers = [] ∧ s.pending = [] ∧ s.departing = []

def State.stopped (s : State w) : Prop :=
  s.outside = true ∨ ∃ p ∈ s.reports, p.cut = true

def State.available (s : State w) (worker : Nat) : Prop :=
  worker ∉ s.pollers ∧ worker ∉ s.pending.map Prod.fst ∧ worker ∉ s.departing

inductive Action where
  | book (worker : Nat)
  | take (worker : Nat)
  | empty (worker : Nat)
  | report (worker : Nat) (post : Post)
  | discard (worker : Nat)
  | leave (worker : Nat)
  | externalStop

def Action.posts : Action → List Post
  | .report _ p => [p]
  | _ => []

/-- Project an interleaving onto one split generation. A successful relaxed update fixes
    the index. The worker can then remain pending through any number of other transitions.
    Only a locked report publishes; observing a stop can instead consume the pending ticket. -/
inductive Step (len : Nat) : State w → Action → State w → Prop where
  | book {s : State w} (openGate : s.booking = true) (noCut : ∀ p ∈ s.reports, p.cut = false)
      (free : s.available worker) :
      Step len s (.book worker) {s with pollers := worker :: s.pollers}
  | take {s : State w} (workerAt : s.pollers = left ++ worker :: right)
      (claimed : advance len s.current = some updated) :
      Step len s (.take worker) {s with
        current := updated
        pollers := left ++ right
        pending := (worker,s.current.toNat) :: s.pending}
  | empty {s : State w} (workerAt : s.pollers = left ++ worker :: right) (exhausted : advance len s.current = none) :
      Step len s (.empty worker) {s with pollers := left ++ right, departing := worker :: s.departing}
  | report {s : State w} {post : Post} (held : s.pending = left ++ (worker,post.report.index) :: right) :
      Step len s (.report worker post) {s with
        pending := left ++ right
        reports := s.reports ++ [post]
        pollers := if post.cut then s.pollers else worker :: s.pollers
        departing := if post.cut then worker :: s.departing else s.departing}
  | discard {s : State w} (held : s.pending = left ++ (worker,index) :: right) (signal : s.stopped) :
      Step len s (.discard worker) {s with
        pending := left ++ right
        discarded := index :: s.discarded
        booking := false
        departing := worker :: s.departing}
  | leave {s : State w} (ending : s.departing = left ++ worker :: right) :
      Step len s (.leave worker) {s with departing := left ++ right, booking := false}
  | externalStop {s : State w} : Step len s .externalStop {s with outside := true}

inductive Run (len : Nat) (start : State w) : State w → Prop where
  | refl : Run len start start
  | step : Run len start s → Step len s event after → Run len start after

variable {s t : State w} {start : BitVec w} {p : Post}

theorem Step.reports_eq (step : Step len s a t) : t.reports = s.reports ++ a.posts := by
  cases step <;> simp [Action.posts]

theorem Step.stopped (step : Step len s a t) (seen : s.stopped) : t.stopped := by
  rcases seen with outside | ⟨p,member,cut⟩
  · left
    cases step <;> simp [outside]
  · exact Or.inr ⟨p,step.reports_eq ▸ List.mem_append_left _ member,cut⟩

/-- A relaxed failed update can read an older value and retry. If the final value
    says that the list is empty, the following modification-order suffix is empty too. -/
theorem Step.empty_of_reads (workerAt : s.pollers = left ++ worker :: right)
    (history : MoveList.ConcurrentTrace.replay len start ops = some s.current)
    (observed : MoveList.ConcurrentTrace.observed len (MoveList.ConcurrentTrace.words start ops) load failures = some value)
    (empty : advance len value = none) :
    Step len s (.empty worker) {s with pollers := left ++ right, departing := worker :: s.departing} :=
  .empty workerAt (MoveList.ConcurrentTrace.declined ops history observed empty).2

def State.indices (s : State w) := s.reports.map (fun p => p.report.index)

private def remaining (len : Nat) (current : BitVec w) :=
  List.range' current.toNat (len - current.toNat)

private def inventory (len : Nat) (s : State w) :=
  s.indices ++ (s.pending.map Prod.snd ++ s.discarded ++ remaining len s.current)

private theorem uncons_remaining (fits : len < 2^w) {c d : BitVec w}
    (take : advance len c = some d) : remaining len c = c.toNat :: remaining len d := by
  have next := MoveList.Concurrent.advance_nat len fits take
  have inside := (MoveList.Concurrent.advance_some len take).1
  have delta : len - c.toNat = (len - d.toNat) + 1 := by omega
  simp [remaining,delta,List.range'_succ,next]

private theorem Step.inventory_perm (op : Step len s act t) (fits : len < 2^w) :
    (inventory len t).Perm (inventory len s) := by
  cases op with
  | book | empty | leave | externalStop => exact .refl _
  | take _ claimed =>
    apply List.perm_iff_count.mpr
    intro n
    simp only [inventory,State.indices,List.map_cons,List.count_append,List.count_cons,uncons_remaining fits claimed]
    omega
  | report held =>
    apply List.perm_iff_count.mpr
    intro n
    simp only [inventory,State.indices,held,List.map_append,List.map_cons,List.map_nil,List.count_append,
      List.count_cons,List.count_nil]
    omega
  | discard held _ =>
    apply List.perm_iff_count.mpr
    intro n
    simp only [inventory,State.indices,held,List.map_append,List.map_cons,List.count_append,List.count_cons]
    omega

private theorem Run.inventory_perm (run : Run len (initial start owner) s) (fits : len < 2^w) :
    (inventory len s).Perm (remaining len start) := by
  induction run with
  | refl => exact .refl _
  | step _ edge ih => exact (edge.inventory_perm fits).trans ih

private structure Live (len : Nat) (s : State w) : Prop where
  kept : ¬ s.stopped → s.discarded = []
  exhausted : ¬ s.stopped → s.pollers = [] → s.pending = [] → len ≤ s.current.toNat

private theorem Run.live (run : Run len (initial start owner) s) : Live len s := by
  induction run with
  | refl => exact ⟨fun _ => rfl,fun _ h => by simp [initial] at h⟩
  | @step beforeState _ _ _ edge ih =>
    constructor
    · intro normal
      have before := ih.kept (fun cut => normal (edge.stopped cut))
      cases edge with
      | discard _ cut => exact False.elim (normal cut)
      | _ => exact before
    · intro normal noPoll noClaim
      have before := ih.exhausted (fun cut => normal (edge.stopped cut))
      cases edge with
      | book => cases noPoll
      | take => cases noClaim
      | empty _ exhausted =>
        by_cases h : len ≤ beforeState.current.toNat
        · exact h
        · simp [advance,Nat.lt_of_not_ge h] at exhausted
      | report held =>
        rename_i post
        cases cut : post.cut with
        | true => exact False.elim (normal (Or.inr ⟨post,List.mem_append_right _ (by simp),cut⟩))
        | false => simp [cut] at noPoll
      | discard _ cut => exact False.elim (normal cut)
      | leave => exact before noPoll noClaim
      | externalStop => exact False.elim (normal (Or.inl rfl))

theorem Run.unique (run : Run len (initial start owner) s) (fits : len < 2^w) : s.indices.Nodup :=
  (List.sublist_append_left _ _).nodup ((run.inventory_perm fits).nodup_iff.mpr (List.nodup_range' ..))

theorem Run.valid (run : Run len (initial start owner) s) (fits : len < 2^w)
    (member : p ∈ s.reports) : start.toNat ≤ p.report.index ∧ p.report.index < len := by
  have listed : p.report.index ∈ s.indices := List.mem_map.mpr ⟨p,member,rfl⟩
  have inside := (run.inventory_perm fits).mem_iff.mp (List.mem_append_left _ listed)
  obtain ⟨offset,upper,eq⟩ := List.mem_range'.mp inside
  simp at eq
  omega

theorem Run.cover (run : Run len (initial start owner) s) (fits : len < 2^w)
    (finished : s.retired) (normal : ¬ s.stopped) :
    s.indices.Perm (List.range' start.toNat (len-start.toNat)) := by
  have beyond := run.live.exhausted normal finished.1 finished.2.1
  have accounted := run.inventory_perm fits
  simpa [inventory,remaining,finished.2.1,run.live.kept normal,Nat.sub_eq_zero_of_le beyond] using accounted

/-- The owner waits for all participants to retire. A missing report then requires a
    published local cut or a stop from outside the split (an ancestor or the abort state). -/
theorem Run.outcome (run : Run len (initial start owner) s) (fits : len < 2^w) (finished : s.retired) :
    s.outside = true ∨ (∃ p ∈ s.reports, p.cut = true) ∨
      s.indices.Perm (List.range' start.toNat (len-start.toNat)) := by
  by_cases stopped : s.stopped
  · exact stopped.elim Or.inl (fun cut => Or.inr (Or.inl cut))
  · exact Or.inr (Or.inr (run.cover fits finished stopped))

end NeuralReversi.PvsParallel.Interleaving.Tickets
