import NeuralReversi.PvsParallel.Interleaving.Tickets
import NeuralReversi.Threading.SplitCompletion

namespace NeuralReversi.PvsParallel.Interleaving.Retirement

open Tickets (Post State Step)
open Threading
open Threading.SplitCompletion (Worker)
open Threading.AtomicWord (Event Write)

def roster (s : State w) := s.pollers ++ (s.pending.map Prod.fst ++ s.departing)

def effect : Tickets.Action → Option (Bool × Nat)
  | .book i => some (true,i)
  | .leave i => some (false,i)
  | _ => none

theorem roster_step (op : Step len s a t) :
    match effect a with
    | none => (roster t).Perm (roster s)
    | some (true,i) => (roster t).Perm (i :: roster s)
    | some (false,i) => (roster s).Perm (i :: roster t) := by
  cases op with
  | book | externalStop => exact .refl _
  | take held _ | empty held _ | discard held _ | leave held =>
    apply List.perm_iff_count.mpr
    intro i
    simp only [roster,held,List.map_append,List.map_cons,List.count_append,List.count_cons]
    omega
  | report held =>
    rename_i post
    cases cut : post.cut <;> apply List.perm_iff_count.mpr <;> intro i <;>
      simp only [roster,held,List.map_append,List.map_cons,List.count_append,List.count_cons,
        Bool.false_eq_true,ite_false,ite_true] <;> omega

/-- A chronological run retains its mask RMWs in modification order. The projected events
    carry their real worker and instruction identity. Other steps leave this location alone. -/
inductive ExecutionWith (outside : Prop) (check : State w → Post → Prop) (len : Nat) (start : BitVec w) (owner : Worker) : List SplitCompletion.Action → State w → Prop where
  | initial : ExecutionWith outside check len start owner [] (Tickets.initial start owner.val)
  | work : ExecutionWith outside check len start owner actions s → Step len s a t → effect a = none → (a = .externalStop → outside) → (∀ p ∈ a.posts, check s p) →
      ExecutionWith outside check len start owner actions t
  | book (worker : Worker) (event : Event) : ExecutionWith outside check len start owner actions s →
      Step len s (.book worker.val) t → ExecutionWith outside check len start owner (actions ++ [.book worker event]) t
  | finish (worker : Worker) (event : Event) : ExecutionWith outside check len start owner actions s →
      Step len s (.leave worker.val) t → ExecutionWith outside check len start owner (actions ++ [.finish worker event]) t

abbrev Execution (len : Nat) (start : BitVec w) (owner : Worker) :=
  ExecutionWith True (fun _ _ => True) len start owner

theorem ExecutionWith.tickets (path : ExecutionWith outside check len start owner actions s) :
    Tickets.Run len (Tickets.initial start owner.val) s := by
  induction path with
  | initial => exact .refl
  | work _ edge _ _ _ ih | book _ _ _ edge ih | finish _ _ _ edge ih => exact ih.step edge

theorem ExecutionWith.external_cause (path : ExecutionWith outside check len start owner actions s)
    (seen : s.outside = true) : outside := by
  induction path with
  | initial => contradiction
  | work _ op _ cause _ ih =>
    cases op <;> first | exact cause rfl | exact ih seen
  | book _ _ _ op ih | finish _ _ _ op ih => cases op; exact ih seen

private structure Tracked (owner : Worker) (actions : List SplitCompletion.Action) (s : State w) : Prop where
  distinct : (roster s).Nodup
  openGate : s.booking = true → ∀ i event, .finish i event ∉ actions
  active : ∀ k ∈ roster s, ∃ i : Worker, i.val = k ∧
    (i = owner ∨ ∃ event, .book i event ∈ actions) ∧ ∀ event, .finish i event ∉ actions

theorem booking_gate (op : Step len s a t) (openAfter : t.booking = true) : s.booking = true := by
  cases op <;> simp_all

theorem opened (op : Step len s (.book i) t) : s.booking = true ∧ i ∉ roster s := by
  cases op with
  | book enabled _ free => exact ⟨enabled,by simpa [roster,Tickets.State.available] using free⟩

theorem closed (op : Step len s (.leave i) t) : t.booking = false := by
  cases op
  rfl

private theorem ExecutionWith.tracked (path : ExecutionWith outside check len start owner actions s) : Tracked owner actions s := by
  induction path with
  | initial =>
    refine ⟨by simp [roster,Tickets.initial],by simp,?_⟩
    intro k member
    have same : k = owner.val := by simpa [roster,Tickets.initial] using member
    exact ⟨owner,same.symm,Or.inl rfl,by simp⟩
  | work _ op silent _ _ ih =>
    have perm := roster_step op
    rw [silent] at perm
    exact ⟨perm.nodup_iff.mpr ih.distinct,fun gate => ih.openGate (booking_gate op gate),
      fun k here => ih.active k (perm.mem_iff.mp here)⟩
  | book worker e _ op ih =>
    have perm := roster_step op
    have born := opened op
    change (roster _).Perm (worker.val :: roster _) at perm
    refine ⟨perm.nodup_iff.mpr (List.nodup_cons.mpr ⟨born.2,ih.distinct⟩),?_,?_⟩
    · intro _ i e
      simpa using ih.openGate born.1 i e
    · intro k member
      rcases List.mem_cons.mp (perm.mem_iff.mp member) with same | prior
      · exact ⟨worker,same.symm,Or.inr ⟨e,by simp⟩,fun evt => by simpa using ih.openGate born.1 worker evt⟩
      · obtain ⟨i,id,birth,still⟩ := ih.active k prior
        exact ⟨i,id,birth.imp_right (fun ⟨e,h⟩ => ⟨e,List.mem_append_left _ h⟩),fun e => by simpa using still e⟩
  | finish worker e _ op ih =>
    have perm := roster_step op
    change (roster _).Perm (worker.val :: roster _) at perm
    obtain ⟨gone,unique⟩ := List.nodup_cons.mp (perm.nodup_iff.mp ih.distinct)
    refine ⟨unique,fun h => by simp [closed op] at h,?_⟩
    intro k member
    obtain ⟨i,id,birth,still⟩ := ih.active k (perm.mem_iff.mpr (List.mem_cons_of_mem _ member))
    have different : i ≠ worker := fun same => gone (same ▸ id ▸ member)
    exact ⟨i,id,birth.imp_right (fun ⟨e,h⟩ => ⟨e,List.mem_append_left _ h⟩),fun evt => by simp [still,different]⟩

/-- A zero Acquire with the native raw mask actions excludes every ticket owner, including
    the initializing owner and a worker that joined after the first ticket was issued. -/
theorem ExecutionWith.retired {activation : SplitCompletion.Activation writes}
    (path : ExecutionWith outside check len start activation.owner activation.actions s)
    (observed : SplitCompletion.Retirement history writes activation) : s.retired := by
  obtain ⟨_,released,late⟩ := SplitCompletion.emptyReadContract_proven _ _ _ observed
  have empty (k) (here : k ∈ roster s) : False := by
    obtain ⟨i,_,birth,still⟩ := path.tracked.active k here
    rcases birth with rfl | ⟨e,booked⟩
    · obtain ⟨e,mem⟩ := released
      exact still e mem
    · obtain ⟨e,mem⟩ := late i e booked
      exact still e mem
  have allGone : roster s = [] := List.eq_nil_iff_forall_not_mem.mpr empty
  simpa [roster,Tickets.State.retired] using allGone

theorem ExecutionWith.outcome {check : State w → Post → Prop} {start : BitVec w} {activation : SplitCompletion.Activation writes}
    (path : ExecutionWith outside check len start activation.owner activation.actions s)
    (observed : SplitCompletion.Retirement history writes activation) (fits : len < 2^w) :
    s.outside = true ∨ (∃ p ∈ s.reports, p.cut = true) ∨
      s.indices.Perm (List.range' start.toNat (len-start.toNat)) :=
  path.tickets.outcome fits (path.retired observed)

end NeuralReversi.PvsParallel.Interleaving.Retirement
