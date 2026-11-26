import NeuralReversi.PvsParallel.Interleaving.FlagSegment
import NeuralReversi.PvsParallel.Interleaving.EventOrder

namespace NeuralReversi.PvsParallel.Interleaving.MaskClock

open Threading
open AtomicWord (Event SequencedBefore)
open CutoffVisibility (History HappensBefore)
open FlagSegment (Transfer)

/-- The finite synchronization relation for objects other than the cutoff epoch.
    Epoch release sequences are reconstructed from that object's own history. -/
structure Sync (h : History) where
  edges : List (Event × Event)
  exact : ∀ a b, (a,b) ∈ edges ↔ h.otherSync a b

private instance {h : History} {i j : Fin h.epochs.length} :
    Decidable (CutoffVisibility.InReleaseSequence h i j) :=
  inferInstanceAs (Decidable ((h.epochs.get i).release = true ∧ i.val ≤ j.val ∧
    ∀ k : Fin h.epochs.length, i.val < k.val → k.val ≤ j.val → (h.epochs.get k).rmw = true))

private def epoch (h : History) (r : {r // r ∈ h.epochReads}) (i : Fin h.epochs.length) :
    Option (Transfer h) :=
  if ok : r.val.acquire = true ∧ CutoffVisibility.InReleaseSequence h i r.val.source then
    some (.epoch i r.val r.property ok.1 ok.2)
  else none

def Sync.transfers (s : Sync h) : List (Transfer h) :=
  s.edges.attach.map (fun z => .handoff z.val.1 z.val.2 ((s.exact ..).mp z.property)) ++
    h.epochReads.attach.flatMap (fun r => (List.finRange h.epochs.length).filterMap (epoch h r))

theorem Sync.contains (s : Sync h) (t : Transfer h) : t ∈ s.transfers := by
  cases t with
  | handoff a b sync =>
    exact List.mem_append_left _ (List.mem_map.mpr ⟨⟨(a,b),(s.exact ..).mpr sync⟩,List.mem_attach ..,rfl⟩)
  | epoch i r mem acq seq =>
    apply List.mem_append_right
    apply List.mem_flatMap.mpr
    refine ⟨⟨r,mem⟩,List.mem_attach ..,List.mem_filterMap.mpr ⟨i,List.mem_finRange _,?_⟩⟩
    simp [epoch,acq,seq]

/-- The index of the last earlier instruction of the same worker. The list is
    reverse chronological. Unobserved instructions still have their original
    worker-local steps and need no synthetic entries. -/
private def block : List Event → Event → Nat
  | [],_ => 0
  | x :: xs,e => if x.worker = e.worker ∧ x.step ≤ e.step then xs.length+1 else block xs e

private theorem bounded (xs : List Event) (e : Event) : block xs e ≤ xs.length := by
  induction xs with
  | nil => exact Nat.le_refl _
  | cons x xs ih => unfold block; split <;> simp_all <;> omega

private theorem monotone (xs : List Event) (sb : SequencedBefore a b) : block xs a ≤ block xs b := by
  induction xs with
  | nil => exact Nat.le_refl _
  | cons x xs ih =>
    by_cases here : x.worker = a.worker ∧ x.step ≤ a.step
    · have there := And.intro (here.1.trans sb.1) (Nat.le_trans here.2 (Nat.le_of_lt sb.2))
      simp only [block,ite_eq_left here,ite_eq_left there,Nat.le_refl]
    · by_cases there : x.worker = b.worker ∧ x.step ≤ b.step
      · simpa only [block,ite_eq_right here,ite_eq_left there] using Nat.le_succ_of_le (bounded xs a)
      · simpa only [block,ite_eq_right here,ite_eq_right there] using ih

def span : List Event → Nat
  | [] => 1
  | e :: es => max (e.step+1) (span es)

/-- A gap between two recorded instructions accommodates any intervening
    local steps. This function is defined on every worker and every local PC. -/
def time (reverseOrder : List Event) (e : Event) : Nat := block reverseOrder e * span reverseOrder + e.step

theorem localTime (xs : List Event) (sb : SequencedBefore a b) : time xs a < time xs b :=
  Nat.add_lt_add_of_le_of_lt (Nat.mul_le_mul_right _ (monotone xs sb)) sb.2

/-- Check only primitive transfers. Program order on arbitrarily large local
    instruction numbers follows directly from the extension of the clock. -/
def ordered (xs : List Event) (s : Sync h) :=
  s.transfers.all fun t => time xs t.edge.val.1 < time xs t.edge.val.2

theorem causal (s : Sync h) (ok : ordered xs s) (hb : HappensBefore h a b) : time xs a < time xs b := by
  have one (t : Transfer h) := of_decide_eq_true (List.all_eq_true.mp ok t (s.contains t))
  induction hb with
  | sequenced sb => exact localTime xs sb
  | other sync => exact one (.handoff _ _ sync)
  | synchronized i r mem acq seq => exact one (.epoch i r mem acq seq)
  | trans _ _ left right => exact Nat.lt_trans left right

structure Capture (h : History) where
  sync : Sync h
  reverseOrder : List Event
  checked : ordered reverseOrder sync = true

def capture (s : Sync h) (order : List Event) : Option (Capture h) :=
  if checked : ordered order.reverse s then some ⟨s,order.reverse,checked⟩ else none

def Capture.clock (c : Capture h) := time c.reverseOrder

theorem Capture.before (c : Capture h) (a b : Event) (hb : HappensBefore h a b) :
    c.clock a < c.clock b := causal c.sync c.checked hb

def Capture.transfers (c : Capture h) : List (Transfer h) :=
  c.sync.transfers.mergeSort (fun x y => decide (c.clock x.edge.val.2 ≤ c.clock y.edge.val.2))

def ReachabilityContract := ∀ (h : History) (c : Capture h) (x y : Event),
  HappensBefore h x y ↔ (FlagSegment.published c.transfers x y).isSome = true

theorem published : ReachabilityContract := by
  intro h c x y
  apply FlagSegment.complete c.clock c.before
  · simpa [Capture.transfers] using (List.pairwise_mergeSort
      (le := fun t u : Transfer h => decide (c.clock t.edge.val.2 ≤ c.clock u.edge.val.2))
      (by simp; omega) (by simp; omega) c.sync.transfers)
  · exact fun t => (List.mergeSort_perm ..).mem_iff.mpr (c.sync.contains t)

private theorem stepBound (xs : List Event) (mem : e ∈ xs) : e.step < span xs := by
  induction xs with
  | nil => cases mem
  | cons z zs ih =>
    rcases List.mem_cons.mp mem with rfl | inside
    · exact Nat.lt_of_lt_of_le (Nat.lt_succ_self _) (Nat.le_max_left ..)
    · exact Nat.lt_of_lt_of_le (ih inside) (Nat.le_max_right ..)

private def Descending (xs : List Event) :=
  xs.Pairwise fun a b => a.worker = b.worker → b.step < a.step

private theorem atBlock (xs : List Event) (sorted : Descending xs) (i : Fin xs.length) :
    block xs (xs.get i) = xs.length - i.val := by
  induction xs with
  | nil => exact Fin.elim0 i
  | cons z zs ih =>
    obtain ⟨prior,rest⟩ := List.pairwise_cons.mp sorted
    obtain ⟨n,inside⟩ := i
    cases n with
    | zero => simp [block]
    | succ n =>
      let j : Fin zs.length := ⟨n,by simpa using inside⟩
      have earlier := prior (zs.get j) (List.get_mem ..)
      have skip : ¬ (z.worker = (zs.get j).worker ∧ z.step ≤ (zs.get j).step) :=
        fun hit => (Nat.not_lt_of_ge hit.2) (earlier hit.1)
      change (if z.worker = (zs.get j).worker ∧ z.step ≤ (zs.get j).step then _ else _) = _
      rw [ite_eq_right skip]
      change block zs (zs.get j) = (zs.length+1) - (n+1)
      simpa [j] using ih rest j

private theorem strictAt (xs : List Event) (sorted : Descending xs) {i j : Fin xs.length}
    (later : j.val < i.val) : time xs (xs.get i) < time xs (xs.get j) := by
  unfold time
  rw [atBlock _ sorted,atBlock _ sorted]
  have small := stepBound xs (List.get_mem xs i)
  have gap : (xs.length-i.val)+1 ≤ xs.length-j.val := by have := i.isLt; omega
  have one := Nat.mul_le_mul_right (span xs) gap
  rw [Nat.add_mul,Nat.one_mul] at one
  omega

private theorem uniqueList (xs : List Event) :
    ∃ ys : List Event, ys.Nodup ∧ ∀ e, e ∈ ys ↔ e ∈ xs := by
  induction xs with
  | nil => exact ⟨[],by simp,fun _ => Iff.rfl⟩
  | cons x xs ih =>
    obtain ⟨ys,distinct,mem⟩ := ih
    by_cases seen : x ∈ ys
    · exact ⟨ys,distinct,fun e => by
        rw [List.mem_cons]; exact ⟨fun h => Or.inr ((mem _).mp h),fun h =>
          h.elim (fun eq => eq ▸ seen) ((mem _).mpr)⟩⟩
    · exact ⟨x::ys,List.nodup_cons.mpr ⟨seen,distinct⟩,fun e => by simp [mem]⟩

private theorem orderExists (h : History) (domain : List Event) (acyclic : ∀ e, ¬ HappensBefore h e e) :
    ∃ xs : List Event, (∀ e, e ∈ xs ↔ e ∈ domain) ∧
      ∀ a ∈ xs, ∀ b ∈ xs, HappensBefore h a b → time xs a < time xs b := by
  classical
  obtain ⟨nodes,unique_,domainMem⟩ := uniqueList domain
  let hb := fun i j : Fin nodes.length => HappensBefore h (nodes.get i) (nodes.get j)
  let rank (e : Event) := if inside : e ∈ nodes then
    (EventOrder.position hb ⟨nodes.idxOf e,List.idxOf_lt_length_of_mem inside⟩).val else 0
  have orderedRank (a b : Event) (ha : a ∈ nodes) (hb_ : b ∈ nodes) (edge : HappensBefore h a b) :
      rank a < rank b := by
    simp only [rank,dite_eq_left ha,dite_eq_left hb_]
    apply EventOrder.extendsHB hb (fun i => acyclic _) (fun x y => x.trans y)
    simpa [hb] using edge
  let xs := nodes.mergeSort (fun a b => decide (rank b ≤ rank a))
  have perm : xs.Perm nodes := List.mergeSort_perm ..
  have sorted : xs.Pairwise (fun a b => rank b ≤ rank a) := by
    simpa [xs] using (List.pairwise_mergeSort (le := fun a b : Event => decide (rank b ≤ rank a))
      (by simp; omega) (by simp; omega) nodes)
  have distinct : xs.Nodup := perm.nodup_iff.mpr unique_
  have local_ : Descending xs := by
    apply (sorted.and distinct).imp_of_mem
    intro a b ha hbb pair same
    by_cases eq : a.step = b.step
    · exact (pair.2 (by cases a; cases b; simp_all)).elim
    · apply Classical.byContradiction
      intro wrong
      have up := orderedRank a b (perm.mem_iff.mp ha) (perm.mem_iff.mp hbb)
        (.sequenced ⟨same,by omega⟩)
      exact Nat.not_lt_of_ge pair.1 up
  refine ⟨xs,fun e => perm.mem_iff.trans (domainMem e),?_⟩
  intro a ma b mb edge
  obtain ⟨i,rfl⟩ := List.get_of_mem ma
  obtain ⟨j,rfl⟩ := List.get_of_mem mb
  apply strictAt xs local_
  have up := orderedRank _ _ (perm.mem_iff.mp ma) (perm.mem_iff.mp mb) edge
  apply Classical.byContradiction
  intro fail
  by_cases eq : i = j
  · exact acyclic _ (eq ▸ edge)
  · have ne := Fin.val_ne_of_ne eq
    have opposite := List.pairwise_iff_getElem.mp sorted i.val j.val i.isLt j.isLt (by omega)
    exact Nat.not_lt_of_ge opposite up

def Contract := ∀ (h : History) (s : Sync h), (∀ e, ¬ HappensBefore h e e) ↔
    ∀ extra : List Event, ∃ xs : List Event, (∀ e ∈ extra, e ∈ xs) ∧ ordered xs s = true

/-- Real executions have a finite linearization. The clock extends this order
    to all local instruction numbers, including gaps and unobserved workers.
    An accepted finite check rules out a cycle anywhere in the whole history. -/
theorem finiteClock : Contract := by
  intro h s
  constructor
  · intro acyclic extra
    let nodes := extra ++ s.transfers.flatMap (fun t => [t.edge.val.1,t.edge.val.2])
    obtain ⟨xs,members,ordered_⟩ := orderExists h nodes acyclic
    refine ⟨xs,fun e mem => (members e).mpr (List.mem_append_left _ mem),List.all_eq_true.mpr ?_⟩
    intro t mem
    have included (e : Event) (one : e ∈ [t.edge.val.1,t.edge.val.2]) : e ∈ xs :=
      (members e).mpr (List.mem_append_right _ (List.mem_flatMap.mpr ⟨t,mem,one⟩))
    exact decide_eq_true (ordered_ _ (included _ (by simp)) _ (included _ (by simp)) t.edge.property)
  · intro all_ e cycle
    obtain ⟨xs,_,ok⟩ := all_ []
    exact Nat.lt_irrefl _ (causal s ok cycle)

end NeuralReversi.PvsParallel.Interleaving.MaskClock
