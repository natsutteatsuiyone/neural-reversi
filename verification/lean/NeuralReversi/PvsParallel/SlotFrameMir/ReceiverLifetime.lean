import NeuralReversi.PvsParallel.SlotFrameMir.ReceiverGuard

namespace NeuralReversi.PvsParallel.SlotFrameMir.Receiver

open Threading
open AtomicWord (Event)
open CutoffVisibility (History HappensBefore)

private def canEnd (s : Slot) (w : Worker) : Instruction → Bool
  | .release target who => decide (w=who ∧ s=target)
  | .abandon _ who => decide (w=who ∧ s.1=who)
  | _ => false

private theorem stays (r : Resources) (op : Instruction) (e : Event) (ok : AtInstruction r op)
    (active : s∈r.frames w) (keep : canEnd s w op=false) :
    s∈(r.exec e op).frames w := by
  cases op with
  | openSlot => exact active
  | reserve => exact active
  | restore => exact active
  | capture _ who =>
    change s∈(if w=who then _::_ else _)
    by_cases equal : w=who
    · subst w;simpa only [ite_true] using List.mem_cons_of_mem _ active
    · simpa only [equal,ite_false] using active
  | release target who =>
    change s∈(if w=who then _ else _)
    by_cases equal : w=who
    · subst w
      have diff : s≠target := by simpa [canEnd] using keep
      obtain ⟨tail,path⟩:=List.head?_eq_some_iff.mp (show (r.frames who).head?=some target from ok)
      rw [path] at active
      simpa [path] using (List.mem_cons.mp active).resolve_left diff
    · simpa only [equal,ite_false] using active
  | abandon taken who =>
    change s∈(if w=who then _ else _)
    by_cases equal : w=who
    · subst w
      have other : s.1≠who := by simpa [canEnd] using keep
      simp [active,other]
    · simpa only [equal,ite_false] using active

private theorem span_keeps (hist : History) (clock : Event → Nat)
    (causal : ∀ x y, HappensBefore hist x y → clock x < clock y)
    (s : Slot) (w : Worker) (past span : List Tick)
    (run : Scheduled hist clock (span++past)) (active : s∈(machine past).frames w)
    (keep : ∀ x ∈ span, canEnd s w x.2=false) :
    s∈(machine (span++past)).frames w ∧ ∀ x ∈ span, x.2≠.openSlot s := by
  induction span with
  | nil => exact ⟨active,by simp⟩
  | cons x xs ih =>
    obtain ⟨stamp,op⟩:=x
    cases run with
    | step prior src later =>
      obtain ⟨live,absent⟩:=ih prior (fun x mem => keep x (List.mem_cons_of_mem _ mem))
      have current := src.checked (fun x cyc => Nat.lt_irrefl _ (causal x x cyc))
      refine ⟨stays _ op stamp current live (keep _ (List.mem_cons_self ..)),?_⟩
      intro z member
      rcases List.mem_cons.mp member with same | old
      · subst z;intro opened
        change op=.openSlot s at opened
        subst op
        exact prior.reuse_has_no_older_worker causal src w live
      · exact absent z old

/-- Relate the physical capture and first nested guard to one generation. Every
    release or abandonment by this worker starts with detach_split taking its
    active Option. No such instruction occurs before its first nested split.
    An older prefix suffices; no validity of the proposed next booking is used. -/
theorem generation_at_first (hist : History) (clock : Event → Nat)
    (causal : ∀ x y, HappensBefore hist x y → clock x < clock y)
    (s : Slot) (w : Worker) (past span : List Tick) (cap : Event)
    (run : Scheduled hist clock (span++(cap,.capture s w)::past)) (localCalls : Calls [] ops)
    (detach : ∀ x ∈ span, (∃ target, x.2=.release target w) ∨ (∃ taken, x.2=.abandon taken w) →
      ∃ stamp, (stamp, Op.effect (.replace none)) ∈ untilSplit ops) :
    ∀ x ∈ span, x.2≠.openSlot s := by
  apply (span_keeps hist clock causal s w ((cap,.capture s w)::past) span run ?_ ?_).2
  · change s∈(if w=w then s::_ else _)
    simp
  · intro x mem
    by_cases ended : canEnd s w x.2=false
    · exact ended
    · have acts : (∃ target, x.2=.release target w) ∨ (∃ taken, x.2=.abandon taken w) := by
        cases action : x.2 <;> simp_all [canEnd]
      obtain ⟨stamp,conflict⟩:=detach x mem acts
      exact ((localCalls.untouched.1 stamp none) conflict).elim

end NeuralReversi.PvsParallel.SlotFrameMir.Receiver
