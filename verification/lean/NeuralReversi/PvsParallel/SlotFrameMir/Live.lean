import NeuralReversi.PvsParallel.SlotFrameMir.Generation

namespace NeuralReversi.PvsParallel.SlotFrameMir

open Threading.SplitCompletion (Action)
open Threading.AtomicWord (Event)

/-- Colour one actual Arc publication. An active work frame and a restored parent
    keep the real pointer while the previous dispatch has consumed its assignment. -/
inductive Pointer
  | empty | pending (slot : Slot) | used (slot : Slot)
  deriving DecidableEq

structure Resources where
  active : Worker → Pointer
  /-- Captured loans, including foreign work abandoned without a normal release. -/
  frames : Worker → List Slot
  mask : Slot → BitVec 64

def Resources.zero : Resources := ⟨fun _ => .empty, fun _ => [], fun _ => 0⟩

inductive Instruction
  | openSlot (s : Slot) | reserve (s : Slot) (w : Worker) | capture (s : Slot) (w : Worker)
  | restore (s : Option Slot) (w : Worker) | release (s : Slot) (w : Worker)
  | abandon (taken : Option Slot) (w : Worker)
  deriving DecidableEq

private def replaced [DecidableEq α] (f : α → β) (key : α) (value : β) (x : α) : β :=
  if x = key then value else f x

def Resources.exec (r : Resources) (e : Event) : Instruction → Resources
  | .openSlot s =>
    {r with active := replaced r.active s.1 (.pending s), mask := replaced r.mask s (Bitboard.squareMask s.1)}
  | .reserve s w =>
    {r with
      active := replaced r.active w (.pending s)
      mask := replaced r.mask s ((Action.book w e).updateBits (r.mask s))}
  | .capture s w =>
    {r with active := replaced r.active w (.used s), frames := replaced r.frames w (s :: r.frames w)}
  | .restore parent w =>
    {r with active := replaced r.active w (parent.elim Pointer.empty Pointer.used)}
  | .release s w =>
    {r with
      active := replaced r.active w .empty
      frames := replaced r.frames w (r.frames w).tail
      mask := replaced r.mask s ((Action.finish w e).updateBits (r.mask s))}
  | .abandon taken w =>
    {active := replaced r.active w .empty
     frames := replaced r.frames w ((r.frames w).filter (fun s => s.1 ≠ w))
     mask := match taken with
       | none => r.mask
       | some s => replaced r.mask s ((Action.finish w e).updateBits (r.mask s))}

def Resources.Retains (r : Resources) : Prop :=
  (∀ s w, r.active w = .pending s → (r.mask s).getLsbD w.val = true ∧ s ∉ r.frames w) ∧
  (∀ s w, s ∈ r.frames w → (r.mask s).getLsbD w.val = true) ∧
  ∀ w, (r.frames w).Nodup

def Instruction.maskSlot : Instruction → Option Slot
  | .openSlot s | .reserve s _ | .release s _ => some s
  | .capture .. | .restore .. => none
  | .abandon taken _ => taken

theorem Resources.unmodified (r : Resources) (e : Event) (op : Instruction)
    (other : op.maskSlot ≠ some s) : (r.exec e op).mask s=r.mask s := by
  cases op with
  | abandon taken w => cases taken <;> simp_all [Instruction.maskSlot,Resources.exec,replaced,ne_comm]
  | _ => simp_all [Instruction.maskSlot,Resources.exec,replaced,ne_comm]

theorem initial_retains : Resources.zero.Retains := by simp [Resources.zero,Resources.Retains]

/-- A reservation may overwrite an unentered assignment. An unwinding worker
    can remove its own slot loans and leave foreign work unreported. -/
def AtInstruction (r : Resources) : Instruction → Prop
  | .openSlot s => r.mask s=0
  | .reserve s w => s ∉ r.frames w
  | .capture s w => r.active w=.pending s
  | .restore .. => True
  | .release s w => (r.frames w).head?=some s
  | .abandon taken w => ∀ s, taken=some s → s.1=w

private theorem after_set (old : BitVec 64) (v w : Worker) (e : Event) :
    ((Action.book v e).updateBits old).getLsbD w.val = (old.getLsbD w.val || decide (w=v)) := by
  simp only [Action.updateBits,BitVec.getLsbD_or,Bitboard.squareMask_bit,Fin.val_inj]

private theorem after_reset (old : BitVec 64) (v w : Worker) (e : Event) :
    ((Action.finish v e).updateBits old).getLsbD w.val = (old.getLsbD w.val && !decide (w=v)) := by
  simp only [Action.updateBits,BitVec.getLsbD_and,BitVec.getLsbD_not,Bitboard.squareMask_bit,
    w.isLt,decide_true,Bool.true_and,Fin.val_inj]

theorem Resources.preserves (r : Resources) (h : r.Retains) (e : Event) (op : Instruction)
    (checked : AtInstruction r op) : (r.exec e op).Retains := by
  obtain ⟨pending,live,distinct⟩ := h
  cases op with
  | openSlot target =>
    have zero : r.mask target = 0 := checked
    have missing (w : Worker) : target ∉ r.frames w := by
      intro mem;have on:=live _ _ mem;rw [zero] at on;simp at on
    refine ⟨?_,?_,distinct⟩
    · intro s w fresh
      by_cases same : w=target.1
      · subst w
        have key : target=s := by simpa [Resources.exec,replaced] using fresh
        subst s
        exact ⟨by simp only [Resources.exec,replaced,ite_true,Bitboard.squareMask_bit,decide_true],missing _⟩
      · have old : r.active w=.pending s := by simpa [Resources.exec,replaced,same] using fresh
        obtain ⟨on,free⟩ := pending _ _ old
        have other : s ≠ target := by intro eq;subst s;rw [zero] at on;simp at on
        exact ⟨by simpa [Resources.exec,replaced,other] using on,free⟩
    · intro s w mem
      have old:=live s w mem
      have other : s ≠ target := by intro eq;subst s;exact missing w mem
      simpa [Resources.exec,replaced,other] using old
  | reserve target who =>
    have bits (s : Slot) (w : Worker) (old : (r.mask s).getLsbD w.val=true) :
        ((r.exec e (.reserve target who)).mask s).getLsbD w.val=true := by
      by_cases same : s=target
      · subst s;change (replaced r.mask _ _ target).getLsbD _=true;rw [replaced,ite_eq_left rfl,after_set,old];rfl
      · simpa [Resources.exec,replaced,same] using old
    refine ⟨?_,fun s w mem => bits s w (live _ _ mem),distinct⟩
    intro s w yes
    by_cases same : w=who
    · subst w
      have slot : target=s := by simpa [Resources.exec,replaced] using yes
      subst s
      refine ⟨?_,checked⟩
      change (replaced r.mask _ _ target).getLsbD _=true
      rw [replaced,ite_eq_left rfl,after_set];simp
    · have old : r.active w=.pending s := by simpa [Resources.exec,replaced,same] using yes
      exact ⟨bits _ _ (pending _ _ old).1,(pending _ _ old).2⟩
  | capture target who =>
    obtain ⟨on,free⟩:=pending target who checked
    refine ⟨?_,?_,?_⟩
    · intro s w yes
      by_cases same : w=who
      · subst w;simp [Resources.exec,replaced] at yes
      · simpa [Resources.exec,replaced,same] using pending s w (by simpa [Resources.exec,replaced,same] using yes)
    · intro s w mem
      by_cases same : w=who
      · subst w
        rcases (by simpa [Resources.exec,replaced] using mem : s=target ∨ s∈r.frames who) with eq|old
        · subst s;exact on
        · exact live _ _ old
      · exact live _ _ (by simpa [Resources.exec,replaced,same] using mem)
    · intro w;by_cases same : w=who
      · subst w;simpa [Resources.exec,replaced] using List.nodup_cons.mpr ⟨free,distinct who⟩
      · simpa [Resources.exec,replaced,same] using distinct w
  | restore parent who =>
    refine ⟨?_,live,distinct⟩
    intro s w yes;by_cases same : w=who
    · subst w;cases parent <;> simp [Resources.exec,replaced] at yes
    · exact pending _ _ (by simpa [Resources.exec,replaced,same] using yes)
  | release target who =>
    have head : (r.frames who).head? = some target := checked
    obtain ⟨tail,old⟩ : ∃ tail,r.frames who=target::tail := List.head?_eq_some_iff.mp head
    have gone : target ∉ tail := (List.nodup_cons.mp (old ▸ distinct who)).1
    have bits (s : Slot) (w : Worker) (oldBit : (r.mask s).getLsbD w.val=true) (ok : s ≠ target ∨ w ≠ who) :
        ((r.exec e (.release target who)).mask s).getLsbD w.val=true := by
      by_cases same : s=target
      · subst s
        have diff:=ok.resolve_left (fun ne => ne rfl)
        change (replaced r.mask _ _ target).getLsbD _=true
        rw [replaced,ite_eq_left rfl,after_reset,oldBit];simp [diff]
      · simpa [Resources.exec,replaced,same] using oldBit
    refine ⟨?_,?_,?_⟩
    · intro s w yes
      by_cases same : w=who
      · subst w;simp [Resources.exec,replaced] at yes
      · have oldPending:=pending s w (by simpa [Resources.exec,replaced,same] using yes)
        exact ⟨bits _ _ oldPending.1 (Or.inr same),by simpa [Resources.exec,replaced,same] using oldPending.2⟩
    · intro s w mem
      by_cases same : w=who
      · subst w
        have inside : s ∈ tail := by simpa [Resources.exec,replaced,old] using mem
        have other : s ≠ target := fun eq => gone (eq ▸ inside)
        exact bits _ _ (live _ _ (old ▸ List.mem_cons_of_mem _ inside)) (Or.inl other)
      · exact bits _ _ (live _ _ (by simpa [Resources.exec,replaced,same] using mem)) (Or.inr same)
    · intro w;by_cases same : w=who
      · subst w;simpa [Resources.exec,replaced,old] using (List.nodup_cons.mp (old ▸ distinct who)).2
      · simpa [Resources.exec,replaced,same] using distinct w
  | abandon taken who =>
    have bits (s : Slot) (w : Worker) (on : (r.mask s).getLsbD w.val=true)
        (safe : w ≠ who ∨ s.1 ≠ who) :
        ((r.exec e (.abandon taken who)).mask s).getLsbD w.val=true := by
      cases taken with
      | none => exact on
      | some target =>
        have own : target.1=who := checked _ rfl
        by_cases same : s=target
        · subst s
          have diff:=safe.resolve_right (fun bad => bad own)
          change (replaced r.mask _ _ target).getLsbD w.val=true
          rw [replaced,ite_eq_left rfl,after_reset,on];simp [diff]
        · simpa [Resources.exec,replaced,same] using on
    refine ⟨?_,?_,?_⟩
    · intro s w yes
      by_cases same : w=who
      · subst w;simp [Resources.exec,replaced] at yes
      · obtain ⟨on,absent⟩:=pending s w (by simpa [Resources.exec,replaced,same] using yes)
        exact ⟨bits s w on (Or.inl same),by simpa [Resources.exec,replaced,same] using absent⟩
    · intro s w retained
      by_cases same : w=who
      · subst w
        have fact : s ∈ r.frames who ∧ s.1 ≠ who := by simpa [Resources.exec,replaced] using retained
        exact bits s who (live _ _ fact.1) (Or.inr fact.2)
      · exact bits s w (live s w (by simpa [Resources.exec,replaced,same] using retained)) (Or.inl same)
    · intro w;by_cases same : w=who
      · subst w
        simp only [Resources.exec,replaced,ite_true]
        exact (distinct who).filter _
      · simpa [Resources.exec,replaced,same] using distinct w

end NeuralReversi.PvsParallel.SlotFrameMir
