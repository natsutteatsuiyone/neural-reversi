import NeuralReversi.PvsParallel.SlotFrameMir.Trace

namespace NeuralReversi.PvsParallel.SlotFrameMir

open Threading
open AtomicWord (Event Read Write ReadWindow)
open CutoffVisibility (History HappensBefore)
open SplitCompletion (Action Activation)

private theorem other_preserves_zero (a : Action) (u : Worker) (old : BitVec 64)
    (zero : old.getLsbD u.val = false) (other : ∀ e, a ≠ .book u e) :
    (a.updateBits old).getLsbD u.val = false := by
  cases a with
  | book w e =>
    have ne : u ≠ w := fun eq => other e (eq ▸ rfl)
    simp only [Action.updateBits,BitVec.getLsbD_or,Bitboard.squareMask_bit,zero,Fin.val_inj,ne,decide_false,Bool.false_or]
  | finish w e => simp only [Action.updateBits,BitVec.getLsbD_and,zero,Bool.false_and]

private theorem still_zero (act : Activation ws) (u : Worker) (other : u ≠ act.owner)
    (n : Nat) (bound : n ≤ act.actions.length)
    (absent : ∀ i : Fin act.actions.length, i.val < n → ∀ e, act.actions.get i ≠ .book u e) :
    (ws.get ⟨act.start.val+n,by have := act.endInBounds; omega⟩).value.getLsbD u.val = false := by
  induction n with
  | zero =>
    change (ws.get act.start).value.getLsbD u.val = false
    rw [act.ownerBit]
    simp only [SplitCompletion.initial,Bitboard.squareMask_bit,Fin.val_inj,other,decide_false]
  | succ n ih =>
    let i : Fin act.actions.length := ⟨n,by omega⟩
    have was := ih (by omega) (fun j lt => absent j (by omega))
    have zero := other_preserves_zero (act.actions.get i) u _ was (absent i (by dsimp [i]; omega))
    have value := (act.updates i).2.2.2
    rw [← value] at zero
    simpa [i,Nat.add_assoc] using zero

private theorem test_implies_bit {bits : BitVec 64} {u : Worker} (h : AtomicBitSet.test bits u = true) :
    bits.getLsbD u.val = true := by
  have low : (bits >>> u.val) &&& 1 ≠ 0 := bne_iff_ne.mp h
  cases actual : bits.getLsbD u.val
  · exfalso; apply low; apply BitVec.zero_iff_eq_false.mpr
    intro i
    by_cases eq : i=0
    · subst i; simpa using actual
    · simp [BitVec.getLsbD_and,BitVec.getLsbD_one,eq]
  · rfl

/-- A successful test for someone other than the bank owner has an actual earlier
    registration of that worker in this activation. AtomicBitSet.test reads Relaxed. -/
theorem prior_book (hist : History) (ws : List Write) (act : Activation ws) (u : Worker)
    (other : u ≠ act.owner) (helpful : Read ws)
    (range : ReadWindow helpful (HappensBefore hist) act.start (act.start.val + act.actions.length))
    (ok : WorkerFrameMir.canJoin searching failed true deepest (AtomicBitSet.test (ws.get helpful.source).value u) = true) :
    ∃ i : Fin act.actions.length, ∃ e, act.actions.get i = .book u e ∧
      (act.updateIndex i).val ≤ helpful.source.val := by
  have grant : AtomicBitSet.test (ws.get helpful.source).value u = true :=
    (Bool.and_eq_true_iff.mp (Bool.and_eq_true_iff.mp ok).2).2
  have bounds := range.bounds
  let n := helpful.source.val-act.start.val
  have bound : n ≤ act.actions.length := by dsimp [n]; omega
  by_cases prior : ∃ i : Fin act.actions.length, i.val < n ∧ ∃ e, act.actions.get i = .book u e
  · obtain ⟨i,lt,e,eq⟩ := prior
    refine ⟨i,e,eq,?_⟩
    change act.start.val + i.val + 1 ≤ helpful.source.val
    dsimp [n] at lt; omega
  · have no := still_zero act u other n bound (fun i lt e eq => prior ⟨i,lt,e,eq⟩)
    have eq : act.start.val + n = helpful.source.val := by dsimp [n]; omega
    have falseBit : (ws.get helpful.source).value.getLsbD u.val = false := by simpa [eq] using no
    rw [test_implies_bit grant] at falseBit; contradiction

/-- The same slot mutex orders its masked modifications, including clear/set on reuse.
    The operation label is the physical MIR call and its bit argument. -/
structure MaskEvents (ws : List Write) (act : Activation ws) where
  tag : Fin ws.length → Option Action
  attached : ∀ i : Fin act.actions.length, tag (act.updateIndex i) = some (act.actions.get i)
  time : Event → Nat
  ordered : ∀ i j : Fin ws.length, i.val < j.val → time (ws.get i).event < time (ws.get j).event

theorem MaskEvents.earlier_duplicate (raw : MaskEvents ws act) (hist : History) (helpful : Read ws)
    (range : ReadWindow helpful (HappensBefore hist) act.start (act.start.val+act.actions.length))
    (other : u ≠ act.owner)
    (accepted : WorkerFrameMir.canJoin searching failed true deepest (AtomicBitSet.test (ws.get helpful.source).value u) = true)
    (idx : Fin ws.length) (posted : Event) (label : raw.tag idx = some (.book u posted))
    (marked : HappensBefore hist helpful.event (ws.get idx).event)
    (j : Fin act.actions.length) (before : raw.time (ws.get idx).event < raw.time (act.actions.get j).event) :
    ∃ first later : Fin act.actions.length, ∃ initial, act.actions.get first = .book u initial ∧
      act.actions.get later = .book u posted ∧ first.val < later.val ∧ later.val < j.val := by
  have prior := range.readWrite idx marked
  have bounds := range.bounds
  have ev : (ws.get (act.updateIndex j)).event = (act.actions.get j).event := (act.updates j).1
  have lim : idx.val < (act.updateIndex j).val := by
    by_cases eq : idx.val = (act.updateIndex j).val
    · have pos := Fin.ext eq; rw [pos,ev] at before; omega
    · by_cases lt : (act.updateIndex j).val < idx.val
      · have reverse := raw.ordered _ _ lt; rw [ev] at reverse; omega
      · omega
  change idx.val < act.start.val + j.val + 1 at lim
  let later : Fin act.actions.length := ⟨idx.val-act.start.val-1,by have := j.isLt; omega⟩
  have atStamp : act.updateIndex later = idx := Fin.ext (by change act.start.val + (idx.val-act.start.val-1)+1=idx.val; omega)
  have eq : act.actions.get later = .book u posted := Option.some.inj ((raw.attached later).symm.trans (atStamp ▸ label))
  obtain ⟨first,initial,booked,bound⟩ := prior_book hist ws act u other helpful range accepted
  refine ⟨first,later,initial,booked,eq,?_,?_⟩
  · change act.start.val + first.val + 1 ≤ helpful.source.val at bound
    dsimp [later]; omega
  · dsimp [later]; omega

end NeuralReversi.PvsParallel.SlotFrameMir
