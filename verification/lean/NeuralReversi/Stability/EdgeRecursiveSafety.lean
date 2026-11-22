import NeuralReversi.Stability.EdgeContracts
import NeuralReversi.Stability.EdgeMoveProgress

namespace NeuralReversi.Stability.Edge

def Step (a b : State) : Prop :=
  ∃ x : Fin 8, emptyAt a.1 a.2 x = true ∧
    (b = firstChild a.1 a.2 x ∨ b = secondChild a.1 a.2 x)

inductive Trace : Nat → State → State → Prop
  | refl (a : State) : Trace 0 a a
  | step {n : Nat} {a b c : State} : Step a b → Trace n b c → Trace (n + 1) a c

theorem stableFuel_subset (fuel : Nat) (p o : Byte) : Subset (stableFuel fuel p o) p := by
  cases fuel with
  | zero => exact subset_refl p
  | succ fuel => exact findWith_subset _ _ _

theorem stableFuel_step (fuel : Nat) {a b : State} (h : Step a b) :
    Subset (stableFuel (fuel + 1) a.1 a.2) (stableFuel fuel b.1 b.2) := by
  obtain ⟨x, empty, hb⟩ := h
  have child := findWith_children (stableFuel fuel) a.1 a.2 x empty
  rcases hb with hb | hb
  · simpa only [hb, stableFuel] using child.1
  · simpa only [hb, stableFuel] using child.2

theorem trace_bounded_safe {n : Nat} {a b : State} (h : Trace n a b)
    (fuel : Nat) (bound : n ≤ fuel) : Subset (stableFuel fuel a.1 a.2) b.1 := by
  induction h generalizing fuel with
  | refl a => exact stableFuel_subset fuel a.1 a.2
  | @step n a b c hab hbc ih =>
    cases fuel with
    | zero => omega
    | succ fuel => exact subset_trans (stableFuel_step fuel hab) (ih fuel (by omega))

theorem step_progress {a b : State} (valid : a.1 &&& a.2 = 0) (h : Step a b) :
    b.1 &&& b.2 = 0 ∧ (b.1 ||| b.2).cpop.toNat = (a.1 ||| a.2).cpop.toNat + 1 := by
  obtain ⟨x, empty, hb⟩ := h
  rcases hb with hb | hb
  · subst b
    exact applyEdgeMove_progress a.1 a.2 x valid empty
  · subst b
    have valid' : a.2 &&& a.1 = 0 := by rw [BitVec.and_comm]; exact valid
    have empty' : emptyAt a.2 a.1 x = true := by simpa only [emptyAt, BitVec.or_comm] using empty
    have h := applyEdgeMove_progress a.2 a.1 x valid' empty'
    simpa only [secondChild, BitVec.and_comm, BitVec.or_comm] using h

theorem trace_progress {n : Nat} {a b : State} (valid : a.1 &&& a.2 = 0) (h : Trace n a b) :
    b.1 &&& b.2 = 0 ∧ (b.1 ||| b.2).cpop.toNat = (a.1 ||| a.2).cpop.toNat + n := by
  induction h with
  | refl a => exact ⟨valid, by omega⟩
  | @step n a b c hab hbc ih =>
    have one := step_progress valid hab
    have rest := ih one.1
    exact ⟨rest.1, by omega⟩

theorem trace_length_le {n : Nat} {a b : State} (valid : a.1 &&& a.2 = 0) (h : Trace n a b) :
    n ≤ 8 := by
  have progress := trace_progress valid h
  have upper := BitVec.toNat_cpop_le (b.1 ||| b.2)
  omega

theorem recursiveStable_trace_safe {n : Nat} {a b : State}
    (valid : a.1 &&& a.2 = 0) (h : Trace n a b) : Subset (recursiveStable a.1 a.2) b.1 := by
  exact trace_bounded_safe h 8 (trace_length_le valid h)

theorem recursiveStable_never_flips {n : Nat} {a b : State}
    (valid : a.1 &&& a.2 = 0) (h : Trace n a b) (i : Fin 8)
    (marked : (recursiveStable a.1 a.2).getLsbD i.val = true) :
    b.1.getLsbD i.val = true := by
  have safe := recursiveStable_trace_safe valid h
  have bit := congrArg (fun x : Byte => x.getLsbD i.val) safe
  simpa only [BitVec.getLsbD_and, marked, Bool.true_and] using bit

end NeuralReversi.Stability.Edge
