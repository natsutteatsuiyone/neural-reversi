module
public import NeuralReversi.ScaledScore.DisplayControl.Consequences

@[expose] public section

namespace NeuralReversi.ScaledScore.DisplayControl
variable {Frame Receiver Ref Payload Stop Context Local World Child Event : Type}
variable (oracle : Oracle Frame Receiver Ref Payload Stop Context Local World Child Event)

def spent (labels : Nat → Label Frame Receiver Ref Payload Stop Child Event) : Nat → Nat
  | 0 => 0
  | n+1 => spent labels n + charge (labels n)

theorem infinite_prefix_bound
    (states : Nat → Configuration (Frame:=Frame) (Receiver:=Receiver) (Ref:=Ref)
      (Payload:=Payload) (Stop:=Stop) (Context:=Context) (Local:=Local) (World:=World))
    (labels : Nat → Label Frame Receiver Ref Payload Stop Child Event)
    (run : ∀ n, Step oracle (states n) (labels n) (states (n+1))) :
    ∀ n, rank (states n).1 + spent labels n ≤ rank (states 0).1 := by
  intro n
  induction n with
  | zero => simp [spent]
  | succ n ih =>
    have hs := step_rank oracle (run n)
    simp only [spent]
    omega

private theorem bounded_eventually_constant (f : Nat → Nat)
    (monotone : ∀ n m, n ≤ m → f n ≤ f m) :
    ∀ bound, (∀ n, f n ≤ bound) → ∃ first, ∀ n, first ≤ n → f n = f first := by
  intro bound
  induction bound with
  | zero =>
    intro bounded
    refine ⟨0,?_⟩
    intro n _
    have hn := bounded n
    have hz := bounded 0
    omega
  | succ bound ih =>
    intro bounded
    classical
    by_cases attained : ∃ n, f n = bound+1
    · obtain ⟨first,exactBound⟩ := attained
      refine ⟨first,?_⟩
      intro n after
      have lower := monotone first n after
      have upper := bounded n
      omega
    · apply ih
      intro n
      have upper := bounded n
      have unequal : f n ≠ bound+1 := fun h => attained ⟨n,h⟩
      omega

theorem eventually_zero_charge
    (states : Nat → Configuration (Frame:=Frame) (Receiver:=Receiver) (Ref:=Ref)
      (Payload:=Payload) (Stop:=Stop) (Context:=Context) (Local:=Local) (World:=World))
    (labels : Nat → Label Frame Receiver Ref Payload Stop Child Event)
    (run : ∀ n, Step oracle (states n) (labels n) (states (n+1))) :
    ∃ first, ∀ n, first ≤ n → charge (labels n) = 0 := by
  have monotone : ∀ n m, n ≤ m → spent labels n ≤ spent labels m := by
    intro n m after
    induction after with
    | refl => exact Nat.le_refl _
    | step _ ih => simp only [spent]; omega
  have bounded : ∀ n, spent labels n ≤ rank (states 0).1 := by
    intro n
    have h := infinite_prefix_bound oracle states labels run n
    omega
  obtain ⟨first,stable⟩ := bounded_eventually_constant (spent labels) monotone _ bounded
  refine ⟨first,?_⟩
  intro n after
  have here := stable n after
  have next := stable (n+1) (by omega)
  simp only [spent] at next
  omega

end NeuralReversi.ScaledScore.DisplayControl
