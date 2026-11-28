module
public import NeuralReversi.ScaledScore.DisplayControl.Progress

@[expose] public section

namespace NeuralReversi.ScaledScore.DisplayControl
open NeuralReversi.ScaledScore (displayChunks)

def ControlContract : Prop :=
  ∀ (Frame Receiver Ref Payload Stop Context Local World Child Event : Type)
    (oracle : Oracle Frame Receiver Ref Payload Stop Context Local World Child Event)
    (a b : Configuration (Frame:=Frame) (Receiver:=Receiver) (Ref:=Ref)
      (Payload:=Payload) (Stop:=Stop) (Context:=Context) (Local:=Local) (World:=World)) label,
    Step oracle a label b ↔ SpecStep oracle a label b

theorem controlContract_proven : ControlContract := by
  intro Frame Receiver Ref Payload Stop Context Local World Child Event oracle a b label
  exact strong_step_equivalence oracle

def TraceContract : Prop :=
  ∀ (Frame Receiver Ref Payload Stop Context Local World Child Event : Type)
    (oracle : Oracle Frame Receiver Ref Payload Stop Context Local World Child Event)
    (scope : Scope Frame Receiver) (world : World) labels
    (last : Configuration (Frame:=Frame) (Receiver:=Receiver) (Ref:=Ref)
      (Payload:=Payload) (Stop:=Stop) (Context:=Context) (Local:=Local) (World:=World)),
    Trace oracle (initialState scope,world) labels last →
      History scope (displayChunks scope.raw) (projectControl labels) last.1 ∧
      (∃ rest, (entries (projectControl labels)).map Invocation.text ++ rest = displayChunks scope.raw) ∧
      (entries (projectControl labels)).length ≤ 4 ∧ stateScope last.1 = scope

theorem traceContract_proven : TraceContract := by
  intro Frame Receiver Ref Payload Stop Context Local World Child Event oracle scope world labels last trace
  have h := initial_history oracle scope world trace
  exact ⟨h,history_texts h,outer_call_bound oracle scope world trace,history_scope h⟩

def ProgressContract : Prop :=
  ∀ (Frame Receiver Ref Payload Stop Context Local World Child Event : Type)
    (oracle : Oracle Frame Receiver Ref Payload Stop Context Local World Child Event),
    (∀ (scope : Scope Frame Receiver) (world : World) labels
      (last : Configuration (Frame:=Frame) (Receiver:=Receiver) (Ref:=Ref)
        (Payload:=Payload) (Stop:=Stop) (Context:=Context) (Local:=Local) (World:=World)),
      Trace oracle (initialState scope,world) labels last → (labels.map charge).sum ≤ 9) ∧
    (∀ (state : ModelState (Frame:=Frame) (Receiver:=Receiver) (Ref:=Ref)
        (Payload:=Payload) (Stop:=Stop) (Context:=Context) (Local:=Local)) (world : World),
      sourceReady state = true → ∃ action next,
        administrativeNext oracle.initial state world = some (action,next) ∧
        Step oracle (state,world) (.control action) (next,world) ∧ rank next < rank state) ∧
    (∀ scope outcome world label next, ¬ Step oracle (.terminal scope outcome,world) label next) ∧
    (∀ (states : Nat → Configuration (Frame:=Frame) (Receiver:=Receiver) (Ref:=Ref)
        (Payload:=Payload) (Stop:=Stop) (Context:=Context) (Local:=Local) (World:=World))
        (labels : Nat → Label Frame Receiver Ref Payload Stop Child Event),
      (∀ n, Step oracle (states n) (labels n) (states (n+1))) →
        ∃ first, ∀ n, first ≤ n → charge (labels n) = 0)

theorem progressContract_proven : ProgressContract := by
  intro Frame Receiver Ref Payload Stop Context Local World Child Event oracle
  exact ⟨fun scope world _ _ trace => charged_bound oracle scope world trace,
    source_progress oracle,terminal_no_step oracle,eventually_zero_charge oracle⟩

end NeuralReversi.ScaledScore.DisplayControl
