import NeuralReversi.ScaledScore.DisplayControl.Contracts

namespace NeuralReversi.ScaledScore.DisplayControl.Replay
variable {Frame Receiver Ref Payload Stop Context Local World Child Event : Type}
variable (oracle : Oracle Frame Receiver Ref Payload Stop Context Local World Child Event)

abbrev Attempt := Option (Label Frame Receiver Ref Payload Stop Child Event ×
  Configuration (Frame:=Frame) (Receiver:=Receiver) (Ref:=Ref)
    (Payload:=Payload) (Stop:=Stop) (Context:=Context) (Local:=Local) (World:=World))

structure Checks where
  internal : ∀ inv ctx loc world event loc' world',
    Decidable (oracle.internal inv ctx loc world event loc' world')
  complete : ∀ inv ctx loc world reply event world',
    Decidable (oracle.complete inv ctx loc world reply event world')

def administrative (before : Configuration (Frame:=Frame) (Receiver:=Receiver) (Ref:=Ref)
    (Payload:=Payload) (Stop:=Stop) (Context:=Context) (Local:=Local) (World:=World)) :
    Attempt (Frame:=Frame) (Receiver:=Receiver) (Ref:=Ref) (Payload:=Payload)
      (Stop:=Stop) (Context:=Context) (Local:=Local) (World:=World) (Child:=Child) (Event:=Event) :=
  match administrativeNext oracle.initial before.1 before.2 with
  | none => none
  | some (action,next) => some (.control action,(next,before.2))

def selectedInternal (checks : Checks oracle)
    (before : Configuration (Frame:=Frame) (Receiver:=Receiver) (Ref:=Ref)
      (Payload:=Payload) (Stop:=Stop) (Context:=Context) (Local:=Local) (World:=World))
    (event : OpenEvent Child Event) (local' : Local) (world' : World) :
    Attempt (Frame:=Frame) (Receiver:=Receiver) (Ref:=Ref) (Payload:=Payload)
      (Stop:=Stop) (Context:=Context) (Local:=Local) (World:=World) (Child:=Child) (Event:=Event) :=
  match before with
  | (.external inv remaining ctx loc,world) =>
    letI := checks.internal inv ctx loc world event local' world'
    if oracle.internal inv ctx loc world event local' world' then
      some (.callback inv event,(.external inv remaining ctx local',world'))
    else none
  | _ => none

def selectedComplete (checks : Checks oracle)
    (before : Configuration (Frame:=Frame) (Receiver:=Receiver) (Ref:=Ref)
      (Payload:=Payload) (Stop:=Stop) (Context:=Context) (Local:=Local) (World:=World))
    (reply : Reply Payload Stop) (event : Event) (world' : World) :
    Attempt (Frame:=Frame) (Receiver:=Receiver) (Ref:=Ref) (Payload:=Payload)
      (Stop:=Stop) (Context:=Context) (Local:=Local) (World:=World) (Child:=Child) (Event:=Event) :=
  match before with
  | (.external inv remaining ctx loc,world) =>
    letI := checks.complete inv ctx loc world reply event world'
    if oracle.complete inv ctx loc world reply event world' then
      some (.completion inv reply event,(resume inv remaining reply,world'))
    else none
  | _ => none

theorem administrative_sound {before label after}
    (accepted : administrative oracle before = some (label,after)) : Step oracle before label after := by
  cases chosen : administrativeNext oracle.initial before.1 before.2 with
  | none => simp [administrative,chosen] at accepted
  | some pair =>
    rcases pair with ⟨action,next⟩
    simp only [administrative,chosen,Option.some.injEq,Prod.mk.injEq] at accepted
    rcases accepted with ⟨rfl,rfl⟩
    exact .administrative before.1 before.2 action next chosen

theorem selectedInternal_sound (checks : Checks oracle) {before event local' world' label after}
    (accepted : selectedInternal oracle checks before event local' world' = some (label,after)) :
    Step oracle before label after := by
  rcases before with ⟨state,world⟩
  cases state with
  | ready => simp [selectedInternal] at accepted
  | terminal => simp [selectedInternal] at accepted
  | external inv remaining ctx loc =>
    simp only [selectedInternal] at accepted
    split at accepted
    · rename_i allowed
      simp only [Option.some.injEq,Prod.mk.injEq] at accepted
      rcases accepted with ⟨rfl,rfl⟩
      exact .internal inv ctx loc remaining world event local' world' allowed
    · cases accepted

theorem selectedComplete_sound (checks : Checks oracle) {before reply event world' label after}
    (accepted : selectedComplete oracle checks before reply event world' = some (label,after)) :
    Step oracle before label after := by
  rcases before with ⟨state,world⟩
  cases state with
  | ready => simp [selectedComplete] at accepted
  | terminal => simp [selectedComplete] at accepted
  | external inv remaining ctx loc =>
    simp only [selectedComplete] at accepted
    split at accepted
    · rename_i allowed
      simp only [Option.some.injEq,Prod.mk.injEq] at accepted
      rcases accepted with ⟨rfl,rfl⟩
      exact .complete inv ctx loc remaining world reply event world' allowed
    · cases accepted

theorem selectedInternal_exact (checks : Checks oracle) (inv ctx loc remaining world event local' world') :
    selectedInternal oracle checks (.external inv remaining ctx loc,world) event local' world' =
      some (.callback inv event,(.external inv remaining ctx local',world')) ↔
      oracle.internal inv ctx loc world event local' world' := by
  by_cases h : oracle.internal inv ctx loc world event local' world' <;> simp [selectedInternal,h]

theorem selectedComplete_exact (checks : Checks oracle) (inv ctx loc remaining world reply event world') :
    selectedComplete oracle checks (.external inv remaining ctx loc,world) reply event world' =
      some (.completion inv reply event,(resume inv remaining reply,world')) ↔
      oracle.complete inv ctx loc world reply event world' := by
  by_cases h : oracle.complete inv ctx loc world reply event world' <;> simp [selectedComplete,h]

theorem selectedInternal_reject (checks : Checks oracle) (inv ctx loc remaining world event local' world')
    (invalid : ¬ oracle.internal inv ctx loc world event local' world') :
    selectedInternal oracle checks (.external inv remaining ctx loc,world) event local' world' = none := by
  simp [selectedInternal,invalid]

theorem selectedComplete_reject (checks : Checks oracle) (inv ctx loc remaining world reply event world')
    (invalid : ¬ oracle.complete inv ctx loc world reply event world') :
    selectedComplete oracle checks (.external inv remaining ctx loc,world) reply event world' = none := by
  simp [selectedComplete,invalid]

inductive Selection (Payload Stop Local World Child Event : Type) where
  | administrative
  | internal (event : OpenEvent Child Event) (local' : Local) (world' : World)
  | complete (reply : Reply Payload Stop) (event : Event) (world' : World)

def step (checks : Checks oracle)
    (before : Configuration (Frame:=Frame) (Receiver:=Receiver) (Ref:=Ref)
      (Payload:=Payload) (Stop:=Stop) (Context:=Context) (Local:=Local) (World:=World))
    (selection : Selection Payload Stop Local World Child Event) :
    Attempt (Frame:=Frame) (Receiver:=Receiver) (Ref:=Ref) (Payload:=Payload)
      (Stop:=Stop) (Context:=Context) (Local:=Local) (World:=World) (Child:=Child) (Event:=Event) :=
  match selection with
  | .administrative => administrative oracle before
  | .internal event local' world' => selectedInternal oracle checks before event local' world'
  | .complete reply event world' => selectedComplete oracle checks before reply event world'

theorem step_sound (checks : Checks oracle) {before selection label after}
    (accepted : step oracle checks before selection = some (label,after)) : Step oracle before label after := by
  cases selection with
  | administrative => exact administrative_sound oracle accepted
  | internal event local' world' => exact selectedInternal_sound oracle checks accepted
  | complete reply event world' => exact selectedComplete_sound oracle checks accepted

def run (checks : Checks oracle)
    (before : Configuration (Frame:=Frame) (Receiver:=Receiver) (Ref:=Ref)
      (Payload:=Payload) (Stop:=Stop) (Context:=Context) (Local:=Local) (World:=World)) :
    List (Selection Payload Stop Local World Child Event) →
    Option (List (Label Frame Receiver Ref Payload Stop Child Event) ×
      Configuration (Frame:=Frame) (Receiver:=Receiver) (Ref:=Ref)
        (Payload:=Payload) (Stop:=Stop) (Context:=Context) (Local:=Local) (World:=World))
  | [] => some ([],before)
  | choice::rest =>
    match step oracle checks before choice with
    | none => none
    | some (label,next) =>
      match run checks next rest with
      | none => none
      | some (labels,last) => some (label::labels,last)

theorem cons_accepted (checks : Checks oracle) {before choice rest label next labels after}
    (first : step oracle checks before choice = some (label,next))
    (tail : run oracle checks next rest = some (labels,after)) :
    run oracle checks before (choice::rest) = some (label::labels,after) := by
  simp only [run,first,tail]

theorem run_sound (checks : Checks oracle) {choices before labels after}
    (accepted : run oracle checks before choices = some (labels,after)) :
    Trace oracle before labels after := by
  induction choices generalizing before labels after with
  | nil =>
    simp only [run,Option.some.injEq,Prod.mk.injEq] at accepted
    rcases accepted with ⟨rfl,rfl⟩
    exact .refl _
  | cons choice rest ih =>
    cases first : step oracle checks before choice with
    | none => simp [run,first] at accepted
    | some pair =>
      rcases pair with ⟨label,next⟩
      cases tail : run oracle checks next rest with
      | none => simp [run,first,tail] at accepted
      | some pair =>
        rcases pair with ⟨later,last⟩
        simp only [run,first,tail,Option.some.injEq,Prod.mk.injEq] at accepted
        rcases accepted with ⟨rfl,rfl⟩
        exact .cons (step_sound oracle checks first) (ih tail)

end NeuralReversi.ScaledScore.DisplayControl.Replay