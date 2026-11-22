import NeuralReversi.ScaledScore.DisplayControl.Contracts

namespace NeuralReversi.ScaledScore.DisplayControl.Witnesses

inductive StopKind where
  | exit (code : Int)
  | abort

abbrev DemoOracle := Oracle Unit Unit Unit Unit StopKind Unit Unit Unit Unit Unit
abbrev DemoScope := Scope Unit Unit
abbrev DemoInv := Invocation Unit Unit Unit
abbrev DemoState := State Unit Unit Unit Unit StopKind Unit Unit
abbrev DemoLabel := Label Unit Unit Unit Unit StopKind Unit Unit

def scope : DemoScope := ⟨(),(),0#32,{}⟩
def invocation (address : BitVec 64) : DemoInv := ⟨scope,0,"0.",⟨(),address⟩⟩
def pending (address : BitVec 64) : DemoState := .external (invocation address) ["00"] () ()

def quietOracle (looping : Bool) (address : BitVec 64) : DemoOracle where
  initial := fun _ _ _ _ => (⟨(),address⟩,(),())
  internal := fun _ _ _ _ _ _ _ => looping = true
  complete := fun _ _ _ _ _ _ _ => False
  environment := fun _ _ _ _ => False
  stopping := fun _ _ _ _ _ => False

theorem reaches_pending (looping : Bool) (address : BitVec 64) :
    Trace (quietOracle looping address) (initialState scope,())
      [.control (.enter (invocation address))] (pending address,()) := by
  apply Trace.cons
  · apply Step.administrative
    rfl
  · exact .refl _

theorem blocking_admitted (address : BitVec 64) (label : DemoLabel) next :
    ¬ Step (quietOracle false address) (pending address,()) label next := by
  intro h
  cases h with
  | administrative state world action next h => simp [pending,administrativeNext] at h
  | complete inv ctx loc rest world reply event world' h => exact h
  | internal inv ctx loc rest world event loc' world' h => cases h
  | environment state world event world' live h => exact h
  | environmentStop state world event reason world' live h => exact h

theorem divergence_admitted (address : BitVec 64) :
    ∀ n : Nat, Step (quietOracle true address) ((fun _ : Nat => (pending address,())) n)
      ((fun _ : Nat => (Label.callback (invocation address) (.effect ()) : DemoLabel)) n)
      ((fun _ : Nat => (pending address,())) (n+1)) := by
  intro _
  exact .internal _ _ _ _ _ _ _ _ rfl

theorem reentrant_child_admitted (address : BitVec 64) :
    Step (quietOracle true address) (pending address,())
      (.callback (invocation address) (.child () ())) (pending address,()) ∧
    projectOne ((Label.callback (invocation address) (.child () ()) : DemoLabel)) = [] := by
  exact ⟨.internal _ _ _ _ _ _ _ _ rfl,rfl⟩

def okOracle (address : BitVec 64) : DemoOracle where
  initial := fun _ _ _ _ => (⟨(),address⟩,(),())
  internal := fun _ _ _ _ _ _ _ => False
  complete := fun _ _ _ _ reply _ _ => reply = .ok
  environment := fun _ _ _ _ => False
  stopping := fun _ _ _ _ _ => False

def okLabels (s : DemoScope) (address : BitVec 64) : Nat → List String → List DemoLabel
  | _, [] => [.control (.returnOk s)]
  | ordinal, chunk::rest =>
    let inv : DemoInv := ⟨s,ordinal,chunk,⟨(),address⟩⟩
    .control (.enter inv) :: .completion inv .ok () :: okLabels s address (ordinal+1) rest

theorem all_ok_execution (s : DemoScope) (address : BitVec 64) (ordinal : Nat) (chunks : List String) :
    Trace (okOracle address) (.ready s ordinal chunks,()) (okLabels s address ordinal chunks)
      (.terminal s .ok,()) := by
  induction chunks generalizing ordinal with
  | nil =>
    exact .cons (.administrative _ _ _ _ rfl) (.refl _)
  | cons chunk rest ih =>
    apply Trace.cons
    · apply Step.administrative
      rfl
    · apply Trace.cons
      · exact Step.complete _ () () _ () .ok () () rfl
      · exact ih (ordinal+1)

theorem complete_display_admitted (s : DemoScope) (address : BitVec 64) :
    Trace (okOracle address) (initialState s,())
      (okLabels s address 0 (NeuralReversi.ScaledScore.displayChunks s.raw)) (.terminal s .ok,()) :=
  all_ok_execution s address 0 _

theorem repeated_address_admitted (address : BitVec 64) :
    Trace (okOracle address) (initialState scope,()) (okLabels scope address 0 ["0.","00"])
      (.terminal scope .ok,()) ∧
    (entries (projectControl (okLabels scope address 0 ["0.","00"]))).map
      (fun inv => inv.reference.address) = [address,address] := by
  exact ⟨complete_display_admitted scope address,rfl⟩

def pointerOracle : DemoOracle where
  initial := fun _ _ _ _ => (⟨(),0x1000#64⟩,(),())
  internal := fun _ _ _ _ _ _ _ => False
  complete := fun inv _ _ _ reply _ _ =>
    if inv.reference.address = 0x1000#64 then reply = .ok else reply = .error
  environment := fun _ _ _ _ => False
  stopping := fun _ _ _ _ _ => False

theorem pointer_sensitive_replies :
    pointerOracle.complete (invocation 0x1000#64) () () () .ok () () ∧
    pointerOracle.complete (invocation 0x1001#64) () () () .error () () ∧
    ¬ pointerOracle.complete (invocation 0x1001#64) () () () .ok () () := by
  simp [pointerOracle,invocation]

def stopOracle (reason : StopKind) : DemoOracle where
  initial := fun _ _ _ _ => (⟨(),0x1000#64⟩,(),())
  internal := fun _ _ _ _ _ _ _ => False
  complete := fun _ _ _ _ reply _ _ => reply = .stopped reason
  environment := fun _ _ _ _ => False
  stopping := fun _ _ _ _ _ => False

theorem exit_admitted :
    Step (stopOracle (.exit 17)) (pending 0x1000#64,())
      (.completion (invocation 0x1000#64) (.stopped (.exit 17)) ())
      (.terminal scope (.stopped (.exit 17)),()) :=
  .complete _ _ _ _ _ _ _ _ rfl

theorem abort_admitted :
    Step (stopOracle .abort) (pending 0x1000#64,())
      (.completion (invocation 0x1000#64) (.stopped .abort) ())
      (.terminal scope (.stopped .abort),()) :=
  .complete _ _ _ _ _ _ _ _ rfl

def ResumeCorrect (candidate : DemoInv → List String → Reply Unit StopKind → DemoState) : Prop :=
  ∀ inv rest reply, Continuation inv rest reply (candidate inv rest reply)

theorem original_resume_correct : ResumeCorrect resume := by
  intro inv rest reply
  exact (resume_exact inv rest reply _).mp rfl

def constantOk (inv : DemoInv) (_ : List String) (_ : Reply Unit StopKind) : DemoState :=
  .terminal inv.scope .ok
def alwaysError (inv : DemoInv) (_ : List String) (_ : Reply Unit StopKind) : DemoState :=
  .terminal inv.scope .error
def swallowError (inv : DemoInv) (rest : List String) (reply : Reply Unit StopKind) : DemoState :=
  match reply with
  | .error => .ready inv.scope (inv.ordinal+1) rest
  | _ => resume inv rest reply
def swallowUnwind (inv : DemoInv) (rest : List String) (reply : Reply Unit StopKind) : DemoState :=
  match reply with
  | .unwind _ => .ready inv.scope (inv.ordinal+1) rest
  | _ => resume inv rest reply
def continueAfterStop (inv : DemoInv) (rest : List String) (reply : Reply Unit StopKind) : DemoState :=
  match reply with
  | .stopped _ => .ready inv.scope (inv.ordinal+1) rest
  | _ => resume inv rest reply
def discardRemainder (inv : DemoInv) (_ : List String) (reply : Reply Unit StopKind) : DemoState :=
  resume inv [] reply

theorem constant_ok_rejected : ¬ ResumeCorrect constantOk := by
  intro h
  have impossible := h (invocation 0x1000#64) ["00"] .error
  cases impossible

theorem always_error_rejected : ¬ ResumeCorrect alwaysError := by
  intro h
  have impossible := h (invocation 0x1000#64) ["00"] .ok
  cases impossible

theorem swallowed_error_rejected : ¬ ResumeCorrect swallowError := by
  intro h
  have impossible := h (invocation 0x1000#64) ["00"] .error
  cases impossible

theorem swallowed_unwind_rejected : ¬ ResumeCorrect swallowUnwind := by
  intro h
  have impossible := h (invocation 0x1000#64) ["00"] (.unwind ())
  cases impossible

theorem continued_after_stop_rejected : ¬ ResumeCorrect continueAfterStop := by
  intro h
  have impossible := h (invocation 0x1000#64) ["00"] (.stopped .abort)
  cases impossible

theorem dropped_chunk_rejected : ¬ ResumeCorrect discardRemainder := by
  intro h
  have impossible := h (invocation 0x1000#64) ["00"] .ok
  cases impossible

abbrev DemoConfiguration := DemoState × Unit
abbrev DemoStep := DemoConfiguration → DemoLabel → DemoConfiguration → Prop

def StepCorrect (candidate : DemoStep) : Prop :=
  ∀ a label b, candidate a label b ↔ SpecStep (okOracle 0x1000#64) a label b

def constantSuccessStep (_ : DemoConfiguration) (_ : DemoLabel) (last : DemoConfiguration) : Prop :=
  last = (.terminal scope .ok,())
def constantErrorStep (_ : DemoConfiguration) (_ : DemoLabel) (last : DemoConfiguration) : Prop :=
  last = (.terminal scope .error,())
def noSteps (_ : DemoConfiguration) (_ : DemoLabel) (_ : DemoConfiguration) : Prop := False

theorem original_step_correct : StepCorrect (Step (okOracle 0x1000#64)) := by
  intro a label b
  exact strong_step_equivalence _

theorem constant_success_step_rejected : ¬ StepCorrect constantSuccessStep := by
  intro h
  have wrong := (h (initialState scope,()) (.control (.returnOk scope)) (.terminal scope .ok,())).mp rfl
  change SpecStep (okOracle 0x1000#64) (.ready scope 0 ["0.","00"],())
    (.control (.returnOk scope)) (.terminal scope .ok,()) at wrong
  cases wrong with
  | administrative state world action next impossible => cases impossible

theorem constant_error_step_rejected : ¬ StepCorrect constantErrorStep := by
  intro h
  have wrong := (h (initialState scope,()) (.control (.returnOk scope)) (.terminal scope .error,())).mp rfl
  change SpecStep (okOracle 0x1000#64) (.ready scope 0 ["0.","00"],())
    (.control (.returnOk scope)) (.terminal scope .error,()) at wrong
  cases wrong with
  | administrative state world action next impossible => cases impossible

theorem source_deadlock_rejected : ¬ StepCorrect noSteps := by
  intro h
  have available : SpecStep (okOracle 0x1000#64) (initialState scope,())
      (.control (.enter (invocation 0x1000#64))) (pending 0x1000#64,()) := by
    apply SpecStep.administrative
    apply (administrative_exact _ _ _ _ _).mp
    rfl
  exact (h _ _ _).mpr available

end NeuralReversi.ScaledScore.DisplayControl.Witnesses
