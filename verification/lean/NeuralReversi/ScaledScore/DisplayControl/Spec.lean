import NeuralReversi.ScaledScore.DisplayControl.Model

namespace NeuralReversi.ScaledScore.DisplayControl
variable {Frame Receiver Ref Payload Stop Context Local World Child Event : Type}

def Matches (scope : Scope Frame Receiver) (ordinal : Nat) (chunk : String)
    (inv : Invocation Frame Receiver Ref) : Prop :=
  inv.scope = scope ∧ inv.ordinal = ordinal ∧ inv.text = chunk

inductive Continuation : Invocation Frame Receiver Ref → List String → Reply Payload Stop →
    ModelState (Frame:=Frame) (Receiver:=Receiver) (Ref:=Ref) (Payload:=Payload)
      (Stop:=Stop) (Context:=Context) (Local:=Local) → Prop
  | next (inv rest) : Continuation inv rest .ok (.ready inv.scope (inv.ordinal+1) rest)
  | error (inv rest) : Continuation inv rest .error (.terminal inv.scope .error)
  | unwind (inv rest payload) : Continuation inv rest (.unwind payload) (.terminal inv.scope (.unwound payload))
  | stop (inv rest reason) : Continuation inv rest (.stopped reason) (.terminal inv.scope (.stopped reason))

inductive AdministrativeSpec
    (initial : Scope Frame Receiver → Nat → String → World → SliceRef Ref × Context × Local) :
    ModelState (Frame:=Frame) (Receiver:=Receiver) (Ref:=Ref) (Payload:=Payload)
      (Stop:=Stop) (Context:=Context) (Local:=Local) → World →
    Control Frame Receiver Ref Payload Stop →
    ModelState (Frame:=Frame) (Receiver:=Receiver) (Ref:=Ref) (Payload:=Payload)
      (Stop:=Stop) (Context:=Context) (Local:=Local) → Prop
  | done (scope ordinal world) : AdministrativeSpec initial (.ready scope ordinal []) world
      (.returnOk scope) (.terminal scope .ok)
  | call (scope ordinal chunk rest world reference context localState)
      (snapshot : initial scope ordinal chunk world = (reference,context,localState)) :
      AdministrativeSpec initial (.ready scope ordinal (chunk::rest)) world
        (.enter ⟨scope,ordinal,chunk,reference⟩)
        (.external ⟨scope,ordinal,chunk,reference⟩ rest context localState)

variable (oracle : Oracle Frame Receiver Ref Payload Stop Context Local World Child Event)

inductive SpecStep : Configuration (Frame:=Frame) (Receiver:=Receiver) (Ref:=Ref)
    (Payload:=Payload) (Stop:=Stop) (Context:=Context) (Local:=Local) (World:=World) →
    Label Frame Receiver Ref Payload Stop Child Event →
    Configuration (Frame:=Frame) (Receiver:=Receiver) (Ref:=Ref)
      (Payload:=Payload) (Stop:=Stop) (Context:=Context) (Local:=Local) (World:=World) → Prop
  | administrative (state world action next)
      (h : AdministrativeSpec oracle.initial state world action next) :
      SpecStep (state,world) (.control action) (next,world)
  | complete (inv ctx loc rest world reply event world' next)
      (external : oracle.complete inv ctx loc world reply event world')
      (continuation : Continuation inv rest reply next) :
      SpecStep (.external inv rest ctx loc,world) (.completion inv reply event) (next,world')
  | internal (inv ctx loc rest world event loc' world')
      (h : oracle.internal inv ctx loc world event loc' world') :
      SpecStep (.external inv rest ctx loc,world) (.callback inv event) (.external inv rest ctx loc',world')
  | environment (state world event world') (live : Live state)
      (h : oracle.environment state world event world') :
      SpecStep (state,world) (.environment (stateScope state) event) (state,world')
  | environmentStop (state world event reason world') (live : Live state)
      (h : oracle.stopping state world event reason world') :
      SpecStep (state,world) (.ambientStop (stateScope state) reason event)
        (.terminal (stateScope state) (.stopped reason),world')

def projectOne : Label Frame Receiver Ref Payload Stop Child Event →
    List (Control Frame Receiver Ref Payload Stop)
  | .control action => [action]
  | .completion inv reply _ => [.finish inv reply]
  | _ => []

def projectControl (labels : List (Label Frame Receiver Ref Payload Stop Child Event)) :
    List (Control Frame Receiver Ref Payload Stop) := labels.flatMap projectOne

inductive SuccessPrefix (scope : Scope Frame Receiver) : Nat → List String →
    List (Control Frame Receiver Ref Payload Stop) → Nat → List String → Prop
  | empty (ordinal chunks) : SuccessPrefix scope ordinal chunks [] ordinal chunks
  | next (ordinal chunk rest inv controls after remaining)
      (matching : Matches scope ordinal chunk inv)
      (tail : SuccessPrefix scope (ordinal+1) rest controls after remaining) :
      SuccessPrefix scope ordinal (chunk::rest)
        (.enter inv :: .finish inv .ok :: controls) after remaining

inductive LivePrefix (scope : Scope Frame Receiver) (chunks : List String) :
    List (Control Frame Receiver Ref Payload Stop) →
    ModelState (Frame:=Frame) (Receiver:=Receiver) (Ref:=Ref) (Payload:=Payload)
      (Stop:=Stop) (Context:=Context) (Local:=Local) → Prop
  | ready (controls ordinal remaining)
      (prior : SuccessPrefix scope 0 chunks controls ordinal remaining) :
      LivePrefix scope chunks controls (.ready scope ordinal remaining)
  | running (controls ordinal chunk rest inv context localState)
      (prior : SuccessPrefix scope 0 chunks controls ordinal (chunk::rest))
      (matching : Matches scope ordinal chunk inv) :
      LivePrefix scope chunks (controls ++ [.enter inv]) (.external inv rest context localState)

inductive History (scope : Scope Frame Receiver) (chunks : List String) :
    List (Control Frame Receiver Ref Payload Stop) →
    ModelState (Frame:=Frame) (Receiver:=Receiver) (Ref:=Ref) (Payload:=Payload)
      (Stop:=Stop) (Context:=Context) (Local:=Local) → Prop
  | live (controls state) (prior : LivePrefix scope chunks controls state) :
      History scope chunks controls state
  | returned (controls ordinal)
      (prior : SuccessPrefix scope 0 chunks controls ordinal []) :
      History scope chunks (controls ++ [.returnOk scope]) (.terminal scope .ok)
  | error (controls ordinal chunk rest inv)
      (prior : SuccessPrefix scope 0 chunks controls ordinal (chunk::rest))
      (matching : Matches scope ordinal chunk inv) :
      History scope chunks (controls ++ [.enter inv,.finish inv .error]) (.terminal scope .error)
  | unwind (controls ordinal chunk rest inv payload)
      (prior : SuccessPrefix scope 0 chunks controls ordinal (chunk::rest))
      (matching : Matches scope ordinal chunk inv) :
      History scope chunks (controls ++ [.enter inv,.finish inv (.unwind payload)])
        (.terminal scope (.unwound payload))
  | callbackStop (controls ordinal chunk rest inv reason)
      (prior : SuccessPrefix scope 0 chunks controls ordinal (chunk::rest))
      (matching : Matches scope ordinal chunk inv) :
      History scope chunks (controls ++ [.enter inv,.finish inv (.stopped reason)])
        (.terminal scope (.stopped reason))
  | ambientStop (controls state reason)
      (prior : LivePrefix (Context:=Context) (Local:=Local) scope chunks controls state) :
      History scope chunks controls (.terminal scope (.stopped reason))

def entries : List (Control Frame Receiver Ref Payload Stop) → List (Invocation Frame Receiver Ref)
  | [] => []
  | .enter inv :: rest => inv :: entries rest
  | _ :: rest => entries rest

end NeuralReversi.ScaledScore.DisplayControl
