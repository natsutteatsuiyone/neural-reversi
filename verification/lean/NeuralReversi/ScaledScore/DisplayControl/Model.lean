import NeuralReversi.ScaledScore.DisplayModel

/-! Formatter::write_fmt calls core::fmt::write, bypassing the sink's write_fmt override. -/
namespace NeuralReversi.ScaledScore.DisplayControl
open NeuralReversi.ScaledScore (Word FormatOptions displayChunks)

structure Scope (Frame Receiver : Type) where
  frame : Frame
  receiver : Receiver
  raw : Word
  options : FormatOptions

structure SliceRef (Ref : Type) where
  token : Ref
  address : BitVec 64

structure Invocation (Frame Receiver Ref : Type) where
  scope : Scope Frame Receiver
  ordinal : Nat
  text : String
  reference : SliceRef Ref

def Invocation.byteLength (inv : Invocation Frame Receiver Ref) : BitVec 64 :=
  BitVec.ofNat 64 inv.text.utf8ByteSize

inductive Reply (Payload Stop : Type) where
  | ok
  | error
  | unwind (payload : Payload)
  | stopped (reason : Stop)

inductive Outcome (Payload Stop : Type) where
  | ok
  | error
  | unwound (payload : Payload)
  | stopped (reason : Stop)

inductive State (Frame Receiver Ref Payload Stop Context Local : Type) where
  | ready (scope : Scope Frame Receiver) (ordinal : Nat) (remaining : List String)
  | external (inv : Invocation Frame Receiver Ref) (remaining : List String)
      (context : Context) (localState : Local)
  | terminal (scope : Scope Frame Receiver) (outcome : Outcome Payload Stop)

variable {Frame Receiver Ref Payload Stop Context Local World Child Event : Type}

abbrev ModelState := State Frame Receiver Ref Payload Stop Context Local
abbrev Configuration := ModelState (Frame:=Frame) (Receiver:=Receiver) (Ref:=Ref)
  (Payload:=Payload) (Stop:=Stop) (Context:=Context) (Local:=Local) × World

def stateScope : ModelState (Frame:=Frame) (Receiver:=Receiver) (Ref:=Ref)
    (Payload:=Payload) (Stop:=Stop) (Context:=Context) (Local:=Local) → Scope Frame Receiver
  | .ready scope _ _ | .terminal scope _ => scope
  | .external inv _ _ _ => inv.scope

def Live (state : ModelState (Frame:=Frame) (Receiver:=Receiver) (Ref:=Ref)
    (Payload:=Payload) (Stop:=Stop) (Context:=Context) (Local:=Local)) : Prop :=
  ∀ scope outcome, state ≠ .terminal scope outcome

inductive OpenEvent (Child Event : Type) where
  | effect (event : Event)
  | child (invocation : Child) (event : Event)

structure Oracle (Frame Receiver Ref Payload Stop Context Local World Child Event : Type) where
  initial : Scope Frame Receiver → Nat → String → World → SliceRef Ref × Context × Local
  internal : Invocation Frame Receiver Ref → Context → Local → World →
    OpenEvent Child Event → Local → World → Prop
  complete : Invocation Frame Receiver Ref → Context → Local → World →
    Reply Payload Stop → Event → World → Prop
  environment : State Frame Receiver Ref Payload Stop Context Local → World → Event → World → Prop
  stopping : State Frame Receiver Ref Payload Stop Context Local → World → Event → Stop → World → Prop

def resume (inv : Invocation Frame Receiver Ref) (remaining : List String)
    (reply : Reply Payload Stop) :
    ModelState (Frame:=Frame) (Receiver:=Receiver) (Ref:=Ref) (Payload:=Payload)
      (Stop:=Stop) (Context:=Context) (Local:=Local) :=
  match reply with
  | .ok => .ready inv.scope (inv.ordinal + 1) remaining
  | .error => .terminal inv.scope .error
  | .unwind payload => .terminal inv.scope (.unwound payload)
  | .stopped reason => .terminal inv.scope (.stopped reason)

inductive Control (Frame Receiver Ref Payload Stop : Type) where
  | enter (inv : Invocation Frame Receiver Ref)
  | finish (inv : Invocation Frame Receiver Ref) (reply : Reply Payload Stop)
  | returnOk (scope : Scope Frame Receiver)

inductive Label (Frame Receiver Ref Payload Stop Child Event : Type) where
  | control (action : Control Frame Receiver Ref Payload Stop)
  | completion (inv : Invocation Frame Receiver Ref) (reply : Reply Payload Stop) (event : Event)
  | callback (inv : Invocation Frame Receiver Ref) (event : OpenEvent Child Event)
  | environment (scope : Scope Frame Receiver) (event : Event)
  | ambientStop (scope : Scope Frame Receiver) (reason : Stop) (event : Event)

def administrativeNext
    (initial : Scope Frame Receiver → Nat → String → World → SliceRef Ref × Context × Local)
    (state : ModelState (Frame:=Frame) (Receiver:=Receiver) (Ref:=Ref)
      (Payload:=Payload) (Stop:=Stop) (Context:=Context) (Local:=Local)) (world : World) :
    Option (Control Frame Receiver Ref Payload Stop × ModelState (Frame:=Frame)
      (Receiver:=Receiver) (Ref:=Ref) (Payload:=Payload) (Stop:=Stop)
      (Context:=Context) (Local:=Local)) :=
  match state with
  | .ready scope _ [] => some (.returnOk scope, .terminal scope .ok)
  | .ready scope ordinal (chunk :: remaining) =>
    let snapshot := initial scope ordinal chunk world
    let inv : Invocation Frame Receiver Ref := ⟨scope, ordinal, chunk, snapshot.1⟩
    some (.enter inv, .external inv remaining snapshot.2.1 snapshot.2.2)
  | _ => none

variable (oracle : Oracle Frame Receiver Ref Payload Stop Context Local World Child Event)

inductive Step : Configuration (Frame:=Frame) (Receiver:=Receiver) (Ref:=Ref)
    (Payload:=Payload) (Stop:=Stop) (Context:=Context) (Local:=Local) (World:=World) →
    Label Frame Receiver Ref Payload Stop Child Event →
    Configuration (Frame:=Frame) (Receiver:=Receiver) (Ref:=Ref)
      (Payload:=Payload) (Stop:=Stop) (Context:=Context) (Local:=Local) (World:=World) → Prop
  | administrative (state world action next)
      (h : administrativeNext oracle.initial state world = some (action,next)) :
      Step (state,world) (.control action) (next,world)
  | complete (inv ctx loc remaining world reply event world')
      (h : oracle.complete inv ctx loc world reply event world') :
      Step (.external inv remaining ctx loc,world) (.completion inv reply event)
        (resume inv remaining reply,world')
  | internal (inv ctx loc remaining world event loc' world')
      (h : oracle.internal inv ctx loc world event loc' world') :
      Step (.external inv remaining ctx loc,world) (.callback inv event)
        (.external inv remaining ctx loc',world')
  | environment (state world event world') (live : Live state)
      (h : oracle.environment state world event world') :
      Step (state,world) (.environment (stateScope state) event) (state,world')
  | environmentStop (state world event reason world') (live : Live state)
      (h : oracle.stopping state world event reason world') :
      Step (state,world) (.ambientStop (stateScope state) reason event)
        (.terminal (stateScope state) (.stopped reason),world')

inductive Trace : Configuration (Frame:=Frame) (Receiver:=Receiver) (Ref:=Ref)
    (Payload:=Payload) (Stop:=Stop) (Context:=Context) (Local:=Local) (World:=World) →
    List (Label Frame Receiver Ref Payload Stop Child Event) →
    Configuration (Frame:=Frame) (Receiver:=Receiver) (Ref:=Ref)
      (Payload:=Payload) (Stop:=Stop) (Context:=Context) (Local:=Local) (World:=World) → Prop
  | refl (a) : Trace a [] a
  | cons {a b c label labels} : Step oracle a label b → Trace b labels c → Trace a (label::labels) c

def initialState (scope : Scope Frame Receiver) :
    ModelState (Frame:=Frame) (Receiver:=Receiver) (Ref:=Ref)
      (Payload:=Payload) (Stop:=Stop) (Context:=Context) (Local:=Local) :=
  .ready scope 0 (displayChunks scope.raw)

end NeuralReversi.ScaledScore.DisplayControl
