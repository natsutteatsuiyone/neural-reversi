import NeuralReversi.Board.Model

namespace NeuralReversi.Board.Transform
open NeuralReversi
abbrev Bits := BitVec 64
variable {Frame Fn Payload Stop Context Local World Event : Type}

inductive Invocation (Frame Fn : Type) where
  | player (frame : Frame) (fn : Fn) (player opponent : Bits)
  | opponent (frame : Frame) (fn : Fn) (player opponent returnedPlayer : Bits)

inductive Reply (Payload Stop : Type) where
  | returned (bits : Bits)
  | unwinding (payload : Payload)
  | stopped (reason : Stop)

inductive Outcome (Payload Stop : Type) where
  | returned (board : NeuralReversi.Board.Board)
  | unwound (payload : Payload)
  | stopped (reason : Stop)

inductive State (Frame Fn Payload Stop Context Local : Type) where
  | readyP (frame : Frame) (fn : Fn) (player opponent : Bits)
  | readyO (frame : Frame) (fn : Fn) (player opponent returnedPlayer : Bits)
  | external (inv : Invocation Frame Fn) (context : Context) (localState : Local)
  | terminal (outcome : Outcome Payload Stop)

abbrev ModelState := State Frame Fn Payload Stop Context Local
abbrev Configuration := ModelState (Frame:=Frame) (Fn:=Fn) (Payload:=Payload)
  (Stop:=Stop) (Context:=Context) (Local:=Local) × World

structure Oracle (Frame Fn Payload Stop Context Local World Event : Type) where
  initial : Invocation Frame Fn → World → Context × Local
  internal : Invocation Frame Fn → Context → Local → World → Event → Local → World → Prop
  complete : Invocation Frame Fn → Context → Local → World → Reply Payload Stop → Event → World → Prop
  environment : State Frame Fn Payload Stop Context Local → World → Event → World → Prop
  stopping : State Frame Fn Payload Stop Context Local → World → Event → Stop → World → Prop

def resume (inv : Invocation Frame Fn) (reply : Reply Payload Stop) :
    ModelState (Frame:=Frame) (Fn:=Fn) (Payload:=Payload) (Stop:=Stop)
      (Context:=Context) (Local:=Local) :=
  match inv, reply with
  | .player frame fn p o, .returned r => .readyO frame fn p o r
  | .opponent _ _ _ _ r, .returned s => .terminal (.returned ⟨r,s⟩)
  | _, .unwinding payload => .terminal (.unwound payload)
  | _, .stopped reason => .terminal (.stopped reason)

inductive Control (Frame Fn Payload Stop : Type) where
  | enter (inv : Invocation Frame Fn)
  | finish (inv : Invocation Frame Fn) (reply : Reply Payload Stop)

inductive Label (Frame Fn Payload Stop Event : Type) where
  | control (action : Control Frame Fn Payload Stop)
  | completion (action : Control Frame Fn Payload Stop) (event : Event)
  | external (event : Event)
  | environment (event : Event)

def administrativeNext
    (initial : Invocation Frame Fn → World → Context × Local)
    (state : ModelState (Frame:=Frame) (Fn:=Fn) (Payload:=Payload) (Stop:=Stop)
      (Context:=Context) (Local:=Local)) (w : World) :
    Option (Control Frame Fn Payload Stop × ModelState (Frame:=Frame) (Fn:=Fn)
      (Payload:=Payload) (Stop:=Stop) (Context:=Context) (Local:=Local)) :=
  let enter inv := some (Control.enter inv,State.external inv (initial inv w).1 (initial inv w).2)
  match state with
  | .readyP frame fn p o => enter (.player frame fn p o)
  | .readyO frame fn p o r => enter (.opponent frame fn p o r)
  | _ => none

variable (oracle : Oracle Frame Fn Payload Stop Context Local World Event)

inductive Step : Configuration (Frame:=Frame) (Fn:=Fn) (Payload:=Payload) (Stop:=Stop)
    (Context:=Context) (Local:=Local) (World:=World) → Label Frame Fn Payload Stop Event →
    Configuration (Frame:=Frame) (Fn:=Fn) (Payload:=Payload) (Stop:=Stop)
      (Context:=Context) (Local:=Local) (World:=World) → Prop
  | administrative (state w action next) (h : administrativeNext oracle.initial state w = some (action,next)) :
      Step (state,w) (.control action) (next,w)
  | complete (inv ctx loc w reply event w') (h : oracle.complete inv ctx loc w reply event w') :
      Step (.external inv ctx loc,w) (.completion (.finish inv reply) event) (resume inv reply,w')
  | internal (inv ctx loc w event loc' w') (h : oracle.internal inv ctx loc w event loc' w') :
      Step (.external inv ctx loc,w) (.external event) (.external inv ctx loc',w')
  | environment (state w event w') (h : oracle.environment state w event w')
      (live : ∀ outcome, state ≠ .terminal outcome) :
      Step (state,w) (.environment event) (state,w')
  | environmentStop (state w event reason w') (h : oracle.stopping state w event reason w')
      (live : ∀ outcome, state ≠ .terminal outcome) :
      Step (state,w) (.environment event) (.terminal (.stopped reason),w')

inductive Trace : Configuration (Frame:=Frame) (Fn:=Fn) (Payload:=Payload) (Stop:=Stop)
    (Context:=Context) (Local:=Local) (World:=World) → List (Label Frame Fn Payload Stop Event) →
    Configuration (Frame:=Frame) (Fn:=Fn) (Payload:=Payload) (Stop:=Stop)
      (Context:=Context) (Local:=Local) (World:=World) → Prop
  | refl (a) : Trace a [] a
  | cons {a b c label labels} : Step oracle a label b → Trace b labels c → Trace a (label::labels) c

end NeuralReversi.Board.Transform
