module
public import NeuralReversi.Board.Model

@[expose] public section

namespace NeuralReversi.Board.Construction
open NeuralReversi
abbrev Bits := BitVec 64
variable {P O Payload Stop Context Local World Event : Type}

inductive Invocation (P O Payload : Type) where
  | intoPlayer (player : P) (retained : O)
  | intoOpponent (player : Bits) (opponent : O)
  | cleanupOpponent (opponent : O) (original : Payload)
  | assertion (player opponent : Bits) (message sourceOrigin : String)
  | fatalCleanup (original fresh : Payload)

inductive Reply (Payload Stop : Type) where
  | converted (bits : Bits)
  | unwinding (payload : Payload)
  | cleaned
  | stopped (reason : Stop)

inductive Outcome (Payload Stop : Type) where
  | returned (board : NeuralReversi.Board.Board)
  | unwound (payload : Payload)
  | stopped (reason : Stop)

inductive State (P O Payload Stop Context Local : Type) where
  | readyP (player : P) (opponent : O)
  | readyO (player : Bits) (opponent : O)
  | check (player opponent : Bits)
  | readyCleanup (opponent : O) (original : Payload)
  | readyAssert (player opponent : Bits)
  | readyFatal (original fresh : Payload)
  | external (inv : Invocation P O Payload) (context : Context) (localState : Local)
  | terminal (outcome : Outcome Payload Stop)

abbrev ModelState := State P O Payload Stop Context Local
abbrev InvocationType := Invocation P O Payload
abbrev ReplyType := Reply Payload Stop
abbrev Configuration := ModelState (P:=P) (O:=O) (Payload:=Payload) (Stop:=Stop) (Context:=Context) (Local:=Local) × World

structure Oracle (P O Payload Stop Context Local World Event : Type) where
  initial : Invocation P O Payload → World → Context × Local
  internal : Invocation P O Payload → Context → Local → World → Event → Local → World → Prop
  complete : Invocation P O Payload → Context → Local → World → Reply Payload Stop → Event → World → Prop
  environment : State P O Payload Stop Context Local → World → Event → World → Prop
  stopping : State P O Payload Stop Context Local → World → Event → Stop → World → Prop

def resume (inv : InvocationType (P:=P) (O:=O) (Payload:=Payload)) (reply : ReplyType (Payload:=Payload) (Stop:=Stop)) :
    Option (ModelState (P:=P) (O:=O) (Payload:=Payload) (Stop:=Stop) (Context:=Context) (Local:=Local)) :=
  match inv, reply with
  | .intoPlayer _ o, .converted p => some (.readyO p o)
  | .intoPlayer _ o, .unwinding payload => some (.readyCleanup o payload)
  | .intoOpponent p _, .converted o => some (.check p o)
  | .intoOpponent _ _, .unwinding payload => some (.terminal (.unwound payload))
  | .cleanupOpponent _ original, .cleaned => some (.terminal (.unwound original))
  | .cleanupOpponent _ original, .unwinding fresh => some (.readyFatal original fresh)
  | .assertion _ _ _ _, .unwinding payload => some (.terminal (.unwound payload))
  | _, .stopped reason => some (.terminal (.stopped reason))
  | _, _ => none

inductive Control (P O Payload Stop : Type) where
  | enter (inv : Invocation P O Payload)
  | finish (inv : Invocation P O Payload) (reply : Reply Payload Stop)
  | disjoint (player opponent : Bits)
  | overlap (player opponent : Bits)

inductive Label (P O Payload Stop Event : Type) where
  | control (action : Control P O Payload Stop)
  | completion (action : Control P O Payload Stop) (event : Event)
  | external (event : Event)
  | environment (event : Event)

def administrativeNext
    (initial : Invocation P O Payload → World → Context × Local)
    (state : ModelState (P:=P) (O:=O) (Payload:=Payload) (Stop:=Stop) (Context:=Context) (Local:=Local)) (w : World) :
    Option (Control P O Payload Stop × ModelState (P:=P) (O:=O) (Payload:=Payload) (Stop:=Stop) (Context:=Context) (Local:=Local)) :=
  let enter inv := some (Control.enter inv,State.external inv (initial inv w).1 (initial inv w).2)
  match state with
  | .readyP p o => enter (.intoPlayer p o)
  | .readyO p o => enter (.intoOpponent p o)
  | .readyCleanup o original => enter (.cleanupOpponent o original)
  | .readyAssert p o => enter (.assertion p o "player and opponent bitboards must not overlap" "f6fd54154c270f6db63ab78953663fe090841e26:crates/reversi-core/src/board.rs:64")
  | .readyFatal old fresh => enter (.fatalCleanup old fresh)
  | .check p o => if p &&& o = 0 then some (.disjoint p o,.terminal (.returned ⟨p,o⟩))
      else some (.overlap p o,.readyAssert p o)
  | _ => none
variable (oracle : Oracle P O Payload Stop Context Local World Event)

inductive Step : Configuration (P:=P) (O:=O) (Payload:=Payload) (Stop:=Stop) (Context:=Context) (Local:=Local) (World:=World) →
    Label P O Payload Stop Event → Configuration (P:=P) (O:=O) (Payload:=Payload) (Stop:=Stop) (Context:=Context) (Local:=Local) (World:=World) → Prop
  | administrative (state w action next) (h : administrativeNext oracle.initial state w = some (action,next)) :
      Step (state,w) (.control action) (next,w)
  | complete (inv ctx loc w reply event w' next)
      (ho : oracle.complete inv ctx loc w reply event w') (hr : resume inv reply = some next) :
      Step (.external inv ctx loc,w) (.completion (.finish inv reply) event) (next,w')
  | internal (inv ctx loc w event loc' w') (h : oracle.internal inv ctx loc w event loc' w') :
      Step (.external inv ctx loc,w) (.external event) (.external inv ctx loc',w')
  | environment (state w event w') (h : oracle.environment state w event w')
      (live : ∀ outcome, state ≠ .terminal outcome) :
      Step (state,w) (.environment event) (state,w')
  | environmentStop (state w event reason w') (h : oracle.stopping state w event reason w')
      (live : ∀ outcome, state ≠ .terminal outcome) :
      Step (state,w) (.environment event) (.terminal (.stopped reason),w')

inductive Trace : Configuration (P:=P) (O:=O) (Payload:=Payload) (Stop:=Stop) (Context:=Context) (Local:=Local) (World:=World) →
    List (Label P O Payload Stop Event) → Configuration (P:=P) (O:=O) (Payload:=Payload) (Stop:=Stop) (Context:=Context) (Local:=Local) (World:=World) → Prop
  | refl (a) : Trace a [] a
  | cons {a b c label labels} : Step oracle a label b → Trace b labels c → Trace a (label::labels) c

end NeuralReversi.Board.Construction
