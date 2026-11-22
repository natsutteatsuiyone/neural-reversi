import NeuralReversi.ScaledScore.DisplayControl.Replay

namespace NeuralReversi.ScaledScore.DisplayControl.Client
open NeuralReversi.ScaledScore (displayChunks)
abbrev W := BitVec 64
abbrev Payload := W × W

inductive Stop where
  | exit | abort
  deriving DecidableEq

structure Config where
  mode : Nat
  stop : Nat
  addresses : Array W

structure Child where
  ordinal : Nat
  address : W
  frame : W
  receiver : W
  length : W
  text : String
  deriving DecidableEq

structure World where
  state : W
  bytes : String := ""
  outer : Nat := 0
  children : Nat := 0
  childFrame : Option W := none
  failed : Bool := false
  payload : Option Payload := none
  deriving DecidableEq

inductive Event where
  | childBegin (receiver frame : W)
  | child (observation : Child)
  | effect
  | payload (address marker : W)
  | finish
  deriving DecidableEq

abbrev Inv := Invocation W W Unit
abbrev Core := State W W Unit Payload Stop Unit Unit
abbrev O := Oracle W W Unit Payload Stop Unit Unit World W Event

def fail (config : Config) (inv : Inv) (w : World) : Bool :=
  (config.mode = 1 && ((inv.reference.address ^^^ w.state) &&& 1#64) = 0) ||
    (decide (3 ≤ config.mode) && decide (inv.ordinal+1 = config.stop))

def mutate (w : World) (address : W) (ordinal : Nat) (text : String) (failed : Bool) : World :=
  {w with
    state := w.state * 3 + address + BitVec.ofNat 64 ordinal,
    bytes := w.bytes ++ (if failed then String.ofList (text.toList.take 1) else text)}

def outerEffect (config : Config) (inv : Inv) (w : World) : World :=
  {mutate w inv.reference.address (inv.ordinal+1) inv.text (fail config inv w) with
    outer := w.outer+1, failed := fail config inv w}

def childEffect (w : World) (child : Child) : World :=
  {mutate w child.address child.ordinal child.text false with
    children := w.children+1, childFrame := some child.frame}

def allowedReply (config : Config) (w : World) : Reply Payload Stop → Prop
  | .ok => w.failed = false
  | .error => w.failed = true ∧ (config.mode = 1 ∨ config.mode = 3)
  | .unwind payload => w.failed = true ∧ config.mode = 4 ∧ w.payload = some payload
  | .stopped .exit => w.failed = true ∧ config.mode = 5
  | .stopped .abort => w.failed = true ∧ config.mode = 6

def expectedReply (config : Config) (w : World) : Option (Reply Payload Stop) :=
  if !w.failed then some .ok
  else if config.mode = 1 ∨ config.mode = 3 then some .error
  else if config.mode = 4 then w.payload.map Reply.unwind
  else if config.mode = 5 then some (.stopped .exit)
  else if config.mode = 6 then some (.stopped .abort)
  else none

def oracle (config : Config) : O where
  initial := fun _ ordinal _ _ => (⟨(),config.addresses[ordinal]?.getD 0⟩,(),())
  internal := fun inv _ _ w event _ next =>
    match event with
    | .child frame (.child observation) =>
      config.mode = 2 ∧ inv.ordinal = 0 ∧ w.outer = 0 ∧ frame = observation.frame ∧
      observation.ordinal = w.children+1 ∧ observation.receiver = inv.scope.receiver ∧
      observation.length.toNat = observation.text.utf8ByteSize ∧
      (displayChunks 256#32)[w.children]? = some observation.text ∧
      w.childFrame = some frame ∧ next = childEffect w observation
    | .effect (.childBegin receiver frame) =>
      config.mode = 2 ∧ inv.ordinal = 0 ∧ w.outer = 0 ∧ w.children = 0 ∧
      w.childFrame = none ∧ receiver = inv.scope.receiver ∧
      next = {w with childFrame := some frame}
    | .effect .effect =>
      w.outer = inv.ordinal ∧
      (config.mode = 2 → inv.ordinal = 0 → w.children = (displayChunks 256#32).length) ∧
      next = outerEffect config inv w
    | .effect (.payload address marker) =>
      config.mode = 4 ∧ w.failed = true ∧ w.outer = inv.ordinal+1 ∧
      w.payload = none ∧ marker = w.state ∧ next = {w with payload := some (address,marker)}
    | _ => False
  complete := fun inv _ _ w reply event next =>
    w.outer = inv.ordinal+1 ∧ allowedReply config w reply ∧ event = .finish ∧ next = w
  environment := fun _ _ _ _ => False
  stopping := fun _ _ _ _ _ => False

def checks (config : Config) : Replay.Checks (oracle config) where
  internal := fun inv ctx loc w event local' next => by
    cases event with
    | effect event => cases event <;> dsimp [oracle] <;> infer_instance
    | child frame event => cases event <;> dsimp [oracle] <;> infer_instance
  complete := fun inv ctx loc w reply event next => by
    cases reply with
    | stopped stop => cases stop <;> dsimp [oracle,allowedReply] <;> infer_instance
    | ok | error | unwind => dsimp [oracle,allowedReply] <;> infer_instance

end NeuralReversi.ScaledScore.DisplayControl.Client
