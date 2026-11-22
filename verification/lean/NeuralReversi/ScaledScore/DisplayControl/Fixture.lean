import NeuralReversi.ScaledScore.DisplayControl.Client
import NeuralReversi.Simd.Fixture

namespace NeuralReversi.ScaledScore.DisplayControl.Fixture
open NeuralReversi.ScaledScore (displayChunks FormatOptions)
open NeuralReversi.Simd.X86 (parseWord)

abbrev W := Client.W
abbrev Payload := Client.Payload
abbrev Stop := Client.Stop
abbrev Core := Client.Core
abbrev Inv := Client.Inv
abbrev Choice := Replay.Selection Payload Stop Unit Client.World W Client.Event
abbrev Child := Client.Child

structure Input where
  raw : BitVec 32
  mode : Nat
  stop : Nat
  seed : W
  ending : String
  lines : List (List String)
  addresses : Array W

structure Session extends Client.World where
  core : Core
  child : Option Child := none
  payloadSeen : Bool := false
  returned : Bool := false
  choices : List Choice := []

def parseNat (text : String) : Except String Nat :=
  match text.toNat? with
  | some n => .ok n
  | none => .error "invalid unsigned replay integer"

def nibble (c : Char) : Except String Nat :=
  if '0' ≤ c ∧ c ≤ '9' then .ok (c.toNat - '0'.toNat)
  else if 'a' ≤ c ∧ c ≤ 'f' then .ok (c.toNat - 'a'.toNat + 10)
  else .error "invalid replay hex"

def unhexBytes : List Char → Except String (List UInt8)
  | [] => .ok []
  | a::b::rest => do
    let a ← nibble a
    let b ← nibble b
    return UInt8.ofNat (16*a+b) :: (← unhexBytes rest)
  | _ => .error "odd replay hex length"

def unhex (text : String) : Except String String := do
  let bytes ← unhexBytes text.toList
  let some result := String.fromUTF8? (ByteArray.mk bytes.toArray) | throw "invalid replay UTF8"
  return result

def options : FormatOptions :=
  { fill := '*', alignment := some 1, sign := some true,
    width := some 19, precision := some 13 }

def config (input : Input) : Client.Config := ⟨input.mode,input.stop,input.addresses⟩

def perform (input : Input) (r : Session) (choice : Choice) : Except String Session := do
  let some (_,next) := Replay.step (Client.oracle (config input)) (Client.checks (config input))
    (r.core,r.toWorld) choice | throw "known-policy source transition rejected"
  return {r with core := next.1, toWorld := next.2, choices := r.choices ++ [choice]}

def enter (input : Input) (r : Session) : Except String (Inv × Session) := do
  let next ← perform input r .administrative
  match next.core with
  | .external inv _ _ _ => return (inv,next)
  | _ => throw "native entry is not a source entry"

def active (r : Session) : Except String (Inv × List String) :=
  match r.core with
  | .external inv rest _ _ => .ok (inv,rest)
  | _ => .error "native callback outside active source call"

def complete (input : Input) (r : Session) : Except String Session := do
  match Client.expectedReply (config input) r.toWorld with
  | some answer => perform input r (.complete answer .finish r.toWorld)
  | none => return r

def checkEntry (r : Session) (inv : Inv) (ordinal : Nat) (receiver frame length : W)
    (text : String) (before : W) : Except String Unit := do
  unless ordinal = inv.ordinal + 1 ∧ receiver = inv.scope.receiver ∧ frame = inv.scope.frame ∧
      text = inv.text ∧ length = inv.byteLength ∧ before = r.state do
    throw "native callback descriptor differs from source input"

def step (input : Input) (r : Session) (fields : List String) : Except String Session := do
  if r.returned then throw "native event after outward return"
  match fields with
  | ["E",depth,ordinal,address,receiver,frame,length,text,before] =>
    let depth ← parseNat depth
    let ordinal ← parseNat ordinal
    let address ← parseWord address
    let receiver ← parseWord receiver
    let frame ← parseWord frame
    let length ← parseWord length
    let text ← unhex text
    let before ← parseWord before
    if depth = 0 then
      if r.child.isSome then throw "outer call during child callback"
      let (inv,next) ← enter input r
      checkEntry r inv ordinal receiver frame length text before
      unless address = inv.reference.address do throw "reference snapshot mismatch"
      return next
    else if depth = 1 then
      let (inv,_) ← active r
      let plan := displayChunks 256#32
      unless input.mode = 2 ∧ inv.ordinal = 0 ∧ r.outer = 0 ∧ r.child.isNone ∧
          ordinal = r.children + 1 ∧ receiver = inv.scope.receiver ∧ before = r.state ∧
          r.childFrame = some frame ∧
          plan[r.children]? = some text ∧ length.toNat = text.utf8ByteSize do
        throw "invalid reentrant child observation"
      return {r with child := some ⟨ordinal,address,frame,receiver,length,text⟩}
    else throw "unsupported native invocation depth"
  | ["C",receiver,frame,raw,before] =>
    let receiver ← parseWord receiver
    let frame ← parseWord frame
    let raw ← parseNat raw
    let before ← parseWord before
    unless raw = 256 ∧ before = r.state ∧ r.child.isNone do throw "bad child invocation boundary"
    let next := {r.toWorld with childFrame := some frame}
    perform input r (.internal (.effect (.childBegin receiver frame)) () next)
  | ["A",state,bytes] =>
    let state ← parseWord state
    let bytes ← unhex bytes
    let next ← match r.child with
      | some child => do
        let next := Client.childEffect r.toWorld child
        let checked ← perform input r (.internal (.child child.frame (.child child)) () next)
        pure {checked with child := none}
      | none => do
        let (inv,_) ← active r
        let next := Client.outerEffect (config input) inv r.toWorld
        let changed ← perform input r (.internal (.effect .effect) () next)
        complete input changed
    unless state = next.state ∧ bytes = next.bytes do throw "native sink mutation mismatch"
    return next
  | ["U",address,marker] =>
    let address ← parseWord address
    let marker ← parseWord marker
    let next := {r.toWorld with payload := some (address,marker)}
    let changed ← perform input r (.internal (.effect (.payload address marker)) () next)
    complete input changed
  | ["P",address,marker] =>
    let address ← parseWord address
    let marker ← parseWord marker
    match r.core with
    | .terminal _ (.unwound payload) =>
      unless payload = (address,marker) ∧ !r.payloadSeen do throw "outward panic payload changed"
      return {r with payloadSeen := true}
    | _ => throw "panic observation without source unwind"
  | ["R",status,preserved,outer,children,fmtCalls,charCalls,state,bytes] =>
    let preserved ← parseNat preserved
    let outer ← parseNat outer
    let children ← parseNat children
    let fmtCalls ← parseNat fmtCalls
    let charCalls ← parseNat charCalls
    let state ← parseWord state
    let bytes ← unhex bytes
    let r ← match r.core with
      | .ready _ _ [] => perform input r .administrative
      | _ => pure r
    let core := r.core
    let expected ← match core with
      | .terminal _ .ok => pure "ok"
      | .terminal _ .error => pure "error"
      | .terminal _ (.unwound _) => do
        unless r.payloadSeen do throw "missing outward payload observation"
        pure "panic"
      | _ => throw "native return before source termination"
    unless status = expected ∧ preserved = 1 ∧ outer = r.outer ∧ children = r.children ∧
        fmtCalls = 0 ∧ charCalls = 0 ∧ state = r.state ∧ bytes = r.bytes do
      throw "native outward result or preserved context mismatch"
    return {r with returned := true}
  | _ => throw "invalid native control event"

def replay (input : Input) : Except String String := do
  let (scope,lines) ← match input.lines with
    | ["B",receiver,frame,raw,mode,stop,seed]::rest => do
      let receiver ← parseWord receiver
      let frame ← parseWord frame
      let some raw := raw.toInt? | throw "invalid signed entry word"
      let mode ← parseNat mode
      let stop ← parseNat stop
      let seed ← parseWord seed
      unless raw = input.raw.toInt ∧ mode = input.mode ∧ stop = input.stop ∧ seed = input.seed do
        throw "native invocation arguments changed"
      pure (({frame := frame, receiver := receiver, raw := input.raw, options := options} : Scope W W),rest)
    | _ => throw "missing caller-supplied invocation context"
  let initial : Session := {core := initialState scope, state := input.seed}
  let final ← lines.foldlM (step input) initial
  let some (_,checked) := Replay.run (Client.oracle (config input)) (Client.checks (config input))
    (initial.core,initial.toWorld) final.choices | throw "parsed replay does not embed in source Trace"
  let final := {final with core := checked.1, toWorld := checked.2}
  unless final.child.isNone do throw "unfinished known child callback"
  match input.ending,final.core with
  | "returned",.terminal _ .ok | "returned",.terminal _ .error |
      "returned",.terminal _ (.unwound _) =>
    unless final.returned do throw "missing returned native result"
  | "exit",.terminal _ (.stopped .exit) =>
    unless !final.returned do throw "exit returned"
  | "abort",.terminal _ (.stopped .abort) =>
    unless !final.returned do throw "abort returned"
  | "pending",.external inv _ _ _ =>
    unless input.mode = 7 ∧ final.failed ∧ final.outer = inv.ordinal+1 ∧ !final.returned do
      throw "pending prefix is not inside the selected callback"
  | _,_ => throw "native stop class disagrees with source control"
  return s!"ok:{final.outer}|{final.children}|{final.state.toNat}"

def request (fields : List String) : Except String String := do
  let [_,raw,mode,stop,seed,ending,trace] := fields |
    throw "expected raw32 mode stop state64 ending tracehex"
  let raw ← parseNat raw
  unless raw < 2^32 do throw "raw outside i32 word domain"
  let mode ← parseNat mode
  let stop ← parseNat stop
  unless mode ≤ 7 ∧ 1 ≤ stop ∧ stop ≤ 4 do throw "control mode outside fixture domain"
  let seed ← parseWord seed
  let text ← unhex trace
  let lines := ((text.splitOn "\n").filter (· != "")).map (·.splitOn "|")
  let addresses ← lines.foldlM (fun values line => do
    match line with
    | ["E","0",_,address,_,_,_,_,_] => return values.push (← parseWord address)
    | _ => return values) (#[] : Array W)
  replay ⟨BitVec.ofNat 32 raw,mode,stop,seed,ending,lines,addresses⟩

end NeuralReversi.ScaledScore.DisplayControl.Fixture
