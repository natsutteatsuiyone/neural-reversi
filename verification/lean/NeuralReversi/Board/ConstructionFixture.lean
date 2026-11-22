import NeuralReversi.Board.ConstructionModel

namespace NeuralReversi.Board.Construction.Fixture

structure Input where
  word : Bits
  mode : Nat
  dropPanics : Bool

structure World where
  word : Bits
  events : List String := []

inductive Payload where
  | playerInto | opponentInto | playerDrop | opponentDrop | overlap | other

inductive Stop where
  | exit (code : Nat)
  | abort

inductive Callback where
  | completed (reply : Reply Payload Stop) (world : World)
  | pending (snapshot : World)

def emit (code : Nat) (text : String) (world : World) : World :=
  ⟨world.word * 3 + BitVec.ofNat 64 code, world.events ++ [text]⟩

def inputName (player : Bool) : String := if player then "P" else "O"

def payloadName : Payload → String
  | .playerInto => "Pi"
  | .opponentInto => "Oi"
  | .playerDrop => "Pd"
  | .opponentDrop => "Od"
  | .overlap => "overlap"
  | .other => "other"

def hook (mode : Nat) (payload : Payload) (world : World) : Callback :=
  let code := match payload with
    | .playerInto => 7 | .opponentInto => 8
    | .playerDrop => 9 | .opponentDrop => 10
    | .overlap => 11 | .other => 13
  let world := emit code ("h:" ++ payloadName payload) world
  match mode with
  | 1 => .completed (.stopped (.exit 31)) (emit 20 "exit:hook" world)
  | 2 => .completed (.stopped .abort) (emit 21 "abort:hook" world)
  | 3 => .pending (emit 22 "park:hook" world)
  | _ => .completed (.unwinding payload) world

def dropInput (player : Bool) (input : Input) (unwinding : Bool)
    (hookMode : Nat) (world : World) : Callback :=
  let code := if player then (if unwinding then 5 else 3) else (if unwinding then 6 else 4)
  let world := emit code s!"d{inputName player}:{if unwinding then 1 else 0}" world
  if input.dropPanics then
    hook hookMode (if player then .playerDrop else .opponentDrop) world
  else .completed .cleaned world

def unwindInput (player : Bool) (input : Input) (payload : Payload)
    (hookMode : Nat) (world : World) : Callback :=
  match dropInput player input true hookMode world with
  | .completed .cleaned world => .completed (.unwinding payload) world
  | .completed (.unwinding _) world => .completed (.stopped .abort) world
  | result => result

def convertInput (player : Bool) (input : Input) (hookMode : Nat) (world : World) : Callback :=
  let world := emit (if player then 1 else 2) ("i" ++ inputName player) world
  let payload := if player then Payload.playerInto else .opponentInto
  match input.mode with
  | 1 => match hook hookMode payload world with
    | .completed (.unwinding original) world => unwindInput player input original hookMode world
    | result => result
  | 3 => unwindInput player input payload hookMode world
  | 5 => .completed (.stopped (.exit 17)) (emit 18 ("exit:" ++ inputName player) world)
  | 6 => .completed (.stopped .abort) (emit 19 ("abort:" ++ inputName player) world)
  | 7 => .pending (emit 23 ("park:" ++ inputName player) world)
  | _ =>
    let word := if input.mode == 4 then input.word ^^^ world.word else input.word
    let world := emit (if player then 14 else 15) s!"v{inputName player}:{word.toNat}" world
    if input.mode == 2 then .completed (.converted word) world
    else match dropInput player input false hookMode world with
      | .completed .cleaned world => .completed (.converted word) world
      | result => result

def callback (hookMode : Nat) (inv : Invocation Input Input Payload) (world : World) : Callback :=
  match inv with
  | .intoPlayer input _ => convertInput true input hookMode world
  | .intoOpponent _ input => convertInput false input hookMode world
  | .cleanupOpponent input _ => dropInput false input true hookMode world
  | .assertion _ _ message _ =>
    hook hookMode (if message == "player and opponent bitboards must not overlap" then .overlap else .other) world
  | .fatalCleanup _ _ => .completed (.stopped .abort) world

abbrev Machine := State Input Input Payload Stop Unit Unit

inductive Observation where
  | returned (outcome : Outcome Payload Stop) (world : World)
  | pending (snapshot : World)
  | invalid

def execute : Nat → Machine → World → Nat → Observation
  | 0, _, _, _ => .invalid
  | fuel + 1, state, world, hookMode =>
    match state with
    | .terminal outcome => .returned outcome world
    | .external inv _ _ => match callback hookMode inv world with
      | .pending snapshot => .pending snapshot
      | .completed reply nextWorld => match resume inv reply with
        | none => .invalid
        | some next => execute fuel next nextWorld hookMode
    | _ => match administrativeNext (fun _ _ => ((), ())) state world with
      | none => .invalid
      | some (_, next) => execute fuel next world hookMode

def output (player opponent : Bits) (flags : Nat) (seed : Bits) : String :=
  let p : Input := ⟨player, flags % 8, flags / 64 % 2 == 1⟩
  let o : Input := ⟨opponent, flags / 8 % 8, flags / 128 % 2 == 1⟩
  let result := execute 8 (.readyP p o) ⟨seed, []⟩ (flags / 256)
  let (status, world) := match result with
    | .invalid => ("invalid", World.mk seed [])
    | .pending snapshot => ("pending", snapshot)
    | .returned (.returned board) world =>
      let text := s!"ok:{board.player.toNat}:{board.opponent.toNat}"
      (text, emit 16 text world)
    | .returned (.unwound payload) world =>
      let text := "unwind:" ++ payloadName payload
      (text, emit 17 text world)
    | .returned (.stopped (.exit code)) world => (s!"exit:{code}", world)
    | .returned (.stopped .abort) world => ("abort", world)
  s!"{status}#{world.word.toNat}#{String.intercalate "," world.events}"

end NeuralReversi.Board.Construction.Fixture
