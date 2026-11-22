import NeuralReversi.Board.TransformModel

namespace NeuralReversi.Board.Transform.Fixture

structure World where
  word : Bits
  seed : Bits
  events : List String := []

abbrev Machine := State Unit Nat (Nat × Nat) Nat Unit Unit

inductive Callback where
  | completed (reply : Reply (Nat × Nat) Nat) (world : World)
  | pending (snapshot : World)

def emit (code : Nat) (event : String) (world : World) : World :=
  { world with
    word := world.word * 3 + BitVec.ofNat 64 code
    events := world.events ++ [event] }

def callback (flags : Nat) (inv : Invocation Unit Nat) (world : World) : Callback :=
  let (identity,index,argument,mode) := match inv with
    | .player _ fn p _ => (fn,1,p,flags / 4 % 7)
    | .opponent _ fn _ o _ => (fn,2,o,flags / 28)
  let world := emit index s!"c:{identity}:{index}:{argument.toNat}" world
  match mode with
  | 1 | 6 => .completed (.unwinding (mode,index)) (emit 13 s!"h:{index}" world)
  | 2 => .completed (.unwinding (2,index)) world
  | 3 => .completed (.stopped 17) (emit 18 s!"exit:{index}" world)
  | 4 => .completed (.stopped 0) (emit 19 s!"abort:{index}" world)
  | 5 => .pending (emit 23 s!"park:{index}" world)
  | _ =>
    let word := match identity with
      | 0 => argument + world.word + BitVec.ofNat 64 index
      | 1 => world.seed
      | 2 => if index == 1 then ~~~argument else argument
      | _ => argument.rotateLeft 17 ^^^ world.word
    .completed (.returned word) (emit (13 + index) s!"v:{index}:{word.toNat}" world)

inductive Observation where
  | returned (outcome : Outcome (Nat × Nat) Nat) (world : World)
  | pending (snapshot : World)
  | invalid

def execute : Nat → Machine → World → Nat → Observation
  | 0, _, _, _ => .invalid
  | fuel+1, state, world, flags =>
    match state with
    | .terminal outcome => .returned outcome world
    | .external inv _ _ => match callback flags inv world with
      | .pending snapshot => .pending snapshot
      | .completed reply nextWorld => execute fuel (resume inv reply) nextWorld flags
    | _ => match administrativeNext (fun _ _ => ((),())) state world with
      | none => .invalid
      | some (_,next) => execute fuel next world flags

def output (player opponent : Bits) (flags : Nat) (seed : Bits) : String :=
  let result := execute 5 (.readyP () (flags % 4) player opponent) ⟨seed,seed,[]⟩ flags
  let (status,world) := match result with
    | .invalid => ("invalid",World.mk seed seed [])
    | .pending snapshot => ("pending",snapshot)
    | .returned (.returned board) world =>
      let text := s!"ok:{board.player.toNat}:{board.opponent.toNat}"
      (text,emit 16 text world)
    | .returned (.unwound (kind,index)) world =>
      let text := s!"unwind:{kind}:{index}"
      (text,emit 17 text world)
    | .returned (.stopped code) world =>
      (if code == 0 then "abort" else s!"exit:{code}",world)
  s!"{status}#{world.word.toNat}#{String.intercalate "," world.events}"

end NeuralReversi.Board.Transform.Fixture
