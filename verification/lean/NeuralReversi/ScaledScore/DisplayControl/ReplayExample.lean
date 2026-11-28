module
public import NeuralReversi.ScaledScore.DisplayControl.Replay

@[expose] public section

namespace NeuralReversi.ScaledScore.DisplayControl.ReplayExample
set_option maxRecDepth 8192
set_option maxHeartbeats 2000000
abbrev Word64 := BitVec 64
abbrev ExampleOracle := Oracle Word64 Word64 Nat Unit Unit Unit Bool Word64 Nat Word64
abbrev ExampleConfiguration := State Word64 Word64 Nat Unit Unit Unit Bool × Word64
abbrev ExampleInvocation := Invocation Word64 Word64 Nat

def effect (inv : ExampleInvocation) (world : Word64) : Word64 :=
  world * 3#64 + inv.reference.address + inv.byteLength + inv.scope.receiver

def replyAllowed (world : Word64) : Reply Unit Unit → Prop
  | .ok => world.getLsbD 0 = false
  | .error => world.getLsbD 0 = true
  | .unwind _ | .stopped _ => False

def expectedReply (world : Word64) : Reply Unit Unit :=
  if world.getLsbD 0 then .error else .ok

theorem expectedReply_allowed (world : Word64) : replyAllowed world (expectedReply world) := by
  unfold expectedReply
  split <;> simp_all [replyAllowed]

def oracle (address : Word64) : ExampleOracle where
  initial := fun _ ordinal _ _ => (⟨ordinal,address⟩,(),false)
  internal := fun inv _ loc world event local' world' =>
    match event with
    | .effect observed =>
      loc = false ∧ observed = effect inv world ∧ local' = true ∧ world' = effect inv world
    | .child _ _ => False
  complete := fun inv _ loc world reply observed world' =>
    loc = true ∧ replyAllowed world reply ∧ observed = (world ^^^ inv.reference.address) ∧ world' = world
  environment := fun _ _ _ _ => False
  stopping := fun _ _ _ _ _ => False

def checks (address : Word64) : Replay.Checks (oracle address) where
  internal := fun inv ctx loc world event local' world' => by
    cases event <;> dsimp [oracle] <;> infer_instance
  complete := fun inv ctx loc world reply observed world' => by
    cases reply <;> dsimp [oracle,replyAllowed] <;> infer_instance

def scope : Scope Word64 Word64 := ⟨0x3000#64,0x2000#64,0#32,{}⟩
def invocation (address : Word64) : ExampleInvocation := ⟨scope,0,"0.",⟨0,address⟩⟩
def start : ExampleConfiguration := (initialState scope,5#64)

def afterEntry (address : Word64) : ExampleConfiguration :=
  (.external (invocation address) ["00"] () false,5#64)

def afterEffect (address : Word64) : ExampleConfiguration :=
  (.external (invocation address) ["00"] () true,effect (invocation address) 5#64)

def choices (address : Word64) : List (Replay.Selection Unit Unit Bool Word64 Nat Word64) :=
  let updated := effect (invocation address) 5#64
  [.administrative,.internal (.effect updated) true updated,
    .complete (expectedReply updated) (updated ^^^ address) updated]

theorem entry_validated (address : Word64) :
    Replay.administrative (oracle address) start =
      some (.control (.enter (invocation address)),afterEntry address) := rfl

theorem effect_validated (address : Word64) :
    Replay.selectedInternal (oracle address) (checks address) (afterEntry address)
      (.effect (effect (invocation address) 5#64)) true (effect (invocation address) 5#64) =
      some (.callback (invocation address) (.effect (effect (invocation address) 5#64)),afterEffect address) := by
  apply (Replay.selectedInternal_exact _ _ _ _ _ _ _ _ _ _).mpr
  exact ⟨rfl,rfl,rfl,rfl⟩

theorem completion_validated (address : Word64) :
    let updated := effect (invocation address) 5#64
    Replay.selectedComplete (oracle address) (checks address) (afterEffect address)
      (expectedReply updated) (updated ^^^ address) updated =
      some (.completion (invocation address) (expectedReply updated) (updated ^^^ address),
        (resume (invocation address) ["00"] (expectedReply updated),updated)) := by
  dsimp only
  apply (Replay.selectedComplete_exact _ _ _ _ _ _ _ _ _ _).mpr
  exact ⟨rfl,expectedReply_allowed _,rfl,rfl⟩

theorem pointer_changes_reply :
    expectedReply (effect (invocation 0x1000#64) 5#64) = .error ∧
    expectedReply (effect (invocation 0x1001#64) 5#64) = .ok := by
  exact ⟨rfl,rfl⟩

theorem wrong_reply_rejected :
    Replay.selectedComplete (oracle 0x1000#64) (checks 0x1000#64) (afterEffect 0x1000#64)
      .ok (effect (invocation 0x1000#64) 5#64 ^^^ 0x1000#64)
      (effect (invocation 0x1000#64) 5#64) = none := rfl

theorem wrong_world_rejected :
    Replay.selectedInternal (oracle 0x1000#64) (checks 0x1000#64) (afterEntry 0x1000#64)
      (.effect (effect (invocation 0x1000#64) 5#64)) true 0#64 = none := rfl

theorem skipped_effect_rejected :
    Replay.selectedComplete (oracle 0x1000#64) (checks 0x1000#64) (afterEntry 0x1000#64)
      .error (5#64 ^^^ 0x1000#64) 5#64 = none := rfl

theorem duplicate_effect_rejected :
    Replay.selectedInternal (oracle 0x1000#64) (checks 0x1000#64) (afterEffect 0x1000#64)
      (.effect (effect (invocation 0x1000#64) (effect (invocation 0x1000#64) 5#64))) true
      (effect (invocation 0x1000#64) (effect (invocation 0x1000#64) 5#64)) = none := rfl

def expectedLabels (address : Word64) : List (Label Word64 Word64 Nat Unit Unit Nat Word64) :=
  let updated := effect (invocation address) 5#64
  [.control (.enter (invocation address)),.callback (invocation address) (.effect updated),
    .completion (invocation address) (expectedReply updated) (updated ^^^ address)]

theorem sequence_validated (address : Word64) :
    let updated := effect (invocation address) 5#64
    Replay.run (oracle address) (checks address) start (choices address) =
      some (expectedLabels address,(resume (invocation address) ["00"] (expectedReply updated),updated)) := by
  dsimp only [choices,expectedLabels]
  apply Replay.cons_accepted
  · exact entry_validated address
  · apply Replay.cons_accepted
    · exact effect_validated address
    · apply Replay.cons_accepted
      · exact completion_validated address
      · rfl

theorem sequence_is_actual_trace (address : Word64) :
    Trace (oracle address) start (expectedLabels address)
      (resume (invocation address) ["00"] (expectedReply (effect (invocation address) 5#64)),
       effect (invocation address) 5#64) :=
  Replay.run_sound _ _ (sequence_validated address)

end NeuralReversi.ScaledScore.DisplayControl.ReplayExample
