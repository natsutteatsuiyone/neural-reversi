module
public import NeuralReversi.Threading.SearchAbort

@[expose] public section

namespace NeuralReversi.Threading.StopToken

open SearchAbort (generation request requestFor Stopped)

structure Run where
  pool : Nat
  generation : BitVec 32
  deriving DecidableEq

structure Token where
  stopped : Bool
  attached : List Run

inductive Action where
  | stop
  | attach (run : Run)
  | detach (run : Run)

def transition (token : Token) : Action → Token × List Run
  | .stop => ({ token with stopped := true }, token.attached)
  | .attach run =>
    ({ token with attached := token.attached ++ [run] }, if token.stopped then [run] else [])
  | .detach run =>
    ({ token with attached := token.attached.filter (· != run) }, [])

def advance (token : Token) (actions : List Action) : Token :=
  actions.foldl (fun state action => (transition state action).1) token

def deliver (run : Run) (pool : Nat) (word : BitVec 64) : BitVec 64 :=
  if run.pool = pool then (requestFor word run.generation).getD word else word

def RegistrationContract : Prop :=
  (∀ token run, run ∈ (transition token .stop).2 ↔ run ∈ token.attached) ∧
  (∀ token run current,
    (current ∈ (transition token (.attach run)).2 ↔ token.stopped = true ∧ current = run) ∧
    (current ∈ (transition token (.attach run)).1.attached ↔ current ∈ token.attached ∨ current = run)) ∧
  (∀ token run current,
    current ∈ (transition token (.detach run)).1.attached ↔ current ∈ token.attached ∧ current ≠ run)

theorem registrationContract_proven : RegistrationContract := by
  refine ⟨fun _ _ => Iff.rfl, ?_, ?_⟩
  · intro token run current
    cases stopped : token.stopped <;> simp [transition, stopped]
  · intro token run current
    simp [transition]

theorem stopped_persists (token : Token) (actions : List Action) (once : token.stopped = true) :
    (advance token actions).stopped = true := by
  induction actions generalizing token with
  | nil => exact once
  | cons first rest ih =>
    apply ih
    cases first <;> simp [transition, once]

def DeliveryContract : Prop :=
  ∀ run pool word,
    (run.pool ≠ pool ∨ generation word ≠ run.generation → deliver run pool word = word) ∧
    (run.pool = pool → generation word = run.generation → Stopped (deliver run pool word))

theorem deliveryContract_proven : DeliveryContract := by
  intro run pool word
  constructor
  · intro stale
    rcases stale with other | generationMismatch
    · simp [deliver, other]
    · simp [deliver, requestFor, generationMismatch]
  · intro same current
    have guarded := (SearchAbort.generationWordContract_proven word).2.2.2.2.1 run.generation
    have sent : requestFor word run.generation = some (request word) := by
      simpa only [current, ite_true] using guarded
    simpa [deliver, same, sent] using SearchAbort.request_stops word

inductive FixtureAction where
  | begin (pool : Fin 2)
  | attach (pool : Fin 2)
  | detach (pool : Fin 2)
  | stop

structure FixtureState where
  token : Token
  word : Fin 2 → BitVec 64
  registered : Fin 2 → Option Run

def fixtureStart : FixtureState :=
  ⟨⟨false, []⟩, fun _ => 0xffffffff#64, fun _ => none⟩

def applyAction (state : FixtureState) (action : Action) : FixtureState :=
  let (token, requests) := transition state.token action
  {state with token, word := fun pool =>
    requests.foldl (fun word run => deliver run pool.val word) (state.word pool)}

def fixtureStep (state : FixtureState) : FixtureAction → Except String FixtureState
  | .begin pool => .ok {state with word := fun p =>
      if p = pool then SearchAbort.begin (state.word p) else state.word p}
  | .attach pool =>
    if (state.registered pool).isSome then .error "duplicate stop attachment"
    else
      let run : Run := ⟨pool.val, generation (state.word pool)⟩
      let next := applyAction state (.attach run)
      .ok {next with registered := fun p => if p = pool then some run else state.registered p}
  | .detach pool =>
    match state.registered pool with
    | none => .error "missing stop attachment"
    | some run =>
      let next := applyAction state (.detach run)
      .ok {next with registered := fun p => if p = pool then none else state.registered p}
  | .stop => .ok (applyAction state .stop)

def fixtureBits (state : FixtureState) : Nat :=
  (if Stopped (state.word 0) then 1 else 0) + (if Stopped (state.word 1) then 2 else 0)

end NeuralReversi.Threading.StopToken
