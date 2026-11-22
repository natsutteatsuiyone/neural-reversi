import NeuralReversi.AlignedBuffer.Construction

namespace NeuralReversi.AlignedBuffer

inductive DropStatus where
  | returned
  | unwound
  | aborted
  deriving DecidableEq, Repr

structure DropReport (Element : Type) where
  status : DropStatus
  visited : List Element
  deriving DecidableEq, Repr

def dropAll (abortPolicy : Bool) (panics : Element → Bool) : Bool → List Element → DropReport Element
  | unwinding, [] => ⟨if unwinding then .unwound else .returned, []⟩
  | unwinding, value :: remaining =>
    if panics value && (abortPolicy || unwinding) then ⟨.aborted, [value]⟩
    else
      let tail := dropAll abortPolicy panics (unwinding || panics value) remaining
      ⟨tail.status, value :: tail.visited⟩

def NonfatalContract : Prop :=
  ∀ (Element : Type) (panics : Element → Bool) (values : List Element) (unwinding : Bool),
    values.countP panics + (if unwinding then 1 else 0) ≤ 1 →
    dropAll false panics unwinding values =
      ⟨if unwinding || values.any panics then .unwound else .returned, values⟩

theorem nonfatalContract_proven : NonfatalContract := by
  intro Element panics values unwinding count
  induction values generalizing unwinding with
  | nil => simp [dropAll]
  | cons item values ih =>
    rw [List.countP_cons] at count
    cases failure : panics item
    · have tail := ih unwinding (by simp [failure] at count; exact count)
      simp [dropAll, failure, tail]
    · cases unwinding
      · have tail := ih true (by simp [failure] at count ⊢; omega)
        simp [dropAll, failure, tail]
      · simp [failure] at count

def release (allocation : Allocation) : Option Allocation :=
  if allocation.backing = .noAllocation then none else some allocation

structure CleanupReport (Element : Type) where
  elements : DropReport Element
  released : Option Allocation
  deriving DecidableEq, Repr

def cleanup (allocation : Allocation) (abortPolicy unwinding : Bool)
    (panics : Element → Bool) (values : List Element) : CleanupReport Element :=
  let elements := dropAll abortPolicy panics unwinding values
  ⟨elements, if elements.status = .aborted then none else release allocation⟩

def ReleaseContract : Prop :=
  ∀ (Element : Type) (allocation : Allocation) (values : List Element) (panics : Element → Bool)
    (unwinding : Bool),
    values.countP panics + (if unwinding then 1 else 0) ≤ 1 →
    let result := cleanup allocation false unwinding panics values
    result.elements.visited = values ∧
    result.elements.status ≠ .aborted ∧
    result.released = release allocation ∧
    (∀ freed, result.released = some freed →
      freed.address = allocation.address ∧ freed.bytes = allocation.capacity * allocation.itemSize ∧
      freed.backing = allocation.backing)

theorem releaseContract_proven : ReleaseContract := by
  intro Element allocation values panics unwinding admissible
  have seen := nonfatalContract_proven Element panics values unwinding admissible
  simp only [cleanup, seen]
  split <;> simp [release, Allocation.bytes, and_self]

def FatalContract : Prop :=
  ∀ (Element : Type) (allocation : Allocation) (panics : Element → Bool) abortPolicy unwinding
      (value : Element) remaining,
    panics value = true → (abortPolicy || unwinding) = true →
    cleanup allocation abortPolicy unwinding panics (value :: remaining) = ⟨⟨.aborted, [value]⟩, none⟩

theorem fatalContract_proven : FatalContract := by
  intro Element allocation panics abortPolicy unwinding value remaining failure policy
  simp [cleanup, dropAll, failure, policy]

def InitializedCleanupContract : Prop :=
  ∀ (Element : Type) (allocation : Allocation) (events : List (Event Element))
    (panics : Element → Bool),
    let filled := fill allocation.capacity events
    (filled.status ≠ .complete → filled.initialized.countP panics = 0) →
    filled.initialized.countP panics ≤ 1 →
    let result := cleanup allocation false (filled.status != .complete) panics filled.initialized
    result.elements.visited = (yielded events).take allocation.capacity ∧
    result.released = release allocation

theorem initializedCleanupContract_proven : InitializedCleanupContract := by
  intro Element allocation events panics filled short atMost
  have admissible : filled.initialized.countP panics + (if filled.status != .complete then 1 else 0) ≤ 1 := by
    by_cases completed : filled.status = .complete
    · simpa [completed] using atMost
    · have none := short completed
      simp [none, completed]
  obtain ⟨visited, _, released, _⟩ :=
    releaseContract_proven Element allocation filled.initialized panics (filled.status != .complete) admissible
  have full := (fillContract_proven Element events allocation.capacity).1
  exact ⟨visited.trans full, released⟩

def LateSourceContract : Prop :=
  ∀ (Element : Type) (allocation : Allocation) (events : List (Event Element))
    (panics : Element → Bool),
    let filled := fill allocation.capacity events
    filled.status = .complete → filled.initialized.countP panics = 0 →
    publishAfterSource allocation events true = none ∧
      let result := cleanup allocation false true panics filled.initialized
      result.elements.visited.length = allocation.capacity ∧ result.released = release allocation

theorem lateSourceContract_proven : LateSourceContract := by
  intro Element allocation events panics filled full noPanic
  obtain ⟨visited, _, released, _⟩ :=
    releaseContract_proven Element allocation filled.initialized panics true (by simp [noPanic])
  obtain ⟨prefixValues, complete, _⟩ :=
    fillContract_proven Element events allocation.capacity
  refine ⟨rfl, ?_, released⟩
  rw [visited, prefixValues, List.length_take, Nat.min_eq_left (complete.mp full)]

structure Observation where
  completed : Bool
  interrupted : Bool
  allocationCount : Nat
  heapFrees : Nat
  largeFrees : Nat
  contents : List Nat
  drops : List Nat
  deriving DecidableEq, Repr

def fixture (large : Bool) (count : Nat) (ending : FillStatus) (stop failingDrop : Nat)
    (sourceDropPanic : Bool) : Observation :=
  let steps := (List.range stop).map Event.item ++
    (match ending with
     | .complete => []
     | .short => [Event.short]
     | .panic => [Event.panic])
  let filled := fill count steps
  let allocation : Allocation := ⟨64, count, 1, origin count large⟩
  let observed := cleanup allocation false (filled.status != .complete || sourceDropPanic)
    (fun item => item == failingDrop) filled.initialized
  let exposed := publishAfterSource allocation steps sourceDropPanic
  ⟨exposed.isSome, observed.elements.status != .returned,
    if allocation.backing = .noAllocation then 0 else 1,
    if observed.released.map Allocation.backing = some .globalHeap then 1 else 0,
    if observed.released.map Allocation.backing = some .largePages then 1 else 0,
    (exposed.map Buffer.contents).getD [], observed.elements.visited⟩

example : fixture false 3 .complete 3 0 false = ⟨true, true, 1, 1, 0, [0, 1, 2], [0, 1, 2]⟩ := by decide
example : fixture true 4 .panic 2 4 false = ⟨false, true, 1, 0, 1, [], [0, 1]⟩ := by decide
example : fixture true 4 .complete 4 4 true = ⟨false, true, 1, 0, 1, [], [0, 1, 2, 3]⟩ := by decide
example : (cleanup ⟨64, 4, 8, .globalHeap⟩ false true (fun i => i == 2) [0, 2, 3]).released =
    none := by decide

end NeuralReversi.AlignedBuffer
