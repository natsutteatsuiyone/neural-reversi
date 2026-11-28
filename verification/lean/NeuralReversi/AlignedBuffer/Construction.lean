module
public import NeuralReversi.AlignedBuffer.Allocation

@[expose] public section

namespace NeuralReversi.AlignedBuffer

inductive Event (Element : Type) where
  | item (value : Element)
  | short
  | panic

inductive FillStatus where
  | complete
  | short
  | panic
  deriving DecidableEq, Repr

structure FillReport (Element : Type) where
  status : FillStatus
  initialized : List Element
  deriving DecidableEq, Repr

def fill : Nat → List (Event Element) → FillReport Element
  | 0, _ => ⟨.complete, []⟩
  | _ + 1, [] | _ + 1, .short :: _ => ⟨.short, []⟩
  | _ + 1, .panic :: _ => ⟨.panic, []⟩
  | n + 1, .item value :: remaining =>
    let report := fill n remaining
    ⟨report.status, value :: report.initialized⟩

def yielded : List (Event Element) → List Element
  | .item value :: rest => value :: yielded rest
  | _ => []

def FillContract : Prop :=
  ∀ (Element : Type) (events : List (Event Element)) (capacity : Nat),
    let seen := fill capacity events
    seen.initialized = (yielded events).take capacity ∧
    (seen.status = .complete ↔ capacity ≤ (yielded events).length) ∧
    seen.initialized.length ≤ capacity

theorem fillContract_proven : FillContract := by
  have copied (Element : Type) (events : List (Event Element)) (capacity : Nat) :
      (fill capacity events).initialized = (yielded events).take capacity ∧
        ((fill capacity events).status = .complete ↔ capacity ≤ (yielded events).length) := by
    induction capacity generalizing events with
    | zero => simp [fill]
    | succ n ih =>
      cases events with
      | nil => simp [fill, yielded]
      | cons head rest =>
        cases head with
        | item value =>
          simpa [fill, yielded] using ih rest
        | short => simp [fill, yielded]
        | panic => simp [fill, yielded]
  intro Element events capacity
  obtain ⟨initial, complete⟩ := copied Element events capacity
  exact ⟨initial, complete, by rw [initial]; exact List.length_take_le _ _⟩

inductive Backing where
  | noAllocation
  | globalHeap
  | largePages
  deriving DecidableEq, Repr

structure Allocation where
  address : Nat
  capacity : Nat
  itemSize : Nat
  backing : Backing
  deriving DecidableEq, Repr

def origin (bytes : Nat) (largeProvider : Bool) : Backing :=
  if bytes = 0 then .noAllocation else if largeProvider then .largePages else .globalHeap

def Allocation.bytes (allocation : Allocation) : Nat := allocation.capacity * allocation.itemSize

structure Buffer (Element : Type) where
  allocation : Allocation
  contents : List Element

def publish (allocation : Allocation) (events : List (Event Element)) : Option (Buffer Element) :=
  let report := fill allocation.capacity events
  if report.status = .complete then some ⟨allocation, report.initialized⟩ else none

def publishAfterSource (allocation : Allocation) (events : List (Event Element))
    (sourcePanics : Bool) : Option (Buffer Element) :=
  if sourcePanics then none else publish allocation events

def TransferContract : Prop :=
  ∀ (Element : Type) (allocation : Allocation) (events : List (Event Element)),
    (∀ buffer, publish allocation events = some buffer →
      buffer.allocation = allocation ∧
      buffer.contents = (yielded events).take allocation.capacity ∧
      buffer.contents.length = allocation.capacity) ∧
    (publish allocation events = none ↔ (yielded events).length < allocation.capacity)

theorem transferContract_proven : TransferContract := by
  intro Element allocation events
  have ⟨copied, finished, _⟩ := fillContract_proven Element events allocation.capacity
  unfold publish
  dsimp only
  split
  · rename_i ready
    have enough := finished.mp ready
    refine ⟨fun buffer found => ?_, ?_⟩
    · cases Option.some.inj found
      exact ⟨rfl, copied, by rw [copied, List.length_take, Nat.min_eq_left enough]⟩
    · constructor
      · intro h
        cases h
      · intro h
        omega
  · rename_i stopped
    have tooShort : (yielded events).length < allocation.capacity :=
      Nat.lt_of_not_ge (stopped ∘ finished.mpr)
    constructor
    · intro _ bad
      cases bad
    · simp [tooShort]

def zeroValues (zeroValue : Element) (count : Nat) : List Element :=
  List.replicate count zeroValue

def ZeroContract : Prop :=
  ∀ (Element : Type) (zeroValue : Element) (count : Nat),
    (zeroValues zeroValue count).length = count ∧
      ∀ item ∈ zeroValues zeroValue count, item = zeroValue

theorem zeroContract_proven : ZeroContract := by
  intro Element zeroValue count
  exact ⟨List.length_replicate, fun item present =>
    (List.mem_replicate.mp present).2⟩

example : (fill 2 [Event.item 17, .item 23, .panic]).initialized = [17, 23] := by decide
example : (fill 2 [Event.item 17, .short, .item 23]).initialized = [17] := by decide
example : (fill 1 [Event.item 17, .panic]).status = .complete := by decide

end NeuralReversi.AlignedBuffer
