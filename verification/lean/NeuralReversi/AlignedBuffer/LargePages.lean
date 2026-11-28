module
public import NeuralReversi.AlignedBuffer.Allocation

@[expose] public section

namespace NeuralReversi.LargePages

open AlignedBuffer (pageRounded)

def nextMultiple (size page : Nat) : Nat :=
  if size % page = 0 then size else size + (page - size % page)

def requestSize (pointerBits size page : Nat) : Option Nat :=
  if page = 0 ∨ size < page then none
  else
    let rounded := nextMultiple size page
    if rounded < 2 ^ pointerBits then some rounded else none

private theorem nextMultiple_eq (size page : Nat) (positive : 0 < page) :
    nextMultiple size page = pageRounded size page := by
  have parts := Nat.mod_add_div size page
  have rem := Nat.mod_lt size positive
  unfold nextMultiple pageRounded
  split
  · rename_i exactPage
    have quotient : (size + page - 1) / page = size / page := by
      apply (Nat.div_eq_iff positive).2
      rw [Nat.mul_comm (size / page)]
      omega
    rw [quotient]
    omega
  · rename_i partialPage
    have quotient : (size + page - 1) / page = size / page + 1 := by
      apply (Nat.div_eq_iff positive).2
      rw [Nat.add_mul, Nat.one_mul, Nat.mul_comm (size / page)]
      omega
    rw [quotient, Nat.mul_add, Nat.mul_one]
    omega

def RoundingContract : Prop :=
  ∀ bits size page,
    (requestSize bits size page = none ↔
      page = 0 ∨ size < page ∨ 2 ^ bits ≤ pageRounded size page) ∧
    ∀ rounded, requestSize bits size page = some rounded ↔
      0 < page ∧ page ≤ size ∧ rounded = pageRounded size page ∧ rounded < 2 ^ bits

theorem roundingContract_proven : RoundingContract := by
  intro bits size page
  by_cases eligible : page = 0 ∨ size < page
  · refine ⟨⟨fun _ => ?_, fun _ => by simp [requestSize, eligible]⟩, ?_⟩
    · rcases eligible with zero | short
      · exact .inl zero
      · exact .inr (.inl short)
    · intro rounded
      simp [requestSize, eligible] <;> omega
  · have positive : 0 < page := by omega
    have large : page ≤ size := by omega
    simp only [requestSize, ite_eq_right eligible, nextMultiple_eq size page positive]
    by_cases fits : pageRounded size page < 2 ^ bits
    · simp [fits, positive, large, eq_comm, show ¬ 2 ^ bits ≤ pageRounded size page by omega]
      omega
    · simp [fits, show 2 ^ bits ≤ pageRounded size page by omega]
      omega

inductive WindowsCall where
  | openToken
  | lookup
  | enable
  | lastError
  | alloc (bytes : Nat)
  | restore (changed : Bool)
  | close
  | free
  deriving DecidableEq

structure WindowsReply where
  opened : Bool
  found : Bool
  adjusted : Bool
  assigned : Bool
  allocated : Bool
  alreadyEnabled : Bool

def windowsPrefix (bytes : Nat) (os : WindowsReply) : List WindowsCall :=
  [.openToken, .lookup] ++
    (if os.found then [.enable] else []) ++
    (if os.found && os.adjusted then [.lastError] else []) ++
    (if os.found && os.adjusted && os.assigned then [.alloc bytes] else [])

def windowsRun (request : Option Nat) (os : WindowsReply) (release : Bool) :
    Bool × List WindowsCall :=
  match request with
  | none => (false, [])
  | some bytes =>
    if os.opened then
      let privilege := os.found && os.adjusted && os.assigned
      let allocated := privilege && os.allocated
      (allocated, windowsPrefix bytes os ++ [.restore (privilege && !os.alreadyEnabled), .close] ++
        (if allocated && release then [.free] else []))
    else (false, [.openToken])

def WindowsContract : Prop :=
  (∀ request os release,
    let (ok, calls) := windowsRun request os release
    let privilege := os.found && os.adjusted && os.assigned
    (∀ bytes, WindowsCall.alloc bytes ∈ calls ↔
      request = some bytes ∧ os.opened = true ∧ privilege = true) ∧
    calls.count .close = (if request.isSome && os.opened then 1 else 0) ∧
    calls.count .free = (if ok && release then 1 else 0)) ∧
  ∀ bytes os release, os.opened = true →
    let (ok, calls) := windowsRun (some bytes) os release
    let privilege := os.found && os.adjusted && os.assigned
    ∃ before, calls = before ++ [.restore (privilege && !os.alreadyEnabled), .close] ++
      (if ok && release then [.free] else [])

theorem windowsContract_proven : WindowsContract := by
  constructor
  · intro request os release
    cases request with
    | none => simp [windowsRun]
    | some rounded =>
      cases opened : os.opened
      · simp [windowsRun, opened]
      · cases found : os.found <;> cases adjusted : os.adjusted <;> cases assigned : os.assigned <;>
          cases allocated : os.allocated <;> cases release <;>
          simp [windowsRun, windowsPrefix, opened, found, adjusted, assigned, allocated, eq_comm]
  · intro bytes os release opened
    rcases os with ⟨openedBit, found, adjusted, assigned, allocated, alreadyEnabled⟩
    cases openedBit
    · cases opened
    · exact ⟨windowsPrefix bytes ⟨true, found, adjusted, assigned, allocated, alreadyEnabled⟩, rfl⟩

inductive LinuxCall where
  | memalign (bytes : Nat)
  | advise (bytes : Nat)
  | zero (bytes : Nat)
  | free
  deriving DecidableEq

def linuxRun (request : Option Nat) (aligned advised release : Bool) : Bool × List LinuxCall :=
  match request with
  | none => (false, [])
  | some bytes =>
    if !aligned then (false, [.memalign bytes])
    else if !advised then (false, [.memalign bytes, .advise bytes, .free])
    else (true, [.memalign bytes, .advise bytes, .zero bytes] ++ (if release then [.free] else []))

def LinuxContract : Prop :=
  ∀ request aligned advised release,
    let (ok, calls) := linuxRun request aligned advised release
    (ok = true ↔ request.isSome = true ∧ aligned = true ∧ advised = true) ∧
    (∀ bytes, LinuxCall.zero bytes ∈ calls ↔ request = some bytes ∧ ok = true) ∧
    calls.count .free =
      (if request.isSome && aligned && (!advised || release) then 1 else 0)

theorem linuxContract_proven : LinuxContract := by
  intro request aligned advised release
  cases request <;> cases aligned <;> cases advised <;> cases release <;>
    simp [linuxRun, eq_comm]

end NeuralReversi.LargePages
