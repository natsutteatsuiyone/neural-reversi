module
public import NeuralReversi.Game.History.Types

@[expose] public section

namespace NeuralReversi.Game.History.VecView

abbrev Word := BitVec 64

structure Allocation where
  identity : Nat
  address : Word
  capacityWord : Word
  deriving DecidableEq, Repr

structure HistoryVec where
  entries : List Entry
  lengthWord : Word
  allocation : Allocation
  deriving DecidableEq, Repr

def AllocationValid (entryBytes : Nat) (a : Allocation) : Prop :=
  0 < entryBytes ∧ entryBytes % 8 = 0 ∧ a.address ≠ 0 ∧ a.address.toNat % 8 = 0 ∧
  a.capacityWord.toNat * entryBytes ≤ 2^63-1 ∧
  a.address.toNat + a.capacityWord.toNat * entryBytes < 2^64

def Valid (entryBytes : Nat) (v : HistoryVec) : Prop :=
  v.lengthWord.toNat = v.entries.length ∧
  v.lengthWord.toNat ≤ v.allocation.capacityWord.toNat ∧
  AllocationValid entryBytes v.allocation

theorem nonempty_of_nonzero (entryBytes : Nat) (v : HistoryVec) (valid : Valid entryBytes v)
    (nonzero : v.lengthWord ≠ 0) : v.entries ≠ [] := by
  intro empty
  apply nonzero
  apply BitVec.eq_of_toNat_eq
  simpa [empty] using valid.1

def pop (entryBytes : Nat) (v : HistoryVec) (valid : Valid entryBytes v) : Option Entry × HistoryVec :=
  if empty : v.lengthWord = 0 then (none,v)
  else
    let nextLength := v.lengthWord - 1
    let nextView := {v with entries := v.entries.dropLast,lengthWord := nextLength}
    let entry := v.entries.getLast (nonempty_of_nonzero entryBytes v valid empty)
    (some entry,nextView)

def confirm (entryBytes : Nat) (v : HistoryVec) (valid : Valid entryBytes v) (side : Disc) :
    Bool × HistoryVec :=
  if empty : v.lengthWord = 0 then (false,v)
  else
    let last := v.entries.getLast (nonempty_of_nonzero entryBytes v valid empty)
    let indexWord := v.lengthWord - 1
    if last.action = .automaticPass then
      if last.side = side then
        (true,{v with entries := v.entries.set indexWord.toNat {last with action := .explicitPass}})
      else (false,v)
    else (false,v)

def project (board : Board) (side : Disc) (v : HistoryVec) : ValueState :=
  ⟨board,side,v.entries⟩

structure LastBorrowView where
  value : Entry
  allocationIdentity : Nat
  indexWord : Word
  address : Word
  deriving DecidableEq, Repr

def lastMut (entryBytes : Nat) (v : HistoryVec) (valid : Valid entryBytes v) : Option LastBorrowView :=
  if empty : v.lengthWord = 0 then none
  else
    let index := v.lengthWord - 1
    some ⟨v.entries.getLast (nonempty_of_nonzero entryBytes v valid empty),
      v.allocation.identity,index,v.allocation.address + index * BitVec.ofNat 64 entryBytes⟩

end NeuralReversi.Game.History.VecView
