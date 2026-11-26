import NeuralReversi.PvsParallel.SplitActivationMir.Roster
import NeuralReversi.Threading.SplitStartup

namespace NeuralReversi.PvsParallel.SplitActivationMir

open Threading
open AtomicWord (Event Write Read SequencedBefore)
open SplitCompletion (Worker Action initial)
open CutoffVisibility (History HappensBefore)

inductive Command
  | clear (atStore : Event)
  | add (w : Worker) (atRmw : Event)
  | remove (w : Worker) (atRmw : Event)

def Command.apply (old : BitVec 64) : Command → Write
  | .clear e => ⟨e, 0, false, false⟩
  | .add w e => ⟨e, old ||| ((1 : BitVec 64) <<< w.val), false, true⟩
  | .remove w e => ⟨e, old &&& ~~~((1 : BitVec 64) <<< w.val), true, true⟩

def toMask (w : Worker) (clear rmw : Event) : Op → Option Command
  | .clear => some (.clear clear)
  | .add => some (.add w rmw)
  | .remove => some (.remove w rmw)
  | _ => none

def fromAction (a : Action) : List Command :=
  match a with
  | .book w e => enrollOps.filterMap (toMask w e e)
  | .finish w e => releaseOps.filterMap (toMask w e e)

def command (a : Action) : Command :=
  match a with
  | .book w e => .add w e
  | .finish w e => .remove w e

private theorem action_one (a : Action) : fromAction a = [command a] := by cases a <;> rfl

def code (owner : Worker) (clear mark : Event) (actions : List Action) : List Command :=
  initialOps.filterMap (toMask owner clear mark) ++ actions.flatMap fromAction

theorem code_eq (owner : Worker) (clear mark : Event) (actions : List Action) :
    code owner clear mark actions = .clear clear :: .add owner mark :: actions.map command := by
  change .clear clear :: .add owner mark :: actions.flatMap fromAction = _
  suffices h : actions.flatMap fromAction = actions.map command by rw [h]
  induction actions with
  | nil => rfl
  | cons a rest ih => simp only [List.flatMap_cons, action_one, List.singleton_append, List.map_cons, ih]

/-- The raw activation occupies one contiguous interval in this atomic object's modification
    order. Every RMW reads the immediately preceding value, including writes from earlier uses. -/
structure ModificationOrder (writes : List Write) (ops : List Command) where
  offset : Nat
  bound : offset + ops.length ≤ writes.length
  apply : ∀ n op, ops[n]? = some op →
    writes[offset+n]? = some (op.apply ((writes[offset+n-1]?).map Write.value |>.getD 0))

private theorem at_index (mo : ModificationOrder ws ops) {op : Command} (n : Nat) (h : ops[n]? = some op)
    (index : Fin ws.length) (eq : index.val = mo.offset + n) :
    ws.get index = op.apply (((ws[mo.offset+n-1]?).map Write.value).getD 0) := by
  have got := mo.apply n op h
  rw [← eq, List.getElem?_eq_getElem index.isLt] at got
  rw [← eq]; exact Option.some.inj got

def activation {writes : List Write} (owner : Worker) (clear mark : Event)
    (early : Startup (begin owner mark) first f) (later : Run f rest)
    (mo : ModificationOrder writes (code owner clear mark (first ++ rest))) :
    SplitCompletion.Activation writes := by
  let actions := first ++ rest
  have size : mo.offset + 2 + actions.length ≤ writes.length := by
    have b := mo.bound
    have len := congrArg List.length (code_eq owner clear mark (first ++ rest))
    simp only [List.length_cons, List.length_map] at len
    dsimp [actions]; omega
  let start : Fin writes.length := ⟨mo.offset+1, by omega⟩
  refine { owner := owner, start := start, actions := actions, endInBounds := by dsimp [start]; omega,
           legal := actions_legal owner mark early later, ownerBit := ?_, updates := ?_ }
  · have clearEq := mo.apply 0 (.clear clear) (by rw [code_eq]; rfl)
    have seed := at_index mo 1 (op := .add owner mark) (by rw [code_eq]; rfl) start rfl
    simp only [Nat.add_zero] at clearEq
    change (writes.get start).value = _
    rw [seed]; simp [clearEq, Command.apply, initial, Bitboard.squareMask]
  · intro i
    let dest : Fin writes.length := ⟨start.val+i.val+1, by dsimp [start]; have := i.isLt; omega⟩
    let prev : Fin writes.length := ⟨start.val+i.val, by dsimp [start]; have := i.isLt; omega⟩
    have getOp : (code owner clear mark (first ++ rest))[i.val+2]? = some (command (actions.get i)) := by
      rw [code_eq]
      change (actions.map command)[i.val]? = _
      simp
    have eq := at_index mo (i.val+2) getOp dest (by dsimp [dest,start]; omega)
    have previous : writes[mo.offset + (i.val+2) - 1]? = some (writes.get prev) := by
      rw [show mo.offset + (i.val+2) - 1 = prev.val by dsimp [prev,start]; omega]
      exact List.getElem?_eq_getElem prev.isLt
    rw [previous] at eq
    change (writes.get dest).event = _ ∧ (writes.get dest).release = _ ∧ (writes.get dest).rmw = _ ∧
      (writes.get dest).value = (actions.get i).updateBits (writes.get prev).value
    rw [eq]
    cases actions.get i <;> exact ⟨rfl,rfl,rfl,rfl⟩

/-- An existing job whose searching flag is still busy at the protected recheck excludes a
    successful booking. If the flag changed under an intervening lock hold, the former worker's
    release happens before the new assignment. -/
theorem booking_orders_old_release (late available : Bool) (history : History) (words : List Write)
    (read : Read words) (oldRelease nextBooking : Event)
    (checked : SplitStartup.BookingCheck history read oldRelease)
    (admitted : gate late available (decide ((words.get read.source).value = 0)) = true)
    (next : SequencedBefore read.event nextBooking) : HappensBefore history oldRelease nextBooking := by
  have idle : (words.get read.source).value = 0 := by cases late <;> simpa [gate] using
    (show decide ((words.get read.source).value = 0) = true from by
      first | exact admitted | exact (Bool.and_eq_true_iff.mp admitted).2)
  exact SplitStartup.rebookingContract_proven history words read oldRelease nextBooking checked idle next

end NeuralReversi.PvsParallel.SplitActivationMir
