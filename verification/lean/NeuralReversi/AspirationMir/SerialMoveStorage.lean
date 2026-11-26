import NeuralReversi.AspirationMir.SerialStack

namespace NeuralReversi.AspirationMir.SerialMoveStorage

open PvsParallel
open MoveList
open MoveList.Initialization (Memory)
open SerialExecution (Inputs Frame)
open PvsBoardTree (Position)

abbrev Storage := Memory Move.Move

/-- Generation initializes the observable header and every candidate before use.
    Unused slots may contain different data in different stack activations. -/
theorem initialized (mc fc) (b : Position) (mask : Bitboard.Bits)
    (selected : Generation.IsSubset mask (Board.getMoves mc b.val)) (p q : Bool) (src dst : Storage) :
    let left := Generation.fill false p fc b.val mask src
    let right := Generation.fill false q fc b.val mask dst
    left.len = right.len ∧ left.wipeout = right.wipeout ∧
      ∀ i, i < (Generation.records fc b.val mask).length → left.slots i = right.slots i := by
  obtain ⟨length, slots, _, _, wipe⟩ := Generation.generationContract_proven mc fc b.val mask false p src
    b.property selected (by simp)
  obtain ⟨length', slots', _, _, wipe'⟩ := Generation.generationContract_proven mc fc b.val mask false q dst
    b.property selected (by simp)
  exact ⟨length.trans length'.symm, wipe.trans wipe'.symm, fun i lt => by rw [slots, slots', ite_eq_left lt, ite_eq_left lt]⟩

def replace (inputs : Inputs mc fc) (storage : Nat → Frame → Storage) : Inputs mc fc :=
  fun i f => {inputs i f with memory := storage i f}

/-- Replacing the incoming backing array in every activation preserves the suspended calls
    and all their continuations, including calls after a re-search. -/
theorem body (c : SerialScout.Config) (inputs : Inputs mc fc) (storage : Nat → Frame → Storage) :
    SerialExecution.body c (replace inputs storage) = SerialExecution.body c inputs := by
  funext i ctx
  let b := ctx.request.board
  have h := initialized mc fc b (Board.getMoves mc b.val) (fun _ x => x) c.paired c.paired
    (storage i ctx) (inputs i ctx).memory
  simp only [SerialExecution.body, replace, SerialRecursion.pvOrder, SerialPvCalls.frame, SerialPvCalls.action,
    PvEntryMir.run, SerialNonPvCalls.frame, SerialNonPvCalls.prelude, GeneralNonPvMir.entryAction,
    GeneralNonPvMir.runPrefix, PvEntryMir.generated]
  simp only [b] at h
  simp only [h.1, h.2.1]
  repeat' first | rfl | split

theorem attempt (c : SerialScout.Config) (inputs : Inputs mc fc) (storage : Nat → Frame → Storage)
    (opening : Nat) (b : Position) (root : SerialRecursion.RootInput) (rows : List RootMoveStorage.Row)
    (idx : Nat) (a z : BitVec 32) :
    SerialRootExecution.attempt c (replace inputs storage) opening b root rows idx a z =
      SerialRootExecution.attempt c inputs opening b root rows idx a z := by
  let mask := Root.remainingMask mc b.val (rows.map RootMoveStorage.Row.square) idx
  have selected : Generation.IsSubset mask (Board.getMoves mc b.val) := by
    intro i h
    simp only [mask, Root.remainingMask, BitVec.getLsbD_and, Bool.and_eq_true] at h
    exact h.1
  have h := initialized mc fc b mask selected c.paired c.paired
    (storage opening (SerialRootExecution.context b a z)) (inputs opening (SerialRootExecution.context b a z)).memory
  simp only [SerialRootExecution.attempt, SerialRootCalls.frame, replace]
  rw [h.1, h.2.1]

def RetryContract : Prop :=
  ∀ mc fc (c : SerialScout.Config) (b : Position) (inputs : Nat → Inputs mc fc)
    (roots : Nat → Word → Word → SerialRecursion.RootInput) (storage : Nat → Nat → Frame → Storage),
    SerialStack.trial c b (fun i => replace (inputs i) (storage i)) roots =
      SerialStack.trial c b inputs roots

theorem retries : RetryContract := by
  intro mc fc c b inputs roots storage
  funext s
  dsimp only [SerialStack.trial]
  rw [body, attempt]

end NeuralReversi.AspirationMir.SerialMoveStorage
