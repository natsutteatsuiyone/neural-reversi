import NeuralReversi.EndgameSearch.EmptyState
import NeuralReversi.Perft
import NeuralReversi.Eval.PatternUpdate.Storage

namespace NeuralReversi.SearchContextState

open PvsBoardTree (Position GeneratedSquare play pass)
open Eval.PatternUpdate
open EndgameSearch (EmptyStateMatches emptyStatePlayContract_proven emptyState_pass)

abbrev Features := Fin 32 → BitVec 16 × BitVec 16

structure EmptyState where
  links : EmptyListLinks.Links
  parity : BitVec 8
  count : BitVec 32

structure State where
  empties : EmptyState
  opponentTurn : Bool
  slots : Nat → Features
  nodes : Nat

def ply (s : State) : Nat := 60 - s.empties.count.toNat

def rebuild (board : Board.Board) : Features :=
  fun lane => wordPair (pairedCode (pattern lane) board)

def current (s : State) (lane : Fin 32) : BitVec 16 :=
  let feature := s.slots (ply s) lane
  if s.opponentTurn then feature.2 else feature.1

def EmptyState.matches (s : EmptyState) (board : Position) : Prop :=
  EmptyStateMatches board s.links s.parity s.count ∧ s.count.toNat ≤ 60

def Matches (s : State) (board : Position) : Prop :=
  s.empties.matches board ∧ s.slots (ply s) = rebuild (moverBoard s.opponentTurn board.val)

def CurrentContract : Prop :=
  ∀ s board, Matches s board → ∀ lane, current s lane = (rebuild board.val lane).1

theorem currentContract_proven : CurrentContract := by
  intro s board agrees lane
  unfold current
  rw [agrees.2]
  cases turn : s.opponentTurn <;> rfl

def withFeatures (empties : EmptyState) (side : Bool) (features : Features) : State :=
  ⟨empties, side, fun i => if i = 60 - empties.count.toNat then features else fun _ => (0, 0), 0⟩

def init (board : Position) : State :=
  let bits := Board.emptySquares board.val
  withFeatures ⟨EmptyListInit.newLinks bits, EmptyListParity.fromEmpties bits, Board.emptyCount board.val⟩
    false (rebuild board.val)

def InitialContract : Prop :=
  (∀ board, (Board.emptyCount board.val).toNat ≤ 60 → Matches (init board) board ∧ (init board).nodes = 0) ∧
  ∀ board empties side features, empties.matches board → features = rebuild (moverBoard side board.val) →
    Matches (withFeatures empties side features) board ∧ (withFeatures empties side features).nodes = 0

theorem initialContract_proven : InitialContract := by
  have seed (board : Position) (empties : EmptyState) (side : Bool) (features : Features)
      (empty : empties.matches board) (view : features = rebuild (moverBoard side board.val)) :
      Matches (withFeatures empties side features) board ∧ (withFeatures empties side features).nodes = 0 := by
    exact ⟨⟨empty, by simpa [withFeatures, ply] using view⟩, rfl⟩
  refine ⟨?_, seed⟩
  intro board fits
  exact seed board _ false _ ⟨EndgameSearch.emptyState_initial board, fits⟩ rfl

def remove (e : EmptyState) (sq : Bitboard.Square) : EmptyState :=
  ⟨EmptyListLinks.remove e.links sq, EmptyListParity.toggle e.parity sq, e.count - 1⟩

def restore (e : EmptyState) (sq : Bitboard.Square) : EmptyState :=
  ⟨EmptyListLinks.restore e.links sq, EmptyListParity.toggle e.parity sq, e.count + 1⟩

def shallow (s : State) (sq : Bitboard.Square) : State :=
  { s with empties := remove s.empties sq, nodes := s.nodes + 1 }

def undoShallow (s : State) (sq : Bitboard.Square) : State :=
  { s with empties := restore s.empties sq }

def placement (s : State) (sq : Bitboard.Square) (flipped : Bitboard.Bits) : State :=
  let next : Features := fun lane =>
    let cells := pattern lane
    let before := s.slots (ply s) lane
    scalarWords s.opponentTurn before.1 before.2
      (BitVec.ofInt 16 (lookupWeight sq.val cells)) (BitVec.ofInt 16 (flipCode cells flipped))
  { shallow s sq with opponentTurn := !s.opponentTurn, slots := storeSuccessor s.slots (ply s) next }

def undo (s : State) (sq : Bitboard.Square) : State :=
  { undoShallow s sq with opponentTurn := !s.opponentTurn }

def passTurn (s : State) : State := { s with opponentTurn := !s.opponentTurn, nodes := s.nodes + 1 }
def undoPass (s : State) : State := { s with opponentTurn := !s.opponentTurn }

def ShallowContract : Prop :=
  ∀ mc fc (board : Position) (move : GeneratedSquare mc board.val) s,
    s.empties.matches board →
    let child := shallow s move.val
    child.empties.matches (play mc fc board move).val ∧
    ply child = ply s + 1 ∧ ply child < 61 ∧
    child.slots = s.slots ∧ child.opponentTurn = s.opponentTurn ∧
    undoShallow child move.val = { s with nodes := s.nodes + 1 }

theorem shallowContract_proven : ShallowContract := by
  intro mc fc board move s agrees
  obtain ⟨positive, _, child, _, fits, links, parity, count⟩ :=
    emptyStatePlayContract_proven mc fc board move s.empties.links s.empties.parity s.empties.count agrees.1
  dsimp only
  have below := fits agrees.2
  have dec : (s.empties.count - 1).toNat = s.empties.count.toNat - 1 := by
    exact BitVec.toNat_sub_of_le (by rw [BitVec.le_def]; exact positive)
  refine ⟨⟨child, Nat.le_of_lt below⟩, ?_, ?_, rfl, rfl, ?_⟩
  · change 60 - (s.empties.count - 1).toNat = (60 - s.empties.count.toNat) + 1
    rw [dec]
    have := agrees.2
    omega
  · change 60 - (s.empties.count - 1).toNat < 61
    omega
  · cases s with
    | mk empties side slots nodes =>
      cases empties
      simp_all only [shallow, undoShallow, restore, remove]

private theorem compatible (mc fc) (board : Position) (move : GeneratedSquare mc board.val) :
    placeCompatible board.val move.val (Flip.Dispatch.resultValue fc move.val board.val.player board.val.opponent) := by
  obtain ⟨_, playerEmpty, opponentEmpty, _⟩ := (Board.getMoves_exact mc board.val _).mp move.property
  refine ⟨board.property, playerEmpty, opponentEmpty, ?_⟩
  intro i captured
  rw [Flip.Dispatch.value_disjoint_scalar _ _ _ _ board.property] at captured
  exact Flip.flipped_is_opponent _ _ _ _ captured

def PlacementContract : Prop :=
  ∀ mc fc (board : Position) (move : GeneratedSquare mc board.val) s, Matches s board →
    let flipped := Flip.Dispatch.resultValue fc move.val board.val.player board.val.opponent
    let child := placement s move.val flipped
    let restored := undo child move.val
    updateAllowed (ply s) flipped = true ∧ Matches child (play mc fc board move).val ∧
    ply child = ply s + 1 ∧ child.nodes = s.nodes + 1 ∧
    restored.empties = s.empties ∧ restored.opponentTurn = s.opponentTurn ∧
    restored.nodes = child.nodes ∧
    (∀ i, i ≠ ply s + 1 → restored.slots i = s.slots i) ∧ Matches restored board

theorem placementContract_proven : PlacementContract := by
  intro mc fc board move s agrees
  dsimp only
  let flipped := Flip.Dispatch.resultValue fc move.val board.val.player board.val.opponent
  let child := placement s move.val flipped
  let next := (play mc fc board move).val
  obtain ⟨emptyChild, plyNext, _, _, _, undoShallowSame⟩ :=
    shallowContract_proven mc fc board move s agrees.1
  have total : ply s + (Board.emptyCount board.val).toNat = 60 := by
    have fits := agrees.1.2
    have count := congrArg BitVec.toNat agrees.1.1.1
    unfold ply
    omega
  have guard := (Perft.plyContract_proven.2.1 mc fc board (ply s) move total).1
  have pairTurn : moverBoard s.opponentTurn (moverBoard s.opponentTurn board.val) = board.val := by
    cases s.opponentTurn <;> rfl
  have childTurn : rootAfter s.opponentTurn (moverBoard s.opponentTurn board.val) move.val flipped =
      moverBoard (!s.opponentTurn) next.val := by cases s.opponentTurn <;> rfl
  have featureChild : child.slots (ply child) = rebuild (moverBoard child.opponentTurn next.val) := by
    have childPly : ply child = ply s + 1 := plyNext
    rw [childPly]
    funext lane
    simp only [child, placement, storeSuccessor]
    rw [agrees.2, rebuild]
    have valid : placeCompatible (moverBoard s.opponentTurn (moverBoard s.opponentTurn board.val))
        move.val flipped := by
      rw [pairTurn]
      exact compatible mc fc board move
    have kernel := kernelRebuildContract_proven s.opponentTurn (moverBoard s.opponentTurn board.val)
      move.val flipped lane valid
    exact kernel.1.trans (congrArg (fun b => rebuild b lane) childTurn)
  have emptyRestore : (undo child move.val).empties = s.empties := by
    have restored := congrArg State.empties undoShallowSame
    exact restored
  have turnRestore : (undo child move.val).opponentTurn = s.opponentTurn := Bool.not_not _
  have parentSlot : ∀ i, i ≠ ply s + 1 → (undo child move.val).slots i = s.slots i := by
    intro i untouched
    simp [child, undo, undoShallow, placement, storeSuccessor, untouched]
  refine ⟨guard, ⟨emptyChild, featureChild⟩, plyNext, rfl, emptyRestore, turnRestore, rfl, parentSlot, ?_⟩
  refine ⟨emptyRestore.symm ▸ agrees.1, ?_⟩
  have samePly : ply (undo child move.val) = ply s := by
    unfold ply
    rw [emptyRestore]
  rw [samePly, parentSlot _ (by omega), turnRestore]
  exact agrees.2

def PassContract : Prop :=
  ∀ s board, Matches s board → Matches (passTurn s) (pass board) ∧
    ply (passTurn s) = ply s ∧ (passTurn s).slots = s.slots ∧
    undoPass (passTurn s) = { s with nodes := s.nodes + 1 }

theorem passContract_proven : PassContract := by
  intro s board agrees
  have turn : moverBoard (!s.opponentTurn) (pass board).val = moverBoard s.opponentTurn board.val := by
    cases s.opponentTurn <;> rfl
  refine ⟨⟨⟨emptyState_pass _ _ _ _ agrees.1.1, agrees.1.2⟩, ?_⟩, rfl, rfl, ?_⟩
  · exact agrees.2.trans (congrArg rebuild turn.symm)
  · simp [undoPass, passTurn]

def SplitContract : Prop :=
  ∀ s board, Matches s board →
    let copy := withFeatures s.empties s.opponentTurn (s.slots (ply s))
    Matches copy board ∧ copy.empties = s.empties ∧ copy.opponentTurn = s.opponentTurn ∧
    ply copy = ply s ∧ (∀ lane, current copy lane = current s lane) ∧ copy.nodes = 0

theorem splitContract_proven : SplitContract := by
  intro s board agrees
  have copied := initialContract_proven.2 board s.empties s.opponentTurn (s.slots (ply s)) agrees.1 agrees.2
  refine ⟨copied.1, rfl, rfl, rfl, ?_, rfl⟩
  intro lane
  exact (currentContract_proven _ _ copied.1 lane).trans (currentContract_proven _ _ agrees lane).symm

inductive Action where
  | move (sq : Bitboard.Square)
  | pass

def ValidTrace (mc : Bitboard.Moves.X86Config) (fc : Flip.Dispatch.X86Config) :
    List Action → Position → State → Position → State → Prop
  | [], board, start, last, finish => board = last ∧ start = finish
  | .move sq :: rest, board, start, last, finish =>
    ∃ move : GeneratedSquare mc board.val, move.val = sq ∧
      ValidTrace mc fc rest (play mc fc board move).val
        (placement start sq (Flip.Dispatch.resultValue fc sq board.val.player board.val.opponent)) last finish
  | .pass :: rest, board, start, last, finish =>
    ValidTrace mc fc rest (pass board) (passTurn start) last finish

def unwind (finish : State) : List Action → State
  | [] => finish
  | .move sq :: rest => undo (unwind finish rest) sq
  | .pass :: rest => undoPass (unwind finish rest)

def RollbackContract : Prop :=
  ∀ mc fc trace board start last finish, ValidTrace mc fc trace board start last finish → Matches start board →
    let restored := unwind finish trace
    Matches finish last ∧ restored.empties = start.empties ∧ restored.opponentTurn = start.opponentTurn ∧
    (∀ i, i ≤ ply start → restored.slots i = start.slots i) ∧ restored.nodes = finish.nodes

theorem rollbackContract_proven : RollbackContract := by
  intro mc fc trace
  induction trace with
  | nil =>
    intro board start last finish valid agrees
    rw [ValidTrace] at valid
    obtain ⟨rfl, rfl⟩ := valid
    exact ⟨agrees, rfl, rfl, fun _ _ => rfl, rfl⟩
  | cons action rest ih =>
    intro board start last finish valid agrees
    cases action with
    | move sq =>
      rw [ValidTrace] at valid
      obtain ⟨move, rfl, tail⟩ := valid
      let flipped := Flip.Dispatch.resultValue fc move.val board.val.player board.val.opponent
      let child := placement start move.val flipped
      obtain ⟨_, childMatches, childPly, _, oldEmpties, oldTurn, _, unchanged, _⟩ :=
        placementContract_proven mc fc board move start agrees
      obtain ⟨terminal, emptyRestored, turnRestored, slotsRestored, nodes⟩ := ih _ _ _ _ tail childMatches
      refine ⟨terminal, ?_, ?_, ?_, nodes⟩
      · change restore (unwind finish rest).empties move.val = start.empties
        rw [emptyRestored]
        exact oldEmpties
      · change (!(unwind finish rest).opponentTurn) = start.opponentTurn
        rw [turnRestored]
        exact oldTurn
      · intro i parent
        change (unwind finish rest).slots i = start.slots i
        have plyLe : i ≤ ply child := by change ply child = ply start + 1 at childPly; omega
        exact (slotsRestored i plyLe).trans (unchanged i (by omega))
    | pass =>
      rw [ValidTrace] at valid
      obtain ⟨child, _, _, _⟩ := passContract_proven start board agrees
      obtain ⟨terminal, empties, turn, slots, nodes⟩ := ih _ _ _ _ valid child
      refine ⟨terminal, empties, ?_, slots, nodes⟩
      change (!(unwind finish rest).opponentTurn) = start.opponentTurn
      rw [turn]
      exact Bool.not_not _

end NeuralReversi.SearchContextState
