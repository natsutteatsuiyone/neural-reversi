module
public import NeuralReversi.EndgameSearch.EmptyState
public import NeuralReversi.Perft
public import NeuralReversi.Eval.PatternUpdate.Storage

@[expose] public section

namespace NeuralReversi.SearchContextState

open PvsBoardTree (Position GeneratedSquare play pass)
open Eval.PatternUpdate
open EndgameSearch (EmptyStateMatches emptyStatePlayContract_proven)

abbrev Features := Fin 32 → BitVec 16 × BitVec 16

/-- `SearchContext::n_empties`. -/
structure EmptyState where
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
  s.count = Board.emptyCount board.val ∧ s.count.toNat ≤ 60

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
  withFeatures ⟨Board.emptyCount board.val⟩ false (rebuild board.val)

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
  exact seed board _ false _ ⟨rfl, fits⟩ rfl

def remove (e : EmptyState) : EmptyState := ⟨e.count - 1⟩

def restore (e : EmptyState) : EmptyState := ⟨e.count + 1⟩

def placement (s : State) (sq : Bitboard.Square) (flipped : Bitboard.Bits) : State :=
  let next : Features := fun lane =>
    let cells := pattern lane
    let before := s.slots (ply s) lane
    scalarWords s.opponentTurn before.1 before.2
      (BitVec.ofInt 16 (lookupWeight sq.val cells)) (BitVec.ofInt 16 (flipCode cells flipped))
  let moved : State := { s with empties := remove s.empties, nodes := s.nodes + 1 }
  { moved with opponentTurn := !s.opponentTurn, slots := storeSuccessor s.slots (ply s) next }

def undo (s : State) : State := { s with empties := restore s.empties, opponentTurn := !s.opponentTurn }

def passTurn (s : State) : State := { s with opponentTurn := !s.opponentTurn, nodes := s.nodes + 1 }
def undoPass (s : State) : State := { s with opponentTurn := !s.opponentTurn }

private theorem count_step (mc : Bitboard.Moves.X86Config) (fc : Flip.Dispatch.X86Config) (board : Position)
    (move : GeneratedSquare mc board.val) (e : EmptyState) (agrees : e.matches board) :
    (remove e).matches (play mc fc board move).val ∧ (remove e).count.toNat = e.count.toNat - 1 ∧
    0 < e.count.toNat ∧ restore (remove e) = e := by
  have aligned : EmptyStateMatches board (Empties.encode (Board.emptySquares board.val)) e.count :=
    ⟨agrees.1, Empties.encodes_encode _⟩
  obtain ⟨positive, child, fits, _, count⟩ := emptyStatePlayContract_proven mc fc board move _ _ aligned
  have dec : (e.count - 1).toNat = e.count.toNat - 1 :=
    BitVec.toNat_sub_of_le (by rw [BitVec.le_def]; exact positive)
  refine ⟨⟨child.1, ?_⟩, dec, positive, ?_⟩
  · have := agrees.2
    change (e.count - 1).toNat ≤ 60
    omega
  · cases e
    simp only [restore, remove] at count ⊢
    rw [count]

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
    let restored := undo child
    updateAllowed (ply s) flipped = true ∧ Matches child (play mc fc board move).val ∧
    ply child = ply s + 1 ∧ child.nodes = s.nodes + 1 ∧
    restored.empties = s.empties ∧ restored.opponentTurn = s.opponentTurn ∧
    restored.nodes = child.nodes ∧
    (∀ i, i ≠ ply s + 1 → restored.slots i = s.slots i) ∧ Matches restored board ∧ ply child < 61

theorem placementContract_proven : PlacementContract := by
  intro mc fc board move s agrees
  dsimp only
  let flipped := Flip.Dispatch.resultValue fc move.val board.val.player board.val.opponent
  let child := placement s move.val flipped
  let next := (play mc fc board move).val
  obtain ⟨emptyChild, dec, positive, restoredCount⟩ := count_step mc fc board move s.empties agrees.1
  have plyNext : ply child = ply s + 1 := by
    change 60 - (remove s.empties).count.toNat = (60 - s.empties.count.toNat) + 1
    rw [dec]
    have := agrees.1.2
    omega
  have total : ply s + (Board.emptyCount board.val).toNat = 60 := by
    have fits := agrees.1.2
    have count := congrArg BitVec.toNat agrees.1.1
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
  have emptyRestore : (undo child).empties = s.empties := restoredCount
  have turnRestore : (undo child).opponentTurn = s.opponentTurn := Bool.not_not _
  have parentSlot : ∀ i, i ≠ ply s + 1 → (undo child).slots i = s.slots i := by
    intro i untouched
    simp [child, undo, placement, storeSuccessor, untouched]
  refine ⟨guard, ⟨emptyChild, featureChild⟩, plyNext, rfl, emptyRestore, turnRestore, rfl, parentSlot, ?_, ?_⟩
  · refine ⟨emptyRestore.symm ▸ agrees.1, ?_⟩
    have samePly : ply (undo child) = ply s := by
      unfold ply
      rw [emptyRestore]
    rw [samePly, parentSlot _ (by omega), turnRestore]
    exact agrees.2
  · rw [plyNext]
    have := agrees.1.2
    unfold ply
    omega

def PassContract : Prop :=
  ∀ s board, Matches s board → Matches (passTurn s) (pass board) ∧
    ply (passTurn s) = ply s ∧ (passTurn s).slots = s.slots ∧
    undoPass (passTurn s) = { s with nodes := s.nodes + 1 }

theorem passContract_proven : PassContract := by
  intro s board agrees
  have turn : moverBoard (!s.opponentTurn) (pass board).val = moverBoard s.opponentTurn board.val := by
    cases s.opponentTurn <;> rfl
  have passCount : Board.emptyCount (pass board).val = Board.emptyCount board.val :=
    Board.switchPlayers_emptyCount board.val
  refine ⟨⟨⟨agrees.1.1.trans passCount.symm, agrees.1.2⟩, ?_⟩, rfl, rfl, ?_⟩
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
  | .move _ :: rest => undo (unwind finish rest)
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
      · change restore (unwind finish rest).empties = start.empties
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
