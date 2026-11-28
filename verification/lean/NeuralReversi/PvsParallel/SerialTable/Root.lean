module
public import NeuralReversi.PvsParallel.SerialTable.Writes
public import NeuralReversi.PvsParallel.SerialRootExecution

@[expose] public section

namespace NeuralReversi.PvsParallel.SerialTable.Root

open PvsBoardTree (Position GeneratedSquare)
open RootMoveStorage (Row)
open SerialCalls (Program Request)
open SerialTable.Writes (Proposal)
open AspirationMir.SmallRoot (Answer candidates)
open Bitboard (Square)

structure Exit where
  answer : Answer
  proposal : Option Proposal

def frame (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : Position) (d : ExactDepth.Depth)
    (paired : Bool) (mem : MoveList.Initialization.Memory Move.Move) (rows : List Row) (idx : Nat)
    (root : SerialRecursion.RootInput) (a z : BitVec 32) : Program Request Exit :=
  let squares := rows.map Row.square
  let g := MoveList.Generation.fill false paired fc b.val (Root.remainingMask mc b.val squares idx) mem
  match SerialRootCalls.entry g.len g.wipeout with
  | none => .fail
  | some (some sq) =>
    let (writes,value) := rootWipeoutExit sq
    .pure ⟨⟨value,writes.foldl (AspirationMir.applyRootWrite root.pvLine) (RootMoveStorage.resetFrom rows idx)⟩,none⟩
  | some none =>
    let moves := candidates mc fc b squares idx root.key root.sort
    ((SerialOwnerCalls.run Subtype.val z.toInt moves (RootReplay.start a.toInt)).mapRequests
      (SerialRootCalls.request mc fc b d)).map fun out =>
        let answer := SerialRootCalls.ownerAnswer rows idx root.pvLine out
        let write := (rootEpilogue idx out.2.state.best).1.map fun score =>
          ⟨b,d,score,TranspositionTable.classify true ⟨BitVec.ofInt 32 score⟩ ⟨a⟩ ⟨z⟩,
            (Writes.pvMove z.toInt out.1).map Subtype.val,true⟩
        ⟨answer,write⟩

theorem erase (mc fc) (b : Position) (d paired mem rows idx root a z) :
    (frame mc fc b d paired mem rows idx root a z).map Exit.answer =
      SerialRootCalls.frame mc fc b d paired mem rows idx root.key root.sort root.pvLine a z := by
  dsimp only [frame,SerialRootCalls.frame]
  generalize SerialRootCalls.entry _ _ = selected
  cases selected with
  | none => rfl
  | some m =>
    cases m with
    | some => rfl
    | none =>
      unfold SerialRootCalls.ordinary
      dsimp only
      generalize (SerialCalls.Program.mapRequests _ _) = code
      induction code with
      | fail | pure => rfl
      | call _ _ ih => exact congrArg (Program.call _) (funext ih)

theorem writes {mc fc b d paired mem rows root a z} (issued : Program.Runs (frame mc fc b d paired mem rows idx root a z) cs out)
    (saved : out.proposal = some w) :
    idx = 0 ∧ w.board = b ∧ w.depth = d ∧ w.score = out.answer.value ∧ w.ttPv = true ∧
      w.bound = TranspositionTable.classify true ⟨BitVec.ofInt 32 out.answer.value⟩ ⟨a⟩ ⟨z⟩ ∧
      TableReads.LegalChoice b w.move := by
  dsimp only [frame] at issued
  split at issued
  · cases issued
  · cases issued; contradiction
  · obtain ⟨owner,_,rfl⟩ := Program.runs_map issued
    simp only [rootEpilogue] at saved
    split at saved <;> rename_i guard
    · cases saved
      refine ⟨guard,rfl,rfl,rfl,rfl,rfl,?_⟩
      intro sq found
      obtain ⟨m,_,rfl⟩ := Option.map_eq_some_iff.mp found
      exact (Board.getMoves_exact mc b.val m.val).mp m.property
    · contradiction

end NeuralReversi.PvsParallel.SerialTable.Root
