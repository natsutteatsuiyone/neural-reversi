import NeuralReversi.TranspositionTable.ReaderMir.SequenceCycle
import NeuralReversi.TranspositionTable.ReaderMir.ProbeProgram
import NeuralReversi.Board.Hash
import NeuralReversi.AspirationMir.PublicScore
import NeuralReversi.AspirationMir.NativeTrials
import NeuralReversi.PvsParallel.RootAspiration

namespace NeuralReversi.TranspositionTable.ReaderMir.SequenceCycleRegression

open PvsBoardTree
open PvsParallel.Cutoffs (exactSelectivity)
open SeqLock (Word)
open ReaderMir.SequenceCycle

private def mc : Bitboard.Moves.X86Config := ⟨false,false⟩
private def fc : Flip.Dispatch.X86Config := ⟨false,false,false⟩

private def first : Position := ⟨⟨0x7f7f676b527a40f8,0x00809894ad85bf07⟩,rfl⟩
private def last : Position := ⟨⟨0xc0c0f0f4daded280,0x3f3f070b25212d7f⟩,rfl⟩
private def moveA : GeneratedSquare mc first.val := ⟨63,by decide⟩
private def moveB : GeneratedSquare mc last.val := ⟨43,by decide⟩

private theorem solved (b : Position) (m : GeneratedSquare mc b.val)
    (onlyMove : legalSquares mc b.val = [m])
    (stuck : Board.hasLegalMoves mc (play mc fc b m).val.val = false)
    (other : Board.hasLegalMoves mc (Board.switchPlayers (play mc fc b m).val.val) = false) :
    minimax mc fc b = max (-PvsMoveLoop.scoreInf) (-(256 * Board.AwardedScore (play mc fc b m).val.val)) := by
  rw [minimax_moves mc fc b m [] onlyMove]
  simp only [PvsMoveLoop.bestValue,terminal_value mc fc _ (nil_of_stuck stuck) other]

set_option maxRecDepth 8192 in
private theorem valueA : minimax mc fc first = 6656 := by
  rw [solved first moveA (by decide) (by decide) (by decide)]
  decide
set_option maxRecDepth 8192 in
private theorem valueB : minimax mc fc last = -512 := by
  rw [solved last moveB (by decide) (by decide) (by decide)]
  decide

private def scoreA := Packing.newData ⟨6656⟩ (some 63) .exact 1 exactSelectivity 0 true true
private def scoreB := Packing.newData ⟨-512⟩ (some 43) .exact 1 exactSelectivity 0 true true

#guard Clear.getClusterIdx (Board.Hash.hash false first.val) 16 == 28
#guard Clear.getClusterIdx (Board.Hash.hash false last.val) 16 == 28
#guard match canCut true scoreA ⟨0⟩ 1 exactSelectivity true with | .ok true => true | _ => false
#guard match canCut true scoreB ⟨0⟩ 1 exactSelectivity true with | .ok true => true | _ => false

private def firstSlot (out : Outcome) : Pass :=
  let seen := out.returned.map fun s => ((s.player,s.opponent),Packing.unpack false s.metadata)
  probePass 0 (ProbeRetry.classify (last.val.player,last.val.opponent) seen) .busy

#guard (schedule false first.val.player first.val.opponent last.val.player last.val.opponent scoreA scoreB 128).any
  fun r => r.1.2.isNone && r.2.returned.isNone

/-- Both entries have the correct minimax scores before publication. Lookup
    and the first slot of probe accept the other board's exact entry. -/
theorem wrong_hits :
    scoreA.score.toInt = minimax mc fc first ∧ scoreB.score.toInt = minimax mc fc last ∧
    ∃ r, schedule false first.val.player first.val.opponent last.val.player last.val.opponent scoreA scoreB (2^63) = some r ∧
      r.1.2 = some scoreB ∧ scoreB.score.toInt ≠ minimax mc fc first ∧
      (firstSlot r.2).effects = [.read 0] ∧ (firstSlot r.2).exit = .done (.hit 0 scoreA) ∧
      scoreA.score.toInt ≠ minimax mc fc last := by
  refine ⟨valueA.symm,valueB.symm,_,full_cycle _ _ _ _ _ _ _ rfl rfl,rfl,?_,rfl,rfl,?_⟩
  · rw [valueA]; decide
  · rw [valueB]; decide

private def firstTrial (seed : EntryData) : AspirationMir.Machine :=
  let w := PvsParallel.RootAspiration.initialWindow 0 seed.score.toInt none
  AspirationMir.NativeTrials.initial (BitVec.ofInt 32 w.1) (BitVec.ofInt 32 w.2)

/-- Feed the other board's returned metadata into the first aspiration window.
    Check the retry loop, row update and public score of the one legal move. -/
private def SeedResult (b : Position) (square : Bitboard.Square) (seed : EntryData) (expected : Int) : Prop :=
  ∀ paired checked (mem : MoveList.Initialization.Memory Move.Move) (pv : Bitboard.Square → List Bitboard.Square)
    (sort : PublicationMir.SliceSort) (nodes : Nat) (fallback : Unit → PublicationMir.SearchResult),
    ∃ writes value last result,
      AspirationMir.oneEmptyExit mc fc b.val [square] paired checked mem = some (writes,value) ∧
      AspirationMir.Returns (fun _ _ _ => BitVec.ofInt 32 value) (fun _ => false)
        (firstTrial seed) (BitVec.ofInt 32 expected) last ∧
      PublicationMir.completed sort (writes.foldl (AspirationMir.applyRootWrite pv)
        (RootMoveStorage.resetFrom [RootMoveStorage.fresh square] 0)) 1 5 true nodes fallback = some result ∧
      PublicationMir.score result = some expected

private theorem oneEmptySeed {b : Position} {m : GeneratedSquare mc b.val} {seed : EntryData}
    (onlyMove : legalSquares mc b.val = [m]) (one : PvsParallel.ExactDepth.Full b 1)
    (window : Aspiration.Window (firstTrial seed).regs.alpha.toInt (firstTrial seed).regs.beta.toInt) :
    SeedResult b m.val seed (minimax mc fc b) := by
  intro paired checked mem pv sort nodes fallback
  let rows := [RootMoveStorage.fresh m.val]
  have covers : PvsParallel.Root.Covers mc b (rows.map RootMoveStorage.Row.square) := by
    unfold PvsParallel.Root.Covers
    rw [onlyMove]
    exact List.Perm.refl _
  obtain ⟨writes,value,result,word,last,used,ran,published,score,correct⟩ :=
    AspirationMir.publicOneEmpty mc fc b rows (by simp [rows]) covers one paired checked mem
      (firstTrial seed) rfl window pv sort nodes fallback
  refine ⟨writes,value,last,result,used,?_,published,score.trans (congrArg some correct)⟩
  simpa only [← correct,BitVec.ofInt_toInt] using ran

set_option maxRecDepth 8192 in
theorem wrong_seed_root_scores :
    SeedResult first 63 scoreB 6656 ∧ SeedResult last 43 scoreA (-512) := by
  rw [← valueA,← valueB]
  constructor
  · apply oneEmptySeed (m := moveA)
    · decide
    · rfl
    · exact ⟨by decide,by decide,by decide⟩
  · apply oneEmptySeed (m := moveB)
    · decide
    · rfl
    · exact ⟨by decide,by decide,by decide⟩

end NeuralReversi.TranspositionTable.ReaderMir.SequenceCycleRegression
