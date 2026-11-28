module
public import NeuralReversi.AspirationMir.SerialTable.SelectiveLadder
public import NeuralReversi.ShallowBoardTree
public import NeuralReversi.MidgameScoreRange.Domain

@[expose] public section

namespace NeuralReversi.AspirationMir.SerialTable.InitialEstimate

open PvsParallel
open PvsParallel.SerialTable (Layout Table Image)
open PvsBoardTree (Position)
open PvsMoveLoop (scoreMax)
open TranspositionTable

/-- A static evaluator's returned network word is clamped; at the last ply the board gives the disc score. -/
def static (raw : Position → Int) (b : Position) : {v : Int // -scoreMax ≤ v ∧ v ≤ scoreMax} :=
  if (Board.emptyCount b.val).toNat = 0 then
    ⟨256 * (Board.discScore b.val).toInt,by
      obtain ⟨_,lo,hi,_⟩ := Board.discScoreContract_proven b.val
      unfold scoreMax
      omega⟩
  else ⟨MidgameScoreRange.evalClamp (raw b),by
    have := MidgameScoreRange.eval_clamp_bounded (raw b)
    unfold scoreMax
    omega⟩

def fallback (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (raw : Position → Int) (b : Position) : Int :=
  if 20 ≤ (Board.emptyCount b.val).toNat then
    ShallowSearch.run (ShallowBoardTree.build mc fc (static raw) 2 b) (-32767) 32767
  else (static raw b).val

private theorem at_full {score reference : Int} (sound : Aspiration.SoundResult reference (-32767) 32767 score) :
    score = reference := sound.exact sound.lower sound.upper

theorem fallback_bounded (mc fc) (raw : Position → Int) (b : Position) :
    -scoreMax ≤ fallback mc fc raw b ∧ fallback mc fc raw b ≤ scoreMax := by
  unfold fallback
  split
  · let tree := ShallowBoardTree.build mc fc (static raw) 2 b
    have sound := ShallowSearch.contract_proven tree (-32767) 32767 ⟨by decide,by decide,by decide⟩
    exact at_full sound ▸ PvsMoveLoop.minimax_range tree
  · exact (static raw b).property

theorem two_ply_agrees (mc fc) (raw : Position → Int) (b : Position) (deep : 20 ≤ (Board.emptyCount b.val).toNat)
    (observed : ShallowBoardTree.Completed mc fc (static raw) 2 b (-32767) 32767 word) :
    fallback mc fc raw b = word := by
  have window : Aspiration.Window (-32767) 32767 := ⟨by decide,by decide,by decide⟩
  rw [fallback,ite_eq_left deep]
  exact (at_full (ShallowSearch.contract_proven _ _ _ window)).trans
    (at_full (ShallowBoardTree.replayContract_proven mc fc (static raw) 2 b (-32767) 32767 word observed window)).symm

/-- The public base estimate uses any matching exact-bound entry, irrespective of its search phase. -/
def estimate (im : Image l) (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (raw : Position → Int) (b : Position) : Word :=
  let data := match im.probe b with | .hit _ e => some e | .miss _ => none
  match data.filter (fun e => e.bound == Packing.boundByte .exact) with
  | some e => BitVec.ofInt 32 e.score.toInt
  | none => BitVec.ofInt 32 (fallback mc fc raw b)

private theorem empty_probe (l : Layout) (g : BitVec 8) (b : Position) : (Table.empty l g).image.probe b = .miss 0 := by
  have stable := ((Table.empty l g).image.stable_probe (Table.empty_supported l g) b).2
  have cell (i : Fin 2) : (Table.empty l g).image.cell b i = TranspositionTable.Clear.emptySlot := by
    simp [Image.cell,Image.get,Table.empty]
  rw [cell,cell] at stable
  have obs : ProbeRetry.classify (Cutoffs.key b) (PvsParallel.SerialTable.sample l.bigEndian TranspositionTable.Clear.emptySlot) = .empty := by
    cases l.bigEndian <;> rfl
  rw [obs] at stable
  exact (TranspositionTable.ReaderMir.ProbeExit.done.inj stable).symm

theorem cold_bounded (l : Layout) (g : BitVec 8) (mc fc) (raw : Position → Int) (b : Position) :
    let value := estimate (Table.empty l g).image mc fc raw b
    (-scoreMax ≤ value.toInt ∧ value.toInt ≤ scoreMax) := by
  dsimp only
  have bounds := fallback_bounded mc fc raw b
  simp only [estimate,empty_probe,Option.filter_none]
  have exactWord : (BitVec.ofInt 32 (fallback mc fc raw b)).toInt = fallback mc fc raw b :=
    BitVec.toInt_ofInt_eq_self (by decide) (by unfold scoreMax at bounds; omega) (by unfold scoreMax at bounds; omega)
  simpa only [exactWord] using bounds

end NeuralReversi.AspirationMir.SerialTable.InitialEstimate
