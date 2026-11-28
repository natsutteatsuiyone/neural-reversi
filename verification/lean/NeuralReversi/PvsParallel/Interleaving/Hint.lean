module
public import NeuralReversi.PvsParallel.Interleaving.Join
public import NeuralReversi.PvsParallel.Interleaving.CausalTable

@[expose] public section

namespace NeuralReversi.PvsParallel.Interleaving

open Threading
open PvsBoardTree (GeneratedSquare Position)
open TranspositionTable

/-- The best-move byte loaded after retirement belongs to the initial candidate or a
    reported move. All candidates in this split have been generated on the saved board. -/
theorem Join.move_operand {b : Position} (j : Join t done_ len start)
    (choices : List (GeneratedSquare mc b.val)) (initial : Option (GeneratedSquare mc b.val))
    (load : AtomicWord.Read words) (last : Fin words.length)
    (read : SplitReturn.Observation.ScoreRead j.history j.wait load last)
    (fields : SplitWindow.MoveStoreWindow Packing.encodeBestMove
      (fun i => choices[i]?.map Subtype.val) t.pv t.beta t.entry (initial.map Subtype.val)
      (Schedule.reports j.state) picked words last) :
    TableReads.LegalChoice b (Packing.decodeBestMove ((words.get load.source).value.setWidth 8)) := by
  have atEnd := (SplitReturn.Observation.scoreReadContract_proven j.history words load last j.wait read).1
  rw [atEnd,fields.lastValue,Packing.decodeBestMove_encodeBestMove]
  have legal (x : Option (GeneratedSquare mc b.val)) : TableReads.LegalChoice b (x.map Subtype.val) := by
    intro sq val
    obtain ⟨m,_,same⟩ := Option.map_eq_some_iff.mp val
    exact same ▸ (Board.getMoves_exact mc b.val _).mp m.property
  rcases fields.member with seed | ⟨r,_,sq⟩
  · exact seed ▸ legal initial
  · exact sq ▸ legal _

/-- The store site classifies the chosen score against the original window. Its input
    gate rejects sentinel scores before narrowing the word to sixteen bits. -/
theorem CausalTable.Input.native {chosen : CausalTable.Values} {b : Position} (pv : Bool) (lo hi score : Score)
    (d : Depth) (mv : Option Bitboard.Square) (gen : BitVec 8) (ttPv : Bool)
    (win : Aspiration.Window lo.raw.toInt hi.raw.toInt)
    (narrow : pv = false → hi.raw.toInt = lo.raw.toInt+1) (picked : chosen b pv lo.raw.toInt hi.raw.toInt score.raw.toInt)
    (moveWord : TableReads.LegalChoice b mv)
    (gate : WriterMir.storeInput (slot : Nat) b score (classify pv score lo hi) d mv Cutoffs.exactSelectivity gen true ttPv =
      some (slot,b,e)) : CausalTable.Input chosen b e := by
  rw [WriterMir.store_input] at gate
  split at gate
  · cases Option.some.inj gate
    exact .searched pv lo hi score d mv gen ttPv win narrow picked moveWord
  · contradiction

end NeuralReversi.PvsParallel.Interleaving
