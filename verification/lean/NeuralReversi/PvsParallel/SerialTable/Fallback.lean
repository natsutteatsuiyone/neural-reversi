import NeuralReversi.PvsParallel.SerialTable.State
import NeuralReversi.PublicationMir.ExactScore

namespace NeuralReversi.PvsParallel.SerialTable

open PvsBoardTree (Position minimax)
open PublicationMir.ExactScore (returned Ensures RowScores)
open TranspositionTable (EntryData)

theorem Image.lookup_source (im : Image l) (support : im.Supported h) (b : Position)
    (hit : im.lookup b = some e) : Nonempty (Cutoffs.Lookup h b e) := by
  cases first : im.lookupSlot b 0 with
  | none =>
    have back : im.lookupSlot b 1 = some e := by simpa [Image.lookup,first] using hit
    exact (back ▸ im.lookup_reads support b 1).published
  | some front =>
    have key : front = e := by simpa [Image.lookup,first] using hit
    cases key
    exact (first ▸ im.lookup_reads support b 0).published

def PublicContract : Prop :=
  ∀ (l : Layout) (t : Table l) mc fc (b : Position) evaluate sort last nodes exit result, t.Good mc fc →
    returned mc fc b evaluate sort last (t.image.lookup b) nodes exit = some result →
    (exit = .search → ∀ it, last = some it → it.endgame = true → it.selectivity = 5 →
      RowScores (minimax mc fc b) it.rows) →
    Ensures mc fc b result

/-- A fallback needs only the state left by the searches that used this table. -/
theorem public_score : PublicContract := by
  intro l t mc fc b evaluate sort last nodes exit result good run rows
  exact PublicationMir.ExactScore.contract_proven _ _ _ _ _ _ _ _ _ _ run rows fun _ _ _ _ hit _ =>
    ⟨t.history,good.publications,t.image.lookup_source good.supported b hit⟩

end NeuralReversi.PvsParallel.SerialTable
