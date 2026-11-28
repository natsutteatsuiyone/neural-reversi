module
public import NeuralReversi.PvsParallel.SerialTable.Prefix
public import NeuralReversi.PvsParallel.SerialTable.State

@[expose] public section

namespace NeuralReversi.PvsParallel.SerialTable

open TranspositionTable
open SerialCalls.ScoreWords (Finite)
open EndgameSearch.CacheHistory (Pool)
open PvsBoardTree (Position)

/-- Weaker searches can use weaker bounds. All accepted engine writes have passed the store's
    finite score guard, including writes from previous generations. -/
def FiniteHistory (history : Cutoffs.History) : Prop :=
  ∀ b e, (b,e) ∈ history → Finite e.score.toInt

/-- The direct endgame seed accepts exact bounds from any selectivity. -/
def ExactScores (history : Cutoffs.History) : Prop :=
  ∀ b e, (b,e) ∈ history → e.bound = Packing.boundByte .exact →
    -PvsMoveLoop.scoreMax ≤ e.score.toInt ∧ e.score.toInt ≤ PvsMoveLoop.scoreMax

theorem proposal_score {b prev data old e} (made : Save.proposal b prev data old = some e) :
    e.score = data.score := by
  rw [(Save.contract_proven ..).1] at made
  split at made
  · contradiction
  · cases Option.some.inj made
    unfold Save.payload
    split <;> rfl

theorem Table.store_finite (t : Table l) (supported : t.image.Supported t.history) (earlier : FiniteHistory t.history)
    (i : l.Index) (b : Position) (score : Score) (bound : Bound) (depth : Depth) (mv : Option Bitboard.Square)
    (sel : Selectivity) (endgame ttPv : Bool) :
    FiniteHistory (t.store i b score bound depth mv sel endgame ttPv).history := by
  rw [Table.store,WriterMir.store_input]
  by_cases ok : score.raw.toInt.natAbs < 32767 <;> simp only [ok,ite_true,ite_false]
  · rw [t.save_eq _ _ _ supported]
    split
    next e made =>
      intro board entry inside
      rcases List.mem_append.mp inside with before | new
      · exact earlier _ _ before
      · obtain ⟨rfl,rfl⟩ := Prod.mk.inj (List.mem_singleton.mp new)
        rw [proposal_score made]
        change Finite (score.raw.setWidth 16).toInt
        have bounds : Finite score.raw.toInt := by
          unfold Finite
          omega
        rwa [StoredBound.narrow_score score bounds.1 bounds.2]
    · exact earlier
  · exact earlier

def Memory.Valid (p : Pool) (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (m : Memory l) : Prop :=
  m.Good p mc fc ∧ FiniteHistory m.tt.history

theorem Memory.cold_valid (l : Layout) (g : BitVec 8) (p : Pool) (mc fc) :
    (Memory.empty l g p).Valid p mc fc :=
  ⟨Memory.empty_good l g p mc fc,fun _ _ mem => by cases mem⟩

theorem Memory.apply_weak (m : Memory l) (valid : m.Valid p mc fc) (slot : l.Index) (sel : Selectivity)
    (weak : sel.val < 5) {proposal : Option Writes.Proposal}
    (legal : ∀ w, proposal = some w → TableReads.LegalChoice w.board w.move) :
    (m.apply slot proposal sel).Valid p mc fc := by
  cases proposal with
  | none => exact valid
  | some w =>
    exact ⟨⟨m.tt.store_other valid.1.1 slot w.board _ w.move _ _ _ true _ (Or.inr weak) (legal w rfl),
      valid.1.2⟩,m.tt.store_finite m.supported valid.2 _ _ _ _ _ _ _ _ _⟩

end NeuralReversi.PvsParallel.SerialTable
