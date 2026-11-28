module
public import NeuralReversi.AspirationMir.SmallRoot

@[expose] public section

namespace NeuralReversi.AspirationMir.SmallRoot

open Bitboard (Square)
open RootMoveStorage (Row)
open PvsBoardTree (Position legalSquares)
open PvsParallel
open EndgameSearch.Cancellation (EmptyState)

/-- The lower-selectivity steps of a line can reorder its suffix and change its scores.
Only the previously selected prefix and the set of candidates are needed here. -/
structure LineInputs where
  prepare : Nat → List Row → List Row
  candidates : ∀ i rows, ((prepare i rows).map Row.square).Perm (rows.map Row.square)
  prefixRows : ∀ i rows, (prepare i rows).take i = rows.take i
  machine : Nat → List Row → Machine
  entry : ∀ i rows, (machine i rows).pc = 0
  window : ∀ i rows, Aspiration.Window (machine i rows).regs.alpha.toInt (machine i rows).regs.beta.toInt
  key : Nat → Nat → Word → Word → Square → BitVec 32
  pv : Nat → Square → List Square

def lineTrial (cfg : SmallPv.Config mc) (board : Position) (state : EmptyState)
    (inputs : LineInputs) (order : RootFrameMir.LibraryOrder) (idx : Nat) (rows : List Row)
    (i : Nat) (a z : Word) :=
  trial cfg board state (inputs.prepare idx rows) idx (inputs.key idx i a z) order (inputs.pv idx) a z

inductive LineRuns (cfg : SmallPv.Config mc) (board : Position) (state : EmptyState)
    (inputs : LineInputs) (order : RootFrameMir.LibraryOrder) : Nat → List Row → List Row → Prop
  | done (rows : List Row) : LineRuns cfg board state inputs order rows.length rows rows
  | next {idx rows final raw last answer} :
      Returns (queryFor (lineTrial cfg board state inputs order idx rows)) (fun _ => false)
        (inputs.machine idx (inputs.prepare idx rows)) raw last →
      lineTrial cfg board state inputs order idx rows last.index last.alpha last.beta = some answer →
      LineRuns cfg board state inputs order (idx + 1) (RootMoveStorage.sortRows answer.rows idx) final →
      LineRuns cfg board state inputs order idx rows final

private theorem finishLines (cfg : SmallPv.Config mc)
    (perm : ∀ b a z, (cfg.order b a z).Perm (legalSquares mc b.val)) (board : Position) (state : EmptyState)
    (aligned : state.matches board) (small : state.count.toNat ≤ 5)
    (inputs : LineInputs) (order : RootFrameMir.LibraryOrder) (remaining idx : Nat) (rows : List Row)
    (size : idx + remaining = rows.length) (covers : Root.Covers mc board (rows.map Row.square)) :
    ∃ final, LineRuns cfg board state inputs order idx rows final ∧ RootSuffix.Lines mc cfg.fc board idx rows final := by
  induction remaining generalizing idx rows with
  | zero =>
    have len : idx = rows.length := by omega
    subst idx
    exact ⟨rows, .done rows, .done rows⟩
  | succ n ih =>
    let prepared := inputs.prepare idx rows
    have names := inputs.candidates idx rows
    have fit : prepared.length = rows.length := by simpa [prepared] using names.length_eq
    have within : idx < prepared.length := by omega
    have legal : Root.Covers mc board (prepared.map Row.square) := names.trans covers
    obtain ⟨_, _, answer, _, ran, produced, _, _, written, inside, sound⟩ :=
      completeLine cfg perm board state prepared idx (inputs.machine idx prepared) (inputs.key idx)
        order (inputs.pv idx) legal within aligned small (inputs.entry idx prepared) (inputs.window idx prepared)
    have same := (written.scores (Root.roots_nodup legal)).1
    have after := (RootMoveStorage.sortRowsContract_proven answer.rows idx).1
    have covered : Root.Covers mc board ((RootMoveStorage.sortRows answer.rows idx).map Row.square) :=
      (after.map Row.square).trans (same ▸ legal)
    have len := after.length_eq
    have len' : answer.rows.length = prepared.length := by simpa using congrArg List.length same
    obtain ⟨final, rest, lines⟩ := ih (idx + 1) _ (by omega) covered
    exact ⟨final, .next ran produced rest, RootSuffix.Lines.acceptedWrites legal names (inputs.prefixRows idx rows)
      within written inside sound lines⟩

theorem publicMulti (c : SmallPv.Config mc) (board : Position) (state : EmptyState) (rows : List Row)
    (inputs : LineInputs) (childKey : Position → Int → Int → Square → BitVec 32)
    (order : RootFrameMir.LibraryOrder) (sort : PublicationMir.SliceSort)
    (nodes minimum : Nat) (prior : Option PublicationMir.Iteration) (fallback : Unit → PublicationMir.SearchResult)
    (reportable : minimum ≤ 5)
    (aligned : state.matches board) (small : state.count.toNat ≤ 5)
    (covers : Root.Covers mc board (rows.map Row.square)) (nonempty : rows ≠ []) :
    ∃ final result, LineRuns (withOrder c childKey order) board state inputs order 0 rows final ∧
      OuterMir.finish minimum 5 sort final state.count.toNat nodes prior fallback = some result ∧
      PublicationMir.score result = some (PvsBoardTree.minimax mc c.fc board) ∧
      ∀ row ∈ final, row.score = Root.squareValue mc c.fc board row.square := by
  obtain ⟨final, ran, lines⟩ := finishLines (withOrder c childKey order) (generatedOrder c childKey order)
    board state aligned small inputs order rows.length 0 rows (by simp) covers
  obtain ⟨result, finished, best⟩ := publicMultiPV lines covers nonempty sort state.count.toNat nodes fallback
  exact ⟨final, result, ran, by simpa [OuterMir.finish, reportable] using finished, best,
    (lines.exactRows covers (by simp)).2⟩

end NeuralReversi.AspirationMir.SmallRoot
