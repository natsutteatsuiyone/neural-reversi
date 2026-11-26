import NeuralReversi.AspirationMir.OneEmpty
import NeuralReversi.PublicationMir.Correct
import NeuralReversi.PvsParallel.RootSuffix.Writes

namespace NeuralReversi.AspirationMir

open Bitboard (Square)
open RootMoveStorage (Row)
open PvsParallel

abbrev RootWrite := Square × Int × Bool

def applyRootWrite (pv : Square → List Square) (rows : List Row) (w : RootWrite) : List Row :=
  RootSuffix.writeFirst w.1 (fun row => RootSuffix.updateRow row w.2.1 w.2.2 (pv w.1)) rows

private def installScore (scores : Square → Int) (w : RootWrite) : Square → Int :=
  RootOrder.writeScore scores w.1 (if w.2.2 then w.2.1 else -PvsMoveLoop.scoreInf)

private theorem singletonScores (writes : List RootWrite) (pv : Square → List Square) (row : Row)
    (scores : Square → Int) (prior : row.score = scores row.square) :
    ∃ final, writes.foldl (applyRootWrite pv) [row] = [final] ∧ final.square = row.square ∧
      final.score = writes.foldl installScore scores row.square := by
  induction writes generalizing row scores with
  | nil => exact ⟨row, rfl, rfl, prior⟩
  | cons w rest ih =>
    obtain ⟨sq, s, visible⟩ := w
    by_cases hit : row.square = sq
    · let new := RootSuffix.updateRow row s visible (pv sq)
      have current : new.score = installScore scores (sq, s, visible) new.square := by
        simp [new, RootSuffix.updateRow, installScore, RootOrder.writeScore, hit]
      simpa [applyRootWrite, RootSuffix.writeFirst, hit, new, RootSuffix.updateRow] using ih new _ current
    · have current : row.score = installScore scores (sq, s, visible) row.square := by
        simpa [installScore, RootOrder.writeScore, hit] using prior
      simpa [applyRootWrite, RootSuffix.writeFirst, hit] using ih row _ current

/-- The physical root rows immediately before the accepted trial may contain arbitrary
scores and history. Apply the trial's reset and actual row updates before publication. -/
def PublicOneEmptyContract : Prop :=
  ∀ (mc : Bitboard.Moves.X86Config) (fc : Flip.Dispatch.X86Config)
    (b : PvsBoardTree.Position) (rows : List Row) (_ : rows ≠ [])
    (_ : Root.Covers mc b (rows.map Row.square)) (_ : ExactDepth.Full b 1)
    (paired checked : Bool) (mem : MoveList.Initialization.Memory Move.Move)
    (m : Machine) (_ : m.pc = 0) (_ : Aspiration.Window m.regs.alpha.toInt m.regs.beta.toInt)
    (pv : Square → List Square) (sort : PublicationMir.SliceSort) (nodes : Nat)
    (fallback : Unit → PublicationMir.SearchResult),
    ∃ writes value result score last,
      oneEmptyExit mc fc b.val (rows.map Row.square) paired checked mem = some (writes, value) ∧
      Returns (fun _ _ _ => BitVec.ofInt 32 value) (fun _ => false) m score last ∧
      PublicationMir.completed sort (writes.foldl (applyRootWrite pv) (RootMoveStorage.resetFrom rows 0))
        1 5 true nodes fallback = some result ∧
      PublicationMir.score result = some score.toInt ∧ score.toInt = PvsBoardTree.minimax mc fc b

theorem publicOneEmpty : PublicOneEmptyContract := by
  intro mc fc b rows nonempty covers one paired checked mem m entry window pv sort nodes fallback
  have size := covers.length_eq
  simp only [List.length_map] at size
  have limit := legalMoveCount_le_empties mc b
  change (1 : BitVec 32) = Board.emptyCount b.val at one
  rw [← one] at limit
  obtain ⟨row, tail, rfl⟩ := List.exists_cons_of_ne_nil nonempty
  have empty : tail = [] := List.length_eq_zero_iff.mp (by simp at size; change _ ≤ 1 at limit; omega)
  subst tail
  obtain ⟨writes, value, generated, score, last, run, report, produced, best, exactScore⟩ :=
    oneEmptyResultContract_proven mc fc b [row.square] paired checked mem m (fun _ => []) 1 nodes
      covers (by simp) one entry window
  obtain ⟨final, updated, _, scoreEq⟩ := singletonScores writes pv (RootMoveStorage.invalidated row)
    (fun _ => -PvsMoveLoop.scoreInf) rfl
  have reportScore : report.best.score = writes.foldl installScore (fun _ => -PvsMoveLoop.scoreInf) row.square := by
    simp [SearchFallback.fromSnapshot, SearchPublication.capture] at produced
    cases produced
    rfl
  have receipt := PublicationMir.completedScore sort [final] 1 5 true nodes fallback
  simp [SearchPublication.capture, SearchFallback.fromSnapshot, PublicationMir.project] at receipt
  obtain ⟨result, returned, actual⟩ := receipt
  refine ⟨writes, value, result, score, last, generated, run, ?_, ?_, exactScore⟩
  · change PublicationMir.completed sort (writes.foldl (applyRootWrite pv) [RootMoveStorage.invalidated row])
      1 5 true nodes fallback = some result
    rw [updated, returned]
  · change final.score = writes.foldl installScore _ row.square at scoreEq
    rw [actual, scoreEq, ← reportScore, best]

/-- The accepted MIR call selects both the physical row writes and the score of this line.
The already selected prefix may be retained across arbitrary earlier selectivities. -/
theorem acceptedLine (m : Machine) (entry : m.pc = 0)
    (initialWindow : Aspiration.Window m.regs.alpha.toInt m.regs.beta.toInt)
    (finite : FiniteQueries query) (ran : Returns query (fun _ => false) m score last)
    (covers : Root.Covers mc board (initial.map Row.square))
    (order : (initial.map Row.square).Perm (before.map Row.square))
    (front : initial.take idx = before.take idx) (within : idx < initial.length)
    (writes : RootSuffix.Writes initial idx last.alpha.toInt last.beta.toInt scored latest)
    (sourceScore : latest.state.best = (query last.index last.alpha last.beta).toInt)
    (attempt : ConcurrentReplay.RootAttempt mc fc endian board (initial.map Row.square) idx
      last.alpha.toInt last.beta.toInt latest.state.best latest.scores)
    (next : RootSuffix.Lines mc fc board (idx + 1) (RootMoveStorage.sortRows scored idx) final) :
    RootSuffix.Lines mc fc board idx before final := by
  obtain ⟨⟨observed, low, high⟩, _⟩ := returned_inside ran entry
  have window := (returned_window ran entry initialWindow finite).2.2
  rw [observed] at sourceScore
  exact RootSuffix.Lines.acceptedWrites covers order front within writes
    (sourceScore.symm ▸ And.intro low high) (attempt.sound covers (by simpa using within) window) next

theorem publicMultiPV (lines : RootSuffix.Lines mc fc b 0 rows final)
    (covers : Root.Covers mc b (rows.map Row.square)) (nonempty : rows ≠ [])
    (sort : PublicationMir.SliceSort) (depth nodes : Nat) (fallback : Unit → PublicationMir.SearchResult) :
    ∃ result, PublicationMir.completed sort final depth 5 true nodes fallback = some result ∧
      PublicationMir.score result = some (PvsBoardTree.minimax mc fc b) := by
  obtain ⟨report, captured, best, _⟩ := RootSuffix.completedResult lines covers nonempty depth nodes
  obtain ⟨result, returned, score⟩ := PublicationMir.completedScore_of_report sort final depth 5 true nodes fallback captured
  exact ⟨result, returned, best ▸ score⟩

/-- Publication reads the accepted trial's physical rows. Only their squares and scores
are used, so arbitrary prior PVs, history values, and ordering ties are allowed. -/
theorem publishSingle (covers : Root.Covers mc board (initial.map Row.square))
    (within : 0 < initial.length)
    (writes : RootSuffix.Writes initial 0 alpha beta rows latest)
    (inside : alpha < latest.state.best ∧ latest.state.best < beta)
    (attempt : RootOrder.SoundAttempt mc fc board (initial.map Row.square) 0 alpha beta
      latest.state.best latest.scores)
    (sort : PublicationMir.SliceSort) (depth nodes : Nat) (fallback : Unit → PublicationMir.SearchResult) :
    ∃ result, PublicationMir.completed sort rows depth 5 true nodes fallback = some result ∧
      PublicationMir.score result = some (PvsBoardTree.minimax mc fc board) := by
  obtain ⟨names, _, stored⟩ := writes.scores (Root.roots_nodup covers)
  simp only [List.drop_zero] at stored
  obtain ⟨head, tail, ordered, selected⟩ := attempt.2 inside.1 inside.2
  obtain ⟨perm, _, _, descending, _⟩ := RootOrder.sortContract_proven (initial.map Row.square) 0 latest.scores
    (Nat.zero_le _)
  simp only [List.drop_zero] at ordered descending
  rw [ordered] at descending perm
  have best := selected.2.1.trans (Root.value_zero covers (by simpa using within))
  obtain ⟨winner, present, eq⟩ := List.mem_map.mp
    (show head ∈ rows.map Row.square from names ▸ perm.mem_iff.mp (by simp))
  obtain ⟨report, captured, top, _, _⟩ := ExactEndgameScore.snapshotScoreContract_proven
    (rows.map PublicationMir.project) ⟨depth, 5⟩ (PvsBoardTree.minimax mc fc board) nodes (by
      intro r member
      obtain ⟨row, mem, rfl⟩ := List.mem_map.mp member
      change row.score ≤ _
      rw [stored row mem, ← best]
      have sq := perm.mem_iff.mpr (names ▸ List.mem_map_of_mem (f := Row.square) mem)
      rcases List.mem_cons.mp sq with same | later
      · rw [same]
        exact Int.le_refl _
      · exact (List.pairwise_cons.mp descending).1 row.square later)
    ⟨PublicationMir.project winner, List.mem_map_of_mem present, (stored winner present).trans (eq ▸ best)⟩
  obtain ⟨result, completed, score⟩ := PublicationMir.completedScore_of_report sort rows depth 5 true nodes
    fallback captured
  exact ⟨result, completed, top ▸ score⟩

end NeuralReversi.AspirationMir
