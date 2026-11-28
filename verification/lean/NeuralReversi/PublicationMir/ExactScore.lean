module
public import NeuralReversi.EndgameTreeCache
public import NeuralReversi.PublicationMir.Correct
public import NeuralReversi.PvsParallel.ExactTableHistory

@[expose] public section

namespace NeuralReversi.PublicationMir.ExactScore

open Bitboard.Moves (X86Config)
open PvsBoardTree (Position minimax)
open TranspositionTable

/-- The caller can decide whether the guarantee applies using the returned fields alone. -/
def scoreIfExact : SearchResult → Option Int
  | .bestMove r => if r.endgame && r.stamp.selectivity == 5 then some r.best.score else none
  | .noLegalMove => none

/-- The public postcondition, independent of how the search obtained its result. -/
def Ensures (mc : X86Config) (fc : Flip.Dispatch.X86Config) (b : Position) (result : SearchResult) : Prop :=
  ∀ v, scoreIfExact result = some v → v = 256 * EndgameTreeCache.discValue mc fc b.val

/-- The root suffix is sorted before the completed snapshot is sorted by score. -/
def afterRootSort (sort : SliceSort) : SliceSort where
  byReverseKey k rows := sort.byReverseKey k (RootMoveStorage.sortRows rows 0)
  permutation k rows :=
    (sort.permutation k _).trans (RootMoveStorage.sortRowsContract_proven rows 0).1
  ordered k rows := sort.ordered k (RootMoveStorage.sortRows rows 0)

theorem completed_tagged (found : completed sort rows depth 5 true nodes fallback = some r) :
    scoreIfExact r = score r := by
  unfold completed record at found
  cases choice : first (capture sort rows depth 5 true) with
  | none => simp [choice] at found
  | some row =>
    simp [choice, intoResult] at found
    subst r
    rfl

def asResult : Option SearchFallback.Report → SearchResult
  | some r => .bestMove r
  | none => .noLegalMove

def cached (mc : X86Config) (board : Position) (nodes : Nat) (entry : EntryData) :
    Option SearchFallback.Report :=
  if entry.bound = Packing.boundByte .exact then do
    let sq ← Packing.decodeBestMove entry.bestMove
    if (Board.getMoves mc board.val).getLsbD sq.val then
      some (SearchFallback.fromCache
        ⟨sq, entry.score.toInt, .exact, ⟨entry.depth.toNat, selectivityRank entry.selectivity⟩, isEndgame entry⟩ nodes)
    else none
  else none

def quick (mc : X86Config) (fc : Flip.Dispatch.X86Config) (b : Position)
    (evaluate : Board.Board → Int) (nodes : Nat) : SearchResult :=
  asResult (SearchFallback.quickMove mc fc b evaluate nodes)

private theorem quick_unmarked (mc fc b evaluate nodes) :
    scoreIfExact (quick mc fc b evaluate nodes) = none := by
  have spec := (SearchFallback.resolveContract_proven mc fc b evaluate none nodes).2.2 (by simp)
  change (SearchFallback.quickMove mc fc b evaluate nodes = none ↔ _) ∧ _ at spec
  cases h : SearchFallback.quickMove mc fc b evaluate nodes with
  | none => simp [quick, h, asResult, scoreIfExact]
  | some r =>
    have flag := (spec.2 r h).2.2.2.2.2.2.1
    simp [quick, h, asResult, scoreIfExact, flag]

/-- The decoded entry supplies a legal move, the reported score and all three tags. -/
def CacheSource (mc : X86Config) (b : Position) (entry : EntryData) (v : Int) : Prop :=
  entry.bound = Packing.boundByte .exact ∧ isEndgame entry = true ∧ selectivityRank entry.selectivity = 5 ∧
    entry.score.toInt = v ∧ ∃ sq, Packing.decodeBestMove entry.bestMove = some sq ∧
      (Board.getMoves mc b.val).getLsbD sq.val = true

private theorem cached_source {mc b entry nodes r v}
    (hit : cached mc b nodes entry = some r) (marked : scoreIfExact (.bestMove r) = some v) :
    CacheSource mc b entry v := by
  by_cases exactTag : entry.bound = Packing.boundByte .exact
  · cases sq : Packing.decodeBestMove entry.bestMove with
    | none => simp [cached, exactTag, sq] at hit
    | some move =>
      rw [cached, ite_eq_left exactTag, sq] at hit
      change (if (Board.getMoves mc b.val).getLsbD move.val then some _ else none) = some r at hit
      split at hit <;> try contradiction
      replace hit := Option.some.inj hit
      subst r
      simp [scoreIfExact, SearchFallback.fromCache] at marked
      obtain ⟨⟨ending, level⟩, rfl⟩ := marked
      exact ⟨exactTag, ending, level, rfl, move, sq, by assumption⟩
  · simp [cached, exactTag] at hit

/-- Rows before the score sort. Single-PV may leave losing rows at the sentinel. -/
def RowScores (truth : Int) (rows : List RootMoveStorage.Row) : Prop :=
  (∀ row ∈ rows, row.score ≤ truth) ∧ ∃ row ∈ rows, row.score = truth

private theorem capture_source (sort : SliceSort) (it : Iteration) (nodes : Nat)
    (fallback : Unit → SearchResult) (r : SearchResult) (v : Int)
    (run : intoResult (some (capture sort it.rows it.depth it.selectivity it.endgame)) nodes fallback = some r)
    (marked : scoreIfExact r = some v) : it.endgame = true ∧ it.selectivity = 5 ∧ RowScores v it.rows := by
  let sorted := sort.byReverseKey key it.rows
  have perm : sorted.Perm it.rows := sort.permutation ..
  have order : sorted.Pairwise (fun a b => b.score ≤ a.score) := sort.ordered key it.rows
  cases h : sorted with
  | nil =>
    have empty : (capture sort it.rows it.depth it.selectivity it.endgame).rows = [] := h
    simp [intoResult, first, empty] at run
  | cons head tail =>
    have rows : (capture sort it.rows it.depth it.selectivity it.endgame).rows = head :: tail := h
    simp [intoResult, first, rows] at run
    subst r
    simp [scoreIfExact, build, capture] at marked
    obtain ⟨⟨ending, level⟩, rfl⟩ := marked
    rw [h] at perm order
    refine ⟨ending, level, ?_, head, perm.mem_iff.mp (by simp), rfl⟩
    intro row member
    rcases List.mem_cons.mp (perm.mem_iff.mpr member) with rfl | later
    · exact Int.le_refl _
    · exact (List.pairwise_cons.mp order).1 row later

inductive Exit where
  | search
  | disconnected
  | noLegalMove
  | randomOpening (sq : Bitboard.Square)

/-- Score and flag projection of the public return path. An iteration is represented
    by its rows immediately before capture and the metadata passed to capture. -/
def returned (mc : X86Config) (fc : Flip.Dispatch.X86Config) (b : Position)
    (evaluate : Board.Board → Int) (sort : SliceSort) (last : Option Iteration)
    (lookup : Option EntryData) (nodes : Nat) : Exit → Option SearchResult
  | .search =>
    let saved := last.map fun it => capture sort it.rows it.depth it.selectivity it.endgame
    intoResult saved nodes fun _ =>
      match lookup.bind (cached mc b nodes) with
      | some r => .bestMove r
      | none => quick mc fc b evaluate nodes
  | .disconnected => some (quick mc fc b evaluate 0)
  | .noLegalMove => some .noLegalMove
  | .randomOpening sq => some (.bestMove ⟨⟨sq, 0, []⟩, [], ⟨0, 5⟩, false, 0⟩)

/-- Classify every flagged public score before using any property of the search or its history. -/
theorem origin (run : returned mc fc b evaluate sort last lookup nodes exit = some result)
    (marked : scoreIfExact result = some v) :
    exit = .search ∧
      ((∃ it, last = some it ∧ it.endgame = true ∧ it.selectivity = 5 ∧ RowScores v it.rows) ∨
       (last = none ∧ ∃ entry, lookup = some entry ∧ CacheSource mc b entry v)) := by
  cases exit with
  | disconnected =>
    simp [returned] at run
    subst result
    rw [quick_unmarked] at marked
    contradiction
  | noLegalMove | randomOpening _ =>
    simp [returned] at run
    subst result
    simp [scoreIfExact] at marked
  | search =>
    cases h : last with
    | some it =>
      simp only [returned, h, Option.map_some] at run
      exact ⟨rfl, .inl ⟨it, rfl, capture_source sort it nodes _ result v run marked⟩⟩
    | none =>
      cases raw : lookup.bind (cached mc b nodes) with
      | none =>
        simp [returned, h, intoResult, raw] at run
        subst result
        rw [quick_unmarked] at marked
        contradiction
      | some r =>
        simp [returned, h, intoResult, raw] at run
        subst result
        obtain ⟨entry, hit, decoded⟩ := Option.bind_eq_some_iff.mp raw
        exact ⟨rfl, .inr ⟨rfl, entry, hit, cached_source decoded marked⟩⟩

/-- Obligations for the selected source only. The table history can include earlier engine searches. -/
def Contract : Prop :=
  ∀ mc fc b evaluate sort last lookup nodes exit result,
    returned mc fc b evaluate sort last lookup nodes exit = some result →
    (exit = .search → ∀ it, last = some it → it.endgame = true → it.selectivity = 5 →
      RowScores (minimax mc fc b) it.rows) →
    (exit = .search → last = none → ∀ entry v, lookup = some entry → CacheSource mc b entry v →
      ∃ history, PvsParallel.ExactTableHistory.Publications mc fc history ∧
        Nonempty (PvsParallel.Cutoffs.Lookup history b entry)) →
    Ensures mc fc b result

theorem contract_proven : Contract := by
  intro mc fc b evaluate sort last lookup nodes exit result run rows reads v marked
  rw [EndgameTreeCache.scaled_discValue mc fc b]
  obtain ⟨searched, saved | table⟩ := origin run marked
  · obtain ⟨it, found, ending, level, upper, row, member, attained⟩ := saved
    obtain ⟨bound, winner, present, best⟩ := rows searched it found ending level
    exact Int.le_antisymm (attained ▸ bound row member) (best ▸ upper winner present)
  · obtain ⟨empty, entry, hit, source⟩ := table
    obtain ⟨history, writes, ⟨read⟩⟩ := reads searched empty entry v hit source
    obtain ⟨exactTag, ending, level, rfl, _⟩ := source
    rw [selectivityRank_eq_min] at level
    have eligible : PvsParallel.Cutoffs.ExactEndgame entry :=
      ⟨(byte_masks entry.flags).1 ▸ ending, by omega⟩
    have bounds := writes.sound b entry read.source eligible
    exact Int.le_antisymm (bounds.1 (by rw [exactTag]; decide)) (bounds.2 (by rw [exactTag]; decide))

end NeuralReversi.PublicationMir.ExactScore
