import NeuralReversi.Aspiration
import NeuralReversi.PrincipalVariation.Replay
import NeuralReversi.PrincipalVariation.Selection
import NeuralReversi.SearchFallback

namespace NeuralReversi.PrincipalVariation

open Bitboard (Square)
open Bitboard.Moves (X86Config)
open PvsBoardTree (Position GeneratedSquare play)
open SearchPublication (Row Stamp Snapshot Completed Attempt Exit capture finish published)

inductive Produced (mc : X86Config) (fc : FlipConfig) : Position → Buffer → Prop where
  | cleared {board : Position} {previous : Buffer} :
      previous.length = capacity → Produced mc fc board (clear previous)
  | forcedPass {board : Position} {line : Buffer} :
      Board.hasLegalMoves mc board.val = false → Board.hasLegalMoves mc (PvsBoardTree.pass board).val = true →
      Produced mc fc (PvsBoardTree.pass board) line → Produced mc fc board line
  | place {board : Position} {child current : Buffer} (square : GeneratedSquare mc board.val) :
      (Board.emptyCount board.val).toNat ≤ 60 → current.length = capacity →
      Produced mc fc (play mc fc board square).val child →
      Produced mc fc board (prepend square.val child current)

def ProducedContract : Prop :=
  ∀ (mc : X86Config) (fc : FlipConfig) (board : Position) (buffer : Buffer),
    Produced mc fc board buffer →
    buffer.length = capacity ∧ ∃ after, Legal mc fc board (visible buffer) after

theorem producedContract_proven : ProducedContract := by
  intro mc fc board buffer trace
  induction trace with
  | @cleared board previous size =>
    refine ⟨?_, board, .empty⟩
    simp only [clear, List.length_cons, List.length_drop, size, capacity]
  | @forcedPass board line stuck other _ ih =>
    obtain ⟨size, after, legal⟩ := ih
    refine ⟨size, ?_⟩
    cases moves : visible line with
    | nil => exact ⟨board, .empty⟩
    | cons first rest =>
      rw [moves] at legal
      exact ⟨after, .forcedPass stuck other legal⟩
  | @place board child current square count size _ ih =>
    obtain ⟨_, after, legal⟩ := ih
    exact ⟨(bufferContract_proven square.val child current size).2.1,
      after, (installContract_proven mc fc board after square child current count size legal).2.2⟩

def ProducerSelectionContract : Prop :=
  ∀ (mc : X86Config) (fc : FlipConfig) (board : Position) (square : GeneratedSquare mc board.val)
    (first root aborted : Bool) (alpha fresh locked scout wide : Int)
    (old completed carried current : Buffer),
    (Board.emptyCount board.val).toNat ≤ 60 → current.length = capacity →
    Produced mc fc (play mc fc board square).val completed →
    (serialChild root first aborted alpha scout wide old completed = some carried ∨
      (alpha ≤ locked ∧ splitChild aborted alpha fresh locked scout wide old completed = some carried)) →
    Produced mc fc board (prepend square.val carried current)

theorem producerSelectionContract_proven : ProducerSelectionContract := by
  intro mc fc board square first root aborted alpha fresh locked scout wide old completed carried current
    fits sized searched selected
  have same : carried = completed := by
    rcases selected with serial | ⟨ordered, split⟩
    · exact ((serialContract_proven root first aborted alpha scout wide old completed carried).2 serial).2
    · exact ((splitContract_proven aborted alpha fresh locked scout wide old completed carried ordered).2 split).2
  rw [same]
  exact .place square fits sized searched

def RowValid (mc : X86Config) (fc : FlipConfig) (board : Position) (row : Row) : Prop :=
  (Board.getMoves mc board.val).getLsbD row.square.val = true ∧
  (row.pv = [] ∨ (row.pv.head? = some row.square ∧ ∃ after, Legal mc fc board row.pv after))

def RowsValid (mc : X86Config) (fc : FlipConfig) (board : Position) (rows : List Row) : Prop :=
  ∀ row ∈ rows, RowValid mc fc board row

def writeRow (row : Row) (score : Int) (line : Option Buffer) : Row :=
  match line with
  | none => { row with score := -PvsMoveLoop.scoreInf }
  | some pv => { row with score := score, pv := visible pv }

def RowContract : Prop :=
  ∀ (mc : X86Config) (fc : FlipConfig) (board after : Position)
    (square : GeneratedSquare mc board.val) (row : Row) (score : Int) (child current : Buffer),
    row.square = square.val → (Board.emptyCount board.val).toNat ≤ 60 → current.length = capacity →
    let installed := prepend square.val child current
    RowValid mc fc board ⟨square.val, -PvsMoveLoop.scoreInf, []⟩ ∧
    (RowValid mc fc board row → RowValid mc fc board (writeRow row score none)) ∧
    (Legal mc fc (play mc fc board square).val (visible child) after →
      RowValid mc fc board (writeRow row score (some installed)) ∧
      (writeRow row score (some installed)).pv = square.val :: visible child)

theorem rowContract_proven : RowContract := by
  intro mc fc board after square row score child current same hCount fits installed
  refine ⟨⟨square.property, .inl rfl⟩, id, fun checked => ?_⟩
  obtain ⟨_, agrees, playable⟩ := installContract_proven mc fc board after square child current hCount fits checked
  have first : (visible installed).head? = some row.square := by rw [agrees, same]; rfl
  have moveLegal : (Board.getMoves mc board.val).getLsbD row.square.val = true := same.symm ▸ square.property
  exact ⟨⟨moveLegal, .inr ⟨first, after, playable⟩⟩, agrees⟩

def SavedValid (mc : X86Config) (fc : FlipConfig) (board : Position) (saved : Option Snapshot) : Prop :=
  ∀ snapshot, saved = some snapshot → RowsValid mc fc board snapshot.rows

private theorem valid_capture (good : RowsValid mc fc board rows) (stamp : Stamp) (endgame : Bool) :
    RowsValid mc fc board (capture rows stamp endgame).rows := by
  intro row included
  exact good row ((SearchPublication.captureContract_proven rows stamp endgame).2.2.1.mem_iff.mp included)

private theorem valid_run (mc : X86Config) (fc : FlipConfig) (board : Position)
    (minimum : Nat) (state : Completed) (attempts : List Attempt)
    (saved : SavedValid mc fc board state.last) (observed : ∀ a ∈ attempts, RowsValid mc fc board a.rows) :
    SavedValid mc fc board (SearchPublication.run minimum state attempts).last := by
  induction attempts generalizing state with
  | nil => exact saved
  | cons attempt rest ih =>
    have next : SavedValid mc fc board (SearchPublication.step minimum state attempt).last := by
      intro snapshot found
      simp only [SearchPublication.step, published] at found
      split at found
      · cases Option.some.inj found
        exact valid_capture (observed attempt (by simp)) _ _
      · exact saved snapshot found
    exact ih _ next (fun a mem => observed a (List.mem_cons_of_mem _ mem))

def SaveContract : Prop :=
  ∀ (mc : X86Config) (fc : FlipConfig) (board : Position) (minimum : Nat) (state : Completed)
    (attempts : List Attempt) (outcome : Exit),
    SavedValid mc fc board state.last → (∀ a ∈ attempts, RowsValid mc fc board a.rows) →
    (∀ stamp rows, outcome = .normal stamp rows → RowsValid mc fc board rows) →
    SavedValid mc fc board (finish minimum (SearchPublication.run minimum state attempts) outcome)

theorem saveContract_proven : SaveContract := by
  intro mc fc board minimum state attempts outcome saved observed passed
  have prior := valid_run mc fc board minimum state attempts saved observed
  cases outcome with
  | interrupted => exact prior
  | normal stamp rows =>
    intro snapshot chosen
    simp only [finish] at chosen
    split at chosen
    · cases Option.some.inj chosen
      exact valid_capture (passed _ _ rfl) stamp true
    · exact prior snapshot chosen

open Aspiration (Trial Trials Accepted runWith)

def RetryContract : Prop :=
  ∀ (mc : X86Config) (fc : FlipConfig) (board : Position) (search : Trials (List Row))
    (fuel index : Nat) (alpha beta delta : Int) (result : Accepted (List Row)),
    (∀ i a b, (search i a b).aborted = false → RowsValid mc fc board (search i a b).payload) →
    runWith search fuel index alpha beta delta = some result →
    result.trial = search result.index result.alpha result.beta ∧
      result.trial.aborted = false ∧ RowsValid mc fc board result.trial.payload

theorem retryContract_proven : RetryContract := by
  intro mc fc board search fuel index alpha beta delta result good found
  induction fuel generalizing index alpha beta delta with
  | zero => cases found
  | succ fuel ih =>
    rw [runWith] at found
    split at found
    next => cases found
    next continued =>
      split at found
      · cases Option.some.inj found
        have completed : (search index alpha beta).aborted = false := Bool.eq_false_iff.mpr continued
        exact ⟨rfl, completed, good _ _ _ completed⟩
      · exact ih _ _ _ _ found

private theorem one_move_valid
    (generated : (Board.getMoves mc board.val).getLsbD sq.val = true) (score : Int) :
    RowValid mc fc board (⟨sq, score, [sq]⟩ : Row) :=
  ⟨generated, .inr ⟨rfl, _, .move ⟨sq, generated⟩ .empty⟩⟩

open SearchFallback (Report CachedMove resolve)

def ResultContract : Prop :=
  ∀ (mc : X86Config) (fc : FlipConfig) (board : Position) (evaluate : Board.Board → Int)
    (saved : Option Snapshot) (cached : Option CachedMove) (nodes : Nat) (result : Report),
    SavedValid mc fc board saved → resolve mc fc board evaluate saved cached nodes = some result →
    RowValid mc fc board result.best ∧ RowsValid mc fc board result.alternatives

theorem resultContract_proven : ResultContract := by
  intro mc fc board evaluate saved cached nodes result available found
  have resolver := SearchFallback.resolveContract_proven mc fc board evaluate cached nodes
  cases saved with
  | some snapshot =>
    cases rows : snapshot.rows with
    | nil => simp [resolve, SearchFallback.fromSnapshot, rows] at found
    | cons best alternatives =>
      rw [resolver.1 snapshot best alternatives rows] at found
      cases Option.some.inj found
      have good := available snapshot rfl
      rw [rows] at good
      exact ⟨good best (by simp), good⟩
  | none =>
    by_cases hit : ∃ move, cached = some move ∧
        Bitboard.Moves.LegalMove board.val.player board.val.opponent move.square.val
    · obtain ⟨move, same, legal⟩ := hit
      rw [resolver.2.1 move same legal] at found
      cases Option.some.inj found
      exact ⟨one_move_valid ((Board.getMoves_exact mc board.val _).mpr legal) _, by simp [RowsValid, SearchFallback.fromCache]⟩
    · have miss : ∀ move, cached = some move → ¬ Bitboard.Moves.LegalMove board.val.player board.val.opponent move.square.val :=
        fun move same legal => hit ⟨move, same, legal⟩
      obtain ⟨legal, _, _, line, alternatives, _, _, _⟩ := (resolver.2.2 miss).2 result found
      refine ⟨?_, by simp [RowsValid, alternatives]⟩
      have shaped (row : Row) (same : row.pv = [row.square]) :
          row = ⟨row.square, row.score, [row.square]⟩ := by
        cases row
        simpa using same
      rw [shaped result.best line]
      exact one_move_valid ((Board.getMoves_exact mc board.val _).mpr legal) _

def ExportContract : Prop :=
  ∀ (mc : X86Config) (fc : FlipConfig) (board : Position) (evaluate : Board.Board → Int)
    (minimum nodes : Nat) (state : Completed) (attempts : List Attempt)
    (outcome : Exit) (cached : Option CachedMove) (result : Report),
    SavedValid mc fc board state.last → (∀ a ∈ attempts, RowsValid mc fc board a.rows) →
    (∀ stamp rows, outcome = .normal stamp rows → RowsValid mc fc board rows) →
    let saved := finish minimum (SearchPublication.run minimum state attempts) outcome
    resolve mc fc board evaluate saved cached nodes = some result →
    RowValid mc fc board result.best ∧ RowsValid mc fc board result.alternatives

theorem exportContract_proven : ExportContract := by
  intro mc fc board evaluate minimum nodes state attempts outcome cached result prior observed final saved found
  exact resultContract_proven mc fc board evaluate saved cached nodes result
    (saveContract_proven mc fc board minimum state attempts outcome prior observed final) found

end NeuralReversi.PrincipalVariation
