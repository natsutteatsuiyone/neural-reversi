import NeuralReversi.AspirationMir.NativeRootBirth
import NeuralReversi.AspirationMir.NativeTrials
import NeuralReversi.AspirationMir.SmallRoot
import NeuralReversi.PvsParallel.RootAspiration
import NeuralReversi.SearchContextState
import NeuralReversi.PublicationMir.ExactScore

namespace NeuralReversi.AspirationMir.SerialEntry

open Bitboard (Square)
open EndgameSearch.Cancellation (EmptyState)
open PvsBoardTree (Position legalSquares)
open PvsParallel
open RootMoveStorage (Row)

/-- The PVS empty-list view of the initial search context. -/
def emptyState (board : Position) : EmptyState :=
  let initial := (SearchContextState.init board).empties
  ⟨initial.links, initial.parity, initial.count⟩

private theorem emptyState_matches (board : Position) (small : (Board.emptyCount board.val).toNat ≤ 5) :
    (emptyState board).matches board :=
  (SearchContextState.initialContract_proven.1 board (by omega)).1.1.1

theorem started_small (c : SmallPv.Config mc) (board : Position)
    (small : (Board.emptyCount board.val).toNat ≤ 5) (legal : legalSquares mc board.val ≠ []) :
    NativeRootBirth.started mc c.fc board c.paired c.memory = some (NativeRootBirth.initial mc board) := by
  have capacity := legalMoveCount_le_empties mc board
  apply (NativeRootBirth.started_iff mc c.fc board c.paired c.memory).2
  exact ⟨by unfold NativeRootBirth.capacity; omega, legal, rfl⟩

def windowMachine (w : Int × Int) : Machine :=
  NativeTrials.initial (BitVec.ofInt 32 w.1) (BitVec.ofInt 32 w.2)

private theorem windowMachine_bounds {w : Int × Int} (valid : Aspiration.Window w.1 w.2) :
    (windowMachine w).regs.alpha.toInt = w.1 ∧ (windowMachine w).regs.beta.toInt = w.2 := by
  have repr (v : Int) (lower : -32767 ≤ v) (upper : v ≤ 32767) : (BitVec.ofInt 32 v).toInt = v :=
    BitVec.toInt_ofInt_eq_self (by decide) (by omega) (by omega)
  obtain ⟨lo, ordered, hi⟩ := valid
  exact ⟨repr w.1 lo (by omega), repr w.2 (by omega) hi⟩

theorem windowMachine_valid {w : Int × Int} (valid : Aspiration.Window w.1 w.2) :
    Aspiration.Window (windowMachine w).regs.alpha.toInt (windowMachine w).regs.beta.toInt := by
  obtain ⟨alpha, beta⟩ := windowMachine_bounds valid
  rwa [alpha, beta]

def firstMachine (base : Word) : Machine :=
  windowMachine (RootAspiration.initialWindow 0 base.toInt none)

def nextMachine (score : Word) : Machine :=
  windowMachine (RootAspiration.nextLevelWindow score.toInt)

theorem firstMachine_window (base : Word) (lo : -PvsMoveLoop.scoreMax ≤ base.toInt)
    (hi : base.toInt ≤ PvsMoveLoop.scoreMax) :
    Aspiration.Window (firstMachine base).regs.alpha.toInt (firstMachine base).regs.beta.toInt :=
  windowMachine_valid (RootAspiration.seedWindowsContract_proven.1 0 base.toInt none lo hi (by simp))

theorem nextMachine_window (score : Word) (lo : -PvsMoveLoop.scoreMax ≤ score.toInt)
    (hi : score.toInt ≤ PvsMoveLoop.scoreMax) :
    Aspiration.Window (nextMachine score).regs.alpha.toInt (nextMachine score).regs.beta.toInt ∧
      (nextMachine score).regs.alpha.toInt < score.toInt ∧ score.toInt < (nextMachine score).regs.beta.toInt := by
  obtain ⟨valid, inside⟩ := RootAspiration.seedWindowsContract_proven.2 score.toInt lo hi
  obtain ⟨alpha, beta⟩ := windowMachine_bounds valid
  exact ⟨windowMachine_valid valid, by simpa only [nextMachine, alpha, beta] using inside⟩

theorem accepted_next_window {regs after : WindowRegs} {score : Word}
    (window : Aspiration.Window regs.alpha.toInt regs.beta.toInt)
    (accepted : applyWiden regs score = some (after,false)) :
    Aspiration.Window (nextMachine score).regs.alpha.toInt (nextMachine score).regs.beta.toInt := by
  have ⟨above,below⟩ := widen_false accepted
  obtain ⟨lo,_,hi⟩ := window
  apply windowMachine_valid
  unfold RootAspiration.nextLevelWindow PvsMoveLoop.scoreInf
  exact ⟨by omega,by omega,by omega⟩

structure Inputs where
  rootKey : Nat → List Row → Nat → Word → Word → Square → BitVec 32
  childKey : Nat → Position → Int → Int → Square → BitVec 32
  order : RootFrameMir.LibraryOrder
  pv : Nat → Square → List Square

def levelTrial (c : SmallPv.Config mc) (inputs : Inputs) (board : Position) (level : Nat)
    (rows : List Row) (i : Nat) (a z : Word) : Option SmallRoot.Answer :=
  SmallRoot.trial (SmallRoot.withOrder c (inputs.childKey level) inputs.order) board (emptyState board) rows 0
    (inputs.rootKey level rows i a z) inputs.order (inputs.pv level) a z

structure Accepted where
  raw : Word
  last : Call
  answer : SmallRoot.Answer
  snap : RootReplay.Snapshot

def Accepted.sorted (a : Accepted) : List Row := RootMoveStorage.sortRows a.answer.rows 0

structure LevelRun (c : SmallPv.Config mc) (inputs : Inputs) (board : Position) (level : Nat)
    (rows : List Row) (start : Machine) (a : Accepted) : Prop where
  returned : Returns (SmallRoot.queryFor (levelTrial c inputs board level rows)) (fun _ => false) start a.raw a.last
  trial : levelTrial c inputs board level rows a.last.index a.last.alpha a.last.beta = some a.answer
  value : a.answer.value = a.raw.toInt
  written : RootSuffix.Writes rows 0 a.last.alpha.toInt a.last.beta.toInt a.answer.rows a.snap
  inside : a.last.alpha.toInt < a.snap.state.best ∧ a.snap.state.best < a.last.beta.toInt
  sound : RootOrder.SoundAttempt mc c.fc board (rows.map Row.square) 0 a.last.alpha.toInt a.last.beta.toInt
    a.snap.state.best a.snap.scores
  exactScore : a.raw.toInt = PvsBoardTree.minimax mc c.fc board

private theorem completeLevel (c : SmallPv.Config mc) (inputs : Inputs) (board : Position)
    (small : (Board.emptyCount board.val).toNat ≤ 5) (level : Nat) (rows : List Row) (m : Machine)
    (covers : Root.Covers mc board (rows.map Row.square)) (positive : 0 < rows.length)
    (entry : m.pc = 0) (window : Aspiration.Window m.regs.alpha.toInt m.regs.beta.toInt) :
    ∃ a, LevelRun c inputs board level rows m a ∧
      Root.Covers mc board (a.sorted.map Row.square) ∧ 0 < a.sorted.length := by
  let cfg := SmallRoot.withOrder c (inputs.childKey level) inputs.order
  obtain ⟨raw, last, answer, snap, returned, trial, value, best, written, inside, sound⟩ :=
    SmallRoot.completeLine cfg (SmallRoot.generatedOrder c (inputs.childKey level) inputs.order)
      board (emptyState board) rows 0 m (inputs.rootKey level rows) inputs.order (inputs.pv level)
      covers positive (emptyState_matches board small) small entry window
  have exactScore := (sound.1.exact inside.1 inside.2).trans (Root.value_zero covers (by simpa using positive))
  have names := (written.scores (Root.roots_nodup covers)).1
  have perm := (RootMoveStorage.sortRowsContract_proven answer.rows 0).1
  have preserved := ((perm.map Row.square).trans (List.Perm.of_eq names)).trans covers
  have len : answer.rows.length = rows.length := by simpa using congrArg List.length names
  exact ⟨⟨raw, last, answer, snap⟩, ⟨returned, trial, value, written, inside, sound,
    value.symm.trans (best.trans exactScore)⟩, preserved, by simpa [Accepted.sorted, perm.length_eq, len] using positive⟩

/-- Each completed level supplies the sorted physical rows and the window for its successor. -/
inductive Levels (c : SmallPv.Config mc) (inputs : Inputs) (board : Position) :
    List Nat → List Row → Machine → List Row → Machine → Prop where
  | done : Levels c inputs board [] rows m rows m
  | next : LevelRun c inputs board level rows m a →
      Levels c inputs board rest a.sorted (nextMachine a.raw) final m' →
      Levels c inputs board (level :: rest) rows m final m'

private theorem completeLevels (c : SmallPv.Config mc) (inputs : Inputs) (board : Position)
    (small : (Board.emptyCount board.val).toNat ≤ 5) (levels : List Nat) (rows : List Row) (m : Machine)
    (covers : Root.Covers mc board (rows.map Row.square)) (positive : 0 < rows.length)
    (entry : m.pc = 0) (window : Aspiration.Window m.regs.alpha.toInt m.regs.beta.toInt) :
    ∃ final m', Levels c inputs board levels rows m final m' ∧
      Root.Covers mc board (final.map Row.square) ∧ 0 < final.length ∧ m'.pc = 0 ∧
      Aspiration.Window m'.regs.alpha.toInt m'.regs.beta.toInt := by
  induction levels generalizing rows m with
  | nil => exact ⟨rows, m, .done, covers, positive, entry, window⟩
  | cons level rest ih =>
    obtain ⟨a, accepted, legal, nonempty⟩ := completeLevel c inputs board small level rows m covers positive entry window
    have range : -PvsMoveLoop.scoreMax ≤ PvsBoardTree.minimax mc c.fc board ∧
        PvsBoardTree.minimax mc c.fc board ≤ PvsMoveLoop.scoreMax :=
      PvsMoveLoop.minimax_range (PvsBoardTree.build mc c.fc board)
    rw [← accepted.exactScore] at range
    obtain ⟨final, m', ran, post⟩ := ih a.sorted (nextMachine a.raw) legal nonempty rfl (nextMachine_window a.raw range.1 range.2).1
    exact ⟨final, m', .next accepted ran, post⟩

/-- The uninterrupted direct single-PV ladder on a mobile board with at most
five empties. The first level takes the evaluated base and each later level
uses the accepted score's inter-selectivity window. -/
def PublicScoreContract : Prop :=
  ∀ (mc : Bitboard.Moves.X86Config) (c : SmallPv.Config mc) (inputs : Inputs) (board : Position) (base : Word)
    (sort : PublicationMir.SliceSort) (nodes : Nat) (fallback : Unit → PublicationMir.SearchResult),
    (Board.emptyCount board.val).toNat ≤ 5 → legalSquares mc board.val ≠ [] →
    -PvsMoveLoop.scoreMax ≤ base.toInt → base.toInt ≤ PvsMoveLoop.scoreMax →
    ∃ rows, NativeRootBirth.started mc c.fc board c.paired c.memory = some rows ∧
      ∃ before m a result, Levels c inputs board (OuterMir.ladder.take 4) rows (firstMachine base) before m ∧
        LevelRun c inputs board 5 before m a ∧
        (∀ prior, OuterMir.finish 1 5 sort a.sorted (Board.emptyCount board.val).toNat nodes prior fallback = some result) ∧
        PublicationMir.ExactScore.scoreIfExact result = some (256 * EndgameTreeCache.discValue mc c.fc board.val)

theorem publicScore : PublicScoreContract := by
  intro mc c inputs board base sort nodes fallback small legal lo hi
  have started := started_small c board small legal
  obtain ⟨covers, nonempty⟩ := NativeRootBirth.covers started
  have window := firstMachine_window base lo hi
  obtain ⟨before, m, ran, legal, positive, entry, valid⟩ := completeLevels c inputs board small (OuterMir.ladder.take 4)
    _ (firstMachine base) covers (List.length_pos_iff.mpr nonempty) rfl window
  obtain ⟨a, accepted, _, _⟩ := completeLevel c inputs board small 5 before m legal positive entry valid
  obtain ⟨result, published, score⟩ := publishSingle legal positive accepted.written accepted.inside accepted.sound
    (PublicationMir.ExactScore.afterRootSort sort) (Board.emptyCount board.val).toNat nodes fallback
  refine ⟨_, started, before, m, a, result, ran, accepted,
    fun _ => by simpa [OuterMir.finish, PublicationMir.completed, PublicationMir.record, PublicationMir.capture,
      PublicationMir.snapshot, PublicationMir.ExactScore.afterRootSort, Accepted.sorted] using published, ?_⟩
  rw [PublicationMir.ExactScore.completed_tagged published, EndgameTreeCache.scaled_discValue]
  exact score

end NeuralReversi.AspirationMir.SerialEntry
