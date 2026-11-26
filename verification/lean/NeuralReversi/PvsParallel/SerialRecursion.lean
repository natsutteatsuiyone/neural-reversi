import NeuralReversi.PvsParallel.SerialScout
import NeuralReversi.PvsParallel.SerialNonPv
import NeuralReversi.PvsParallel.SerialChildState
import NeuralReversi.PvsParallel.SerialPv
import NeuralReversi.PvsParallel.EndgameOrderMir
import NeuralReversi.PvsParallel.SerialRoot
import NeuralReversi.PvsParallel.ContextPathMir
import NeuralReversi.AspirationMir.NativeRootBirth

namespace NeuralReversi.PvsParallel.SerialRecursion

open Bitboard.Moves (X86Config)
open PvsBoardTree (Position GeneratedSquare build minimax legalSquares play pass)
open EndgameSearch.Cancellation (EmptyState)
open ExactDepth (Depth)
open Aspiration (Window)

/-- The full call path distinguishes visits to transpositions and PV re-searches.
    A separate root path can be supplied for each aspiration trial. -/
structure Call where
  board : Board.Board
  pv : Bool
  cut : Bool
  alpha : Int
  beta : Int

structure Operands (mc : X86Config) (fc : Cutoffs.FlipConfig) (b : Position) where
  memory : MoveList.Initialization.Memory Move.Move
  cache : List EndgameTreeCache.Store
  table : Cutoffs.History
  probe : List TranspositionTable.ProbeRetry.RawAttempt
  etc : List (TableReads.ChildLookupFrom mc fc b (·.publication table))
  ordering : Bitboard.Square → EndgameOrderMir.Heuristic
  tie : GeneratedSquare mc b.val → Fin 64

abbrev Inputs (mc : X86Config) (fc : Cutoffs.FlipConfig) :=
  List Call → (b : Position) → Operands mc fc b

/-- These are the cache and shared-table obligations, stated independently of
    the control flow or the answer of this search. -/
structure Admissible (inputs : Inputs mc fc) : Prop where
  cache : ∀ p b, EndgameTreeCache.soundHistory mc fc (inputs p b).cache
  probe : ∀ p b, TableReads.ProbeReads (inputs p b).table (inputs p b).probe
  table : ∀ p b, ExactTableHistory.Publications mc fc (inputs p b).table
  hints : ∀ p b, TableReads.LegalHints (inputs p b).table

def pvOrder (b : Position) (op : Operands mc fc b) :=
  let hint := GeneralNonPvMir.checkedHint mc b (GeneralNonPvMir.probeHint b op.probe)
  EndgameOrderMir.ordinary (fun (m : GeneratedSquare mc b.val) => EndgameOrderMir.priority mc fc b op.ordering m.val)
    op.tie false hint (legalSquares mc b.val)

private def raw (v : Int) : ScoreRefresh.Score := ⟨BitVec.ofInt 32 v⟩

def runFuel (c : SerialScout.Config) (inputs : Inputs mc fc) :
    Nat → List Call → Position → Depth → EmptyState → Bool → Bool → Int → Int → Option Int
  | 0, _, _, _, _, _, _, _, _ => none
  | fuel + 1, prior, b, d, s, pv, cut, a, z =>
    let path := ⟨b.val, pv, cut, a, z⟩ :: prior
    let op := inputs path b
    let descend := fun next childDepth (childPv childCut : Bool) lo hi => do
      let state ← SerialChildState.nextState mc fc b d s next childDepth
      runFuel c inputs fuel path next childDepth state childPv childCut lo hi
    if pv then
      SerialPv.runFrame mc fc b d c.paired c.checked op.memory a z
        (fun next cd => descend next cd true false) (pvOrder b op)
        (fun m cp => descend (play mc fc b m).val (d - 1) cp (!cp && !cut))
    else if d.toNat ≤ 12 then
      (SerialScout.run c mc fc b s (raw a) (raw z) op.cache).map Prod.fst
    else
      SerialNonPv.runFrame mc fc b d (raw a) (raw z) c.paired c.checked cut op.memory op.probe op.etc
        (EndgameOrderMir.framePriority mc fc b op.ordering op.tie)
        (fun next cd => descend next cd false)

private theorem fuel_returns (c : SerialScout.Config) (inputs : Inputs mc fc) (valid : Admissible inputs)
    (fuel : Nat) (path : List Call) (b : Position) (d : Depth) (s : EmptyState) (pv cut : Bool) (a z : Int)
    (rank : Board.playRank mc b.val < fuel) (full : ExactDepth.Full b d) (aligned : s.matches b)
    (w : Window a z) (null : pv = false → z = a + 1) :
    ∃ score, runFuel c inputs fuel path b d s pv cut a z = some score ∧
      boundedReplay fuel (build mc fc b) pv a z score := by
  induction fuel generalizing path b d s pv cut a z with
  | zero => omega
  | succ fuel ih =>
    let here : List Call := ⟨b.val, pv, cut, a, z⟩ :: path
    let op := inputs here b
    let descend := fun next cd cp cc lo hi => do
      let state ← SerialChildState.nextState mc fc b d s next cd
      runFuel c inputs fuel here next cd state cp cc lo hi
    have child {next cd} (edge : ExactDepth.RecursiveCall mc fc b d next cd)
        (cp cc : Bool) (lo hi : Int) (window : Window lo hi) (narrow : cp = false → hi = lo+1) :
        ∃ score, descend next cd cp cc lo hi = some score ∧
          boundedReplay fuel (build mc fc next) cp lo hi score := by
      obtain ⟨state, found, fitted⟩ := SerialChildState.nextState_matches edge aligned
      have smaller := edge.earlier
      obtain ⟨v, ran, replay⟩ := ih here next cd state cp cc lo hi (by omega) (edge.full full) fitted window narrow
      exact ⟨v, by simp [descend, found, ran], replay⟩
    obtain ⟨low, ordered, high⟩ := w
    have av : (raw a).raw.toInt = a := BitVec.toInt_ofInt_eq_self (by decide) (by omega) (by omega)
    have zv : (raw z).raw.toInt = z := BitVec.toInt_ofInt_eq_self (by decide) (by omega) (by omega)
    cases pv with
    | true =>
      apply SerialPv.frame_returns mc fc b d c.paired c.checked op.memory a z
        (fun next cd => descend next cd true false) (pvOrder b op)
        (fun m cp => descend (play mc fc b m).val (d-1) cp (!cp && !cut)) fuel full
        (EndgameOrderMir.ordinary_perm _ _ _ _ _) ⟨low,ordered,high⟩
      · intro stuck mobile
        exact child (.forcedPass b d ⟨PvsBoardTree.stuck_of_nil stuck, mobile⟩) true false _ _
          (PvsMoveLoop.reflectedWindow ⟨low,ordered,high⟩) (by simp)
      · intro m cp lo hi window narrow
        exact child (.ply b m d) cp _ _ _ window narrow
    | false =>
      have narrow : (raw z).raw.toInt = (raw a).raw.toInt + 1 := by rw [av,zv,null rfl]
      by_cases small : d.toNat ≤ 12
      · have size : s.count.toNat ≤ 12 := by rw [aligned.1, ← full]; exact small
        obtain ⟨v, after, answer, evidence, _, _⟩ := SerialScout.returns c mc fc b s (raw a) (raw z)
          op.cache aligned size (valid.cache here b) narrow
        rw [av, zv] at evidence
        refine ⟨v, ?_, boundedReplay_mono (by omega : 1 ≤ fuel + 1) _ _ _ _ _ evidence⟩
        simp only [runFuel, Bool.false_eq_true, ↓reduceIte, small]
        change (SerialScout.run c mc fc b s (raw a) (raw z) op.cache).map Prod.fst = some v
        rw [answer]
        rfl
      · have current := SerialNonPv.frame_returns_from_history mc fc b d (raw a) (raw z) c.paired
          c.checked cut op.memory op.probe op.etc (EndgameOrderMir.framePriority mc fc b op.ordering op.tie)
          (fun next cd => descend next cd false) fuel full (by omega)
          (by rw [av,zv]; exact ⟨low,ordered,high⟩) narrow (valid.probe here b) (valid.table here b)
          (valid.hints here b) (fun m _ => EndgameOrderMir.framePriority_low _ _ _ _ _ m)
          (fun edge childCut lo hi window n => child edge false childCut lo hi window (fun _ => n))
        obtain ⟨v, returned, replay, _⟩ := current
        rw [av,zv] at replay
        exact ⟨v, by simpa only [runFuel, Bool.false_eq_true, ↓reduceIte, small] using returned, replay⟩

def run (c : SerialScout.Config) (inputs : Inputs mc fc) (path : List Call) (b : Position) (depth : Depth)
    (s : EmptyState) (pv cut : Bool) (alpha beta : Int) : Option Int :=
  runFuel c inputs (Board.playRank mc b.val + 1) path b depth s pv cut alpha beta

theorem returns (c : SerialScout.Config) (inputs : Inputs mc fc) (valid : Admissible inputs)
    (path : List Call) (b : Position) (depth : Depth) (s : EmptyState) (pv cut : Bool) (alpha beta : Int)
    (full : ExactDepth.Full b depth) (aligned : s.matches b) (w : Window alpha beta)
    (null : pv = false → beta = alpha + 1) :
    ∃ score, run c inputs path b depth s pv cut alpha beta = some score ∧
      Replayed (build mc fc b) pv alpha beta score ∧ Aspiration.SoundResult (minimax mc fc b) alpha beta score := by
  obtain ⟨score, ran, trace⟩ := fuel_returns c inputs valid _ path b depth s pv cut alpha beta
    (by omega : Board.playRank mc b.val < Board.playRank mc b.val + 1) full aligned w null
  exact ⟨_, ran, ⟨_,trace⟩, replayContract_proven _ _ _ _ _ _ w null trace⟩

structure RootInput where
  key : Bitboard.Square → BitVec 32
  sort : RootFrameMir.LibraryOrder
  pvLine : Bitboard.Square → List Bitboard.Square

def rootReplies (c : SerialScout.Config) (inputs : Inputs mc fc) (path : List Call)
    (b : Position) (alpha beta : BitVec 32) : RootOwnerMir.Replies (GeneratedSquare mc b.val) :=
  let s := ContextPathMir.leafState (ContextPathMir.start b)
  let here := ⟨b.val, true, false, alpha.toInt, beta.toInt⟩ :: path
  fun m pv => run c inputs here (play mc fc b m).val (s.count - 1) (s.remove m.val) pv (!pv)

def rootTrial (c : SerialScout.Config) (inputs : Inputs mc fc) (path : List Call) (b : Position)
    (root : RootInput) (rows : List RootMoveStorage.Row) (idx : Nat) (alpha beta : BitVec 32) :=
  let state := ContextPathMir.start b
  let op := inputs (⟨b.val, true, false, alpha.toInt, beta.toInt⟩ :: path) b
  SerialRoot.trial mc fc b (ContextPathMir.rootDepth state) c.paired op.memory rows idx root.key root.sort
    root.pvLine alpha beta (rootReplies c inputs path b alpha beta)

def RootTrialContract : Prop :=
  ∀ mc fc (c : SerialScout.Config) (inputs : Inputs mc fc) (path : List Call) (b : Position)
    (root : RootInput) (rows : List RootMoveStorage.Row) idx (a z : BitVec 32),
    Admissible inputs → (Board.emptyCount b.val).toNat ≤ 60 →
    Root.Covers mc b (rows.map RootMoveStorage.Row.square) → idx < rows.length → Window a.toInt z.toInt →
    ∃ answer snap, rootTrial c inputs path b root rows idx a z = some answer ∧ answer.value = snap.state.best ∧
      RootSuffix.Writes rows idx a.toInt z.toInt answer.rows snap ∧
      RootEvidence.Attempt mc fc b (rows.map RootMoveStorage.Row.square) idx a.toInt z.toInt answer.value snap.scores ∧
      RootOrder.SoundAttempt mc fc b (rows.map RootMoveStorage.Row.square) idx a.toInt z.toInt answer.value snap.scores

theorem rootTrialContract_proven : RootTrialContract := by
  intro mc fc c inputs path b root rows idx a z valid size covers within window
  have ⟨aligned, full, depth, _⟩ := (ContextPathMir.At.initial (mc := mc) (fc := fc) (root := b)).frameInputs size
  apply SerialRoot.trial_returns _ _ _ _ _ _ _ _ _ _ _ _ _ _ covers within full window
  intro m pv lo hi w null
  have childFull := (ExactDepth.move_depth mc fc b _ m full).2.1
  rw [depth] at childFull
  have placed := (EndgameSearch.emptyStatePlayContract_proven mc fc b m _ _ _ aligned).2.2.1
  obtain ⟨v, ran, replay, _⟩ := returns c inputs valid (⟨b.val, true, false, a.toInt, z.toInt⟩ :: path)
    (play mc fc b m).val _ ((ContextPathMir.leafState (ContextPathMir.start b)).remove m.val) pv (!pv) lo hi
    childFull placed w null
  exact ⟨v, ran, replay⟩

/-- Start at the direct Rust root-vector constructor. An in-window result is
    the minimax value; both directions of a failed window retain their bounds. -/
theorem fromBoard (c : SerialScout.Config) (inputs : Inputs mc fc) (path : List Call)
    (b : Position) (root : RootInput) (a z : BitVec 32) (valid : Admissible inputs)
    (fits : (Board.emptyCount b.val).toNat ≤ 60) (mobile : legalSquares mc b.val ≠ [])
    (capacity : (legalSquares mc b.val).length ≤ AspirationMir.NativeRootBirth.capacity)
    (window : Window a.toInt z.toInt) :
    let mem := (inputs (⟨b.val, true, false, a.toInt, z.toInt⟩ :: path) b).memory
    ∃ rows answer, AspirationMir.NativeRootBirth.started mc fc b c.paired mem = some rows ∧
      rootTrial c inputs path b root rows 0 a z = some answer ∧
      Aspiration.SoundResult (minimax mc fc b) a.toInt z.toInt answer.value := by
  intro mem
  have started := (AspirationMir.NativeRootBirth.started_iff mc fc b c.paired mem).2 ⟨capacity, mobile, rfl⟩
  obtain ⟨covers, nonempty⟩ := AspirationMir.NativeRootBirth.covers started
  have pos := List.length_pos_iff.mpr nonempty
  obtain ⟨answer, _, returned, _, _, _, sound⟩ :=
    rootTrialContract_proven mc fc c inputs path b root _ 0 a z valid fits covers pos window
  exact ⟨_, answer, started, returned, Root.value_zero covers (by simpa using pos) ▸ sound.1⟩

end NeuralReversi.PvsParallel.SerialRecursion
