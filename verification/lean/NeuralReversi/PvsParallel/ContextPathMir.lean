import NeuralReversi.PvsParallel.ContextPathMir.Program
import NeuralReversi.PvsParallel.NonPvMir
import NeuralReversi.PvsParallel.DeepLeafMir

namespace NeuralReversi.PvsParallel.ContextPathMir

open PvsBoardTree (Position GeneratedSquare play pass)
open SearchContextState (State Matches ply)
open Bitboard.Moves (X86Config)

theorem start_eq (root : Position) : start root = SearchContextState.init root := by
  let bits := Board.emptySquares root.val
  have perm := EmptyListParity.scan_perm.filter (fun sq => bits.getLsbD sq.val)
  have size := perm.length_eq.trans (Solve34.EmptyBoard.ofBoard root.val root.property).length_eq
  have word : BitVec.ofNat 32 (EmptyListParity.presorted.filter (fun sq => bits.getLsbD sq.val)).length =
      Board.emptyCount root.val := by rw [size]; simp
  unfold start SearchContextState.init
  change SearchContextState.withFeatures ⟨_, _, BitVec.ofNat 32 (EmptyListParity.presorted.filter (fun sq => bits.getLsbD sq.val)).length⟩ _ _ = _
  rw [word]
  rfl

private def flips (fc : Flip.Dispatch.X86Config) (b : Position) (sq : Bitboard.Square) :=
  Flip.Dispatch.resultValue fc sq b.val.player b.val.opponent

/-- Normal nested returns can leave scratch feature slots above the parent ply and increment counters. -/
structure Restored (before after : State) : Prop where
  empties : after.empties = before.empties
  turn : after.opponentTurn = before.opponentTurn
  slots : ∀ i, i ≤ ply before → after.slots i = before.slots i

theorem Restored.ply_eq (h : Restored s t) : ply t = ply s := by
  simp only [ply, h.empties]

theorem Restored.agrees (h : Restored s t) (aligned : Matches s b) : Matches t b := by
  exact ⟨h.empties ▸ aligned.1, by rw [h.ply_eq, h.slots _ (Nat.le_refl _), h.turn]; exact aligned.2⟩

/-- A completed series of child calls, including siblings and repeated PVS searches at one child. -/
inductive Calls (mc : X86Config) (fc : Flip.Dispatch.X86Config) :
    Position → State → State → Prop
  | done : Calls mc fc b s s
  | next : Calls mc fc b s t → Calls mc fc b t u → Calls mc fc b s u
  | move {b : Position} {s t : State} (m : GeneratedSquare mc b.val) :
      Calls mc fc (play mc fc b m).val (place s m.val (flips fc b m.val)) t →
      Calls mc fc b s (restore t m.val)
  | forcedPass {b : Position} {s t : State} (forced : Board.ForcedPass mc b.val) :
      Calls mc fc (pass b) (passTurn s) t → Calls mc fc b s (unpass t)
  | split : Calls mc fc b (splitCopy s) t → Calls mc fc b s s

theorem Calls.restores (calls : Calls mc fc b s t) (initial : Matches s b) : Restored s t := by
  induction calls with
  | done => exact ⟨rfl, rfl, fun _ _ => rfl⟩
  | next _ _ first second =>
    have f := first initial
    have g := second (f.agrees initial)
    exact ⟨g.empties.trans f.empties, g.turn.trans f.turn,
      fun i le => (g.slots i (f.ply_eq ▸ le)).trans (f.slots i le)⟩
  | @move board s t move _ ih =>
    obtain ⟨_, child, nextPly, _, oldEmpty, oldTurn, _, untouched, _⟩ :=
      SearchContextState.placementContract_proven mc fc board move s initial
    have h := ih child
    refine ⟨?_, ?_, ?_⟩
    · change SearchContextState.restore t.empties move.val = s.empties
      rw [h.empties]
      exact oldEmpty
    · change (!t.opponentTurn) = s.opponentTurn
      rw [h.turn]
      exact oldTurn
    · intro i le
      have depth : ply (place s move.val (flips fc board move.val)) = ply s + 1 := nextPly
      exact (h.slots i (by omega)).trans (untouched i (by omega))
  | @forcedPass board s t _ _ ih =>
    have h := ih (SearchContextState.passContract_proven _ _ initial).1
    exact ⟨h.empties, (congrArg Bool.not h.turn).trans (Bool.not_not _), h.slots⟩
  | split => exact ⟨rfl, rfl, fun _ _ => rfl⟩

/-- Frames encountered from one initialized root, including calls reached after siblings return
    and frames running on a worker reconstructed from a published split task. -/
inductive At (mc : X86Config) (fc : Flip.Dispatch.X86Config) (root : Position) :
    Position → ExactDepth.Depth → State → Prop
  | initial : At mc fc root root (rootDepth (start root)) (start root)
  | move (parent : At mc fc root b d s) (m : GeneratedSquare mc b.val) :
      At mc fc root (play mc fc b m).val (d - 1) (place s m.val (flips fc b m.val))
  | forcedPass (parent : At mc fc root b d s) (forced : Board.ForcedPass mc b.val) :
      At mc fc root (pass b) d (passTurn s)
  | after (parent : At mc fc root b d s) (calls : Calls mc fc b s t) : At mc fc root b d t
  | worker (parent : At mc fc root b d s) : At mc fc root b d (splitCopy s)

inductive Argument (mc : X86Config) (b : Position) where
  | move (m : GeneratedSquare mc b.val)
  | forcedPass (forced : Board.ForcedPass mc b.val)

def enter (mc : X86Config) (fc : Flip.Dispatch.X86Config) (worker : Bool) (pc : Nat)
    (b : Position) (d : ExactDepth.Depth) (s : State) (arg : Argument mc b) :
    Option (Position × ExactDepth.Depth × State) :=
  match childSite worker pc, arg with
  | some .move, .move m => some ((play mc fc b m).val, d - 1, place s m.val (flips fc b m.val))
  | some .pass, .forcedPass _ => some (pass b, d, passTurn s)
  | _, _ => none

theorem At.entered {worker : Bool} (parent : At mc fc root b d s)
    (edge : enter mc fc worker pc b d s arg = some (child, cd, t)) : At mc fc root child cd t := by
  unfold enter at edge
  split at edge
  next m _ => cases edge; exact parent.move m
  next forced _ => cases edge; exact parent.forcedPass forced
  next => contradiction

/-- The depth word follows the number of empties even before an evaluator
    feature slot is read. Returning from siblings or copying to a worker preserves it. -/
theorem At.full (path : At mc fc root b d s) : ExactDepth.Full b d := by
  induction path with
  | initial => rw [start_eq]; rfl
  | move _ m ih => exact (ExactDepth.move_depth mc fc _ _ m ih).2.1
  | forcedPass _ _ ih => exact ExactDepth.pass_depth _ _ ih
  | after _ _ ih | worker _ ih => exact ih

theorem At.invariant (path : At mc fc root b d s) (fits : (Board.emptyCount root.val).toNat ≤ 60) :
    Matches s b ∧ ExactDepth.Full b d := by
  induction path with
  | initial => rw [start_eq]; exact ⟨(SearchContextState.initialContract_proven.1 root fits).1, rfl⟩
  | move _ m ih => exact ⟨(SearchContextState.placementContract_proven _ _ _ m _ ih.1).2.1,
      (ExactDepth.move_depth _ _ _ _ m ih.2).2.1⟩
  | forcedPass _ _ ih => exact ⟨(SearchContextState.passContract_proven _ _ ih.1).1, ExactDepth.pass_depth _ _ ih.2⟩
  | after _ calls ih => exact ⟨(calls.restores ih.1).agrees ih.1, ih.2⟩
  | worker _ ih => exact ⟨(SearchContextState.splitContract_proven _ _ ih.1).1, ih.2⟩

theorem At.frameInputs (path : At mc fc root b d s) (fits : (Board.emptyCount root.val).toNat ≤ 60) :
    (leafState s).matches b ∧ ExactDepth.Full b d ∧ d = (leafState s).count ∧
    ply s + d.toNat = 60 ∧ (∀ lane, SearchContextState.current s lane = (SearchContextState.rebuild b.val lane).1) := by
  obtain ⟨aligned, full⟩ := path.invariant fits
  have eq := full.trans aligned.1.1.1.symm
  refine ⟨aligned.1.1, full, eq, ?_, SearchContextState.currentContract_proven s b aligned⟩
  rw [eq, ply]
  have bound := aligned.1.2
  omega

theorem At.fromOpening (path : At mc fc root b d s)
    (played : Board.CheckedPlacementTrace Board.newBoard placements root.val) :
    (leafState s).matches b ∧ ExactDepth.Full b d := by
  have bound := (Board.checkedPlacementTrace_initial_bound played).2.1
  have inputs := path.frameInputs (by omega)
  exact ⟨inputs.1, inputs.2.1⟩

theorem At.shallowSource (path : At mc fc root b d s) (fits : (Board.emptyCount root.val).toNat ≤ 60)
    (target : Flip.Dispatch.TargetConfig) (bmi2 : Bool) (address : EndgameTreeCache.Address)
    (history : List EndgameTreeCache.Store) (alpha : ScoreRefresh.Score) (checked : Bool)
    (cache : 5 ≤ s.empties.count.toNat → EndgameTreeCache.soundHistory mc fc history) :
    NonPvMir.sourceRun target bmi2 mc fc address history b s.empties.links s.empties.parity d s.empties.count alpha checked =
      NonPvMir.run target bmi2 mc fc address history b s.empties.links s.empties.parity d s.empties.count alpha checked :=
  NonPvMir.sourceRun_eq _ _ _ _ _ _ _ _ _ _ _ _ _ (path.frameInputs fits).1 cache

theorem At.deepSource (path : At mc fc root b d s) (fits : (Board.emptyCount root.val).toNat ≤ 60)
    (target : Flip.Dispatch.TargetConfig) (bmi2 paired : Bool) (address : EndgameTreeCache.Address)
    (history : List EndgameTreeCache.Store) (polls : Threading.SplitCutoff.Polling)
    (alpha beta : ScoreRefresh.Score) (attempts : List TranspositionTable.ProbeRetry.RawAttempt)
    (parallel active checked aborted : Bool) (cache : EndgameTreeCache.soundHistory mc fc history) :
    let initial : EndgameSearch.Cancellation.State := ⟨leafState s, history, polls⟩
    DeepLeafMir.sourceRun target bmi2 paired mc fc address b alpha beta d initial attempts parallel active checked aborted =
      DeepLeafMir.run target bmi2 mc fc address b alpha beta d initial attempts parallel active checked aborted :=
  DeepLeafMir.sourceRun_eq _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ (path.frameInputs fits).1 cache

end NeuralReversi.PvsParallel.ContextPathMir
