import NeuralReversi.PvsParallel.Interleaving.ObservedJoin
import NeuralReversi.PvsParallel.Interleaving.RootCausal

namespace NeuralReversi.PvsParallel.Interleaving.Nested

open PvsMoveLoop (ChildRelation)
open Aspiration (Window)
open AspirationMir
open Threading.SplitWindow (State)
open PvsBoardTree (Position GeneratedSquare)
open EndgameOrderMir (Heuristic pvOrder framePriority)
open TranspositionTable (Depth)
open TranspositionTable hiding Score
open TranspositionTable.ProbeRetry (RawAttempt)
open ScoreRefresh (Score)
open BoardFrame (Children)
open RootMoveStorage (Row)
open NativeNodes (Returns Probed)
open TableReads (ChildLookupFrom)

structure Split (done_ : ChildRelation M) (pv cut : Bool) (depth width : Nat) (beta : Int)
    (moves : List M) (idx : Nat) (entry : State) where
  fallback : M
  trace : ObservedJoin.Trace ⟨pv,false,cut,depth,beta,entry⟩ (fun i => done_ (moves[i]?.getD fallback))
    moves.length (BitVec.ofNat width idx)

abbrev Splits (done_ : ChildRelation M) (pv cut : Bool) (depth width : Nat) (beta : Int) :=
  ∀ moves idx entry, Option (Split done_ pv cut depth width beta moves idx entry)

variable {M : Type} {done_ : ChildRelation M}

private def range (entry : State) (beta : Int) := -32767 ≤ entry.alpha ∧ entry.alpha < beta ∧ beta ≤ 32767
private instance (entry : State) (beta : Int) : Decidable (range entry beta) := inferInstanceAs (Decidable (_ ∧ _ ∧ _))

private theorem window (ok : range entry beta) : Window entry.alpha beta := ⟨ok.1,ok.2.1,ok.2.2⟩

/-- Replay reads the owner's actual score word and its restored-ancestor stop poll.
    The caller's entry alpha and task beta are read from the handoff operands. -/
def call (s : Splits done_ pv cut depth width beta) (moves : List M) (idx : Nat) (entry : State) : Option Int := do
  let split ← s moves idx entry
  if range entry beta then split.trace.answer else none

theorem call_of_window (s : Splits done_ pv cut depth width beta) (w : Window entry.alpha beta) :
    call s moves idx entry = (s moves idx entry).bind (fun split => split.trace.answer) := by
  have ok : range entry beta := ⟨w.lower,w.ordered,w.upper⟩
  simp [call,ok]

/-- A handoff reached by either native owner loop passes the numeric check, for any
    scores returned by the visited children. The parent only needs its entry window. -/
theorem call_after_owner (s : Splits done_ pv cut depth width beta)
    (ran : Prefix.Serial.Run done_ reply pv beta ⟨0,moves,[]⟩ ⟨alpha,-PvsMoveLoop.scoreInf⟩ visited remaining (.handoff entry))
    (w : Window alpha beta) :
    let labels := visited ++ remaining.pending
    call s labels remaining.idx entry = (s labels remaining.idx entry).bind (fun split => split.trace.answer) := by
  obtain ⟨_,_,continued,_⟩ := ran.reproduced
  exact call_of_window s (Prefix.continued_window pv reply alpha beta visited _ _ _ w (Int.le_max_left ..) continued).1

private noncomputable def replay (s : Splits done_ pv cut depth width beta) : NativeNodes.Splits done_ pv cut depth width beta :=
  fun moves idx entry => do
    let raw ← s moves idx entry
    if h : range entry beta then
      let evidence := raw.trace.exit.observes entry.alpha (window h) (Int.le_max_left ..)
      return ⟨raw.fallback,Classical.choose evidence⟩
    else none

private theorem replay_call (s : Splits done_ pv cut depth width beta) :
    NativeNodes.call (replay s) moves idx entry = call s moves idx entry := by
  unfold NativeNodes.call replay call
  cases hit : s moves idx entry with
  | none => rfl
  | some op =>
    by_cases ok : range entry beta
    · simp only [ok,↓reduceDIte,↓reduceIte,pure,bind,Option.bind]
      exact (Classical.choose_spec (op.trace.exit.observes entry.alpha (window ok) (Int.le_max_left ..))).1
    · simp [ok]

def pvSplit (s : Splits done_ true cut depth width beta) (e : RootOwnerMir.Slice.Exit M) : Option Int :=
  call s e.moves e.idx e.snapshot.state

def scoutSplit (s : Splits done_ false cut depth width beta) : GeneralOwnerMir.SplitReturn M :=
  fun ordered idx r => match r.outcome with
    | .returned _ => none
    | .handoff entry => call s (ordered.take idx ++ r.visited ++ r.remaining.pending) r.remaining.idx entry

private theorem pv_eq (s : Splits done_ true cut depth width beta) : NativeNodes.pvSplit (replay s) = pvSplit s :=
  funext fun _ => replay_call s

private theorem scout_eq (s : Splits done_ false cut depth width beta) : NativeNodes.scoutSplit (replay s) = scoutSplit s := by
  funext moves idx r
  unfold NativeNodes.scoutSplit scoutSplit
  cases r.outcome with
  | returned => rfl
  | handoff => exact replay_call s

inductive Body (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (prior : CausalTable.Values)
    (reads : CausalTable.Lookups) (b : Position) : Bool → Int → Int → Int → Prop where
  | pv (d : Depth) (paired checked cut : Bool) (mem : MoveList.Initialization.Memory Move.Move) (x : Returns)
      (inside : x.InPrefix prior) (mode : Bitboard.Square → Heuristic) (tie : GeneratedSquare mc b.val → Fin 64)
      (hint : Option Bitboard.Square) (schedule : Nat → Bool × Bool)
      (splits : Splits (Children mc fc b x.prior) true cut d.toNat width beta)
      (full : ExactDepth.Full b d) (wide : 7 ≤ width)
      (run : PvSerialMir.scheduled mc fc b d paired checked mem alpha beta
        (fun next cd lo hi => x.reply next cd true false lo hi) (pvOrder mc fc b mode tie hint)
        (x.pvReplies mc fc b d cut) schedule (pvSplit splits) = some v) : Body mc fc prior reads b true alpha beta v
  | scout (d : Depth) (alpha beta : Score) (paired checked cut : Bool) (mem : MoveList.Initialization.Memory Move.Move)
      (attempts : List RawAttempt) (rows : List (ChildLookupFrom mc fc b src)) (x : Returns)
      (inside : x.InPrefix prior) (mode : Bitboard.Square → Heuristic) (tie : GeneratedSquare mc b.val → Fin 64)
      (schedule : GeneralOwnerMir.Schedule (GeneratedSquare mc b.val))
      (splits : Splits (Children mc fc b x.prior) false cut d.toNat width beta.raw.toInt)
      (full : ExactDepth.Full b d) (wide : 7 ≤ width) (probed : Probed mc fc b reads attempts rows)
      (run : GeneralOwnerMir.scheduled mc fc b d alpha beta paired checked cut mem attempts rows
        (framePriority mc fc b mode tie) x.scout schedule (scoutSplit splits) = some v) :
      Body mc fc prior reads b false alpha.raw.toInt beta.raw.toInt v
  | direct (alpha beta raw : Score) (aligned : EndgameSearch.EmptyStateMatches b mask count)
      (small : count.toNat ≤ 4)
      (result : EndgameSearch.Direct.Mir.run target bmi2 b.val ⟨mask,count⟩ alpha checked = some raw) :
      Body mc fc prior reads b false alpha.raw.toInt beta.raw.toInt raw.raw.toInt
  | boundary (p : EndgameSearch.CacheHistory.Pool) (timeline : EndgameSearch.CacheHistory.Timeline p) (worker atStep : Nat)
      (alpha beta : Score) (d : Depth) (state : EndgameSearch.Cancellation.State) (attempts : List RawAttempt)
      (history : state.history = (timeline.readAt mc fc worker atStep).history) (aligned : state.empties.matches b)
      (parent : ∀ slot e, ProbeRetry.probe (attempts.map (ProbeRetry.observe (Cutoffs.key b))) = .hit slot e → reads b e)
      (result : DeepLeafMir.sourceRun target bmi2 paired mc fc p.address b alpha beta d state attempts
        (DeepLeafMir.parallelPool size) active checked aborted = some v) :
      Body mc fc prior reads b false alpha.raw.toInt beta.raw.toInt v
  | boundarySave (p : EndgameSearch.CacheHistory.Pool) (timeline : EndgameSearch.CacheHistory.Timeline p)
      (worker opening : Nat) (i : EndgameSearch.CacheHistory.Input) (board : i.board = b)
      (depth : i.depth.toNat = 12) (count : i.depth = i.empties.count)
      (parallel : i.parallel = true) (running : i.aborted = false)
      (enough : DeepLeafMir.enoughNodes i.checked before after = true)
      (localRun : i.arm = .deep i.active)
      (result : (i.run mc fc p.address (timeline.readAt mc fc worker opening).history).score = some v) :
      Body mc fc prior reads b false i.alpha.raw.toInt i.beta.raw.toInt v
  | fixed (result : ScoreRefresh.DiscScore) (alpha beta raw : Score)
      (solved : EndgameSearch.Direct.FixedDisc target bmi2 b (ScoreRefresh.toDiscDiff alpha).raw result.raw)
      (converted : ScoreRefresh.fromDiscDiff checked result = .ok raw) :
      Body mc fc prior reads b false alpha.raw.toInt beta.raw.toInt raw.raw.toInt
  | shallow (p : EndgameSearch.CacheHistory.Pool) (timeline : EndgameSearch.CacheHistory.Timeline p) (worker atStep : Nat)
      (alpha beta raw : Score) (lo : 5 ≤ (Board.emptyCount b.val).toNat) (hi : (Board.emptyCount b.val).toNat ≤ 6)
      (solved : let h := (timeline.readAt mc fc worker atStep).history
        let result := EndgameSearch.run mc fc (EndgameSearch.orderedOptions target bmi2 mc fc p.address) .shallow b
          (ScoreRefresh.toDiscDiff alpha).raw.toInt h
        ScoreRefresh.fromDiscDiff checked ⟨BitVec.ofInt 32 result.score⟩ = .ok raw) :
      Body mc fc prior reads b false alpha.raw.toInt beta.raw.toInt raw.raw.toInt
  | cached (p : EndgameSearch.CacheHistory.Pool) (timeline : EndgameSearch.CacheHistory.Timeline p) (worker atStep : Nat)
      (alpha beta raw : Score) (s : EndgameSearch.Cancellation.State)
      (localState : s.history = (timeline.readAt mc fc worker atStep).history) (aligned : s.empties.matches b)
      (solved : let opts := EndgameSearch.orderedOptions target bmi2 mc fc p.address
        let out := EndgameSearch.Cancellation.run mc fc opts b (ScoreRefresh.toDiscDiff alpha).raw.toInt s
        ScoreRefresh.fromDiscDiff checked ⟨BitVec.ofInt 32 (out.score.getD 0)⟩ = .ok raw ∧
          EndgameSearch.Cancellation.publish out aborted raw = some raw) :
      Body mc fc prior reads b false alpha.raw.toInt beta.raw.toInt raw.raw.toInt

  | root (s : SerialStack.State) (op : RootProgram.Operands mc b (BoardFrame.Children mc fc b prior) s.regs.beta.toInt)
      (covers : Root.Covers mc b (s.rows.map Row.square)) (positive : 0 < s.rows.length)
      (answer : RootProgram.run fc 0 s op = some ans) :
      Body mc fc prior reads b true s.regs.alpha.toInt s.regs.beta.toInt ans.value

theorem Body.toBody {pv : Bool} (body : Body mc fc prior reads b pv a z v) : RootCausal.Body mc fc prior reads b pv a z v := by
  cases body with
  | pv d paired checked cut mem x inside mode tie hint sched captures full width run =>
    exact .node (.pv d paired checked cut mem x inside mode tie hint sched (replay captures) full width (pv_eq captures ▸ run))
  | scout d a z paired checked cut mem probes rows x inside mode tie sched captures full width seen run =>
    exact .node (.scout d a z paired checked cut mem probes rows x inside mode tie sched (replay captures) full width seen (scout_eq captures ▸ run))
  | direct a z raw aligned small result => exact .node (.direct a z raw aligned small result)
  | boundary p timeline worker atStep a z d state attempts history aligned parent result =>
    exact .node (.boundary p timeline worker atStep a z d state attempts history aligned parent result)
  | boundarySave p timeline worker opening i board depth count parallel running enough arm result =>
    exact .node (.boundarySave p timeline worker opening i board depth count parallel running enough arm result)
  | fixed result a z v solved convert => exact .node (.fixed result a z v solved convert)
  | shallow p log worker step a z v lo hi run => exact .node (.shallow p log worker step a z v lo hi run)
  | cached p log worker step a z v state localState aligned run =>
    exact .node (.cached p log worker step a z v state localState aligned run)
  | root s op cover pos answer => exact .root s op cover pos answer

theorem toPrefix (run : CausalTable.Prefix x init frames before (Body mc fc)) :
    CausalTable.Prefix x init frames before (RootCausal.Body mc fc) :=
  {run with body := fun f member => (run.body f member).toBody}

end NeuralReversi.PvsParallel.Interleaving.Nested
