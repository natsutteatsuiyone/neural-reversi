import NeuralReversi.PvsParallel.ConcurrentReplay
import NeuralReversi.PvsParallel.DeepLeafMir
import NeuralReversi.PvsParallel.EndgameOrderMir
import NeuralReversi.PvsParallel.GeneralOwnerMir
import NeuralReversi.PvsParallel.NonPvMir
import NeuralReversi.PvsParallel.PvSerialMir
import NeuralReversi.PvsParallel.RootWipeout
import NeuralReversi.PvsParallel.TraceCheck.Frames

namespace NeuralReversi.PvsParallel.FrameReturns

open PvsBoardTree (Position build play pass GeneratedSquare legalSquares)
open Threading.AtomicWord (Event SequencedBefore)
open ConcurrentReplay (UsedChild)

structure Frame extends TraceCheck.FrameReturn where
  id : Nat
  cut : Bool
  scoreStep : Nat
  returnStep : Option Nat
  scoreBeforeReturn : ∀ step, returnStep = some step → scoreStep < step
  boardValid : Board.wellFormed toFrameReturn.board

def Frame.result (f : Frame) (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) : CausalReplay.ScoreResult :=
  ⟨build mc fc ⟨f.toFrameReturn.board, f.boardValid⟩, f.pv, f.alpha, f.beta, f.score⟩

def Frame.actions (f : Frame) (identify : Event → Nat) (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) :
    List ConcurrentReplay.Event :=
  ⟨identify ⟨f.worker, f.scoreStep⟩, .scored f.id (f.result mc fc)⟩ ::
    f.returnStep.toList.map (fun step => ⟨identify ⟨f.worker, step⟩, .returned f.id (f.result mc fc)⟩)

structure Log (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) where
  frames : List Frame
  writes : List ConcurrentReplay.Event
  identify : Event → Nat
  before : Nat → Nat → Prop
  sequenced : ∀ a b, SequencedBefore a b → before (identify a) (identify b)

def Log.events (log : Log mc fc) : List ConcurrentReplay.Event :=
  log.frames.flatMap (fun f => f.actions log.identify mc fc) ++ log.writes

theorem Log.scored (log : Log mc fc) (f : Frame) (mem : f ∈ log.frames) :
    ⟨log.identify ⟨f.worker,f.scoreStep⟩, .scored f.id (f.result mc fc)⟩ ∈ log.events := by
  apply List.mem_append_left
  exact List.mem_flatMap.mpr ⟨f, mem, by simp [Frame.actions]⟩

theorem Log.returned (log : Log mc fc) (f : Frame) (mem : f ∈ log.frames) (finished : f.returnStep = some n) :
    ⟨log.identify ⟨f.worker,n⟩, .returned f.id (f.result mc fc)⟩ ∈ log.events := by
  apply List.mem_append_left
  exact List.mem_flatMap.mpr ⟨f, mem, by simp [Frame.actions, finished]⟩

theorem Log.returnValid (log : Log mc fc) (f : Frame) (mem : f ∈ log.frames) (finished : f.returnStep = some n) :
    (ConcurrentReplay.Action.returned f.id (f.result mc fc)).Valid mc fc endian children reads
      (ConcurrentReplay.predecessors log.before log.events (log.identify ⟨f.worker,n⟩)) := by
  exact ⟨_, ConcurrentReplay.mem_predecessors.mpr ⟨log.scored f mem, log.sequenced _ _ ⟨rfl, f.scoreBeforeReturn _ finished⟩⟩, rfl⟩

def matchesCall (f : Frame) (caller : Nat) (use : Event) (board : Position) (depth : TranspositionTable.Depth)
    (pv cut : Bool) (a z : Int) : Bool :=
  f.returnStep.any fun step => decide (f.parent = caller ∧ f.worker = use.worker ∧ step < use.step ∧
    f.toFrameReturn.board = board.val ∧ f.depth = depth.toNat ∧ f.pv = pv ∧ f.cut = cut ∧ f.alpha = a ∧ f.beta = z)

def Log.reply (log : Log mc fc) (caller : Nat) (use : Event) (board : Position) (depth : TranspositionTable.Depth)
    (pv cut : Bool) (a z : Int) : Option Int :=
  (log.frames.find? (fun f => matchesCall f caller use board depth pv cut a z)).map (·.score)

theorem Log.reply_used (log : Log mc fc) (caller : Nat) (use : Event) (board : Position)
    (depth : TranspositionTable.Depth) (pv cut : Bool) (a z v : Int)
    (answer : log.reply caller use board depth pv cut a z = some v) :
    UsedChild log.identify log.before log.events (log.identify use) .serial (build mc fc board) pv a z v := by
  unfold Log.reply at answer
  cases row : log.frames.find? (fun f => matchesCall f caller use board depth pv cut a z) with
  | none => simp [row] at answer
  | some f =>
    have ok := List.find?_some row
    obtain ⟨step,finished,_,worker,earlier,position,_,node,_,lo,hi⟩ :
        ∃ step, f.returnStep = some step ∧ f.parent = caller ∧ f.worker = use.worker ∧ step < use.step ∧
          f.toFrameReturn.board = board.val ∧ f.depth = depth.toNat ∧ f.pv = pv ∧ f.cut = cut ∧
          f.alpha = a ∧ f.beta = z := by
      cases h : f.returnStep with
      | none => simp [matchesCall,h] at ok
      | some n => exact ⟨n,rfl,by simpa [matchesCall,h] using ok⟩
    have value : f.score = v := by simpa [row] using answer
    have boardEq : (⟨f.toFrameReturn.board, f.boardValid⟩ : Position) = board := Subtype.ext position
    refine ⟨_, log.returned f (List.mem_of_find?_eq_some row) finished, log.sequenced _ _ ⟨worker, earlier⟩, f.id, ?_⟩
    simp only [Frame.result, value, node, lo, hi, boardEq]

def Log.nonPv (log : Log mc fc) (caller : Nat) (use : Event) : GeneralNonPvMir.Child :=
  fun board depth cut => log.reply caller use board depth false cut

theorem Log.scouted (log : Log mc fc) (caller : Nat) (use : Event) (b : Position)
    (depth : TranspositionTable.Depth) (cut : Bool) (a z : Int) (window : Aspiration.Window a z)
    (prior : ∀ stage tree pv x y v, UsedChild log.identify log.before log.events (log.identify use)
      stage tree pv x y v → Replayed tree pv x y v)
    (m : GeneratedSquare mc b.val) (v : Int)
    (answer : GeneralOwnerMir.scoutReply mc fc b depth cut (log.nonPv caller use) a m = some v) :
    UsedChild log.identify log.before log.events (log.identify use) .serial (build mc fc (play mc fc b m).val)
      false (-(a+1)) (-a) v ∧ v < PvsMoveLoop.scoreInf := by
  have seen := log.reply_used caller use (play mc fc b m).val (depth-1) false (!cut) (-(a+1)) (-a) v answer
  obtain ⟨n, replay⟩ := prior .serial _ _ _ _ _ seen
  exact ⟨seen, (replayContract_proven _ _ _ _ _ n (PvsMoveLoop.scoutWindow window) (by omega) replay).upper⟩

def Frame.position (f : Frame) : Position := ⟨f.toFrameReturn.board, f.boardValid⟩
def Frame.use (f : Frame) : Event := ⟨f.worker, f.scoreStep⟩
abbrev Samples (log : Log mc fc) (endian : Bool) (f : Frame) :=
  ConcurrentReplay.ReadAt log.before log.events (log.identify f.use) endian
abbrev Children (log : Log mc fc) (f : Frame) :=
  UsedChild log.identify log.before log.events (log.identify f.use)

def Log.pvReplies (log : Log mc fc) (f : Frame) (d : TranspositionTable.Depth) :
    RootOwnerMir.Replies (GeneratedSquare mc f.position.val) :=
  fun m pv => log.reply f.id f.use (play mc fc f.position m).val (d-1) pv (!pv && !f.cut)

def rootOrder (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : Position)
    (roots : List Bitboard.Square) (idx : Nat) (mode : Bitboard.Square → EndgameOrderMir.Heuristic)
    (tie : Move.Move → Fin 64) (hint : Option Bitboard.Square) : List (GeneratedSquare mc b.val) :=
  (EndgameOrderMir.rootRecords mc fc b roots idx mode tie hint).filterMap (RootFrameMir.asChoice mc b)

structure RootOwner (log : Log mc fc) (f : Frame) (d : TranspositionTable.Depth)
    (roots : List Bitboard.Square) (idx : Nat) (a z : Int) where
  mode : Bitboard.Square → EndgameOrderMir.Heuristic
  tie : Move.Move → Fin 64
  hint : Option Bitboard.Square
  schedule : Nat → Bool × Bool
  receipt : RootOwnerMir.Result (GeneratedSquare mc f.position.val)
  normal : let order := rootOrder mc fc f.position roots idx mode tie hint
    RootOwnerMir.fromMoves Subtype.val (log.pvReplies f d) schedule d.toNat rootMinimumSplitDepth a z order = some receipt

def RootOwner.order {log : Log mc fc} (owner : RootOwner log f d roots idx a z) :=
  rootOrder mc fc f.position roots idx owner.mode owner.tie owner.hint

theorem RootOwner.candidates {log : Log mc fc} (owner : RootOwner log f d roots idx a z) :
    owner.order.Perm (Root.choices mc f.position roots idx) :=
  (EndgameOrderMir.root_perm mc fc f.position roots idx owner.mode owner.tie owner.hint).1

inductive RootStep (log : Log mc fc) (f : Frame) (d : TranspositionTable.Depth)
    (roots : List Bitboard.Square) (idx : Nat) (a z : Int) : Int → (Bitboard.Square → Int) → Prop
  | serial (owner : RootOwner log f d roots idx a z) : owner.receipt.handoff = false →
      RootStep log f d roots idx a z owner.receipt.snapshot.state.best owner.receipt.snapshot.scores
  | split (owner : RootOwner log f d roots idx a z)
      (first : GeneratedSquare mc f.position.val) (rest : List (GeneratedSquare mc f.position.val))
      (division : SplitExecution ((first :: rest).map (fun m => build mc fc (play mc fc f.position m).val)).length
        true a z v) :
      owner.receipt.handoff = true → first :: rest = owner.receipt.visited ++ owner.receipt.remaining.pending →
      division.start.toNat = owner.receipt.remaining.idx → division.entry = owner.receipt.snapshot.state →
      ReportCalls (Children log f) ((first :: rest).map (fun m => build mc fc (play mc fc f.position m).val)) division →
      RootStep log f d roots idx a z v (RootReplay.Split.replay (fun i => (((first :: rest)[i]?).getD first).val)
        z division.reports owner.receipt.snapshot).scores
  | wipe (paired : Bool) (mem : MoveList.Initialization.Memory Move.Move) (sq : Bitboard.Square) :
      (MoveList.Generation.fill false paired fc f.position.val (Root.remainingMask mc f.position.val roots idx) mem).wipeout =
        some (some sq) → RootStep log f d roots idx a z PvsMoveLoop.scoreMax
          ((RootReplay.start a).record z ⟨sq,a,PvsMoveLoop.scoreMax⟩).scores

theorem RootStep.attempt {log : Log mc fc} (step : RootStep log f d roots idx a z v rows)
    (covers : Root.Covers mc f.position roots) :
    RootEvidence.AttemptWithUses (Children log f) mc fc f.position roots idx a z v rows := by
  have calls : ∀ m pv x y value, log.pvReplies f d m pv x y = some value →
      Children log f .serial (build mc fc (play mc fc f.position m).val) pv x y value :=
    fun m pv x y value h => log.reply_used f.id f.use _ _ pv _ x y value h
  cases step with
  | serial owner finish => exact RootOwnerMir.returnedAttempt owner.candidates calls owner.normal finish
  | split owner first rest division handoff labels start entry reports =>
    exact RootOwnerMir.handoffAttempt owner.candidates calls owner.normal handoff first rest labels division start entry reports
  | wipe paired mem sq cache =>
    obtain ⟨m, member, same, wipe⟩ := ConcurrentReplay.generatedWipeout mc fc f.position roots idx paired mem sq covers cache
    rw [← same]
    exact .wipeout m member wipe

/-- Local inputs and the normal return from the branch taken by one exact search frame. -/
inductive ScoreStep (log : Log mc fc) (endian : Bool) (f : Frame) (depth : TranspositionTable.Depth)
    (a z : ScoreRefresh.Score) : Prop
  | leaf (target : Flip.Dispatch.TargetConfig) (bmi2 checked : Bool) (address : EndgameTreeCache.Address)
      (s : EndgameSearch.Cancellation.State) :
      f.pv = false → s.empties.matches f.position → s.empties.count.toNat ≤ 6 →
      (5 ≤ s.empties.count.toNat → EndgameTreeCache.soundHistory mc fc s.history) →
      (NonPvMir.sourceRun target bmi2 mc fc address s.history f.position s.empties.mask depth
        s.empties.count a checked).map (·.raw.toInt) = some f.score → ScoreStep log endian f depth a z
  | deepLeaf (target : Flip.Dispatch.TargetConfig) (bmi2 paired : Bool) (address : EndgameTreeCache.Address)
      (s : EndgameSearch.Cancellation.State) (attempts : List TranspositionTable.ProbeRetry.RawAttempt)
      (parallel active checked aborted : Bool) :
      f.pv = false → s.empties.matches f.position → EndgameTreeCache.soundHistory mc fc s.history →
      TableReads.ProbeReadsFrom (Samples log endian f) attempts →
      DeepLeafMir.sourceRun target bmi2 paired mc fc address f.position a z depth s attempts parallel active checked aborted =
        some f.score → ScoreStep log endian f depth a z
  | nonPv (paired checked : Bool) (mem : MoveList.Initialization.Memory Move.Move)
      (attempts : List TranspositionTable.ProbeRetry.RawAttempt)
      (reads : List (TableReads.ChildLookupFrom mc fc f.position (Samples log endian f)))
      (mode : Bitboard.Square → EndgameOrderMir.Heuristic) (tie : GeneratedSquare mc f.position.val → Fin 64)
      (schedule : GeneralOwnerMir.Schedule (GeneratedSquare mc f.position.val))
      (split : GeneralOwnerMir.SplitReturn (GeneratedSquare mc f.position.val)) :
      f.pv = false → TableReads.ProbeReadsFrom (Samples log endian f) attempts →
      (∀ ordered idx r state v, r.outcome = .handoff state → split ordered idx r = some v →
        GeneralOwnerMir.Completed (Children log f) (fun m => build mc fc (play mc fc f.position m).val)
          a.raw.toInt z.raw.toInt (ordered.take idx) r v) →
      GeneralOwnerMir.scheduled mc fc f.position depth a z paired checked f.cut mem attempts reads
        (EndgameOrderMir.framePriority mc fc f.position mode tie)
        (log.nonPv f.id f.use) schedule split = some f.score → ScoreStep log endian f depth a z
  | pv (paired checked : Bool) (mem : MoveList.Initialization.Memory Move.Move)
      (mode : Bitboard.Square → EndgameOrderMir.Heuristic) (tie : GeneratedSquare mc f.position.val → Fin 64)
      (hint : Option Bitboard.Square) (schedule : Nat → Bool × Bool)
      (splitCall : RootOwnerMir.Slice.Exit (GeneratedSquare mc f.position.val) → Option Int) :
      f.pv = true →
      (∀ r v, r.handoff = true → splitCall r = some v →
        let labels := r.moves.map (fun m => build mc fc (play mc fc f.position m).val)
        ∃ split : SplitExecution labels.length true a.raw.toInt z.raw.toInt v,
          split.start.toNat = r.idx ∧ split.entry = r.snapshot.state ∧ ReportCalls (Children log f) labels split) →
      PvSerialMir.scheduled mc fc f.position depth paired checked mem a.raw.toInt z.raw.toInt
        (fun b d => log.reply f.id f.use b d true false)
        (EndgameOrderMir.pvOrder mc fc f.position mode tie hint) (log.pvReplies f depth) schedule splitCall =
          some f.score → ScoreStep log endian f depth a z
  | firstRoot (roots : List Bitboard.Square) (rows : Bitboard.Square → Int) :
      f.pv = true → f.cut = false → Root.Covers mc f.position roots → 0 < roots.length →
      RootStep log f depth roots 0 a.raw.toInt z.raw.toInt f.score rows → ScoreStep log endian f depth a z

theorem ScoreStep.valid (log : Log mc fc) (f : Frame) (source : ScoreStep log endian f depth a z)
    (full : ExactDepth.Full f.position depth) (lo : f.alpha = a.raw.toInt) (hi : f.beta = z.raw.toInt)
    (window : Aspiration.Window f.alpha f.beta) (null : f.pv = false → f.beta = f.alpha+1)
    (prior : ∀ stage tree pv x y v, Children log f stage tree pv x y v → Replayed tree pv x y v) :
    (ConcurrentReplay.Action.scored f.id (f.result mc fc)).Valid mc fc endian (Children log f)
      (Samples log endian f) past := by
  change ExecutionWithUses _ _ (build mc fc f.position) f.pv f.alpha f.beta f.score
  rw [lo, hi] at window null ⊢
  cases source with
  | leaf target bmi2 checked address s tag aligned size cache result =>
    rw [NonPvMir.sourceRun_eq _ _ _ _ _ _ _ _ _ _ _ _ aligned cache] at result
    obtain ⟨v, same, valid, _⟩ := NonPvMir.scoreContract_proven (CausalReplay.PrefixExitFrom mc fc (Samples log endian f))
      (Children log f) target bmi2 mc fc address s.history f.position s.empties.mask depth
      s.empties.count a z checked full aligned size cache (null tag)
    rw [same] at result
    simpa only [tag, Option.map_some, Option.some.injEq] using
      (show ExecutionWithUses _ _ _ false _ _ f.score from (Option.some.inj result) ▸ valid)
  | deepLeaf target bmi2 paired address s attempts parallel active checked aborted tag aligned cache sampled answer =>
    rw [DeepLeafMir.sourceRun_eq target bmi2 paired mc fc address f.position a z depth s attempts
      parallel active checked aborted aligned cache] at answer
    rw [tag]
    exact DeepLeafMir.observedExecution target bmi2 mc fc address f.position a z depth s attempts
      parallel active checked aborted sampled aligned cache (null tag) answer
  | nonPv paired checked mem attempts etc mode tie schedule split tag sampled reports answer =>
    rw [tag]
    exact GeneralOwnerMir.scheduledExecution mc fc f.position depth a z paired checked f.cut mem attempts etc _
      (log.nonPv f.id f.use) schedule split full (null tag) sampled
      (fun m _ => EndgameOrderMir.framePriority_low mc fc f.position mode tie m)
      (fun _ _ v ret => log.reply_used _ _ _ _ _ _ _ _ v ret)
      (log.scouted f.id f.use f.position depth f.cut _ _ window prior) reports answer
  | pv paired checked mem mode tie hint schedule splitCall tag reports answer =>
    rw [tag]
    exact PvSerialMir.scheduledExecution mc fc f.position depth paired checked mem _ _
      (fun b d => log.reply f.id f.use b d true false) _ (log.pvReplies f depth) schedule splitCall full
      (EndgameOrderMir.pvOrder_perm mc fc f.position mode tie hint)
      (fun _ _ v ret => log.reply_used _ _ _ _ _ _ _ _ v ret)
      (fun _ _ _ _ v ret => log.reply_used _ _ _ _ _ _ _ _ v ret) reports answer
  | firstRoot roots rows pv _ covers positive root =>
    rw [pv]
    exact (root.attempt covers).boardExecution covers positive (fun m legal wipe => .wipeout m legal wipe)

structure Decoded (log : Log mc fc) (endian : Bool) : Prop where
  irrefl : ∀ i, ¬log.before i i
  trans : ∀ {i j k}, log.before i j → log.before j k → log.before i k
  frames : ∀ f ∈ log.frames, ∃ depth a z, ExactDepth.Full f.position depth ∧
    f.depth = depth.toNat ∧ f.alpha = a.raw.toInt ∧ f.beta = z.raw.toInt ∧
    Aspiration.Window f.alpha f.beta ∧ (f.pv = false → f.beta = f.alpha + 1) ∧ ScoreStep log endian f depth a z
  writes : ∀ e ∈ log.writes, e.action.Words endian ∧
    (∀ call, e.call = some call → e.id = call.finish) ∧
    (TableReads.LegalHints (ConcurrentReplay.history (ConcurrentReplay.predecessors log.before log.events e.id)) →
    (∀ raw, ConcurrentReplay.ReadAt log.before log.events e.id endian raw →
      raw.publication (ConcurrentReplay.history (ConcurrentReplay.predecessors log.before log.events e.id))) →
    e.action.Valid mc fc endian (UsedChild log.identify log.before log.events e.id)
      (ConcurrentReplay.ReadAt log.before log.events e.id endian)
      (ConcurrentReplay.predecessors log.before log.events e.id))

private theorem Log.cases_event (log : Log mc fc) (e : ConcurrentReplay.Event) (member : e ∈ log.events) :
    e ∈ log.writes ∨ ∃ f ∈ log.frames,
      e = ⟨log.identify f.use, .scored f.id (f.result mc fc)⟩ ∨
      ∃ n, f.returnStep = some n ∧ e = ⟨log.identify ⟨f.worker,n⟩, .returned f.id (f.result mc fc)⟩ := by
  rcases List.mem_append.mp member with frame | write
  · obtain ⟨f, present, action⟩ := List.mem_flatMap.mp frame
    refine Or.inr ⟨f,present,?_⟩
    cases h : f.returnStep <;> simpa [Frame.actions,Frame.use,h] using action
  · exact Or.inl write

def Decoded.observations {log : Log mc fc} (data : Decoded log endian) : ConcurrentReplay.Observations mc fc endian log.events := by
  have words (e : ConcurrentReplay.Event) (member : e ∈ log.events) : e.action.Words endian := by
    rcases log.cases_event e member with write | ⟨f, _, x | ⟨_,_,x⟩⟩
    · exact (data.writes e write).1
    all_goals subst e; trivial
  have publications (e : ConcurrentReplay.Event) (member : e ∈ log.events) :
      ∀ call, e.call = some call → e.id = call.finish := by
    rcases log.cases_event e member with write | ⟨f, _, x | ⟨_,_,x⟩⟩
    · exact (data.writes e write).2.1
    all_goals subst e; simp [ConcurrentReplay.Event.call]
  apply ConcurrentReplay.Observations.ofPriorReturns log.identify log.before data.irrefl data.trans words publications
  intro e member hints prior
  rcases log.cases_event e member with write | ⟨f, present, rfl | ⟨_,finished,rfl⟩⟩
  · exact (data.writes e write).2.2 hints (fun raw sampled =>
      ConcurrentReplay.sampleContract_proven _ _ _ _ raw sampled data.trans publications words)
  · obtain ⟨d, a, z, full, _, lo, hi, window, null, step⟩ := data.frames f present
    exact step.valid log f full lo hi window null prior
  · exact log.returnValid f present finished

theorem Decoded.rootAttempt {log : Log mc fc} (data : Decoded log endian)
    (step : RootStep log f depth roots idx a z v rows) (covers : Root.Covers mc f.position roots) :
    ConcurrentReplay.RootAttempt mc fc endian f.position roots idx a z v rows :=
  ⟨log.events, data.observations, log.identify f.use, step.attempt covers⟩

theorem Decoded.returnedScore {log : Log mc fc} (data : Decoded log endian)
    (covers : Root.Covers mc f.position roots) (positive : 0 < roots.length) (window : Aspiration.Window a z)
    (accepted : Aspiration.runWith search fuel 0 a z 256 = some line)
    (step : RootStep log f d roots 0 line.alpha line.beta line.trial.score line.trial.payload)
    (pv : Bitboard.Square → List Bitboard.Square) (nodes : Nat) :
    let rows := roots.map (fun sq => (⟨sq,line.trial.payload sq,pv sq⟩ : SearchPublication.Row))
    ∃ result, SearchFallback.fromSnapshot (SearchPublication.capture rows ⟨d.toNat,5⟩ true) nodes = some result ∧
      result.best.score = PvsBoardTree.minimax mc fc f.position :=
  ConcurrentReplay.singleReturnedContract_proven mc fc endian f.position roots search fuel a z line pv d.toNat nodes
    covers positive window accepted (data.rootAttempt step covers)

theorem Decoded.returnedMultiScore {log : Log mc fc} (data : Decoded log endian) (board : Position)
    (observe : RootMultiPV.Observations) (seed : RootMultiPV.Seed)
    (covers : Root.Covers mc board roots) (positive : 0 < roots.length)
    (lines : RootMultiPV.LineCalls observe seed (fun order idx _ =>
      ∃ f d accepted, f.position = board ∧ Aspiration.Window (seed order idx).1 (seed order idx).2 ∧
        Aspiration.runWith (observe order idx) 11 0 (seed order idx).1 (seed order idx).2 256 = some accepted ∧
        RootStep log f d order idx accepted.alpha accepted.beta accepted.trial.score accepted.trial.payload)
      roots.length 0 roots)
    (pv : Bitboard.Square → List Bitboard.Square) (depth nodes : Nat) :
    ∃ batch result, RootMultiPV.run observe seed roots.length 0 roots = some batch ∧
      RootMultiPV.Ranked (Root.squareValue mc fc board) roots batch.picks ∧
      let rows := batch.picks.map (fun p => (⟨p.square,p.score,pv p.square⟩ : SearchPublication.Row))
      SearchFallback.fromSnapshot (SearchPublication.capture rows ⟨depth,5⟩ true) nodes = some result ∧
        result.best.score = PvsBoardTree.minimax mc fc board ∧ result.endgame = true := by
  obtain ⟨batch, found, count, _, _, _, ranked⟩ := lines.ranked (mc := mc) (fc := fc) (board := board) (by
    intro order idx line covered inside ran checked
    obtain ⟨f, d, accepted, boardEq, window, chosen, step⟩ := checked
    have attempt := data.rootAttempt step (boardEq ▸ covered)
    rw [boardEq] at attempt
    have witness := Aspiration.runWith_witness (observe order idx) window (by decide) chosen
    obtain ⟨expected, finished, _, _, valid⟩ := RootAspiration.line_of_accepted inside witness
      (attempt.sound covered inside witness.window)
    have same := Option.some.inj (finished.symm.trans (by simpa [RootAspiration.runLine, chosen] using ran))
    exact same ▸ valid) covers (by simp)
  have present : batch.picks ≠ [] := by
    intro empty
    simp [empty] at count
    omega
  obtain ⟨result, current, score, endgame, _⟩ :=
    ExactEndgameScore.rootResult_of_ranked covers batch.picks present ranked pv depth nodes [] (by simp)
  exact ⟨batch,result,found,by simpa using ranked,by simpa using current,score,endgame⟩

end NeuralReversi.PvsParallel.FrameReturns
