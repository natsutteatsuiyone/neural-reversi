module
public import NeuralReversi.PvsParallel.Interleaving.JournalScoutOwner
public import NeuralReversi.PvsParallel.Interleaving.JournalNestedHelpers
meta import NeuralReversi.PvsParallel.Interleaving.JournalNestedHelpers
meta import NeuralReversi.PvsParallel.Interleaving.JournalScoutOwner

@[expose] public section

namespace NeuralReversi.PvsParallel.Interleaving.JournalPvSplit

open PvsBoardTree (Position GeneratedSquare legalSquares)
open CausalTable (Frame Values Lookups)
open TranspositionTable (Depth)
open CallReturns (Scope Owner)
open EndgameOrderMir (pvOrder)
open JournalBodies (Check)

def run (sc : Scope) (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : Position)
    (d : Depth) (cut : Bool) (a z : Int) (op : JournalPv.Operands mc b) (hint : Option Bitboard.Square) :=
  let moves := pvOrder mc fc b op.mode op.tie hint
  let q : PvsPostpone.Queue _ := ⟨0,moves,[]⟩
  CallReturns.ownerRun Subtype.val (sc.ownerReplies (CallReturns.pvRequest mc fc b d cut)) op.schedule
    d.toNat rootMinimumSplitDepth z (q.budget+1) 0 0 moves.length q (RootReplay.start a)

def dispatch (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : Position) (d : Depth) :=
  let g := PvEntryMir.generated mc fc b false (fun _ => none)
  PvEntryMir.dispatch d.toNat (g.len.getD 0 != 0) (Board.hasLegalMoves mc (PvsBoardTree.pass b).val) g.wipeout.join.isSome

private theorem replay (sc : Scope) (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig)
    (b : Position) (d : Depth) (cut : Bool) (a z score : Int) (op : JournalPv.Operands mc b) (hint : Option Bitboard.Square)
    (full : ExactDepth.Full b d) (w : Aspiration.Window a z) (mobile : legalSquares mc b.val ≠ [])
    (path : dispatch mc fc b d = .continueSearch) (out : Owner (GeneratedSquare mc b.val))
    (ran : run sc mc fc b d cut a z op hint = some out) (hands : out.result.handoff = true) (wide : 7 ≤ width)
    (splits : Nested.Splits (BoardFrame.Children mc fc b (sc.nativePlan (CallReturns.boardPlan mc fc b out.plan)).prior) true cut d.toNat width z)
    (word : Nested.pvSplit splits (RootOwnerMir.Slice.layout [] out.result) = some score) :
    let x := sc.nativePlan (CallReturns.boardPlan mc fc b out.plan)
    ∀ (prior : Values) (reads : Lookups), x.InPrefix prior → Nested.Body mc fc prior reads b true a z score ∧
      WeakFrame.Frame mc fc prior reads b true a z score := by
  let moves := pvOrder mc fc b op.mode op.tie hint
  let x := sc.nativePlan (CallReturns.boardPlan mc fc b out.plan)
  let r := out.result
  have perm := EndgameOrderMir.pvOrder_perm mc fc b op.mode op.tie hint
  obtain ⟨distinct,size⟩ := EndgameOrderMir.generated_layout mc b
  have fromMoves := sc.pvRun_fromMoves mc fc b d cut op.schedule rootMinimumSplitDepth a z moves (perm.nodup_iff.mpr distinct) ran
  dsimp only
  intro prior reads inside
  constructor
  · apply Nested.Body.pv d false false cut (fun _ => none) x inside op.mode op.tie hint op.schedule splits full wide
    unfold PvSerialMir.scheduled
    change (match dispatch mc fc b d with | .continueSearch => _ | _ => _) = _
    rw [path]
    dsimp only
    rw [RootOwnerMir.Slice.fromMoves_eq,fromMoves]
    simpa [RootOwnerMir.Slice.layout,hands] using word
  · obtain ⟨trace,_,_⟩ := RootOwnerMir.refines Subtype.val (x.pvReplies mc fc b d cut) (BoardFrame.Children mc fc b prior)
      (fun _ _ _ _ _ got => inside _ _ _ _ _ (x.hit got)) _ _ _ _ _ _ _ _ _ _ (by simp) rfl fromMoves
    rw [show r.outcome = .handoff r.snapshot.state by simp [r,RootOwnerMir.Result.outcome,hands]] at trace
    let labels := r.visited ++ r.remaining.pending
    have window := (Prefix.continued_window true _ a z r.visited true ⟨a,-PvsMoveLoop.scoreInf⟩ r.snapshot.state
      w (Int.le_max_left ..) trace.reproduced.2.2.1).1
    change Nested.call splits labels r.remaining.idx r.snapshot.state = some score at word
    rw [Nested.call_of_window splits window] at word
    obtain ⟨physical,_,answer⟩ := Option.bind_eq_some_iff.mp word
    obtain ⟨join,normal,_⟩ := physical.trace.exit.observes _ window (Int.le_max_left ..)
    let joined : Join _ (fun i => BoardFrame.Children mc fc b prior (labels[i]?.getD physical.fallback)) _ _ :=
      join.mapCalls (fun _ _ _ _ _ child => inside _ _ _ _ _ child)
    have count : labels.length < 2^width := by
      have complete : labels.Perm moves := by simpa [labels,PvsPostpone.Queue.pending] using trace.reproduced.1
      rw [complete.length_eq,perm.length_eq]
      exact Nat.lt_of_le_of_lt size (Nat.lt_of_lt_of_le (by decide : 64 < 2^7) (Nat.pow_le_pow_right (by decide) wide))
    cases gen : legalSquares mc b.val with
    | nil => contradiction
    | cons first rest =>
      apply WeakFrame.Frame.moves first rest moves gen (gen ▸ perm)
      exact .joined _ physical.fallback false cut d.toNat width trace count joined (normal.trans answer)

/-- This owner performed its direct recursive calls, entered `Thread::split` at the PVS
    call site, then returned from that split before choosing its score. -/
def span (es : Joint.Tape Op) (sc : Scope) (worker after : Nat) : Option Nat := do
  let direct := sc.calls.toList.zipIdx.filter fun (a,_) => a.parent == some sc.parent && decide (a.entered < sc.consumer)
  let (split,id) :: prev := direct.reverse | none
  guard (split.worker = worker ∧ split.callee = .split ∧ split.site = 6 ∧ after < split.entered)
  guard (prev.all fun (a,_) => a.worker == worker && a.callee == .pvs && decide (after < a.entered) &&
    a.returned.any (fun t => decide (t < split.entered)))
  let [closed] := (es.zipIdx.filterMap fun (ev,k) => match ev with
    | .engine (.localCall w (.leave _ first .split _ none) _) => if w = worker ∧ first = split.first then some k else none
    | _ => none) | none
  guard (split.entered < closed ∧ closed < sc.consumer)
  return id

structure Candidate (sc : Scope) (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : Position)
    (cut : Bool) (depth : Nat) (beta : Int) where
  width : Nat
  splits : Nested.Splits (BoardFrame.Children mc fc b (sc.native .unique).prior) true cut depth width beta

def finish (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (s : CallJournal.State Op) (f : Frame)
    (acts : CallReturns.State) (parent : Nat) (reads : Lookups) (d : Depth) (cut : Bool) (op : JournalPv.Operands mc f.board)
    (hint : Option Bitboard.Square) (pv : f.pv = true) (full : ExactDepth.Full f.board d) (w : Aspiration.Window f.alpha f.beta)
    (path : dispatch mc fc f.board d = .continueSearch)
    (packet : Candidate (JournalPv.scope s f acts parent) mc fc f.board cut d.toNat f.beta) : Check (JournalPv.Bodies mc fc s f reads) :=
  let sc := JournalPv.scope s f acts parent
  match success : run sc mc fc f.board d cut f.alpha f.beta op hint with
  | none => .rejected
  | some out =>
    if ok : out.result.handoff = true ∧ out.next = sc.slots.length ∧ 7 ≤ packet.width ∧ legalSquares mc f.board.val ≠ [] ∧
        Nested.pvSplit packet.splits (RootOwnerMir.Slice.layout [] out.result) = some f.score then .proved (by
      obtain ⟨seed,reg⟩ := JournalPv.projections s f acts parent (CallReturns.boardPlan mc fc f.board out.plan)
      have result := replay sc mc fc f.board d cut f.alpha f.beta f.score op hint full w ok.2.2.2.1 path out success
        ok.1 ok.2.2.1 packet.splits ok.2.2.2.2
      constructor
      · unfold JournalBodies.SeedObligation
        rw [pv]
        exact .internal (result _ reads seed).2
      · intro weak
        unfold JournalBodies.WeakObligation
        rw [pv]
        simpa only [JournalPv.registry,weak,↓reduceIte] using (result _ reads reg).2
      · intro exact_
        unfold JournalBodies.ExactObligation
        rw [pv]
        simpa only [JournalPv.registry,exact_,↓reduceIte] using (result _ reads reg).1)
    else .rejected

def evaluate (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (es : Joint.Tape Op) (s : CallJournal.State Op)
    (f : Frame) (reads : Lookups) (op : JournalPv.Operands mc f.board)
    (probe : Option (Nat × Nat × TranspositionTable.ProbeRetry.Result))
    (get : (sc : Scope) → (splitId : Nat) → (cut : Bool) → (depth : Nat) → Option (Candidate sc mc fc f.board cut depth f.beta)) :
    Check (JournalPv.Bodies mc fc s f reads) :=
  match s.choices.find? (fun owner => owner.stamp == f.chosen) with
  | none => .rejected
  | some owner =>
    if !owner.pv || !owner.endgame || owner.rootLine.isSome then .outside else
    if shape : f.pv = true ∧ owner.level = s.level f ∧ owner.depth = Board.emptyCount f.board.val ∧
        -32767 ≤ f.alpha ∧ f.alpha < f.beta ∧ f.beta ≤ 32767 then
      if path : dispatch mc fc f.board owner.depth = .continueSearch then
        match CallReturns.captureTimed (es.map Joint.Event.frameSample) with
        | none => .rejected
        | some acts => match JournalPv.parent acts owner f with
          | none => .rejected
          | some (parent,start) =>
            let sc := JournalPv.scope s f acts parent
            if !(acts.calls.any fun a => a.parent == some parent && a.callee == .split && decide (a.entered < f.chosen)) then .outside else
            match JournalPrefix.parentAt es owner.worker start f.chosen f.board with
            | none => .rejected
            | some (opened,closed,raw) =>
              if probe != some (opened,closed,raw) then .rejected else
              match span es sc owner.worker closed with
              | none => .rejected
              | some id => match get sc id owner.cut owner.depth.toNat with
                | none => .rejected
                | some pkt => finish mc fc s f acts parent reads owner.depth owner.cut op (TranspositionTable.ReaderMir.bestMove raw)
                    shape.1 shape.2.2.1 ⟨shape.2.2.2.1,shape.2.2.2.2.1,shape.2.2.2.2.2⟩ path pkt
      else .outside
    else .rejected

/-- The captured primitive-word trace and iterator ticket width for this specific nested
    split. Its immutable operands are checked by `JournalNestedHelpers.splits`. -/
abbrev Feed (es : Joint.Tape Op) (s : CallJournal.State Op) (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) :=
  (f : Frame) → (sc : Scope) → (splitId : Nat) → (cut : Bool) → (depth : Nat) →
    (width : Nat) × JournalNestedHelpers.Feed es sc (s.level f).val mc fc f.board true cut depth width f.beta

variable {x : NativeTableWrites.Journal.Execution c endian rs} {es : Joint.Tape Op} {m : Joint.Atomics x es}

/-- Check the owner's direct child ordinal prefix and its own physical split. Each helper
    scout or wide retry is a completed recursive call in this same event history. -/
def check (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig)
    (init : ∀ a, NativeTableWrites.Journal.Even (x.memory a)) (probes : JournalBodies.Probes m) (s : CallJournal.State Op)
    (f : Frame) (op : JournalPv.Operands mc f.board) (feed : Feed es s mc fc) : Check (JournalPv.Bodies mc fc s f (CausalTable.lookup x init f.chosen)) :=
  evaluate mc fc es s f _ op ((probes f.chosen f.board).map fun got =>
    (got.opening,got.finish,TranspositionTable.ProbeRetry.probe
      ((JointReads.attempts f.board (some got)).map (TranspositionTable.ProbeRetry.observe (Cutoffs.key f.board)))))
    fun sc id cut depth => let ⟨width,actual⟩ := feed f sc id cut depth; some ⟨width,JournalNestedHelpers.splits actual⟩

def pending (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig)
    (init : ∀ a, NativeTableWrites.Journal.Even (x.memory a)) (probes : JournalBodies.Probes m) (look : JournalScout.ChildReads m)
    (pool : EndgameSearch.CacheHistory.Pool) (checked : Bool) (s : CallJournal.State Op) (pv : JournalPv.Options mc)
    (scouts : JournalScoutOwner.Inputs mc) (feed : Feed es s mc fc) (f : Frame) (frames : List Frame) (body : Prop) :=
  JournalScoutOwner.pending mc fc init probes look pool checked s pv scouts f frames ((check mc fc init probes s f (pv f) feed).requires body)

structure Remaining (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig)
    (init : ∀ a, NativeTableWrites.Journal.Even (x.memory a)) (probes : JournalBodies.Probes m) (look : JournalScout.ChildReads m)
    (pool : EndgameSearch.CacheHistory.Pool) (checked : Bool) (s : CallJournal.State Op) (pv : JournalPv.Options mc)
    (scouts : JournalScoutOwner.Inputs mc) (feed : Feed es s mc fc) : Prop where
  seed : ∀ f ∈ s.frames, pending mc fc init probes look pool checked s pv scouts feed f (JournalPass.seedRows s f)
    (JournalBodies.SeedObligation mc fc s f (CausalTable.lookup x init f.chosen))
  weak : ∀ f ∈ s.weakChildren, pending mc fc init probes look pool checked s pv scouts feed f s.weakChildren
    (JournalBodies.WeakObligation mc fc s f (CausalTable.lookup x init f.chosen))
  exact : ∀ f ∈ s.exactFrames, pending mc fc init probes look pool checked s pv scouts feed f s.exactFrames
    (JournalBodies.ExactObligation mc fc s f (CausalTable.lookup x init f.chosen))

private theorem through (test : Check P) (use : A → B) : test.requires A → test.requires B := by
  cases test with | outside => exact use | rejected => exact id | proved _ => exact id

private theorem discharge (body : JournalPv.Bodies mc fc s f (CausalTable.lookup x init f.chosen) → Q)
    (h : pending mc fc init probes look pool checked s pv scouts feed f frames Q) :
    JournalScoutOwner.pending mc fc init probes look pool checked s pv scouts f frames Q :=
  through _ (through _ (through _ (through _ (through _ (through _ (through _ (Check.derive _ body))))))) h

theorem Remaining.owners (h : Remaining mc fc init probes look pool checked s pv scouts feed) :
    JournalScoutOwner.Remaining mc fc init probes look pool checked s pv scouts where
  seed f mem := discharge (fun r => r.seed) (h.seed f mem)
  weak f mem := discharge (fun r => r.weak (bne_iff_ne.mp (Bool.and_eq_true_iff.mp (List.mem_filter.mp mem).2).1)) (h.weak f mem)
  exact f mem := discharge (fun r => r.exact (beq_iff_eq.mp (List.mem_filter.mp mem).2)) (h.exact f mem)

namespace Regression

private def mc : Bitboard.Moves.X86Config := ⟨false,false⟩
private def fc : Cutoffs.FlipConfig := ⟨false,false,false⟩
private def b : Position := ⟨⟨0x810000000,0x1008000000⟩,rfl⟩
private def calls : NativeTableWrites.Journal.Calls := fun _ => ⟨⟨0,0⟩,⟨0,rfl⟩,.clear,false,id⟩
private def score (s : Int) : ScoreRefresh.Score := ⟨BitVec.ofInt 32 s⟩
private def op (board : Position) (available : Bool) : JournalPv.Operands mc board :=
  ⟨fun _ => .skip,fun sq => sq.val,fun i => (available && i == 1,false)⟩

structure Change where
  available : Bool := true
  site : Nat := 6
  childBeta : Int := 0
  childLevel : Option TranspositionTable.Selectivity := none
  scoreWord : Int := 5
  resultBank : Nat := 2
  reportedCut : Option Bool := none
  left : Bool := true

/-- The wide eldest child raises alpha to 2, then the physical owner loads 5 at the
    split. The full frame also needs the actual ticket/packet evidence for that load. -/
private def trial (level : TranspositionTable.Selectivity) (cut : Bool) (c : Change) : Bool := (show Option Unit from do
  let first ← (pvOrder mc fc b (op b true).mode (op b true).tie none).head?
  let child := (PvsBoardTree.play mc fc b first).val
  let parentArgs : NativeFrames.Args := ⟨b.val,60,score 0,score 100,true,level,true,cut,none⟩
  let childArgs : NativeFrames.Args := ⟨child.val,59,score (-100),score c.childBeta,true,c.childLevel.getD level,true,false,none⟩
  let es : Joint.Tape Unit := [.engine (.localCall 0 (.entry 0 9 .pvs .none none) (some parentArgs)),
    .enter 0 ⟨.probe (Cutoffs.key b),0,16,0⟩,.leave 0 (.probe (.miss 0)),
    .engine (.localCall 0 (.entry 1 9 .pvs .none none) (some childArgs)),.engine (.localCall 0 (.scored 2 (score (-2))) none),
    .engine (.localCall 0 (.leave 3 1 .pvs none (some (score (-2)))) none),
    .engine (.localCall 0 (.entry 4 c.site .split .none none) none),.splitWord 0 5 (.mark 2),
    .splitWord 0 6 (.result c.resultBank 5)] ++ (if c.left then [.engine (.localCall 0 (.leave 7 4 .split none none) none)] else []) ++
    [.engine (.localCall 0 (.scored 8 (score c.scoreWord)) none)]
  let s ← CallJournal.run mc fc calls false (es.map Joint.Event.engineEvent)
  let acts ← CallReturns.captureTimed (es.map Joint.Event.frameSample)
  let f ← s.frames.head?
  let owner ← s.choices.head?
  let (parent,opened) ← JournalPv.parent acts owner f
  let (_,close,_) ← JournalPrefix.parentAt es owner.worker opened f.chosen f.board
  let sc := JournalPv.scope s f acts parent
  let _ ← span es sc owner.worker close
  let actual ← run sc mc fc f.board owner.depth cut f.alpha f.beta (op f.board c.available) none
  guard (actual.result.handoff ∧ actual.next = 1 ∧ actual.result.remaining.idx = 1 ∧
    actual.result.snapshot.state.alpha = 2 ∧ actual.result.snapshot.state.best = 2)
  guard (JournalHelpers.ownerWindow es sc f.board false true (c.reportedCut.getD cut) level.val 60 100 2 ⟨0,5⟩ ⟨0,6⟩ 5 false)
  let status := evaluate mc fc es s f (fun _ _ => False) (op f.board c.available) (some (1,2,.miss 0))
    (fun _ _ _ _ => some ⟨7,fun _ _ _ => none⟩)
  match status with | .rejected => return () | _ => none) |>.isSome

#guard trial 5 false {}
#guard trial 2 true {}
#guard !(trial 5 false {available := false})
#guard !(trial 5 false {childBeta := 1})
#guard !(trial 5 false {childLevel := some 2})
#guard !(trial 5 false {site := 5})
#guard !(trial 5 false {scoreWord := 2})
#guard !(trial 2 true {resultBank := 3})
#guard !(trial 2 true {reportedCut := some false})
#guard !(trial 5 false {left := false})

end Regression

end NeuralReversi.PvsParallel.Interleaving.JournalPvSplit
