import NeuralReversi.PvsParallel.Interleaving.CallReturns
import NeuralReversi.PvsParallel.Interleaving.JournalPass

namespace NeuralReversi.PvsParallel.Interleaving.JournalPv

open PvsBoardTree (Position GeneratedSquare legalSquares)
open TranspositionTable (Depth)
open CausalTable (Frame Lookups)
open EndgameOrderMir (Heuristic pvOrder)
open CallReturns (Row Scope Owner)
open JournalBodies (Check)

private def sameFrame (g f : Frame) := decide (g.chosen = f.chosen ∧ g.returned = f.returned ∧ g.board = f.board ∧
  g.pv = f.pv ∧ g.alpha = f.alpha ∧ g.beta = f.beta ∧ g.score = f.score)

private theorem sameFrame_iff (g f : Frame) : sameFrame g f = true ↔ g = f := by
  cases g; cases f; simp [sameFrame]

private def sameLevel (s : CallJournal.State Op) (f : Frame) := s.frames.filter (fun g => s.level g == s.level f)

def registry (s : CallJournal.State Op) (f : Frame) :=
  if s.level f = Cutoffs.exactSelectivity then s.exactFrames else s.weakChildren

private def eligible (s : CallJournal.State Op) (f : Frame) (r : Row) :=
  r.entry.endgame && r.entry.rootLine.isNone && (r.entry.level == (s.level f).val) &&
    (sameLevel s f).any (sameFrame r.frame) && (registry s f).any (sameFrame r.frame)

def scope (s : CallJournal.State Op) (f : Frame) (acts : CallReturns.State) (parent : Nat) : Scope :=
  ⟨acts.allRows.filter (eligible s f),parent,f.chosen,acts.calls⟩

theorem projections (s : CallJournal.State Op) (f : Frame) (acts : CallReturns.State) (parent : Nat)
    (plan : CallReturns.Plan Position) :
    let x := (scope s f acts parent).nativePlan plan
    x.InPrefix (CausalTable.returned (· < ·) (sameLevel s f) f.chosen) ∧
    x.InPrefix (CausalTable.returned (· < ·) (registry s f) f.chosen) := by
  have both {g} (listed : g ∈ (scope s f acts parent).rows.map Row.frame) : g ∈ sameLevel s f ∧ g ∈ registry s f := by
    obtain ⟨row,h,rfl⟩ := List.mem_map.mp listed
    have includes := (List.mem_filter.mp h).2
    simp only [eligible,Bool.and_eq_true,List.any_eq_true,sameFrame_iff] at includes
    obtain ⟨⟨_,a,mem,rfl⟩,b,mem',eq⟩ := includes
    exact ⟨mem,eq ▸ mem'⟩
  constructor <;> intro b p a z v result <;> obtain ⟨g,member,t,ret,lt,rest⟩ := result
  · exact ⟨g,(both member).1,t,ret,of_decide_eq_true lt,rest⟩
  · exact ⟨g,(both member).2,t,ret,of_decide_eq_true lt,rest⟩

structure Bodies (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (s : CallJournal.State Op)
    (f : Frame) (reads : Lookups) : Prop where
  seed : JournalBodies.SeedObligation mc fc s f reads
  weak : s.level f ≠ Cutoffs.exactSelectivity → JournalBodies.WeakObligation mc fc s f reads
  exact : s.level f = Cutoffs.exactSelectivity → JournalBodies.ExactObligation mc fc s f reads

/-- Flags observed at the owner's header, and move-ordering operands. Each successful
    lookup still has to match the next physical recursive activation on the same worker. -/
structure Operands (mc : Bitboard.Moves.X86Config) (b : Position) where
  mode : Bitboard.Square → Heuristic
  tie : GeneratedSquare mc b.val → Fin 64
  schedule : Nat → Bool × Bool

private def dispatch (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : Position) (d : Depth) :=
  let generated := PvEntryMir.generated mc fc b false (fun _ => none)
  PvEntryMir.dispatch d.toNat (generated.len.getD 0 != 0) (Board.hasLegalMoves mc (PvsBoardTree.pass b).val)
    generated.wipeout.join.isSome

private def run (sc : Scope) (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : Position)
    (d : Depth) (cut : Bool) (a z : Int) (op : Operands mc b) (hint : Option Bitboard.Square) :=
  let moves := pvOrder mc fc b op.mode op.tie hint
  let q : PvsPostpone.Queue _ := ⟨0,moves,[]⟩
  CallReturns.ownerRun Subtype.val (sc.ownerReplies (CallReturns.pvRequest mc fc b d cut)) op.schedule
    d.toNat rootMinimumSplitDepth z (q.budget+1) 0 0 moves.length q (RootReplay.start a)

private theorem replay (sc : Scope) (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : Position)
    (d : Depth) (cut : Bool) (a z score : Int) (op : Operands mc b) (hint : Option Bitboard.Square)
    (full : ExactDepth.Full b d) (mobile : legalSquares mc b.val ≠ []) (path : dispatch mc fc b d = .continueSearch)
    (out : Owner (GeneratedSquare mc b.val)) (ran : run sc mc fc b d cut a z op hint = some out)
    (serial : out.result.handoff = false) (word : out.result.snapshot.state.best = score) :
    let x := sc.nativePlan (CallReturns.boardPlan mc fc b out.plan)
    ∀ (prior : CausalTable.Values) (reads : Lookups), x.InPrefix prior →
      Nested.Body mc fc prior reads b true a z score ∧ WeakFrame.Frame mc fc prior reads b true a z score := by
  let moves := pvOrder mc fc b op.mode op.tie hint
  let x := sc.nativePlan (CallReturns.boardPlan mc fc b out.plan)
  have perm := EndgameOrderMir.pvOrder_perm mc fc b op.mode op.tie hint
  obtain ⟨distinct,size⟩ := EndgameOrderMir.generated_layout mc b
  have fromMoves := sc.pvRun_fromMoves mc fc b d cut op.schedule rootMinimumSplitDepth a z moves (perm.nodup_iff.mpr distinct) ran
  have finish (splitCall : RootOwnerMir.Slice.Exit (GeneratedSquare mc b.val) → Option Int) :
      (do let owner ← RootOwnerMir.Slice.run Subtype.val (x.pvReplies mc fc b d cut) op.schedule d.toNat rootMinimumSplitDepth z
            0 0 moves.length moves (RootReplay.start a)
          if owner.handoff then splitCall owner else some owner.snapshot.state.best) = some score := by
    rw [RootOwnerMir.Slice.fromMoves_eq,fromMoves]
    simp [RootOwnerMir.Slice.layout,serial,word]
  dsimp only
  intro prior reads inside
  constructor
  · apply Nested.Body.pv (width := 7) d false false cut (fun _ => none) x inside op.mode op.tie hint op.schedule
      (fun _ _ _ => none) full (by decide)
    unfold PvSerialMir.scheduled
    change (match dispatch mc fc b d with | .continueSearch => _ | _ => _) = _
    rw [path]
    exact finish _
  · cases gen : legalSquares mc b.val with
    | nil => contradiction
    | cons first rest =>
      apply WeakFrame.Frame.moves first rest moves gen (gen ▸ perm)
      apply NativeNodes.pvTurn (cut := cut) (width := 7) Subtype.val (x.pvReplies mc fc b d cut)
        (fun _ _ _ _ _ got => inside _ _ _ _ _ (x.hit got)) op.schedule (fun _ _ _ => none) _ (finish _)
      exact Nat.lt_of_le_of_lt (perm.length_eq ▸ size) (by decide : 64 < 2^7)

def parent (calls : CallReturns.State) (owner : CallJournal.Choice) (f : Frame) : Option (Nat × Nat) := do
  let (activation,id) ← calls.calls.toList.zipIdx.find? fun (pair : CallReturns.Activation × Nat) => pair.1.worker == owner.worker && pair.1.first == owner.opening
  let r ← activation.row id
  guard (activation.callee == .pvs && sameFrame r.frame f && r.entry.rootLine.isNone && r.entry.endgame &&
    r.entry.level == owner.level.val && r.entry.request.depth == owner.depth && r.entry.request.cut == owner.cut &&
    r.entry.request.board == owner.board && r.entry.request.pv == owner.pv && r.entry.request.alpha == owner.alpha.raw.toInt &&
    r.entry.request.beta == owner.beta.raw.toInt && r.score == owner.word.raw.toInt && r.chosen == owner.stamp)
  return (id,activation.entered)

/-- Replay a non-root endgame PV owner from the supplied native ordering and scheduling operands.
    A parent reader's own key and closing word select the TT ordering hint. Child returns are
    counted even when an identical board is requested twice; a split or other side call is outside. -/
private def evaluate (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (es : Joint.Tape Op)
    (s : CallJournal.State Op) (f : Frame) (reads : Lookups) (op : Operands mc f.board)
    (probe : Option (Nat × Nat × TranspositionTable.ProbeRetry.Result)) :
    Check (Bodies mc fc s f reads) :=
  match s.choices.find? (fun owner => owner.stamp == f.chosen) with
  | none => .rejected
  | some owner =>
    if !owner.pv || !owner.endgame || owner.rootLine.isSome then .outside else
    if shape : f.pv = true ∧ owner.level = s.level f ∧ owner.depth = Board.emptyCount f.board.val then
      if path : dispatch mc fc f.board owner.depth = .continueSearch then
        match CallReturns.captureTimed (es.map Joint.Event.frameSample) with
        | none => .rejected
        | some acts => match parent acts owner f with
          | none => .rejected
          | some (id,opened) => match JournalPrefix.parentAt es owner.worker opened f.chosen f.board with
            | none => .rejected
            | some (opening,closed,raw) =>
              if probe != some (opening,closed,raw) then .rejected else
              let direct := acts.calls.toList.filter fun a => a.parent == some id && decide (a.entered < f.chosen)
              if direct.any (fun a => a.callee != .pvs) then .outside else
              if direct.any (fun a => decide (a.entered ≤ closed)) then .rejected else
              let sc := scope s f acts id
              let hint := TranspositionTable.ReaderMir.bestMove raw
              match got : run sc mc fc f.board owner.depth owner.cut f.alpha f.beta op hint with
              | none => .rejected
              | some out =>
                if serial : out.result.handoff = false then
                  if ok : out.next = sc.slots.length ∧ out.result.snapshot.state.best = f.score ∧ legalSquares mc f.board.val ≠ [] then
                    .proved (by
                      have physical := replay sc mc fc f.board owner.depth owner.cut f.alpha f.beta f.score op hint
                        shape.2.2 ok.2.2 path out got serial ok.2.1
                      obtain ⟨seed,reg⟩ := projections s f acts id (CallReturns.boardPlan mc fc f.board out.plan)
                      constructor
                      · unfold JournalBodies.SeedObligation
                        rw [shape.1]
                        exact .internal (physical _ reads seed).2
                      · intro weak
                        unfold JournalBodies.WeakObligation
                        rw [shape.1]
                        simpa only [registry,weak,↓reduceIte] using (physical _ reads reg).2
                      · intro exact_
                        unfold JournalBodies.ExactObligation
                        rw [shape.1]
                        simpa only [registry,exact_,↓reduceIte] using (physical _ reads reg).1)
                  else .rejected
                else .rejected
      else .outside
    else .rejected

variable {x : NativeTableWrites.Journal.Execution c endian rs} {es : Joint.Tape Op} {m : Joint.Atomics x es}

/-- The table hint is recovered from the physical probe replay, including the frame-local
    reader opening, close and raw return checked by `parentAt`. -/
def check (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig)
    (init : ∀ a, NativeTableWrites.Journal.Even (x.memory a)) (probes : JournalBodies.Probes m)
    (s : CallJournal.State Op) (f : Frame) (op : Operands mc f.board) :
    Check (Bodies mc fc s f (CausalTable.lookup x init f.chosen)) :=
  evaluate mc fc es s f _ op ((probes f.chosen f.board).map fun got =>
    (got.opening,got.finish,TranspositionTable.ProbeRetry.probe
      ((JointReads.attempts f.board (some got)).map (TranspositionTable.ProbeRetry.observe (Cutoffs.key f.board)))))

abbrev Options (mc : Bitboard.Moves.X86Config) := (f : Frame) → Operands mc f.board

def pending (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig)
    (init : ∀ a, NativeTableWrites.Journal.Even (x.memory a)) (probes : JournalBodies.Probes m)
    (pool : EndgameSearch.CacheHistory.Pool) (checked : Bool) (s : CallJournal.State Op) (op : Options mc)
    (f : Frame) (frames : List Frame) (body : Prop) :=
  JournalPass.pending mc fc init probes pool checked s f frames ((check mc fc init probes s f (op f)).requires body)

structure Remaining (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig)
    (init : ∀ a, NativeTableWrites.Journal.Even (x.memory a)) (probes : JournalBodies.Probes m)
    (pool : EndgameSearch.CacheHistory.Pool) (checked : Bool) (s : CallJournal.State Op) (op : Options mc) : Prop where
  seed : ∀ f ∈ s.frames, pending mc fc init probes pool checked s op f (JournalPass.seedRows s f)
    (JournalBodies.SeedObligation mc fc s f (CausalTable.lookup x init f.chosen))
  weak : ∀ f ∈ s.weakChildren, pending mc fc init probes pool checked s op f s.weakChildren
    (JournalBodies.WeakObligation mc fc s f (CausalTable.lookup x init f.chosen))
  exact : ∀ f ∈ s.exactFrames, pending mc fc init probes pool checked s op f s.exactFrames
    (JournalBodies.ExactObligation mc fc s f (CausalTable.lookup x init f.chosen))

private theorem through (test : Check P) (use : A → B) : test.requires A → test.requires B := by
  cases test with | outside => exact use | rejected => exact id | proved _ => exact id

private theorem discharge (body : Bodies mc fc s f (CausalTable.lookup x init f.chosen) → Q)
    (h : pending mc fc init probes pool checked s op f frames Q) :
    JournalPass.pending mc fc init probes pool checked s f frames Q :=
  through _ (through _ (through _ (Check.derive _ body))) h

theorem Remaining.passes (h : Remaining mc fc init probes pool checked s op) :
    JournalPass.Remaining mc fc init probes pool checked s where
  seed f mem := discharge (fun r => r.seed) (h.seed f mem)
  weak f mem := discharge (fun r => r.weak (bne_iff_ne.mp (Bool.and_eq_true_iff.mp (List.mem_filter.mp mem).2).1)) (h.weak f mem)
  exact f mem := discharge (fun r => r.exact (beq_iff_eq.mp (List.mem_filter.mp mem).2)) (h.exact f mem)

namespace Regression

private def mc : Bitboard.Moves.X86Config := ⟨false,false⟩
private def fc : Cutoffs.FlipConfig := ⟨false,false,false⟩
private def board : Position := ⟨⟨0x100000001,0x200000002⟩,rfl⟩
private def order (b : Position) : Operands mc b := ⟨fun _ => .skip,fun sq => sq.val,fun _ => (false,false)⟩
private def word (v : Int) : ScoreRefresh.Score := ⟨BitVec.ofInt 32 v⟩
private def entered (t site : Nat) (b : Position) (level : TranspositionTable.Selectivity) (pv cut : Bool)
    (a z : Int) : Joint.Event Unit := .engine (.localCall 0 (.entry t site .pvs .none none)
    (some ⟨b.val,Board.emptyCount b.val,word a,word z,true,level,pv,cut,none⟩))
private def returned (t : Nat) (v : Int) : List (Joint.Event Unit) :=
  [.engine (.localCall 0 (.scored (t+1) (word v)) none),.engine (.localCall 0 (.leave (t+2) t .pvs none (some (word v))) none)]
private def calls : NativeTableWrites.Journal.Calls := fun _ => ⟨⟨0,0⟩,⟨0,rfl⟩,.clear,false,id⟩

/-- The same second board first returns -4 to the scout, then -5 to the wide retry.
    Multiple simultaneous frames on other workers do not change the owner's ordinals. -/
private def trial (level : TranspositionTable.Selectivity) (score : Int) (retry : Bool) (site : Nat)
    (retryLevel : TranspositionTable.Selectivity) (retryAlpha : Int) (tail : Bool) : Option Bool := do
  let [m,n] := pvOrder mc fc board (order board).mode (order board).tie none | none
  let b := (PvsBoardTree.play mc fc board m).val
  let c := (PvsBoardTree.play mc fc board n).val
  let es : Joint.Tape Unit := [entered 0 44 board level true false 0 100,
    .enter 0 ⟨.probe (Cutoffs.key board),0,16,0⟩,.leave 0 (.probe (.miss 0)),
    .engine (.localCall 1 (.entry 0 9 .helper .none none) none),entered 1 8 b level true false (-100) 0] ++
    returned 1 (-2) ++ [entered 4 6 c level false true (-3) (-2)] ++ returned 4 (-4) ++
    (if retry then [entered 7 site c retryLevel true false retryAlpha (-2)] ++ returned 7 (-5) else []) ++
    (if tail then [entered 10 8 c level true false (-100) (-2)] ++ returned 10 (-5) else []) ++
    [.engine (.localCall 0 (.scored 20 (word score)) none)]
  let s ← CallJournal.run mc fc calls false (es.map Joint.Event.engineEvent)
  let f ← s.frames.head?
  match evaluate mc fc es s f (fun _ _ => False) (order f.board) (some (1,2,.miss 0)) with
  | .outside => none | .rejected => some false | .proved _ => some true

#guard trial 5 5 true 8 5 (-100) false = some true
#guard trial 2 5 true 8 2 (-100) false = some true
#guard trial 5 4 true 8 5 (-100) false = some false
#guard trial 5 5 false 8 5 (-100) false = some false
#guard trial 5 5 true 6 5 (-100) false = some false
#guard trial 5 5 true 8 2 (-100) false = some false
#guard trial 2 5 true 8 2 (-99) false = some false
#guard trial 2 5 true 8 2 (-100) true = some false

end Regression

end NeuralReversi.PvsParallel.Interleaving.JournalPv
