module
public import NeuralReversi.PvsParallel.Interleaving.JournalBodies
meta import NeuralReversi.PvsParallel.Interleaving.JournalBodies

@[expose] public section

namespace NeuralReversi.PvsParallel.Interleaving.JournalPrefix

open TranspositionTable hiding Score
open PvsBoardTree (Position GeneratedSquare)
open CausalTable (Frame Lookups)
open JournalBodies (Check Bodies)
open ScoreRefresh (Score)

def noCalls : NativeNodes.Returns := ⟨[],fun _ => 0,fun _ => false,fun _ _ => false,0,fun _ _ _ _ _ _ => none⟩

theorem noCalls_inside (prior : CausalTable.Values) : noCalls.InPrefix prior := by
  intro b pv a z v h
  obtain ⟨_,never,_⟩ := h
  contradiction

def priority (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : Position) :=
  EndgameOrderMir.framePriority mc fc b (fun _ => .skip) (fun m => m.val)

def frozen : GeneralOwnerMir.Schedule M := ⟨fun _ _ => false,fun _ _ => false⟩

def noSplit (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : Position) (cut : Bool)
    (d : Depth) (beta : Score) : Nested.Splits (BoardFrame.Children mc fc b noCalls.prior) false cut d.toNat 7 beta.raw.toInt :=
  fun _ _ _ => none

variable {x : NativeTableWrites.Journal.Execution c endian rs} {es : Joint.Tape Op} {m : Joint.Atomics x es}

def exactRun mc fc (b : Position) (d : Depth) (a z : Score) (cut checked : Bool)
    (attempts : List ProbeRetry.RawAttempt) :=
  GeneralOwnerMir.scheduled mc fc b d a z false checked cut (fun _ => none) attempts
    ([] : List (TableReads.ChildLookupFrom mc fc b (fun _ => False))) (priority mc fc b)
    noCalls.scout frozen (Nested.scoutSplit (noSplit mc fc b cut d z))

def frame (f : Frame) (a z : Score) (pv : Bool := false) := {f with pv := pv,alpha := a.raw.toInt,beta := z.raw.toInt}

structure Parent (reads : Lookups) (b : Position) where
  opening : Nat
  finish : Nat
  attempts : List ProbeRetry.RawAttempt
  seen : ∀ slot e, ProbeRetry.probe (attempts.map (ProbeRetry.observe (Cutoffs.key b))) = .hit slot e → reads b e

def absent (b : Position) (reads : Lookups) : Parent reads b :=
  ⟨0,0,[],fun _ _ hit => by contradiction⟩

def certify (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (reads : Lookups)
    (f : Frame) (d : Depth) (a z : Score) (cut checked : Bool) (level : Selectivity) (p : Parent reads f.board)
    (full : ExactDepth.Full f.board d)
    (weak : ∀ prior, WeakFrame.Frame mc fc prior reads f.board false a.raw.toInt z.raw.toInt f.score) :
    Check (Bodies mc fc level reads (frame f a z)) :=
  if exactLevel : level = Cutoffs.exactSelectivity then
    if ran : exactRun mc fc f.board d a z cut checked p.attempts = some f.score then
      .proved ⟨fun _ prior =>
        .scout (width := 7) d a z false checked cut (fun _ => none) _ ([] : List (TableReads.ChildLookupFrom mc fc f.board (fun _ => False)))
          noCalls (noCalls_inside prior) (fun _ => .skip) (fun sq => sq.val) frozen (noSplit mc fc f.board cut d z) full (by decide)
          ⟨p.seen,fun _ member => (List.not_mem_nil member).elim⟩ ran,weak⟩
    else .rejected
  else .proved ⟨fun same => (exactLevel same).elim,weak⟩

def nonNeutral (e : Joint.Event Op) : Bool := match e with | .engine (.other _) => false | _ => true

def segment (es : Joint.Tape Op) (worker start finish : Nat) :=
  (es.zipIdx.drop (start+1)).takeWhile (fun x => x.2 < finish) |>.filter fun row =>
    row.1.worker == worker && nonNeutral row.1

def entry (es : Joint.Tape Op) (w opening : Nat) :=
  es.zipIdx.findSome? fun (ev,i) => match ev.engineEvent with
    | .localCall atWorker (.entry tick _ .pvs ..) (some args) =>
      if atWorker = w ∧ tick = opening then some (i,args) else none
    | _ => none

/-- The first reader call following this frame's entry. Events on other workers
    can intervene. A nested search frame or an earlier reader blocks the match. -/
def parentAt (es : Joint.Tape Op) (worker start finish : Nat) (b : Position) : Option (Nat × Nat × ProbeRetry.Result) := do
  let (first,opening) :: tail := segment es worker start finish | none
  let .enter _ args := first | none
  let .probe q := args.method | none
  guard (q = Cutoffs.key b)
  let (last,closed) ← (tail.dropWhile fun row => match row.1 with | .issue .. => true | _ => false).head?
  let .leave _ (.probe raw) := last | none
  return (opening,closed,raw)

structure Local (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (level : Selectivity)
    (reads : Lookups) (f : Frame) : Prop extends Bodies mc fc level reads f where
  range : SeedTable.Range f.score

theorem Local.finite (result : Local mc fc level reads f) : FiniteTable.Finite f.score := by
  obtain ⟨lo,hi⟩ := result.range
  unfold PvsMoveLoop.scoreMax at lo hi
  change -32767 < f.score ∧ f.score < 32767
  omega

def onlyBody : Check (Local mc fc level reads f) → Check (Bodies mc fc level reads f)
  | .outside => .outside | .rejected => .rejected | .proved local_ => .proved local_.toBodies

/-- Replay the move-generation exits at their native location. PV reaches this
    code directly; a scout reaches it after the stability test falls through. -/
def localResult (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (es : Joint.Tape Op) (reads : Lookups)
    (f : Frame) (worker start : Nat) (d : Depth) (a z : Score) (pv cut checked : Bool) (level : Selectivity)
    (full : ExactDepth.Full f.board d) (win : Aspiration.Window a.raw.toInt z.raw.toInt)
    (narrow : pv = false → z.raw.toInt = a.raw.toInt+1) :
    Option (Check (Local mc fc level reads (frame f a z pv))) := by
  let leaf (v : {word : Int // SeedTable.Range word}) (why : BoardFrame.Frame mc fc (fun _ _ _ _ _ => False) (fun _ _ => False)
       f.board pv a.raw.toInt z.raw.toInt v.val) : Check (Local mc fc level reads (frame f a z pv)) := by
    if ok : f.score = v.val ∧ (segment es worker start f.chosen).isEmpty then
      have weak (prior) : WeakFrame.Frame mc fc prior reads f.board pv a.raw.toInt z.raw.toInt f.score :=
        ok.1 ▸ .localReturn win narrow why
      have range : SeedTable.Range f.score := ok.1 ▸ v.property
      cases pv with
      | false =>
        match certify mc fc reads f d a z cut checked level (absent f.board reads) full weak with
        | .outside => exact .outside
        | .rejected => exact .rejected
        | .proved result => exact .proved ⟨result,range⟩
      | true =>
        let splits : Nested.Splits (BoardFrame.Children mc fc f.board noCalls.prior) true cut d.toNat 7 z.raw.toInt :=
          fun _ _ _ => none
        if fullLevel : level = Cutoffs.exactSelectivity then
          if run : PvSerialMir.scheduled mc fc f.board d false checked (fun _ => none) a.raw.toInt z.raw.toInt
              (fun next cd lo hi => noCalls.reply next cd true false lo hi)
              (EndgameOrderMir.pvOrder mc fc f.board (fun _ => .skip) (fun m => m.val) none)
              (noCalls.pvReplies mc fc f.board d cut) (fun _ => (false,false)) (Nested.pvSplit splits) = some f.score then
            exact .proved ⟨⟨fun _ prior => .pv (width := 7) d false checked cut (fun _ => none) noCalls (noCalls_inside prior)
              (fun _ => .skip) (fun m => m.val) none (fun _ => (false,false)) splits full (by decide) run,weak⟩,range⟩
          else exact .rejected
        else exact .proved ⟨⟨fun yes => (fullLevel yes).elim,weak⟩,range⟩
    else exact .rejected
  let generated := PvEntryMir.generated mc fc f.board false (fun _ => none)
  if noMoves : generated.len = some 0 then
    if other : Board.hasLegalMoves mc (PvsBoardTree.pass f.board).val = false then
      have still : PvsBoardTree.legalSquares mc f.board.val = [] := by
        have count := PvEntryMir.generated_count mc fc f.board false (fun _ => none)
        exact List.length_eq_zero_iff.mp (Option.some.inj (count.symm.trans noMoves))
      match solved : (ScoreRefresh.BoardBridge.solve checked f.board.val d).map (fun word => word.raw.toInt) with
      | .error _ => exact some .rejected
      | .ok word =>
        have native := PvEntryMir.terminal_solve f.board checked
        rw [←full,solved] at native
        have score : word = (PvsBoardTree.terminalScore f.board).val := Except.ok.inj native
        exact some (leaf ⟨word,score ▸ (PvsBoardTree.terminalScore f.board).property⟩ (by
          change BoardFrame.Frame _ _ _ _ _ _ _ _ word
          rw [score]
          exact .terminal still other))
    else exact some .outside
  else
    match wiped : generated.wipeout with
    | some (some sq) =>
      exact some (leaf ⟨PvsMoveLoop.scoreMax,by unfold SeedTable.Range PvsMoveLoop.scoreMax; decide⟩ (by
        obtain ⟨move,member,all⟩ := PvEntryMir.cached_wipe mc fc f.board false (fun _ => none) wiped
        exact .wipe move member all))
    | _ => exact none

def scoreNode (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig)
    (es : Joint.Tape Op) (reads : Lookups) (probes : (b : Position) → Option (Parent reads b))
    (f : Frame) (worker start : Nat) (d : Depth) (a z : Score) (cut checked : Bool) (level : Selectivity)
    (full : ExactDepth.Full f.board d) (win : Aspiration.Window a.raw.toInt z.raw.toInt)
    (null : z.raw.toInt = a.raw.toInt+1) :
    Check (Bodies mc fc level reads (frame f a z)) :=
  match st : EndgameStability.tryCutoff f.board.val (Board.emptyCount f.board.val).toNat ⟨(ScoreRefresh.toDiscDiff a).raw.toInt⟩ with
  | some discs =>
    match converted : ScoreRefresh.fromDiscDiff checked ⟨BitVec.ofInt 32 discs.value⟩ with
    | .error _ => .rejected
    | .ok word =>
      if hit : f.score = word.raw.toInt ∧ (segment es worker start f.chosen).isEmpty then
        certify mc fc reads f d a z cut checked level (absent f.board reads) full (fun _ => by
          rw [hit.1]
          apply WeakFrame.Frame.localReturn win (fun _ => null)
          rw [null]
          exact .stable a discs word checked st converted)
      else .rejected
  | none =>
    match localResult mc fc es reads f worker start d a z false cut checked level full win (fun _ => null) with
    | some local_ => onlyBody local_
    | none => match parentAt es worker start f.chosen f.board with
      | none => .rejected
      | some (opening,closed,raw) =>
        match probes f.board with
        | none => .rejected
        | some got =>
          let data := ProbeRetry.probe (got.attempts.map (ProbeRetry.observe (Cutoffs.key f.board)))
          if located : data = raw ∧ got.opening = opening ∧ got.finish = closed then
            match raw with
            | .miss _ => .outside
            | .hit slot e =>
              match gate : canCut checked e ⟨z.raw⟩ d level true with
              | .error _ => .rejected
              | .ok false => .outside
              | .ok true =>
                if cutWord : (segment es worker closed f.chosen).isEmpty ∧ f.score = e.score.toInt then
                  certify mc fc reads f d a z cut checked level got full (fun _ => by
                    rw [cutWord.2]
                    exact .table ⟨z.raw⟩ (got.seen slot e located.1) gate)
                else .rejected
          else .rejected


/-- Bind the score to the entry registers. The scout checks stability before move generation.
    The PV frame reaches terminal and wipeout directly. Both precede a parent TT cutoff. -/
def replay (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig)
    (es : Joint.Tape Op) (reads : Lookups) (probes : (b : Position) → Option (Parent reads b))
    (checked : Bool) (s : CallJournal.State Op) (f : Frame) :
    Check (Bodies mc fc (s.level f) reads f) :=
  match s.choices.find? (fun owner => owner.stamp == f.chosen) with
  | none => .rejected
  | some owner =>
    let rootWipe := owner.rootLine == some 0 && owner.pv &&
      (PvEntryMir.generated mc fc f.board false (fun _ => none)).wipeout.join.isSome
    if !owner.endgame || (owner.rootLine.isSome && !rootWipe) || (!owner.pv && decide (owner.depth.toNat ≤ 12)) then .outside else
    match es[f.chosen]?, entry es owner.worker owner.opening with
    | some (.engine (.localCall w (.scored _ word) _)), some (start,args) =>
      if same : w = owner.worker ∧ owner.stamp = f.chosen ∧ start < f.chosen ∧ args.board = f.board.val ∧
          args.board = owner.board.val ∧ args.depth = owner.depth ∧ args.depth = Board.emptyCount f.board.val ∧
          args.pv = owner.pv ∧ args.endgame = true ∧ args.rootLine = owner.rootLine ∧ args.selectivity = s.level f ∧
          args.selectivity = owner.level ∧ args.cut = owner.cut ∧ args.alpha.raw = owner.alpha.raw ∧ args.beta.raw = owner.beta.raw ∧
          f.pv = args.pv ∧ f.alpha = args.alpha.raw.toInt ∧ f.beta = args.beta.raw.toInt ∧
          word.raw = owner.word.raw ∧ word.raw.toInt = f.score ∧
          (-32767 ≤ f.alpha ∧ f.alpha < f.beta ∧ f.beta ≤ 32767) ∧ (args.pv = false → f.beta = f.alpha+1) then by
        obtain ⟨_,_,_,_,_,_,full,_,_,_,_,_,_,_,_,pv,lo,hi,_,_,valid,narrow⟩ := same
        have win : Aspiration.Window args.alpha.raw.toInt args.beta.raw.toInt := by
          rw [←lo,←hi]; exact ⟨valid.1,valid.2.1,valid.2.2⟩
        have null : args.pv = false → args.beta.raw.toInt = args.alpha.raw.toInt+1 := by simpa [←lo,←hi] using narrow
        have normalized : frame f args.alpha args.beta args.pv = f := by cases f; simp_all [frame]
        suffices val : Check (Bodies mc fc (s.level f) reads (frame f args.alpha args.beta args.pv)) from normalized ▸ val
        cases hp : args.pv with
        | false => exact scoreNode mc fc es reads probes f w start args.depth args.alpha args.beta args.cut checked (s.level f) full win (null hp)
        | true =>
          exact onlyBody ((localResult mc fc es reads f w start args.depth args.alpha args.beta true args.cut checked (s.level f)
            full win (by simp)).getD .outside)
      else .rejected
    | _,_ => .rejected

def check (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig)
    (init : ∀ a, NativeTableWrites.Journal.Even (x.memory a)) (probes : JournalBodies.Probes m)
    (checked : Bool) (s : CallJournal.State Op) (f : Frame) :
    Check (Bodies mc fc (s.level f) (CausalTable.lookup x init f.chosen) f) :=
  replay mc fc es _ (fun b => (probes f.chosen b).map fun got =>
    ⟨got.opening,got.finish,JointReads.attempts b (some got),fun _ _ observed => JointReads.parent init b (some got) observed⟩)
    checked s f

private theorem requires (first next : Check P) (prove : P → Q)
    (ok : first.requires (next.requires Q)) : first.requires Q := by
  cases first with
  | outside => exact next.derive prove ok
  | rejected => exact ok
  | proved _ => trivial

structure Remaining (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig)
    (init : ∀ addr, NativeTableWrites.Journal.Even (x.memory addr)) (probes : JournalBodies.Probes m)
    (pool : EndgameSearch.CacheHistory.Pool) (checked : Bool) (s : CallJournal.State Op) : Prop where
  seed : ∀ f ∈ s.frames, (JournalBodies.check mc fc init probes pool s f).requires
    ((check mc fc init probes checked s f).requires (JournalBodies.SeedObligation mc fc s f (CausalTable.lookup x init f.chosen)))
  weak : ∀ f ∈ s.weakChildren, (JournalBodies.check mc fc init probes pool s f).requires
    ((check mc fc init probes checked s f).requires (JournalBodies.WeakObligation mc fc s f (CausalTable.lookup x init f.chosen)))
  exact : ∀ f ∈ s.exactFrames, (JournalBodies.check mc fc init probes pool s f).requires
    ((check mc fc init probes checked s f).requires (JournalBodies.ExactObligation mc fc s f (CausalTable.lookup x init f.chosen)))

theorem Remaining.bodies (h : Remaining mc fc init probes pool checked s) :
    JournalBodies.Remaining mc fc init probes pool s where
  seed f member := requires _ _ (fun body => .internal (body.weak _)) (h.seed f member)
  weak f member := requires _ _ (fun body => body.weak _) (h.weak f member)
  exact f member := requires _ _ (fun body => body.exact (beq_iff_eq.mp (List.mem_filter.mp member).2) _) (h.exact f member)

namespace Regression

private def board : Position := ⟨⟨0,0xffffffffffff⟩,rfl⟩
private def mc : Bitboard.Moves.X86Config := ⟨false,false⟩
private def fc : Cutoffs.FlipConfig := ⟨false,false,false⟩
private def args (level : Selectivity) : NativeFrames.Args :=
  ⟨board.val,16,⟨16384⟩,⟨16385⟩,true,level,false,true,none⟩
private def trace (level : Selectivity) (score : Int) (extra : Bool) : Joint.Tape Unit :=
  [.engine (.localCall 0 (.entry 0 31 .pvs .none none) (some (args level)))] ++
  (if extra then [.enter 0 ⟨.probe (Cutoffs.key board),0,16,0⟩] else []) ++
  [.engine (.localCall 0 (.scored 1 ⟨BitVec.ofInt 32 score⟩) none)]
private def calls : NativeTableWrites.Journal.Calls := fun _ => ⟨⟨0,0⟩,⟨0,rfl⟩,.clear,false,id⟩

private def result (level : Selectivity) (score : Int) (extra : Bool) : Option Bool := do
  let events := trace level score extra
  let s ← CallJournal.run mc fc calls false (events.map Joint.Event.engineEvent)
  let f ← s.frames.head?
  match replay mc fc events (fun _ _ => False) (fun _ => none) true s f with
  | .outside => none | .rejected => some false | .proved _ => some true

-- A physical probe supplies no operand on this branch. Sixteen empties and 48
-- stable opposing stones force -32 discs at alpha = 64, at either selectivity.
#guard result 5 (-8192) false = some true
#guard result 2 (-8192) false = some true
#guard result 5 (-8191) false = some false
#guard result 5 (-8192) true = some false

private def mid : Position := ⟨⟨0x0000000810000000,0x0000001008000000⟩,rfl⟩
private def metadata (level : Selectivity) :=
  Packing.newData ⟨0⟩ none .exact 60 level 0 true false

/-- The physical reader theorem supplies the decoded slot attempt in production.
    These fixtures exercise its consumer and vary the return register and address. -/
private def probeCase (level entryLevel : Selectivity) (rawScore : Int) (opening : Nat) (miss : Bool) : Option Bool := do
  let data := metadata entryLevel
  let op : NativeFrames.Args := ⟨mid.val,60,⟨0⟩,⟨1⟩,true,level,false,true,none⟩
  let output : ProbeRetry.Result := if miss then .miss 0 else .hit 0 data
  let tape : Joint.Tape Unit := [
    .engine (.localCall 0 (.entry 0 44 .pvs .none none) (some op)), .enter 0 ⟨.probe (Cutoffs.key mid),0,16,0⟩,
    .issue 0 (.load ⟨0,0⟩ .sequence .acquire) 2 2,
    .issue 0 (.load ⟨0,0⟩ (.payload .metadata) .relaxed) 1 (Packing.pack false data),
    .issue 0 (.load ⟨0,0⟩ (.payload .player) .relaxed) 1 mid.val.player,
    .issue 0 (.load ⟨0,0⟩ (.payload .opponent) .relaxed) 1 mid.val.opponent,
    .issue 0 (.fence .acquire) 0 0, .issue 0 (.load ⟨0,0⟩ .sequence .relaxed) 2 2,
    .leave 0 (.probe output), .engine (.localCall 0 (.scored 1 ⟨BitVec.ofInt 32 rawScore⟩) none)]
  let st ← CallJournal.run mc fc calls false (tape.map Joint.Event.engineEvent)
  let f ← st.frames.head?
  let at_ := fun i : Nat => (ReaderWindow.positions (Joint.worker tape) (Joint.opcode tape) 1 8)[i]?.getD 9
  let regs : FrameTableReads.Operations.Supply := ⟨fun _ i => Joint.word tape (at_ i),fun _ i => (Joint.word tape (at_ i)).setWidth 8⟩
  let decoded := ReaderWindow.check (Joint.worker tape) (Joint.marker tape) (Joint.opcode tape) (fun _ => Joint.word tape)
    (fun _ i => (Joint.word tape i).setWidth 8) false 8 9
  let supply : (b : Position) → Option (Parent (fun _ _ => True) b) := fun b => decoded.map fun _ =>
    let rounds := (FrameTableReads.Operations.probe regs 0 16 0 false (Cutoffs.key b) 0).rounds
    ⟨opening,8,rounds.map (fun round_ => round_.read regs false),fun _ _ _ => trivial⟩
  match replay mc fc tape _ supply true st f with
  | .outside => none | .rejected => some false | .proved _ => some true

#guard probeCase 5 5 0 1 false = some true
#guard probeCase 2 2 0 1 false = some true
#guard probeCase 5 2 0 1 false = none
#guard probeCase 5 5 1 1 false = some false
#guard probeCase 5 5 0 2 false = some false
#guard probeCase 5 5 0 1 true = some false

private def wipe : Position := ⟨⟨0xc01,2⟩,rfl⟩
private def forced : Position := ⟨⟨2,0x0600000000001⟩,rfl⟩

private def localCase (b : Position) (pv : Bool) (level : Selectivity) (alpha beta score : Int) (spurious : Bool)
    (rootLine : Option Nat := none) : Option Bool := do
  let op : NativeFrames.Args := ⟨b.val,Board.emptyCount b.val,⟨BitVec.ofInt 32 alpha⟩,⟨BitVec.ofInt 32 beta⟩,true,level,pv,false,rootLine⟩
  let start := if rootLine.isSome then Joint.Event.rootEnter 0 0 45 .none op [] else .engine (.localCall 0 (.entry 0 45 .pvs .none none) (some op))
  let tape : Joint.Tape Unit := [start] ++
    (if spurious then [.enter 0 ⟨.probe (Cutoffs.key b),0,16,0⟩] else []) ++
    [.engine (.localCall 0 (.scored 1 ⟨BitVec.ofInt 32 score⟩) none)]
  let st ← CallJournal.run mc fc calls false (tape.map Joint.Event.engineEvent)
  let f ← st.frames.head?
  match replay mc fc tape (fun _ _ => False) (fun _ => none) true st f with
  | .outside => none | .rejected => some false | .proved _ => some true

-- A full PV board takes the zero-depth evaluation. A board ending early
-- receives the remaining empty squares under the engine's scoring convention.
private def fullBoard : Position := ⟨⟨0xffffffffffffffff,0⟩,rfl⟩
private def oneEmpty : Position := ⟨⟨0xfffffffffffffffe,0⟩,rfl⟩
#guard localCase fullBoard true 5 (-100) 100 16384 false = some true
#guard localCase fullBoard true 5 (-100) 100 16128 false = some false
#guard localCase oneEmpty true 2 (-100) 100 16384 false = some true
#guard localCase oneEmpty true 5 (-100) 100 16128 false = some false

-- The same terminal board takes stability at non-PV alpha 64 and board.solve at PV.
#guard localCase board false 5 0 1 (-16384) false = some true
#guard localCase board true 2 16384 16390 (-16384) false = some true
#guard localCase board true 5 16384 16390 (-8192) false = some false
#guard localCase board false 5 16384 16385 (-16384) false = some false
#guard localCase board true 5 0 50 (-16384) true = some false
#guard localCase wipe false 2 0 1 16384 false = some true
#guard localCase wipe true 5 (-1024) 512 16384 false = some true
#guard localCase wipe false 5 0 1 16383 false = some false
#guard localCase wipe true 5 0 256 16384 true = some false
#guard localCase forced false 5 0 1 (-16384) false = none
#guard localCase forced true 5 (-512) 512 (-16384) false = none

-- A legal wipeout suffices for the full-board value even at the root.
-- A later Multi-PV line searches only the remaining candidate moves.
#guard localCase wipe true 5 (-1024) 512 16384 false (some 0) = some true
#guard localCase wipe true 2 (-1024) 512 16384 false (some 0) = some true
#guard localCase wipe true 5 (-1024) 512 16383 false (some 0) = some false
#guard localCase wipe true 5 (-1024) 512 16384 true (some 0) = some false
#guard localCase wipe true 5 (-1024) 512 16384 false (some 1) = none
#guard localCase mid true 5 (-1024) 512 16384 false (some 0) = none

end Regression

end NeuralReversi.PvsParallel.Interleaving.JournalPrefix
