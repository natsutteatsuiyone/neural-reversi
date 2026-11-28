module
public import NeuralReversi.PvsParallel.Interleaving.JournalPv
public import NeuralReversi.PvsParallel.Interleaving.SolveEntry
public import NeuralReversi.PvsParallel.Interleaving.JournalNestedHelpers
meta import NeuralReversi.PvsParallel.Interleaving.JournalNestedHelpers
meta import NeuralReversi.PvsParallel.Interleaving.JournalPv
meta import NeuralReversi.PvsParallel.Interleaving.SolveEntry

@[expose] public section

namespace NeuralReversi.PvsParallel.Interleaving.JournalRootStore

open PvsBoardTree (Position GeneratedSquare legalSquares)
open CausalTable (Frame Lookups)
open RootMoveStorage (Row)
open AspirationMir (SerialStack.State)
open JournalBodies (Check)
open TranspositionTable hiding Score

structure Options where
  memory : MoveList.Initialization.Memory Move.Move
  paired : Bool
  order : SerialRecursion.RootInput
  schedule : Nat → Bool × Bool

/-- Identify the chosen first-line root by its physical worker and activation, even while
    its TT store is in progress. Abandoned retries still occupy their solve-local ordinal. -/
def origin (es : Joint.Tape Op) (acts : CallReturns.State) (owner : CallJournal.Choice) (f : Frame) :
    Option (Nat × AspirationMir.SerialStack.State) := do
  let (a,id) ← acts.calls.toList.zipIdx.find? fun (a,_) => a.worker == owner.worker && a.first == owner.opening
  let row ← a.row id
  let .rootEnter w opening site _ args rows ← es[a.entered]? | none
  let entry := CallReturns.rootEntry f.board 0 owner.level.val f.alpha f.beta
  guard (w = owner.worker ∧ opening = owner.opening ∧ site = 0 ∧ a.callee = .pvs ∧ a.site = 0 ∧
    row.entry = entry ∧ CallReturns.Entry.read args = some entry ∧ row.returned = f.returned ∧
    row.chosen = f.chosen ∧ row.chosen = owner.stamp ∧ row.score = f.score ∧ row.score = owner.word.raw.toInt ∧
    owner.board = f.board ∧ owner.depth = Board.emptyCount f.board.val ∧ !owner.cut ∧ owner.pv ∧ owner.endgame ∧
    owner.rootLine = some 0 ∧ owner.alpha.raw.toInt = f.alpha ∧ owner.beta.raw.toInt = f.beta)
  let window ← a.parent.bind fun i => acts.calls[i]?
  let solveId ← window.parent
  let solve ← acts.calls[solveId]?
  let .solveEnter _ _ pc _ seed ← es[solve.entered]? | none
  guard (seed.board = f.board ∧ window.worker = w ∧ solve.worker = w)
  let found ← if pc = 0 then SolveEntry.atEntry es acts solve.entered .endRoot 0 seed
    else SolveEntry.timed es acts solve.entered seed
  let ids := acts.rootCallIds solveId
  let ordinal := ids.idxOf id
  guard (found = solveId ∧ ids[ordinal]? = some id)
  if ordinal = 0 then
    guard ((f.alpha,f.beta) = RootAspiration.initialWindow 0 seed.base.toInt none ∧ rows = seed.rows)
  else
    guard ((rows.map Row.square).Perm (seed.rows.map Row.square))
  return (id,⟨ordinal,a.entered,⟨args.alpha.raw,args.beta.raw,0⟩,rows⟩)

def generated (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : Position)
    (st : AspirationMir.SerialStack.State) (op : Options) :=
  MoveList.Generation.fill false op.paired fc b.val (Root.remainingMask mc b.val (st.rows.map Row.square) 0) op.memory

def moves (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : Position)
    (st : AspirationMir.SerialStack.State) (op : Options) :=
  AspirationMir.SmallRoot.candidates mc fc b (st.rows.map Row.square) 0 op.order.key op.order.sort

def compiled (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : Position) (x : NativeNodes.Returns)
    (st : AspirationMir.SerialStack.State) (op : Options) (inside : x.InPrefix prior) :
    RootProgram.Operands mc b (BoardFrame.Children mc fc b prior) st.regs.beta.toInt :=
  ⟨op.memory,op.paired,op.order.key,op.order.sort,op.order.pvLine,x.pvReplies mc fc b (Board.emptyCount b.val) false,
   fun _ _ _ _ _ used => inside _ _ _ _ _ (x.hit used),op.schedule,rootMinimumSplitDepth,7,by decide,fun _ => none⟩

private theorem ownerMoves (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : Position)
    (sc : CallReturns.Scope) (st : AspirationMir.SerialStack.State) (op : Options)
    (out : CallReturns.Owner (GeneratedSquare mc b.val)) (distinct : (moves mc fc b st op).Nodup)
    (ran : sc.rootRun mc fc b op.schedule rootMinimumSplitDepth st.regs.alpha.toInt st.regs.beta.toInt
      (moves mc fc b st op) = some out) :
    RootOwnerMir.fromMoves Subtype.val ((sc.nativePlan (CallReturns.boardPlan mc fc b out.plan)).pvReplies mc fc b (Board.emptyCount b.val) false)
      op.schedule (Board.emptyCount b.val).toNat rootMinimumSplitDepth st.regs.alpha.toInt st.regs.beta.toInt (moves mc fc b st op) = some out.result := by
  have aligned := sc.pvRun_fromMoves mc fc b (Board.emptyCount b.val) false op.schedule rootMinimumSplitDepth
    st.regs.alpha.toInt st.regs.beta.toInt (moves mc fc b st op) distinct (out := out)
  have req (m : GeneratedSquare mc b.val) (p : Bool) (a z : Int) :
      CallReturns.pvRequest mc fc b (Board.emptyCount b.val) false m p a z =
      SerialRootCalls.request mc fc b (Board.emptyCount b.val) ⟨m,p,a,z⟩ := by cases p <;> rfl
  rw [show CallReturns.pvRequest mc fc b (Board.emptyCount b.val) false = _ from funext fun m => funext fun p => funext fun a => funext fun z => req m p a z] at aligned
  exact aligned ran

private theorem replay (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : Position)
    (sc : CallReturns.Scope) (st : AspirationMir.SerialStack.State) (op : Options)
    (out : CallReturns.Owner (GeneratedSquare mc b.val)) (prior : CausalTable.Values) (reads : Lookups)
    (inside : (sc.nativePlan (CallReturns.boardPlan mc fc b out.plan)).InPrefix prior)
    (cover : Root.Covers mc b (st.rows.map Row.square)) (pos : 0 < st.rows.length)
    (win : Aspiration.Window st.regs.alpha.toInt st.regs.beta.toInt)
    (ordinary : SerialRootCalls.entry (generated mc fc b st op).len (generated mc fc b st op).wipeout = some none)
    (ran : sc.rootRun mc fc b op.schedule rootMinimumSplitDepth st.regs.alpha.toInt st.regs.beta.toInt
      (moves mc fc b st op) = some out) (serial : out.result.handoff = false) (level : Selectivity) :
    let v := out.result.snapshot.state.best
    Nested.Body mc fc prior reads b true st.regs.alpha.toInt st.regs.beta.toInt v ∧
    SeedHistory.Body mc fc level prior reads b true st.regs.alpha.toInt st.regs.beta.toInt v ∧
    WeakFrame.Frame mc fc prior reads b true st.regs.alpha.toInt st.regs.beta.toInt v := by
  let x := sc.nativePlan (CallReturns.boardPlan mc fc b out.plan)
  let code := compiled mc fc b x st op inside
  have perm : (moves mc fc b st op).Perm (legalSquares mc b.val) := by
    rw [← Root.choices_zero cover]
    exact (RootFrameMir.ordered_candidates mc fc b _ 0 op.order.key op.order.sort).1
  obtain ⟨distinct,size⟩ := EndgameOrderMir.generated_layout mc b
  have fromMoves := ownerMoves mc fc b sc st op out (perm.nodup_iff.mpr distinct) ran
  let ans : AspirationMir.SmallRoot.Answer := ⟨out.result.snapshot.state.best,
    RootTrial.ownerRows st.regs.alpha.toInt st.regs.beta.toInt st.rows 0 out.result code.child op.order.pvLine⟩
  have result : RootProgram.run fc 0 st code = some ans := by
    unfold RootProgram.run
    dsimp only [code,compiled]
    unfold generated at ordinary
    unfold moves at fromMoves
    rw [ordinary]
    dsimp only [Bind.bind,Option.bind,x]
    rw [fromMoves]
    dsimp only
    rw [serial]
    rfl
  refine ⟨Nested.Body.root st code cover pos result,SeedHistory.Body.root st (fun _ => some code) cover pos win result,?_⟩
  cases gen : legalSquares mc b.val with
  | nil => have h := cover.length_eq; simp only [List.length_map,gen,List.length_nil] at h; omega
  | cons first rest =>
    apply WeakFrame.Frame.moves first rest (moves mc fc b st op) gen (gen ▸ perm)
    apply NativeNodes.pvTurn (depth := (Board.emptyCount b.val).toNat) (limit := rootMinimumSplitDepth) (cut := false) (width := 7) Subtype.val code.child
      (fun _ _ _ _ _ got => code.returned _ _ _ _ _ got) op.schedule (fun _ _ _ => none)
    · exact Nat.lt_of_le_of_lt (perm.length_eq ▸ size) (by decide : 64 < 2^7)
    · change RootOwnerMir.fromMoves Subtype.val code.child _ _ _ _ _ _ = some out.result at fromMoves
      rw [RootOwnerMir.Slice.fromMoves_eq,fromMoves]
      simp [RootOwnerMir.Slice.layout,serial]

structure Packet (es : Joint.Tape Op) (sc : CallReturns.Scope) (level : Nat) (mc : Bitboard.Moves.X86Config)
    (fc : Cutoffs.FlipConfig) (b : Position) (beta : Int) where
  width : Nat
  wide : 7 ≤ width
  feed : JournalNestedHelpers.RootFeed es sc level mc fc b width beta

abbrev Splits (es : Joint.Tape Op) (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : Position) :=
  (sc : CallReturns.Scope) → (level : Nat) → (beta : Int) → Option (Packet es sc level mc fc b beta)

private theorem resize {cursor : BitVec width} (j : Join t children n cursor) (same : n = m) (h : j.answer = some score) :
    ∃ k : Join t children m cursor, k.answer = some score := by subst m; exact ⟨j,h⟩

def joined (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : Position)
    (sc : CallReturns.Scope) (st : AspirationMir.SerialStack.State) (op : Options)
    (out : CallReturns.Owner (GeneratedSquare mc b.val)) (prior : CausalTable.Values) (reads : Lookups)
    (inside : (sc.nativePlan (CallReturns.boardPlan mc fc b out.plan)).InPrefix prior)
    (cover : Root.Covers mc b (st.rows.map Row.square)) (pos : 0 < st.rows.length)
    (win : Aspiration.Window st.regs.alpha.toInt st.regs.beta.toInt) (level : Selectivity) (score : Int)
    (ran : sc.rootRun mc fc b op.schedule rootMinimumSplitDepth st.regs.alpha.toInt st.regs.beta.toInt
      (moves mc fc b st op) = some out) (handoff : out.result.handoff = true)
    (packet : Packet es sc level.val mc fc b st.regs.beta.toInt) : Option (PLift (
      Nested.Body mc fc prior reads b true st.regs.alpha.toInt st.regs.beta.toInt score ∧
      SeedHistory.Body mc fc level prior reads b true st.regs.alpha.toInt st.regs.beta.toInt score ∧
      WeakFrame.Frame mc fc prior reads b true st.regs.alpha.toInt st.regs.beta.toInt score)) := do
  let x := sc.nativePlan (CallReturns.boardPlan mc fc b out.plan)
  let code := { compiled mc fc b x st op inside with
    width := packet.width
    supported := packet.wide
    split := JournalNestedHelpers.rootSplit packet.feed inside}
  let r := out.result
  let some physical := code.split r | none
  match normal : RootProgram.run fc 0 st code with
  | none => none
  | some ans =>
    if ok : ans.value = score ∧ physical.answer = some score then
      return ⟨by
        refine ⟨ok.1 ▸ Nested.Body.root st code cover pos normal,
          ok.1 ▸ SeedHistory.Body.root st (fun _ => some code) cover pos win normal,?_⟩
        have perm : (moves mc fc b st op).Perm (legalSquares mc b.val) := by
          rw [← Root.choices_zero cover]
          exact (RootFrameMir.ordered_candidates mc fc b _ 0 op.order.key op.order.sort).1
        obtain ⟨distinct,size⟩ := EndgameOrderMir.generated_layout mc b
        have fromMoves := ownerMoves mc fc b sc st op out (perm.nodup_iff.mpr distinct) ran
        obtain ⟨trace,_,_⟩ := RootOwnerMir.refines Subtype.val code.child (BoardFrame.Children mc fc b prior)
          code.returned _ _ _ _ _ _ _ _ _ _ (by simp) rfl fromMoves
        rw [show r.outcome = .handoff r.snapshot.state by simp [r,RootOwnerMir.Result.outcome,handoff]] at trace
        let labels := r.visited ++ r.remaining.pending
        obtain ⟨window,cap,_⟩ := Prefix.continued_window true _ st.regs.alpha.toInt st.regs.beta.toInt r.visited _
          _ r.snapshot.state win (Int.le_max_left ..) trace.reproduced.2.2.1
        obtain ⟨join,answer,_⟩ := physical.exit.observes st.regs.alpha.toInt window cap
        have fits : labels.length < 2^packet.width := by
          have order : labels.Perm (moves mc fc b st op) := by simpa [labels,PvsPostpone.Queue.pending] using trace.reproduced.1
          rw [order.length_eq,perm.length_eq]
          exact Nat.lt_of_le_of_lt size (Nat.lt_of_lt_of_le (by decide : 64 < 2^7) (Nat.pow_le_pow_right (by decide) packet.wide))
        cases gen : legalSquares mc b.val with
        | nil => have h := cover.length_eq; simp only [List.length_map,gen,List.length_nil] at h; omega
        | cons first rest =>
          apply WeakFrame.Frame.moves first rest (moves mc fc b st op) gen (gen ▸ perm)
          let j := join.mapCalls (q := fun i => BoardFrame.Children mc fc b prior (labels[i]?.getD first)) (by
            intro i pv a z v got
            obtain ⟨m,e,done_⟩ := got
            change labels[i]? = some m at e
            simpa [e] using done_)
          have length : (RootTrial.labels r).length = labels.length := List.length_map ..
          obtain ⟨j,done_⟩ := resize j length (answer.trans ok.2)
          exact .joined _ first true false (Board.emptyCount b.val).toNat packet.width trace fits j done_⟩
    else none

def evaluate (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (es : Joint.Tape Op)
    (s : CallJournal.State Op) (f : Frame) (reads : Lookups) (op : Options)
    (probe : Option (Nat × Nat × ProbeRetry.Result)) (splits : Splits es mc fc f.board) : Check (JournalPv.Bodies mc fc s f reads) :=
  match s.choices.find? (fun owner => owner.stamp == f.chosen) with
  | none => .rejected
  | some owner =>
    if !owner.endgame || owner.rootLine != some 0 then .outside else
    match CallReturns.captureTimed (es.map Joint.Event.frameSample) with
    | none => .rejected
    | some acts => match origin es acts owner f with
      | none => .rejected
      | some (id,st) =>
        if shape : f.pv = true ∧ s.level f = owner.level ∧ f.alpha = st.regs.alpha.toInt ∧ f.beta = st.regs.beta.toInt ∧
            -32767 ≤ st.regs.alpha.toInt ∧ st.regs.alpha.toInt < st.regs.beta.toInt ∧ st.regs.beta.toInt ≤ 32767 then
          let g := generated mc fc f.board st op
          match ordinary : SerialRootCalls.entry g.len g.wipeout with
          | none => .rejected
          | some (some _) => .outside
          | some none =>
            match JournalPrefix.parentAt es owner.worker st.opening f.chosen f.board with
            | none => .rejected
            | some (opening,close,raw) =>
              if probe != some (opening,close,raw) then .rejected else
              let direct := acts.calls.toList.filter fun a => a.parent == some id && decide (a.entered < f.chosen)
              let handoff := direct.any (fun a => a.callee == .split)
              let priorCalls := if handoff then direct.dropLast else direct
              if priorCalls.any (fun a => a.callee != .pvs) then .outside else
              if direct.any (fun a => decide (a.entered ≤ close)) then .rejected else
              if handoff && !(direct.getLast?.any fun a => a.callee == .split && a.site == 6 &&
                  priorCalls.all (fun g => g.returned.any (fun ret => decide (ret < a.entered)))) then .rejected else
              let sc := JournalPv.scope s f acts id
              match got : sc.rootRun mc fc f.board op.schedule rootMinimumSplitDepth st.regs.alpha.toInt st.regs.beta.toInt
                (moves mc fc f.board st op) with
              | none => .rejected
              | some out =>
                if ok : out.next = sc.slots.length ∧ Root.Covers mc f.board (st.rows.map Row.square) ∧ 0 < st.rows.length then
                  if handoff != out.result.handoff then .rejected else
                  let plan := CallReturns.boardPlan mc fc f.board out.plan
                  have window : Aspiration.Window st.regs.alpha.toInt st.regs.beta.toInt :=
                    ⟨shape.2.2.2.2.1,shape.2.2.2.2.2.1,shape.2.2.2.2.2.2⟩
                  have make (first : SeedHistory.Body mc fc (s.level f)
                      (CausalTable.returned (· < ·) (JournalPass.seedRows s f) f.chosen) reads f.board true
                      st.regs.alpha.toInt st.regs.beta.toInt f.score)
                      (other : let past := CausalTable.returned (· < ·) (JournalPv.registry s f) f.chosen
                               Nested.Body mc fc past reads f.board true st.regs.alpha.toInt st.regs.beta.toInt f.score ∧
                               WeakFrame.Frame mc fc past reads f.board true st.regs.alpha.toInt st.regs.beta.toInt f.score) :
                      JournalPv.Bodies mc fc s f reads := by
                    constructor
                    · unfold JournalBodies.SeedObligation
                      rw [shape.1,shape.2.2.1,shape.2.2.2.1]
                      exact first
                    · intro weak
                      unfold JournalBodies.WeakObligation
                      rw [shape.1,shape.2.2.1,shape.2.2.2.1]
                      simpa only [JournalPv.registry,weak,↓reduceIte] using other.2
                    · intro exact_
                      unfold JournalBodies.ExactObligation
                      rw [shape.1,shape.2.2.1,shape.2.2.2.1]
                      simpa only [JournalPv.registry,exact_,↓reduceIte] using other.1
                  have origins := JournalPv.projections s f acts id plan
                  match kind : out.result.handoff with
                  | false =>
                    if word : out.result.snapshot.state.best = f.score then
                      .proved (by
                        have physical := fun prior inside => replay mc fc f.board sc st op out prior reads inside
                          ok.2.1 ok.2.2 window ordinary got kind (s.level f)
                        exact make (word ▸ (physical _ origins.1).2.1)
                          ⟨word ▸ (physical _ origins.2).1,word ▸ (physical _ origins.2).2.2⟩)
                    else .rejected
                  | true =>
                    match splits sc (s.level f).val st.regs.beta.toInt with
                    | none => .rejected
                    | some packet =>
                      let run := fun prior inside => joined mc fc f.board sc st op out prior reads inside ok.2.1 ok.2.2
                        window (s.level f) f.score got kind packet
                      match run _ origins.1,run _ origins.2 with
                      | some first,some other => .proved (make first.down.2.1 ⟨other.down.1,other.down.2.2⟩)
                      | _,_ => .rejected
                else .rejected
        else .rejected

variable {x : NativeTableWrites.Journal.Execution c endian rs} {es : Joint.Tape Op} {m : Joint.Atomics x es}

/-- Certify a chosen TT publisher at index zero using its entry registers, enclosing solve,
    same-worker probe and immediate child returns. The parent may still be in its store. -/
def check (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig)
    (init : ∀ a, NativeTableWrites.Journal.Even (x.memory a)) (probes : JournalBodies.Probes m)
    (s : CallJournal.State Op) (f : Frame) (op : Options) (splits : Splits es mc fc f.board := fun _ _ _ => none) :
    Check (JournalPv.Bodies mc fc s f (CausalTable.lookup x init f.chosen)) :=
  evaluate mc fc es s f _ op ((probes f.chosen f.board).map fun p =>
    (p.opening,p.finish,ProbeRetry.probe ((JointReads.attempts f.board (some p)).map (ProbeRetry.observe (Cutoffs.key f.board))))) splits


namespace Regression

private def mc : Bitboard.Moves.X86Config := ⟨false,false⟩
private def fc : Cutoffs.FlipConfig := ⟨false,false,false⟩
private def b : Position := ⟨⟨0x100000001,0x200000002⟩,rfl⟩
private def opt : Options := ⟨fun _ => none,false,⟨fun _ => 0,fun xs => ⟨xs,.refl _⟩,fun _ => []⟩,fun _ => (false,false)⟩
private def word (i : Int) : ScoreRefresh.Score := ⟨BitVec.ofInt 32 i⟩
private def args (b : Position) (pv cut : Bool) (a z : Int) (level : Selectivity) (idx : Option Nat) : NativeFrames.Args :=
  ⟨b.val,Board.emptyCount b.val,word a,word z,true,level,pv,cut,idx⟩
private def enter (t site : Nat) (n : EndgameSearch.CacheOwner.Node) (arg : Option NativeFrames.Args := none) : Joint.Event Unit :=
  .engine (.localCall 0 (.entry t site n .none none) arg)
private def scored (t : Nat) (v : Int) : Joint.Event Unit := .engine (.localCall 0 (.scored t (word v)) none)
private def closed (t start : Nat) (v : Int) : Joint.Event Unit := .engine (.localCall 0 (.leave t start .pvs none (some (word v))) none)
private def calls : NativeTableWrites.Journal.Calls := fun _ => ⟨⟨0,0⟩,⟨0,rfl⟩,.clear,false,id⟩

structure Mutation where
  level : Selectivity := 5
  otherLevel : Option Selectivity := none
  base : Int := 0
  root : Int := 5
  retrySite : Nat := 9
  retry : Bool := true
  endRoot : Bool := false
  rows : Bool := true
  corruptRows : Bool := false
  idx : Nat := 0
  extra : Bool := false
  rawProbe : Fin 2 := 0
  retryWindow : Int := -256

private def trial (p : Mutation) : Option Bool := do
  let rows := AspirationMir.NativeRootBirth.initial mc b
  let rows' := if p.corruptRows then rows.take 1 else rows
  let root := args b true false (-256) 256 p.level (some p.idx)
  let [m,n] := moves mc fc b ⟨0,0,⟨0,0,0⟩,rows⟩ opt | none
  let b1 := (PvsBoardTree.play mc fc b m).val
  let b2 := (PvsBoardTree.play mc fc b n).val
  let es : Joint.Tape Unit := [enter 0 0 .root,enter 1 0 .endRoot,
    .solveEnter 0 2 0 .none ⟨b,(word p.base).raw,rows,none,false,1,5⟩,enter 3 0 .endWindow,
    if p.rows then .rootEnter 0 4 0 .none root rows' else enter 4 0 .pvs (some root),
    .enter 0 ⟨.probe (Cutoffs.key b),0,16,0⟩,.leave 0 (.probe (.miss 0)),
    enter 5 9 .pvs (args b1 true false (-256) 256 p.level none),scored 6 (-2),closed 7 5 (-2),
    enter 8 7 .pvs (args b2 false true (-3) (-2) p.level none),scored 9 (-4),closed 10 8 (-4)] ++
    (if p.retry then [enter 11 p.retrySite .pvs (args b2 true false p.retryWindow (-2) (p.otherLevel.getD p.level) none),
      scored 12 (-5),closed 13 11 (-5)] else []) ++
    (if p.extra then [enter 14 9 .pvs (args b2 true false (-256) (-2) p.level none),scored 15 (-5),closed 16 14 (-5)] else []) ++
    [scored 17 p.root] ++ (if p.endRoot then [closed 18 4 p.root] else [])
  let state ← CallJournal.run mc fc calls false (es.map Joint.Event.engineEvent)
  let choice ← state.choices.find? (fun c => c.opening == 4)
  let f := choice.frame (CallJournal.returnOf state.closed)
  match evaluate mc fc es state f (fun _ _ => False) opt (some (5,6,.miss p.rawProbe)) (fun _ _ _ => none) with
  | .outside => none | .rejected => some false | .proved _ => some true

#guard trial {} = some true
#guard trial {endRoot := true} = some true
#guard trial {level := 2} = some true
#guard trial {root := 4} = some false
#guard trial {otherLevel := some 2} = some false
#guard trial {retry := false} = some false
#guard trial {retrySite := 7} = some false
#guard trial {retryWindow := -255} = some false
#guard trial {extra := true} = some false
#guard trial {base := 1} = some false
#guard trial {rows := false} = some false
#guard trial {corruptRows := true} = some false
#guard trial {idx := 1} = none
#guard trial {rawProbe := 1} = some false

/-- A three-move board reaches the native split after the first PV child. The
    close and chosen word remain distinct events, and the helper feed is empty. -/
private def splitTrial (site bank : Nat) (load score : Int) (schedule : Bool) : Option (Bool × Bool × Bool) := do
  let board : Position := ⟨⟨0x100010001,0x200020002⟩,rfl⟩
  let rows := AspirationMir.NativeRootBirth.initial mc board
  let regs : AspirationMir.SerialStack.State := ⟨0,4,⟨(word (-256)).raw,256,0⟩,rows⟩
  let op := {opt with schedule := fun _ => (schedule,false)}
  let order := moves mc fc board regs op
  let first ← order.head?
  let next := (PvsBoardTree.play mc fc board first).val
  let root := args board true false (-256) 256 2 (some 0)
  let es : Joint.Tape Unit := [enter 0 0 .root,enter 1 0 .endRoot,
    .solveEnter 0 2 0 .none ⟨board,0,rows,none,false,1,5⟩,enter 3 0 .endWindow,
    .rootEnter 0 4 0 .none root rows,.enter 0 ⟨.probe (Cutoffs.key board),0,16,0⟩,.leave 0 (.probe (.miss 0)),
    enter 5 9 .pvs (args next true false (-256) 256 2 none),scored 6 (-2),closed 7 5 (-2),
    enter 8 site .split,.splitWord 0 9 (.mark 0),.splitWord 0 10 (.result bank (word load).raw),
    .engine (.localCall 0 (.leave 11 8 .split none none) none),scored 12 score]
  let s ← CallJournal.run mc fc calls false (es.map Joint.Event.engineEvent)
  let c ← s.choices.find? (fun owner => owner.opening == 4)
  let f := c.frame (CallJournal.returnOf s.closed)
  let act ← CallReturns.captureTimed (es.map Joint.Event.frameSample)
  let (id,_) ← origin es act c f
  let sc := JournalPv.scope s f act id
  let hands := (sc.rootRun mc fc board op.schedule rootMinimumSplitDepth (-256) 256 order).any (·.result.handoff)
  let observed := JournalHelpers.ownerWindow es sc board true true false 2 (Board.emptyCount board.val).toNat 256
    0 ⟨0,9⟩ ⟨0,10⟩ score false
  let rejects := match evaluate mc fc es s f (fun _ _ => False) op (some (5,6,.miss 0)) (fun _ _ _ => none) with
    | .rejected => true | _ => false
  return (hands,observed,rejects)

#guard splitTrial 6 0 7 7 true = some (true,true,true)
#guard splitTrial 5 0 7 7 true = some (true,false,true)
#guard splitTrial 6 1 7 7 true = some (true,false,true)
#guard splitTrial 6 0 6 7 true = some (true,false,true)
#guard splitTrial 6 0 7 7 false = some (false,true,true)

end Regression
end NeuralReversi.PvsParallel.Interleaving.JournalRootStore
