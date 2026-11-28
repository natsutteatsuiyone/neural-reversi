module
public import NeuralReversi.PvsParallel.Interleaving.Entry
public import NeuralReversi.PvsParallel.Interleaving.Hint

@[expose] public section

namespace NeuralReversi.PvsParallel.Interleaving.CallJournal

open NativeTableWrites.Journal (Calls Record Pc Site)
open TranspositionTable hiding Score
open ScoreRefresh (Score)
open PvsBoardTree (Position)
open CausalTable (Frame)

structure Choice where
  worker : Nat
  opening : Nat
  parent : Option Nat
  stamp : Nat
  level : Selectivity
  endgame : Bool
  rootLine : Option Nat
  board : Position
  pv : Bool
  cut : Bool
  depth : Depth
  alpha : Score
  beta : Score
  word : Score
  deriving DecidableEq

structure Completion where
  choice : Choice
  stamp : Nat
  later : choice.stamp < stamp

def returnOf (closed : List Completion) (f : Choice) : Option Nat :=
  (closed.find? (fun v => v.choice == f)).map Completion.stamp

def Choice.frame (f : Choice) (returns : Choice → Option Nat) : Frame :=
  ⟨f.stamp,returns f,f.board,f.pv,f.alpha.raw.toInt,f.beta.raw.toInt,f.word.raw.toInt⟩

def Choice.whole (f : Choice) := f.endgame && f.rootLine.all (· == 0)

def choice (worker opening stamp : Nat) (parent : Option Nat) (a : NativeFrames.Args) (v : Score) : Option Choice :=
  if valid : a.board.player &&& a.board.opponent = 0 then
    some ⟨worker,opening,parent,stamp,a.selectivity,a.endgame,a.rootLine,⟨a.board,valid⟩,a.pv,a.cut,a.depth,a.alpha,a.beta,v⟩
  else none

structure Activation where
  opening : Nat
  node : EndgameSearch.CacheOwner.Node
  args : Option NativeFrames.Args
  boundary : Option EndgameSearch.CacheHistory.NativeTrace.Operands
  picked : Option Choice

/-- Each writer invocation is begun by a worker, not by a frame ID supplied by the caller.
    Its instruction stream may stop at any program counter. -/
inductive Event (Operand : Type) where
  | other (worker : Nat)
  | localCall (worker : Nat) (native : NativeFrames.Event) (args : Option NativeFrames.Args)
  | boundaryScore (worker : Nat) (pool : EndgameSearch.CacheHistory.Pool) (timeline : EndgameSearch.CacheHistory.Timeline pool)
      (nodesBefore nodesNow : BitVec 64) (word : Score)
  | beginWrite (worker invocation : Nat) (raw : Operand)
  | atomic (worker invocation : Nat)

structure Writer (Operand : Type) where
  invocation : Nat
  next : Nat
  owner : Option Choice
  raw : Operand

inductive Issued (Operand : Type) where
  | localCall (worker : Nat) (native : NativeFrames.Event)
  | atomic (writer : Writer Operand) (pc : Pc)

def Issued.record : Issued Operand → Record
  | .localCall worker e => .localCall worker e
  | .atomic w pc => .writer ⟨w.invocation,pc⟩

structure State (Operand : Type) where
  stacks : Nat → List Activation
  localClock : Nat → Nat
  writing : Nat → Option (Writer Operand)
  started : List Nat
  choices : List Choice
  closed : List Completion
  issued : List (Issued Operand)

def start : State Operand := ⟨fun _ => [],fun _ => 0,fun _ => none,[],[],[],[]⟩

def put (f : Nat → α) (i : Nat) (v : α) := fun j => if i = j then v else f j

def localStep (s : State Operand) (time worker : Nat) (e : NativeFrames.Event)
    (args : Option NativeFrames.Args) : Option (State Operand) := do
  let stack := s.stacks worker
  match e with
  | .entry opening _ node _ boundary =>
    guard (args.isSome = (node == .pvs))
    return {s with stacks := put s.stacks worker (⟨opening,node,args,boundary,none⟩ :: stack)}
  | .scored _ word =>
    let top :: rest := stack | none
    guard (top.node == .pvs && top.picked.isNone)
    let a ← top.args
    let parent := rest.head?.bind (fun p => if p.node == .pvs then some p.opening else none)
    let f ← choice worker top.opening time parent a word
    return {s with stacks := put s.stacks worker ({top with picked := some f} :: rest), choices := f :: s.choices}
  | .leave _ opening node _ normal =>
    let top :: rest := stack | none
    guard (top.opening == opening && top.node == node)
    let restState := {s with stacks := put s.stacks worker rest}
    match normal with
    | none => return restState
    | some word =>
      let f ← top.picked
      if before : f.stamp < time then
        guard (f.word.raw == word.raw)
        return {restState with closed := ⟨f,time,before⟩ :: s.closed}
      else none
  | .discarded _ opening _ =>
    let top :: rest := stack | none
    guard (top.opening == opening && top.node == .pvs)
    return {s with stacks := put s.stacks worker rest}

def deep (i : EndgameSearch.CacheHistory.Input) : Bool :=
  match i.arm with | .deep poll => poll == i.active | _ => false

theorem deep_arm (i : EndgameSearch.CacheHistory.Input) (yes : deep i = true) : i.arm = .deep i.active := by
  cases h : i.arm <;> simp [deep,h] at yes
  exact congrArg _ yes

def BoundaryChecks (i : EndgameSearch.CacheHistory.Input) (before after : BitVec 64) : Bool :=
  i.depth.toNat == 12 && i.depth == i.empties.count && i.parallel && !i.aborted &&
    DeepLeafMir.enoughNodes i.checked before after && deep i &&
    decide (-32767 ≤ i.alpha.raw.toInt ∧ i.alpha.raw.toInt < i.beta.raw.toInt ∧ i.beta.raw.toInt ≤ 32767 ∧
      i.beta.raw.toInt = i.alpha.raw.toInt+1)

/-- A twelve-empty save issues at the strategy boundary itself, including when its
    enclosing PVS activation was selective and has no chosen score yet. -/
theorem boundaryBodies (p : EndgameSearch.CacheHistory.Pool) (timeline : EndgameSearch.CacheHistory.Timeline p)
    (i : EndgameSearch.CacheHistory.Input) (worker opening : Nat)
    (allowed : BoundaryChecks i before after = true)
    (run : (i.run mc fc p.address (timeline.readAt mc fc worker opening).history).score = some v) :
    Nested.Body mc fc prior reads i.board false i.alpha.raw.toInt i.beta.raw.toInt v ∧
    SeedHistory.Body mc fc Cutoffs.exactSelectivity earlier seen i.board false i.alpha.raw.toInt i.beta.raw.toInt v := by
  simp only [BoundaryChecks,Bool.and_eq_true,beq_iff_eq,Bool.not_eq_true',decide_eq_true_eq] at allowed
  obtain ⟨⟨⟨⟨⟨⟨depth,count⟩,parallel⟩,running⟩,enough⟩,active⟩,lo,mid,hi,narrow⟩ := allowed
  exact ⟨.boundarySave p timeline worker opening i rfl depth count parallel running enough (deep_arm i active) run,
    .boundary p timeline worker opening i rfl ⟨lo,mid,hi⟩ narrow (deep_arm i active) run⟩

def boundaryStep (s : State Operand) (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig)
    (time worker : Nat) (p : EndgameSearch.CacheHistory.Pool) (timeline : EndgameSearch.CacheHistory.Timeline p)
    (before after : BitVec 64) (word : Score) : Option (State Operand) := do
  let top :: rest := s.stacks worker | none
  guard (top.node == .boundary && top.picked.isNone && (s.writing worker).isNone)
  let args ← top.boundary
  let i := args.input
  guard (BoundaryChecks i before after)
  let scored ← (i.run mc fc p.address (timeline.readAt mc fc worker top.opening).history).score
  guard (scored == word.raw.toInt)
  let f : Choice := ⟨worker,top.opening,none,time,Cutoffs.exactSelectivity,true,none,i.board,false,false,i.depth,i.alpha,i.beta,word⟩
  return {s with stacks := put s.stacks worker ({top with picked := some f} :: rest), choices := f :: s.choices}

def step (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (c : Calls) (endian : Bool) (s : State Operand) (time : Nat) : Event Operand → Option (State Operand)
  | .other _ => some s
  | .localCall worker e args => do
    guard ((s.writing worker).isNone && s.localClock worker ≤ e.step)
    let next ← localStep s time worker e args
    return {next with localClock := put s.localClock worker (e.step+1), issued := s.issued ++ [.localCall worker e]}
  | .boundaryScore worker p timeline before after word => boundaryStep s mc fc time worker p timeline before after word
  | .beginWrite worker k raw => do
    guard ((s.writing worker).isNone && !(s.started.contains k))
    let owner := (s.stacks worker).head?.bind Activation.picked
    return {s with writing := put s.writing worker (some ⟨k,0,owner,raw⟩), started := k :: s.started}
  | .atomic worker k => do
    let w ← s.writing worker
    guard (w.invocation == k)
    let pc ← (c k).fetch endian w.next
    guard ((c k).event pc.index == time)
    let next := if ((c k).fetch endian (w.next+1)).isSome then some {w with next := w.next+1} else none
    return {s with writing := put s.writing worker next,issued := s.issued ++ [.atomic w pc]}

theorem localWriter (ok : step mc fc c endian s time (.localCall worker e args) = some t) :
    t.writing = s.writing ∧ t.issued = s.issued ++ [.localCall worker e] := by
  simp only [step, guard, Bind.bind, Option.bind, Pure.pure] at ok
  split at ok <;> try contradiction
  split at ok <;> try contradiction
  rename_i st h
  cases ok
  refine ⟨?_,rfl⟩
  unfold localStep at h
  cases e <;> simp only [guard, Bind.bind, Option.bind, Pure.pure] at h <;>
    repeat' first | (split at h <;> try dsimp only at h) | contradiction | (cases h; rfl)

theorem boundaryWriter (ok : step mc fc c endian s time (.boundaryScore worker p timeline before after word) = some t) :
    t.writing = s.writing ∧ t.issued = s.issued := by
  simp only [step, boundaryStep, guard, Bind.bind, Option.bind, Pure.pure] at ok
  repeat' first | (split at ok <;> try dsimp only at ok) | contradiction | (cases ok; exact ⟨rfl,rfl⟩)

/-- The global event number is the logical position of an observed instruction.
    Local Rust entry/return stamps can differ across workers. -/
def run (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (c : Calls) (endian : Bool) (events : List (Event Operand)) :=
  events.zipIdx.foldlM (fun state ev => step mc fc c endian state ev.2 ev.1) start

def State.frames (s : State Operand) := (s.choices.filter Choice.whole).map (fun f => f.frame (returnOf s.closed))

def State.level (s : State Operand) (f : Frame) :=
  ((s.choices.find? (fun d => d.stamp == f.chosen)).map Choice.level).getD Cutoffs.exactSelectivity

def State.exactFrames (s : State Operand) := s.frames.filter (fun f => s.level f == Cutoffs.exactSelectivity)

def State.records (s : State Operand) := s.issued.map Issued.record

theorem released (s : State Operand) (recorded : Record.writer ⟨k,.finish⟩ ∈ s.records) :
    ∃ w : Writer Operand, Issued.atomic w .finish ∈ s.issued ∧ w.invocation = k := by
  obtain ⟨op,seen,eq⟩ := List.mem_map.mp recorded
  cases op with
  | localCall => contradiction
  | atomic w pc =>
    obtain ⟨rfl,rfl⟩ := Site.mk.inj (Record.writer.inj eq)
    exact ⟨w,seen,rfl⟩

open NativeTableWrites.Journal (Origin Even)
open SeqLock (Payload)

structure StoreArgs where
  board : Position
  score : TranspositionTable.Score
  bound : Bound
  depth : Depth
  move : Option Bitboard.Square
  level : Selectivity
  generation : BitVec 8
  phase : Bool
  ttPv : Bool

def StoreArgs.data (a : StoreArgs) := Packing.newData a.score a.move a.bound a.depth a.level a.generation a.phase a.ttPv

def StoreArgs.gate (a : StoreArgs) (slot : Nat) :=
  WriterMir.storeInput slot a.board a.score a.bound a.depth a.move a.level a.generation a.phase a.ttPv

def pickedAt (s : State Operand) (t : Nat) (lv : Selectivity) :=
  CausalTable.chosen (· < ·) (s.frames.filter (fun f => s.level f == lv)) t

def exactAt (s : State Operand) (t : Nat) := CausalTable.chosen (· < ·) s.exactFrames t

structure Inputs (s : State Operand) (t : Nat) (b : Position) (e : EntryData) : Prop where
  seed : SeedHistory.StoreInput (pickedAt s t) b e
  exact : CausalTable.Input (exactAt s t) b e

def legalMove (mc : Bitboard.Moves.X86Config) (a : StoreArgs) : Bool :=
  a.move.all (fun sq => (Board.getMoves mc a.board.val).getLsbD sq.val)

theorem legal_of_test (ok : legalMove mc a = true) : TableReads.LegalChoice a.board a.move := by
  intro m choice
  unfold legalMove at ok
  rw [choice] at ok
  exact (Board.getMoves_exact mc a.board.val m.val).mp ok

def classified (s : State Operand) (slot t : Nat) (a : StoreArgs)
    (owner : Option Choice) (entered : a.gate slot = some (slot,a.board,a.data))
    (legal : TableReads.LegalChoice a.board a.move) : Option (PLift (Inputs s t a.board a.data)) := do
  let f ← owner
  if h : f ∈ s.choices ∧ f.whole = true ∧ s.level (f.frame (returnOf s.closed)) = a.level ∧ f.level = a.level ∧
      f.stamp < t ∧ f.board = a.board ∧ f.word.raw = a.score.raw ∧ f.endgame = a.phase ∧
      f.depth = a.depth ∧ a.bound = classify f.pv a.score ⟨f.alpha.raw⟩ ⟨f.beta.raw⟩ ∧
      -32767 ≤ f.alpha.raw.toInt ∧ f.alpha.raw.toInt < f.beta.raw.toInt ∧ f.beta.raw.toInt ≤ 32767 ∧
      (f.pv = false → f.beta.raw.toInt = f.alpha.raw.toInt+1) then
    let ⟨present,whole,level,_,older,board,word,_,_,bound,low,ordered,high,narrow⟩ := h
    have member : f.frame (returnOf s.closed) ∈ s.frames := List.mem_map.mpr ⟨f,List.mem_filter.mpr ⟨present,whole⟩,rfl⟩
    have used : pickedAt s t a.level a.board f.pv f.alpha.raw.toInt f.beta.raw.toInt a.score.raw.toInt :=
      ⟨f.frame (returnOf s.closed),List.mem_filter.mpr ⟨member,beq_iff_eq.mpr level⟩,older,board,rfl,rfl,rfl,congrArg BitVec.toInt word⟩
    have got : WriterMir.storeInput slot a.board a.score (classify f.pv a.score ⟨f.alpha.raw⟩ ⟨f.beta.raw⟩) a.depth
        a.move a.level a.generation a.phase a.ttPv = some (slot,a.board,a.data) := by simpa only [StoreArgs.gate,←bound] using entered
    let win : Aspiration.Window f.alpha.raw.toInt f.beta.raw.toInt := ⟨low,ordered,high⟩
    have seed := SeedHistory.StoreInput.classified slot a.score ⟨f.alpha.raw⟩ ⟨f.beta.raw⟩ f.pv a.depth a.move
      a.level a.generation a.phase a.ttPv win used got
    if flags : a.level = Cutoffs.exactSelectivity ∧ a.phase = true then
      have truly : exactAt s t a.board f.pv f.alpha.raw.toInt f.beta.raw.toInt a.score.raw.toInt := by
        change pickedAt s t Cutoffs.exactSelectivity _ _ _ _ _
        simpa only [flags.1] using used
      have full := CausalTable.Input.native (slot := slot) f.pv ⟨f.alpha.raw⟩ ⟨f.beta.raw⟩ a.score a.depth a.move a.generation a.ttPv
        win narrow truly legal (by simpa only [flags.1,flags.2] using got)
      return ⟨⟨seed,full⟩⟩
    else if weak : ¬(a.data.flags.getLsbD 0 = true ∧ 5 ≤ a.data.selectivity.toNat) then
      return ⟨⟨seed,.weaker a.data weak (TableReads.newHint legal)⟩⟩
    else none
  else none

def midgame (s : State Operand) (slot t : Nat) (a : StoreArgs) (owner : Option Choice)
    (entered : a.gate slot = some (slot,a.board,a.data))
    (legal : TableReads.LegalChoice a.board a.move) : Option (PLift (Inputs s t a.board a.data)) := do
  let f ← owner
  if ok : f ∈ s.choices ∧ f.stamp < t ∧ f.board = a.board ∧ f.word.raw = a.score.raw ∧
      f.rootLine.all (· == 0) ∧ f.endgame = false ∧ f.level = a.level ∧
      a.phase = false ∧ a.bound = .exact ∧
      classify f.pv a.score ⟨f.alpha.raw⟩ ⟨f.beta.raw⟩ = .exact ∧
      -PvsMoveLoop.scoreMax ≤ a.score.raw.toInt ∧ a.score.raw.toInt ≤ PvsMoveLoop.scoreMax then
    let ⟨_,_,_,_,_,_,_,phase,bound,_,lo,hi⟩ := ok
    have weak : ¬Cutoffs.ExactEndgame a.data := by
      cases tag : a.ttPv <;> simp [Cutoffs.ExactEndgame,StoreArgs.data,Packing.newData,phase,tag]
    have seed := SeedHistory.StoreInput.midgame (picked := pickedAt s t) slot a.score a.depth a.move a.level a.generation
      a.ttPv ⟨lo,hi⟩ (by simpa only [StoreArgs.gate,bound,phase] using entered)
    return ⟨⟨seed,.weaker a.data weak (TableReads.newHint legal)⟩⟩
  else none

def inputs (s : State Operand) (mc : Bitboard.Moves.X86Config) (slot t : Nat) (a : StoreArgs) (owner : Option Choice) :
    Option (PLift (Inputs s t a.board a.data)) := do
  if entered : a.gate slot = some (slot,a.board,a.data) then
    if legal : legalMove mc a then
      let hint := legal_of_test legal
      if weak : a.bound ≠ .exact ∧ ¬(a.data.flags.getLsbD 0 = true ∧ 5 ≤ a.data.selectivity.toNat) then
        return ⟨⟨.other slot a.score a.bound a.depth a.move a.level a.generation a.phase a.ttPv weak.1 entered,
          .weaker a.data weak.2 (TableReads.newHint hint)⟩⟩
      else if a.phase then classified s slot t a owner entered hint else midgame s slot t a owner entered hint
    else none
  else none

/-- A selected old payload is an actual read at this slot. The supplied observation
    is needed only when native save inherits a move or penalize uses its old depth. -/
inductive Operand (c : Calls) (endian : Bool) (seen : Nat → Payload → Prop) where
  | clear
  | save (k : Nat) (a : StoreArgs) (obs : Save.Observation)
      (older : ∀ snap, a.data.bound ≠ Packing.boundByte .exact → a.data.bestMove = Packing.noMove →
        Snapshot.read endian obs.first obs.last obs.sample = some snap → (snap.player,snap.opponent) = Cutoffs.key a.board →
        (Packing.unpack endian snap.metadata).bound ≠ 0 → seen k ⟨snap.player,snap.opponent,snap.metadata⟩)
  | penalty (k : Nat) (b : Position) (expected : EntryData) (s : Option SeqLock.Snapshot) (current : SeqLock.Word)
      (older : ∀ snap, s = some snap → (Packing.unpack endian snap.metadata).bound ≠ 0 → seen k ⟨snap.player,snap.opponent,snap.metadata⟩)

abbrev Output (s : State Operand') (c : Calls) (endian : Bool) (seen : Nat → Payload → Prop) (k : Nat) :=
  Table.Operands endian (Inputs s ((c k).event 5)) (seen k) (c k).origin

def issueOutput (mc : Bitboard.Moves.X86Config) (s : State (Operand c endian seen)) (w : Writer (Operand c endian seen)) :
    Option (PLift (Output s c endian seen w.invocation)) := do
  let k := w.invocation
  match w.raw with
  | .clear =>
    match o : (c k).origin with
    | .clear => return ⟨by change Table.Operands _ _ _ (c k).origin; rw [o]; exact .clear⟩
    | .store .. => none
  | .save id a obs older =>
    if same : id = k then
      let ⟨incoming⟩ ← inputs s mc (c k).address.index ((c k).event 5) a w.owner
      match org : (c k).origin with
      | .clear => none
      | .store frame board entry =>
        if key : board = a.board then
          if wrote : Save.attempt endian (Cutoffs.key a.board) a.data obs = .written (SeqLock.publish (c k).expected.val) entry then
            return ⟨by
              change Table.Operands _ _ _ (c k).origin
              rw [org,key]
              exact .save frame a.board a.data incoming obs _ entry wrote (by simpa only [same,k] using older)⟩
          else none
        else none
    else none
  | .penalty id b expected snapshot current older =>
    if same : id = k then
      match org : (c k).origin with
      | .clear => none
      | .store frame board entry =>
        if key : board = b then
          if wrote : current = (c k).expected.val ∧ Penalty.tryPenalize endian b.val.player b.val.opponent expected snapshot current = some entry then
            return ⟨by
              change Table.Operands _ _ _ (c k).origin
              rw [org,key]
              exact .penalty frame b expected snapshot current entry wrote.2 (by simpa only [same,k] using older)⟩
          else none
        else none
    else none

def accepted (mc : Bitboard.Moves.X86Config) (s : State (Operand c endian seen)) : Issued (Operand c endian seen) → Bool
  | .atomic w .finish => (issueOutput mc s w).isSome
  | _ => true

/-- This second pass reads only the frame index and immutable data captured at writer
    entry. Unfinished invocations have no release and never contribute a TT payload. -/
def decode (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (c : Calls) (endian : Bool) (seen : Nat → Payload → Prop)
    (events : List (Event (Operand c endian seen))) : Option (State (Operand c endian seen)) := do
  let state ← run mc fc c endian events
  if state.issued.all (accepted mc state) then some state else none

structure Trace (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (c : Calls) (endian : Bool) (seen : Nat → Payload → Prop) where
  events : List (Event (Operand c endian seen))
  state : State (Operand c endian seen)
  replay : decode mc fc c endian seen events = some state

theorem Trace.output (trace : Trace mc fc c endian seen) (recorded : Record.writer ⟨k,.finish⟩ ∈ trace.state.records) :
    Output trace.state c endian seen k := by
  obtain ⟨state,_,ok⟩ := Option.bind_eq_some_iff.mp trace.replay
  split at ok
  next acceptedAll =>
    cases Option.some.inj ok
    obtain ⟨w,atIndex,rfl⟩ := released trace.state recorded
    have checked := List.all_eq_true.mp acceptedAll _ atIndex
    obtain ⟨witness,_⟩ := Option.isSome_iff_exists.mp (show (issueOutput mc trace.state w).isSome from checked)
    exact witness.down
  next => contradiction

theorem Trace.seedWrites (trace : Trace mc fc c endian seen) (released : Record.writer ⟨k,.finish⟩ ∈ trace.state.records) :
    Table.Operands endian (SeedHistory.StoreInput (pickedAt trace.state ((c k).event 5))) (seen k) (c k).origin :=
  (trace.output released).mapInputs (fun _ _ witness => witness.seed)

theorem Trace.exactWrites (trace : Trace mc fc c endian seen) (released : Record.writer ⟨k,.finish⟩ ∈ trace.state.records) :
    Table.Operands endian (CausalTable.Input (exactAt trace.state ((c k).event 5))) (seen k) (c k).origin :=
  (trace.output released).mapInputs (fun _ _ witness => witness.exact)

/-- A returned row has a matching normal leave of the worker's active stack top.
    An aborted leave or a suspended frame contributes no return record. -/
theorem return_source {s : State Operand'} {f : Frame} (member : f ∈ s.frames) (normal : f.returned = some t) :
    ∃ completed ∈ s.closed, completed.choice.frame (returnOf s.closed) = f ∧ completed.stamp = t ∧ f.chosen < t := by
  obtain ⟨d,_,rfl⟩ := List.mem_map.mp member
  obtain ⟨ret,found,rfl⟩ := Option.map_eq_some_iff.mp normal
  have ok := List.find?_some (p := fun v : Completion => v.choice == d) found
  have same : ret.choice = d := by simpa using ok
  exact ⟨ret,List.mem_of_find?_eq_some found,congrArg (fun f : Choice => f.frame (returnOf s.closed)) same,rfl,
    by simpa only [Choice.frame,←same] using ret.later⟩

theorem State.ordered {s : State Operand'} (member : f ∈ s.frames) (returned : f.returned = some t) : f.chosen < t :=
  (return_source member returned).choose_spec.2.2.2

/-- Select a normally returned direct recursive call from its actual parent's
    stack entry, callee arguments and word. An inactive child cannot answer. -/
def State.reply (s : State Operand') (caller : Choice) (b : Position) (depth : Depth) (pv cut : Bool) (a z : Int) :=
  (s.choices.find? (fun f => f.whole && f.worker == caller.worker && f.parent == some caller.opening &&
    s.level (f.frame (returnOf s.closed)) == caller.level && f.board == b && f.depth == depth && f.pv == pv &&
    f.cut == cut && f.alpha.raw.toInt == a && f.beta.raw.toInt == z &&
    (returnOf s.closed f).any (· < caller.stamp))).map (fun f => f.word.raw.toInt)

theorem State.reply_used (s : State Operand') (caller : Choice)
    (returned : s.reply caller b depth pv cut a z = some v) :
    CausalTable.returned (· < ·) (s.frames.filter (fun f => s.level f == caller.level)) caller.stamp b pv a z v := by
  obtain ⟨f,found,score⟩ := Option.map_eq_some_iff.mp returned
  have checked := List.find?_some found
  simp only [Bool.and_eq_true,beq_iff_eq] at checked
  obtain ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨whole,sameWorker⟩,parent⟩,level⟩,board⟩,sameDepth⟩,kind⟩,sameCut⟩,lo⟩,hi⟩,checkTime⟩ := checked
  obtain ⟨t,closed,earlier⟩ : ∃ t, returnOf s.closed f = some t ∧ t < caller.stamp := by
    cases h : returnOf s.closed f with
    | none => simp [h] at checkTime
    | some t => exact ⟨t,rfl,by simpa [h] using checkTime⟩
  refine ⟨f.frame (returnOf s.closed),List.mem_filter.mpr ⟨?_,beq_iff_eq.mpr level⟩,t,closed,earlier,board,kind,lo,hi,score⟩
  exact List.mem_map.mpr ⟨f,List.mem_filter.mpr ⟨List.mem_of_find?_eq_some found,whole⟩,rfl⟩

def State.weakChildren (s : State Operand') := s.frames.filter (fun f =>
  s.level f != Cutoffs.exactSelectivity && (s.choices.find? (fun d => d.stamp == f.chosen)).any (·.rootLine.isNone))

def observed {c : Calls} {rs : List Record} (x : NativeTableWrites.Journal.Execution c endian rs)
    (init : ∀ a, Even (x.memory a)) (k : Nat) := Table.Seen x (c k).address (init _) ((c k).event 5)

variable {c : Calls} {endian : Bool} {rs : List Record}
variable {x : NativeTableWrites.Journal.Execution c endian rs} {init : ∀ a, Even (x.memory a)}

def SeedBodies (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig)
    (s : State (Operand c endian (observed x init))) : Prop :=
  ∀ f ∈ s.frames, SeedHistory.Body mc fc (s.level f)
    (CausalTable.returned (· < ·) (s.frames.filter (fun g => s.level g == s.level f)) f.chosen)
    (CausalTable.lookup x init f.chosen) f.board f.pv f.alpha f.beta f.score

def WeakBodies (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig)
    (s : State (Operand c endian (observed x init))) : Prop :=
  ∀ f ∈ s.weakChildren, WeakFrame.Frame mc fc (CausalTable.returned (· < ·) s.weakChildren f.chosen)
    (CausalTable.lookup x init f.chosen) f.board f.pv f.alpha f.beta f.score

def ExactBodies (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig)
    (s : State (Operand c endian (observed x init))) : Prop :=
  ∀ f ∈ s.exactFrames, Nested.Body mc fc (CausalTable.returned (· < ·) s.exactFrames f.chosen)
    (CausalTable.lookup x init f.chosen) f.board f.pv f.alpha f.beta f.score

theorem Trace.levels (trace : Trace mc fc c endian (observed x init)) (same : trace.state.records = rs)
    (bodies : SeedBodies mc fc trace.state) :
    SeedHistory.Prefix mc fc x init trace.state.frames trace.state.level (· < ·) where
  irrefl := Nat.lt_irrefl
  trans := Nat.lt_trans
  left := fun _ member _ normal => State.ordered member normal
  body := bodies
  writes := fun _ mem _ => trace.seedWrites (by rwa [same])

theorem Trace.weakPrefix (trace : Trace mc fc c endian (observed x init)) (bodies : WeakBodies mc fc trace.state) :
    FiniteTable.Prefix x init trace.state.weakChildren (· < ·) (WeakFrame.Frame mc fc) where
  irrefl := Nat.lt_irrefl
  trans := Nat.lt_trans
  left := fun _ member _ normal => State.ordered (List.mem_filter.mp member).1 normal
  body := bodies

theorem Trace.exactPrefix (trace : Trace mc fc c endian (observed x init)) (same : trace.state.records = rs)
    (hb : ∀ a i j, (x.memory a).hb i j → i < j) (bodies : ExactBodies mc fc trace.state) :
    CausalTable.Prefix x init trace.state.exactFrames (· < ·) (Nested.Body mc fc) where
  irrefl := Nat.lt_irrefl
  trans := Nat.lt_trans
  observes := hb
  left := fun _ member _ normal => State.ordered (List.mem_filter.mp member).1 normal
  body := bodies
  store := fun _ mem _ => trace.exactWrites (by rwa [same])

/-- Build the public entry's three histories from the same native worker stack
    and writer stream. The remaining body arguments describe frame control flow. -/
def Trace.capture (trace : Trace mc fc c endian (observed x init)) (same : trace.state.records = rs)
    (resident : ∀ a, Entry.Resident mc fc endian ((x.slot a (init a)).payloadAt 0))
    (order : ∀ a i j, (x.memory a).hb i j → i < j)
    (seeds : SeedBodies mc fc trace.state) (weak : WeakBodies mc fc trace.state) (exactFrames : ExactBodies mc fc trace.state) :
    Entry.Capture mc fc :=
  let lt := fun i j : Nat => decide (i < j)
  have eq : (fun i j => lt i j = true) = (· < ·) := by funext i j; simp [lt]
  ⟨c,endian,rs,x,init,resident,lt,trace.state.frames,trace.state.level,
    by simpa only [eq] using trace.levels same seeds,trace.state.weakChildren,trace.state.exactFrames,
    by simpa only [eq] using trace.weakPrefix weak,by simpa only [eq] using trace.exactPrefix same order exactFrames⟩

end NeuralReversi.PvsParallel.Interleaving.CallJournal
