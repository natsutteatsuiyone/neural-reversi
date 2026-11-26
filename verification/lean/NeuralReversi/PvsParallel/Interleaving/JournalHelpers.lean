import NeuralReversi.PvsParallel.Interleaving.Joint
import NeuralReversi.PvsParallel.Interleaving.CallReturns
import NeuralReversi.PvsParallel.Interleaving.AlphaOrigin
import NeuralReversi.PvsParallel.Interleaving.BestPublisher

namespace NeuralReversi.PvsParallel.Interleaving.JournalHelpers

open Threading
open SplitWorkerMir
open CallReturns (Row Scope State Activation)
open PvsBoardTree (Position GeneratedSquare)
open AtomicWord (Event)
open RootOwnerMir (Result)
open IteratorFrames (Observation)

/-- A successful CAS carries both the opening of next() and the instruction that
    installed its successor. A reused iterator can repeat both counter words. -/
private def claimAt (es : Joint.Tape Op) (w opening owner bank old stored : Nat) : Option (Nat × Nat) :=
  match es.zipIdx.filterMap fun (e,t) => match e with
    | .splitWord worker step (.claim k o b i j) =>
      if worker = w ∧ k = opening ∧ o = owner ∧ b = bank ∧ i = old ∧ j = stored then some (t,step) else none
    | _ => none with
  | [x] => some x
  | _ => none

def lastMark (es : Joint.Tape Op) (owner bank time : Nat) : Option (Nat × Nat) :=
  ((es.take time).zipIdx.filterMap fun (ev,k) => match ev with
    | .splitWord w step (.mark b) => if w = owner ∧ b = bank then some (k,step) else none
    | _ => none).getLast?

private def ticks (es : Joint.Tape Op) (worker : Nat) :=
  es.filterMap fun e => if e.worker = worker then match e with
    | .splitWord _ step _ => some step
    | _ => e.frameSample.bind (fun s => match s.record with | .localCall _ x => some x.step | _ => none)
  else none

private def stackAt (es : Joint.Tape Op) (worker t : Nat) : Option (List Nat) := do
  let state ← CallReturns.captureTimed ((es.take t).map Joint.Event.frameSample)
  return (state.workers worker).stack

/-- Match the actual claim and the last generation mark at that address. The
    mark must execute under this root's split. The CAS must execute under the
    matching specialization of dispatch_search. Stamps below are worker-local. -/
private def location (es : Joint.Tape Op) (state : State) (root bank : Nat) (mark : Event)
    (obs : Observation) (old stored dispatchSite : Nat) : Option (Nat × Nat × Nat) := do
  let (claim,stamp) ← claimAt es obs.worker obs.opening mark.worker bank old stored
  let (created,markStep) ← lastMark es mark.worker bank claim
  guard (markStep = mark.step ∧ obs.opening ≤ stamp ∧ stamp < obs.scoutOpening)
  let split :: caller :: _ ← stackAt es mark.worker created | none
  let parent ← state.calls[split]?
  guard (caller = root ∧ parent.callee = .split ∧ parent.parent = some root ∧ parent.site = 5)
  let helper :: method :: _ ← stackAt es obs.worker claim | none
  let op ← state.calls[helper]?
  let from_ ← state.calls[method]?
  guard (op.worker = obs.worker ∧ op.first = obs.dispatch ∧ op.callee = .splitSearch ∧ op.site = dispatchSite ∧
    op.parent = some method ∧ from_.callee = .dispatch)
  guard ((ticks es mark.worker).Pairwise (· < ·) ∧ (ticks es obs.worker).Pairwise (· < ·))
  return (helper,claim,stamp)

/-- Count *entries*, including an earlier invocation that never returned a
    score. Only the physical children after this successful CAS and before its
    lock may supply the one or two return words. -/
private def interval (s : Scope) (claim stamp : Nat) (obs : Observation) :=
  let children := s.calls.toList.zipIdx.filter fun (a,_) =>
    a.callee == .pvs && a.parent == some s.parent && decide (a.entered < s.consumer)
  let before := (children.takeWhile fun (a,_) => decide (a.entered < claim)).length
  let active := (children.drop before).takeWhile fun (a,_) => decide (a.first < obs.acquired.step)
  (before,active.filterMap fun (a,_) => if stamp < a.first then some (a.first,a.returned) else none,active.length)

private def lockTime (es : Joint.Tape Op) (event : Event) (instruction : Joint.SplitWord) : Option Nat :=
  (es.zipIdx.find? fun (row,_) => match row with
    | .splitWord w tick op => w == event.worker && tick == event.step && op == instruction
    | _ => false).map Prod.snd

/-- Bind the final relaxed score load to the only direct split in this PVS activation,
    even when every helper stopped. The split returns before the owner chooses the word. -/
def ownerWindow (es : Joint.Tape Op) (sc : Scope) (b : Position) (rootNode pv cut : Bool)
    (level depth : Nat) (beta : Int) (bank : Nat) (mark load : Event) (value : Int) (stopped : Bool) : Bool := (show Option Unit from do
  let state ← CallReturns.captureTimed (es.map Joint.Event.frameSample)
  let owner ← state.calls[sc.parent]?
  let entry ← owner.input
  let (decision,word) ← owner.chosen
  guard (owner.callee = .pvs ∧ owner.worker = mark.worker ∧ load.worker = mark.worker ∧
    entry.endgame ∧ entry.request.board = b ∧ entry.rootLine.isSome = rootNode ∧ entry.level = level ∧
    entry.request.pv = pv ∧ entry.request.cut = cut ∧ entry.request.depth.toNat = depth ∧ entry.request.beta = beta ∧
    decision = sc.consumer ∧ word.raw.toInt = (if stopped then 0 else value))
  let [(a,id)] := state.calls.toList.zipIdx.filter (fun (r : Activation × Nat) => r.1.parent == some sc.parent && r.1.callee == .split) | none
  guard (a.site = 5 ∧ a.worker = owner.worker)
  let read ← lockTime es load (.result bank (BitVec.ofInt 32 value))
  let (created,markStep) ← lastMark es mark.worker bank read
  guard (markStep = mark.step ∧ (ticks es mark.worker).Pairwise (· < ·))
  let head ← stackAt es mark.worker created
  let atRead ← stackAt es mark.worker read
  guard (head.take 2 = [id,sc.parent] ∧ atRead.take 2 = [id,sc.parent])
  let next := (es.zipIdx.drop (read+1)).filter fun (r : Joint.Event Op × Nat) => r.1.worker == owner.worker &&
    match r.1 with | .splitWord .. => true | ev => ev.frameSample.isSome
  let (.engine (.localCall _ (.leave _ first .split _ none) _),_) :: (_,chosen) :: _ := next | none
  guard (first = a.first ∧ chosen = decision)
  return ()) |>.isSome

def ownerExit {t : NativeSplit.Tickets rc pv beta} (es : Joint.Tape Op) (sc : Scope) (b : Position) (level : Nat)
    (raw : AlphaOrigin.Capture t rootNode cut depth done_) : Bool :=
  match BestPublisher.publisher t with
  | none => false
  | some (store,_) =>
    ownerWindow es sc b rootNode pv cut level depth beta raw.bankIndex (raw.mask.get raw.generation.mark).event
      (raw.owner.exit.clock.atStep (raw.owner.exit.log.length+5)) store.word.toInt raw.owner.check.stopped

/-- The executable caller checks the operand obtained by replaying protected reports.
    This is the same register value as the authenticated final owner load. -/
theorem ownerExit_read {t : NativeSplit.Tickets rc pv beta} (es : Joint.Tape Op) (sc : Scope) (b : Position)
    (level : Nat) (raw : AlphaOrigin.Capture t rootNode cut depth done_) :
    ownerExit es sc b level raw =
    let phys := raw.score.frame.physical
    ownerWindow es sc b rootNode pv cut level depth beta raw.bankIndex (raw.mask.get raw.generation.mark).event
      phys.load.event (AtomicWord.signed32At phys.scores phys.load.source) phys.stop.stopped := by
  have read := BestPublisher.returnWord raw.score
  cases src : BestPublisher.publisher t with
  | none => simp [src] at read
  | some result =>
    obtain ⟨store,k⟩ := result
    simp only [src,Option.map_some,Option.some.injEq] at read
    simp only [ownerExit,src,read]
    rfl

/-- Both mutex operations belong to the dispatch that claimed the move. Check the last
    generation mark through unlock, including the interval after the recursive return. -/
private def locked (es : Joint.Tape Op) (helper claim decision bank : Nat) (mark : Event)
    (obs : Observation) : Option Unit := do
  guard (obs.acquired.worker = obs.worker ∧ obs.published.worker = obs.worker)
  let acquire ← lockTime es obs.acquired (.lock mark.worker bank)
  let release ← lockTime es obs.published (.unlock mark.worker bank)
  guard (claim < acquire ∧ acquire < release ∧ release < decision)
  let atLock ← stackAt es obs.worker acquire
  let atRelease ← stackAt es obs.worker release
  guard (atLock.head? = some helper ∧ atRelease.head? = some helper)
  guard (lastMark es mark.worker bank release = lastMark es mark.worker bank claim)

private def rawLeave (es : Joint.Tape Op) (worker opening : Nat) (finish : Event) (when_ : Option Nat) (word : Word) : Bool :=
  match when_.bind (fun t => es[t]?) with
  | some (.engine (.localCall w (.leave t first .pvs _ (some value)) _)) =>
    w == worker && finish.worker == w && first == opening && finish.step == t && value.raw == word
  | _ => false

structure Receipt (rows : List Row) where
  scope : Scope
  first : Nat
  input : Input
  registry : scope.rows = rows

private def decode (es : Joint.Tape Op) (state : State) (rows : List Row) (root bank level : Nat)
    (mark : Event) (obs : Observation) (next : Position) (old stored : Nat) (i : Input) (rootNode : Bool := true) : Option (Receipt rows) := do
  let rootFrame ← state.calls[root]?
  let entry ← rootFrame.input
  let (decision,_) ← rootFrame.chosen
  guard (rootFrame.callee = .pvs ∧ entry.endgame = true ∧ entry.level = level ∧ entry.rootLine.isSome = rootNode ∧ entry.request.pv = i.pv ∧ entry.request.cut = i.cut ∧
    entry.request.beta = i.beta.toInt)
  guard (old = i.index ∧ stored = old+1 ∧ i.depth = obs.task.depth.toNat ∧ i.root = rootNode ∧ (rootNode = true → i.pv = true ∧ i.cut = false) ∧
    i.depth = entry.request.depth.toNat ∧ i.beta = obs.task.beta ∧ i.alpha = obs.alpha ∧ obs.task.root = i.root ∧
    obs.task.pv = i.pv ∧ obs.task.cut = i.cut ∧ (retry i = true → i.fresh = obs.fresh))
  let (parent,claim,stamp) ← location es state root bank mark obs old stored (if i.root then 0 else if i.pv then 1 else 2)
  locked es parent claim decision bank mark obs
  guard (obs.scoutLoad.worker = obs.worker ∧ stamp < obs.scoutLoad.step ∧ obs.scoutLoad.step < obs.scoutOpening)
  let scope : Scope := ⟨rows,parent,decision,state.calls⟩
  let (first,children,count) := interval scope claim stamp obs
  let dispatch := CallReturns.Dispatch.mk i.index i.depth i.pv i.root i.cut i.beta i.alpha i.fresh
  let (input,after) ← scope.dispatch next dispatch first
  guard (ScoreOrigin.operands input = ScoreOrigin.operands i ∧ after = first+count ∧
    obs.scoutReturn.step < obs.acquired.step)
  let same := fun n => (scope.atOrdinal n).any (fun row => row.entry.level == level && row.entry.endgame && row.entry.rootLine.isNone)
  match children with
  | [(s,t)] =>
    guard (!retry input && s == obs.scoutOpening && rawLeave es obs.worker s obs.scoutReturn t input.scout && same first)
  | [(s,t),(w,u)] =>
    guard (retry input && obs.wideLoad.worker == obs.worker && s == obs.scoutOpening && w == obs.wideOpening && same first && same (first+1) &&
      decide (obs.scoutReturn.step < obs.wideLoad.step ∧ obs.wideLoad.step < w ∧ obs.wideReturn.step < obs.acquired.step) &&
      rawLeave es obs.worker s obs.scoutReturn t input.scout && rawLeave es obs.worker w obs.wideReturn u input.wide)
  | _ => none
  return ⟨scope,first,input,rfl⟩

/-- The report's position in the mutex log, its iterator index and the CAS
    actor all identify the same physical lock record. -/
def report (es : Joint.Tape Op) (rows : List Row) (root bank level : Nat) (mark : Event)
    (t : NativeSplit.Tickets rc pv beta) (rank actor : Nat) (packet : FieldStores.Raw)
    (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : Position)
    (moves : List (GeneratedSquare mc b.val)) (rootNode : Bool := true) : Option (Receipt rows) := do
  let state ← CallReturns.captureTimed (es.map Joint.Event.frameSample)
  let rootFrame ← state.calls[root]?
  guard (rootFrame.input.any fun e => e.request.board == b)
  let op ← t.state.writes.find? (fun op => op.old.toNat == packet.input.index)
  let (k,obs) ← IteratorFrames.findLock t.journal op.worker op.opening
  guard (k = rank ∧ op.worker = actor ∧ op.stored.toNat = op.old.toNat+1 ∧ packet.stopped = false)
  let mv ← moves[packet.input.index]?
  let cell ← rc[packet.input.index]?
  guard (cell.sq = mv.val ∧ cell.flipped = Flip.Dispatch.resultValue fc mv.val b.val.player b.val.opponent)
  decode es state rows root bank level mark obs (PvsBoardTree.play mc fc b mv).val op.old.toNat op.stored.toNat packet.input rootNode

private def one {mc : Bitboard.Moves.X86Config} {fc : Cutoffs.FlipConfig} (h : RootNative.History mc fc)
    (r : Receipt rows) (b : Position) (mv : GeneratedSquare mc b.val) (level : Nat)
    (registry : RootCausal.frames h.weak h.exact level = rows.map Row.frame) (order : h.order = CallReturns.earlier)
    (wide : Bool) (lo hi score : Int) : Option (PLift (h.completed b level mv wide lo hi score)) := do
  let q := if wide then wideQuery r.input else scoutQuery r.input
  guard (q.alpha.toInt = lo ∧ q.beta.toInt = hi ∧ q.result.toInt = score)
  let access := CallReturns.Access.callSite (if wide then r.first+1 else r.first) .splitSearch (if wide then 1 else 0)
  let expected := SerialRootCalls.request mc fc b (Board.emptyCount b.val) ⟨mv,wide,lo,hi⟩
  guard (CallReturns.queryRequest (PvsBoardTree.play mc fc b mv).val q = expected)
  if found : (r.scope.selected access expected).map Row.score = some score then
    return ⟨⟨r.scope.sites access,by
      rw [order,r.scope.root access mc fc b mv (registry.trans (congrArg (List.map Row.frame) r.registry.symm)) wide lo hi]
      exact found⟩⟩
  else none

structure Uses (done_ : PvsMoveLoop.ChildRelation Nat) (beta : Int) (i : Input) : Prop where
  scout : done_ i.index false (-(i.alpha.toInt+1)) (-i.alpha.toInt) i.scout.toInt
  wide : retry i = true → done_ i.index true (-beta) (-i.fresh.toInt) i.wide.toInt
  again : retry i = SplitWindow.reSearch i.pv i.alpha.toInt i.scout.toInt

private def livePacket {mc : Bitboard.Moves.X86Config} {fc : Cutoffs.FlipConfig} (h : RootNative.History mc fc)
    (es : Joint.Tape Op) (rows : List Row) (root bank level : Nat) (mark : Event)
    (t : NativeSplit.Tickets rc pv beta) (rank : Nat) (p : FieldStores.Raw)
    (b : Position) (r : Result (GeneratedSquare mc b.val))
    (registry : RootCausal.frames h.weak h.exact level = rows.map Row.frame) (order : h.order = CallReturns.earlier) :
    Option (PLift (Uses (RootSchedule.indexed (h.completed b level) r) beta p.input)) := do
  let i := p.input
  let actor ← t.state.writes.find? (fun op => op.old.toNat == i.index)
  let moves := r.visited ++ r.remaining.pending
  let out ← report es rows root bank level mark t rank actor.worker p mc fc b moves
  if again : retry i = SplitWindow.reSearch i.pv i.alpha.toInt i.scout.toInt then
    match elem : moves[i.index]? with
    | none => none
    | some mv =>
      let scout ← one h out b mv level registry order false (-(i.alpha.toInt+1)) (-i.alpha.toInt) i.scout.toInt
      if used : retry i = true then
        let wide ← one h out b mv level registry order true (-beta) (-i.fresh.toInt) i.wide.toInt
        return ⟨⟨⟨mv,elem,scout.down⟩,fun _ => ⟨mv,elem,wide.down⟩,again⟩⟩
      else return ⟨⟨⟨mv,elem,scout.down⟩,fun yes => (used yes).elim,again⟩⟩
  else none

/-- A finite Boolean check over the actual mutex log. No unobserved commit
    or schedule is quantified over by the execution interface. -/
def checkPackets {mc : Bitboard.Moves.X86Config} {fc : Cutoffs.FlipConfig} (h : RootNative.History mc fc)
    (es : Joint.Tape Op) (rows : List Row) (root bank level : Nat) (mark : Event)
    (t : NativeSplit.Tickets rc pv beta) (b : Position) (r : Result (GeneratedSquare mc b.val))
    (registry : RootCausal.frames h.weak h.exact level = rows.map Row.frame) (order : h.order = CallReturns.earlier) :=
  t.calls.zipIdx.all fun (p,k) => p.stopped || (livePacket h es rows root bank level mark t k p b r registry order).isSome

private theorem normalized {a b : Input} (same : ScoreOrigin.operands a = ScoreOrigin.operands b) :
    a.index = b.index ∧ a.alpha = b.alpha ∧ a.scout = b.scout ∧ a.pv = b.pv ∧
    (retry a = true → a.fresh = b.fresh ∧ a.wide = b.wide) := by
  have again (i : Input) : retry (ScoreOrigin.operands i) = retry i := by
    unfold ScoreOrigin.operands; split <;> rfl
  have eq := (again a).symm.trans ((congrArg retry same).trans (again b))
  cases flag : retry a with
  | false =>
    simp only [ScoreOrigin.operands,flag,←eq,Bool.false_eq_true,ite_false] at same
    simp only [Input.mk.injEq] at same
    obtain ⟨idx,_,pv,_,_,_,alpha,_,scout,_⟩ := same
    exact ⟨idx,alpha,scout,pv,fun x => by contradiction⟩
  | true =>
    simp only [ScoreOrigin.operands,flag,←eq,ite_true] at same
    subst b; exact ⟨rfl,rfl,rfl,rfl,fun _ => ⟨rfl,rfl⟩⟩

private theorem value32 (s : RawSchedule.Sample hist ws task posts) :
    (BitVec.ofInt 32 s.value).toInt = s.value := by simp [RawSchedule.Sample.value,AtomicWord.signed32At]

theorem reportsUsing {t : NativeSplit.Tickets rc pv beta} {raw : AlphaOrigin.Capture t rootNode cut depth oldChild}
    (test : (k : Nat) → (p : FieldStores.Raw) → Option (PLift (Uses done_ beta p.input)))
    (checked : t.calls.zipIdx.all (fun (p,k) => p.stopped || (test k p).isSome) = true) :
    ScoreOrigin.ReportCalls t (AlphaOrigin.source (root := rootNode) (cut := cut) (depth := depth) raw.registers.reports) done_ t.calls := by
  intro pre p post split ps obs _worker _actor compatible normal
  let c := (AlphaOrigin.source (root := rootNode) (cut := cut) (depth := depth) raw.registers.reports).commit ps obs
  have entry : (p,pre.length) ∈ t.calls.zipIdx := by
    apply List.mk_mem_zipIdx_iff_getElem?.mpr
    rw [split,List.getElem?_append_right (by omega)]
    simp
  have tested := List.all_eq_true.mp checked _ entry
  simp only [normal,Bool.false_or] at tested
  have calls := ((test _ p).get tested).down
  obtain ⟨idx,alpha,scout,kind,wide⟩ := normalized compatible
  change p.input.index = c.index at idx
  change p.input.pv = pv at kind
  have lo : p.input.alpha.toInt = c.scout.value := (congrArg BitVec.toInt alpha).trans (value32 c.scout)
  have round (useWide : Bool) (a z : Int) :
      (BitVec.ofInt 32 (c.child c.index useWide a z)).toInt = c.child c.index useWide a z := by
    let i := (t.post ((AlphaOrigin.next t.reports ps).get obs.down).position).input
    change (BitVec.ofInt 32 (AlphaOrigin.children i c.index useWide a z)).toInt = AlphaOrigin.children i c.index useWide a z
    cases useWide <;> simp [AlphaOrigin.children]
  have first : p.input.scout.toInt = c.child c.index false (-(c.scout.value+1)) (-c.scout.value) :=
    (congrArg BitVec.toInt scout).trans (round false _ _)
  have recalc := calls.again
  rw [kind,lo,first] at recalc
  have again : retry p.input = c.wide.isSome := recalc.trans c.reload.symm
  constructor
  · simpa only [idx,lo,first] using calls.scout
  · intro used
    have enabled := again.trans used
    obtain ⟨fresh,wideWord⟩ := wide enabled
    have value : p.input.fresh.toInt = c.fresh := (congrArg BitVec.toInt fresh).trans (value32 (c.wide.getD c.scout))
    have gate : SplitWindow.reSearch pv c.scout.value (c.child c.index false (-(c.scout.value+1)) (-c.scout.value)) = true :=
      c.reload.symm.trans used
    have word : p.input.wide.toInt = c.child c.index true (-beta) (-c.fresh) := by
      refine (congrArg BitVec.toInt wideWord).trans ?_
      change (BitVec.ofInt 32 (if SplitWindow.reSearch pv c.scout.value
          (c.child c.index false (-(c.scout.value+1)) (-c.scout.value)) then
          c.child c.index true (-beta) (-c.fresh) else 0)).toInt = _
      simp only [gate,ite_true]
      exact round true _ _
    change done_ c.index true (-beta) (-c.fresh) (c.child c.index true (-beta) (-c.fresh))
    simpa only [idx,value,word] using calls.wide enabled

theorem packetReports {h : RootNative.History mc fc} {es : Joint.Tape Op} {rows : List Row} {root level : Nat}
    {b : Position} {r : Result (GeneratedSquare mc b.val)}
    {registry : RootCausal.frames h.weak h.exact level = rows.map Row.frame} {order : h.order = CallReturns.earlier}
    {t : NativeSplit.Tickets rc pv beta} {raw : AlphaOrigin.Capture t rootNode cut depth oldChild}
    (checked : checkPackets h es rows root raw.bankIndex level (raw.mask.get raw.generation.mark).event t b r registry order = true) :
    ScoreOrigin.ReportCalls t (AlphaOrigin.source (root := rootNode) (cut := cut) (depth := depth) raw.registers.reports)
      (RootSchedule.indexed (h.completed b level) r) t.calls := by
  exact reportsUsing (raw := raw) (fun k p => livePacket h es rows root raw.bankIndex level
    (raw.mask.get raw.generation.mark).event t k p b r registry order) checked

namespace Fixture

private def board : Position := ⟨⟨0x810000000,0x1008000000⟩,rfl⟩
private def nextBoard : Position := ⟨⟨0x1000000000,0x81c000000⟩,rfl⟩
private def word (x : Int) := BitVec.ofInt 32 x

private structure Edit where
  interior : Bool := false
  nonPv : Bool := false
  cut : Bool := false
  wrongDispatch : Bool := false
  wrongParentCut : Bool := false
  staleCaller : Bool := false
  priorHole : Bool := false
  insideHole : Bool := false
  abandon : Bool := false
  wide : Bool := true
  oldGeneration : Bool := false
  reuseDuringCall : Bool := false
  reuseAfterReturn : Bool := false
  reuseWhileLocked : Bool := false
  omitLock : Bool := false
  wrongUnlockBank : Bool := false
  unlockAfterExit : Bool := false
  pending : Bool := false
  wrongActor : Bool := false
  wrongSite : Bool := false
  wrongLevel : Bool := false
  forgedWide : Bool := false

private def packet (m : Edit) : Input :=
  ⟨1,60,!m.nonPv,!m.interior,m.cut,(if m.nonPv then 1 else 8),0,2,word (if m.wide then -4 else 4),word (-6)⟩

private def observation (m : Edit) : Observation :=
  ⟨1,5,2,none,10,14,⟨1,12⟩,⟨1,16⟩,⟨1,7⟩,⟨1,13⟩,⟨1,18⟩,⟨1,19⟩,⟨1,20⟩,
    ⟨60,!m.nonPv,!m.interior,m.cut,(packet m).beta⟩,0,2,⟨false,0,0⟩⟩

private def tape (m : Edit) : Joint.Tape Unit :=
  let i := packet m
  let args := fun (q : Query) =>
    NativeFrames.Args.mk nextBoard.val 59 ⟨q.alpha⟩ ⟨q.beta⟩ true (if m.wrongLevel then 2 else 5) q.pv q.cut none
  let root : NativeFrames.Args := ⟨board.val,60,⟨0⟩,⟨i.beta⟩,true,5,i.pv,(if m.wrongParentCut then !i.cut else i.cut),(if m.interior then none else some 0)⟩
  let enter := fun w step site node registers => Joint.Event.engine (.localCall w (.entry step site node .none none) registers)
  let score := fun w step raw => Joint.Event.engine (.localCall w (.scored step ⟨raw⟩) none)
  let leave := fun w step first (value : Option ScoreRefresh.Score) =>
    Joint.Event.engine (.localCall w (.leave step first .pvs none value) none)
  let scout := args (scoutQuery i)
  let wide := args (wideQuery i)
  let close : Joint.Event Unit := .engine (.localCall 1 (.leave (if m.unlockAfterExit then 19 else 21) 2 .splitSearch none none) none)
  let unlock : Joint.Event Unit := .splitWord 1 20 (.unlock 0 (if m.wrongUnlockBank then 4 else 3))
  let site := if !m.interior then 0 else if m.nonPv then 2 else 1
  [enter 0 0 0 .pvs (some root),enter 0 1 5 .split none,.splitWord 0 2 (.mark 3),
    enter 1 0 0 .active none,enter 1 1 0 .dispatch none,enter 1 2 (if m.wrongDispatch then site+1 else site) .splitSearch none] ++
    (if m.priorHole then [enter 1 3 0 .pvs (some scout),leave 1 4 3 none] else []) ++
    (if m.oldGeneration then [.splitWord 0 3 (.mark 3)] else []) ++
    [.splitWord (if m.wrongActor then 2 else 1) 6 (.claim 5 0 3 1 2)] ++
    (if m.insideHole then [enter 1 8 0 .pvs (some scout),leave 1 9 8 none] else []) ++
    [enter 1 10 (if m.wrongSite then 1 else 0) .pvs (some scout),score 1 11 i.scout] ++
    (if m.reuseDuringCall then [.splitWord 0 3 (.mark 3)] else []) ++
    (if m.pending then [] else [leave 1 12 10 (if m.abandon then none else some ⟨i.scout⟩)]) ++
    (if m.wide && !m.pending then [enter 1 14 1 .pvs (some wide),score 1 15 i.wide,leave 1 16 14 (some ⟨i.wide⟩)] else []) ++
    (if m.reuseAfterReturn then [.splitWord 0 3 (.mark 3)] else []) ++
    (if m.omitLock then [] else [.splitWord 1 18 (.lock 0 3)]) ++
    (if m.reuseWhileLocked then [.splitWord 0 3 (.mark 3)] else []) ++
    (if m.pending then [] else if m.unlockAfterExit then [close,unlock] else [unlock,close]) ++
    [.splitWord 0 19 (.result 3 (word 6)),.engine (.localCall 0 (.leave 20 1 .split none none) none)] ++
    (if m.staleCaller then [enter 0 21 5 .split none,.engine (.localCall 0 (.leave 22 21 .split none none) none)] else []) ++
    [score 0 23 (word 6)]

private def observed (m : Edit) : Option (Nat × Int × Int) := do
  let es := tape m
  let state ← CallReturns.captureTimed (es.map Joint.Event.frameSample)
  let i := packet m
  let out ← decode es state (state.rows true) 0 3 5 ⟨0,2⟩ (observation m) nextBoard 1 2
    (if m.forgedWide then {i with wide := word (-7)} else i) (!m.interior)
  return (out.first,out.input.scout.toInt,out.input.wide.toInt)

#guard observed {} = some (0,-4,-6)
#guard observed {priorHole := true} = some (1,-4,-6)
#guard observed {wide := false,priorHole := true} = some (1,4,0)
#guard observed {oldGeneration := true} = none
#guard observed {reuseDuringCall := true} = none
#guard observed {reuseAfterReturn := true} = none
#guard observed {reuseWhileLocked := true} = none
#guard observed {omitLock := true} = none
#guard observed {wrongUnlockBank := true} = none
#guard observed {unlockAfterExit := true} = none
#guard observed {pending := true} = none
#guard observed {wrongActor := true} = none
#guard observed {wrongSite := true} = none
#guard observed {wrongLevel := true} = none
#guard observed {forgedWide := true} = none
#guard observed {abandon := true} = none
#guard observed {insideHole := true} = none

private def exitChecked (m : Edit) (noPacket : Bool) (result : Int) : Bool :=
  let es := (tape m).filter fun ev => !noPacket || ev.worker == 0
  let sc : Scope := ⟨[],0,es.length-1,#[]⟩
  ownerWindow es sc board (!m.interior) (!m.nonPv) m.cut 5 60 (packet m).beta.toInt
    3 ⟨0,2⟩ ⟨0,19⟩ result false

#guard observed {interior := true,cut := true,priorHole := true} = some (1,-4,-6)
#guard observed {interior := true,nonPv := true,cut := true,wide := false} = some (0,4,0)
#guard observed {interior := true,wrongDispatch := true} = none
#guard observed {interior := true,nonPv := true,wide := false,wrongParentCut := true} = none
#guard observed {interior := true,reuseAfterReturn := true} = none
#guard exitChecked {} true 6
#guard exitChecked {interior := true} true 6
#guard exitChecked {interior := true,nonPv := true,cut := true,wide := false} false 6
#guard !exitChecked {interior := true} true 5
#guard !exitChecked {interior := true,staleCaller := true} true 6

end Fixture

end NeuralReversi.PvsParallel.Interleaving.JournalHelpers
