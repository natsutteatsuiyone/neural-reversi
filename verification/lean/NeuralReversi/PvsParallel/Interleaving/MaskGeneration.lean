import NeuralReversi.PvsParallel.Interleaving.TaskRetention
import NeuralReversi.PvsParallel.Interleaving.OwnerMaskGuards
import NeuralReversi.PvsParallel.Interleaving.MaskClock

namespace NeuralReversi.PvsParallel.Interleaving.MaskGeneration

open Threading
open AtomicWord (Event Write Read SequencedBefore InReleaseSequence)
open CutoffVisibility (History HappensBefore)
open SplitCompletion (Action Worker)
open SlotFrameMir (RawMask ClosedAfterFirst)

private local instance {ws : List Write} : DecidableEq (SplitSchedulerMir.Guard ws)
  | .shutdown a, .shutdown b => if eq : a=b then isTrue (eq ▸ rfl) else isFalse (fun h => eq (SplitSchedulerMir.Guard.shutdown.inj h))
  | .mask a, .mask b => if eq : a=b then isTrue (eq ▸ rfl) else isFalse (fun h => eq (SplitSchedulerMir.Guard.mask.inj h))
  | .shutdown _, .mask _ | .mask _, .shutdown _ => isFalse (by intro eq; cases eq)

inductive Op where
  | clear | mark
  | change (who : Nat) (book : Bool) (old stored : BitVec 64)
  | shutdown (closed : Bool)
  | load (source : Event) (word : BitVec 64) (acquire : Bool)
  | returned | pc (block : Fin 80)
  deriving DecidableEq

structure Row where
  event : Event
  op : Op
  deriving DecidableEq

def rows (owner : Worker) (bank : Nat) (es : Joint.Tape Operand) : List Row :=
  es.filterMap fun row =>
    let emit (w t : Nat) (owns : Bool) (op : Op) : Option Row :=
      if owns then some ⟨⟨w,t⟩,op⟩ else none
    match row with
    | .maskWord w t (.clear b) => emit w t (w == owner.val && b == bank) .clear
    | .splitWord w t (.mark b) => emit w t (w == owner.val && b == bank) .mark
    | .splitWord w t (.maskRmw o b who book old stored) =>
      emit w t (o == owner.val && b == bank) (.change who book old stored)
    | .maskWord w t (.shutdown b stop) => emit w t (w == owner.val && b == bank) (.shutdown stop)
    | .maskWord w t (.load b src word acquire) => emit w t (w == owner.val && b == bank) (.load src word acquire)
    | .maskWord w t (.returned b) => emit w t (w == owner.val && b == bank) .returned
    | .maskWord w t (.pc b block) => emit w t (w == owner.val && b == bank) (.pc block)
    | _ => none

def Row.write (owner : Worker) (row : Row) : Option Write :=
  match row.op with
  | .clear => some ⟨row.event,0,false,false⟩
  | .mark => some ⟨row.event,Bitboard.squareMask owner,false,true⟩
  | .change _ book _ word => some ⟨row.event,word,!book,true⟩
  | _ => none

def Row.guardRow (row : Row) : Option OwnerMaskGuards.Operand :=
  match row.op with
  | .shutdown b => some (.shutdown row.event b)
  | .load s value order => some (.mask row.event s value order)
  | _ => none

def Row.pc? (r : Row) : Option (Fin 80) := match r.op with | .pc pc => some pc | _ => none

private def delim (r : Row) := r.op == .clear || r.op == .mark

private def slice (log : List Row) (mark : Event) : Option (List Row × List Row) := do
  let before := log.takeWhile (fun r => r != ⟨mark,.mark⟩)
  let chosen :: rest := log.drop before.length | none
  guard (chosen == ⟨mark,.mark⟩)
  guard (((before.filter delim).getLast?).any (fun r => r.op == .clear))
  let current := rest.takeWhile (fun r => !delim r)
  guard (((rest.drop current.length).head?).all (fun r => r.op == .clear))
  return (before,current)

private def change (row : Row) : Option (Option (Action × BitVec 64)) :=
  match row.op with
  | .change who book old _ =>
    if bound : who < 64 then
      let w : Worker := ⟨who,bound⟩
      if book then some (some (.book w row.event,old))
      else if who == row.event.worker then some (some (.finish w row.event,old)) else none
    else none
  | _ => some none

private def actor (a : Action) := match a with | .book w _ | .finish w _ => w

structure Decoded (owner : Worker) (log : List Row) where
  mask : RawMask (log.filterMap (Row.write owner))
  closed : ClosedAfterFirst mask
  actors : ∀ i w e, mask.actions.get i = .finish w e → e.worker = w.val
  span : List Row

private def raw (owner : Worker) (log : List Row) (mark : Event) : Option (Decoded owner log) := do
  let (pre,body) ← slice log mark
  let changes := (← body.mapM change).reduceOption
  let as_ := changes.map Prod.fst
  let ws := log.filterMap (Row.write owner)
  guard ((ws.map Write.event).Nodup)
  let start ← OwnerMaskGuards.source ws mark
  let prior := pre.filterMap (Row.write owner)
  guard ((prior.getLast?).any (fun wr => !wr.rmw && !wr.release && wr.value == 0 &&
    wr.event.worker == mark.worker && wr.event.step < mark.step))
  guard (start.val = prior.length)
  if top : start.val+as_.length < ws.length then
    if init : (ws.get start).value = Bitboard.squareMask owner then
      if actual : ∀ i : Fin as_.length,
          let pos : Fin ws.length := ⟨start.val+i.val+1,by have := i.isLt; omega⟩
          let prev : Fin ws.length := ⟨start.val+i.val,by have := i.isLt; omega⟩
          (ws.get pos).event = (as_.get i).event ∧ (ws.get pos).release = (as_.get i).release ∧
            (ws.get pos).rmw = true ∧ (ws.get pos).value = (as_.get i).updateBits (ws.get prev).value then
        let g : RawMask ws := ⟨owner,as_,start,top,init,actual⟩
        guard ((List.finRange changes.length).all (fun i =>
          (ws[start.val+i.val]?).any (fun wr => wr.value == (changes.get i).2)))
        if close : ∀ i j : Fin as_.length, i.val < j.val → (as_.get i).release = true → (as_.get j).release = true then
          if actors : ∀ i : Fin as_.length, (as_.get i).release = true → (as_.get i).event.worker = (actor (as_.get i)).val then
            return ⟨g,close,fun i w e reset => by
              change as_.get i = .finish w e at reset
              have ok := actors i (by rw [reset];rfl)
              rw [reset] at ok; exact ok,body⟩
  none

private def idleRow : Row → Bool
  | ⟨_,.shutdown _⟩ | ⟨_,.load ..⟩ | ⟨_,.returned⟩ | ⟨_,.pc _⟩ => true
  | _ => false

private def shortPath (ws : List Write) (guards : List (SplitSchedulerMir.Guard ws)) : List (Fin 80) :=
  ([1,2,3,4,5,6,7] : List (Fin 80)) ++ guards.flatMap (β := Fin 80) fun
    | .shutdown _ => [9,10,11,12,13,14,15]
    | .mask src => if (ws.get src).value = 0 then [16,78,79] else
      [17,18,19,20,21,22,23,40,41,42,43,44,76,77,4,5,6,7]

private abbrev Cursor := Fin 80 × Nat

private def successors (ws : List Write) (guards : Array (SplitSchedulerMir.Guard ws)) (c : Cursor) : List Cursor :=
  match SplitSchedulerMir.idle c.1 with
  | .done => []
  | .any targets => targets.map (fun pc => (pc,c.2))
  | .shutdown run stop => match guards[c.2]? with
    | some (.shutdown b) => [(if b then stop else run,c.2+1)]
    | _ => []
  | .mask busy zero => match guards[c.2]? with
    | some (.mask src) => [(if (ws.get src).value = 0 then zero else busy,c.2+1)]
    | _ => []

/-- With no PC capture, search the finite product of the 80 MIR blocks and the
    guard cursor. Mark states at insertion so a silent cycle cannot exhaust the
    search before another outgoing edge has been tried. -/
private def findPath (ws : List Write) (gs : Array (SplitSchedulerMir.Guard ws)) :
    Nat → List (Cursor × List (Fin 80)) → Std.HashSet Cursor → Option (List (Fin 80))
  | 0, _, _ | _, [], _ => none
  | fuel+1, (c,path)::todo, seen =>
    match SplitSchedulerMir.idle c.1 with
    | .done => if c.2 = gs.size then some path.reverse else findPath ws gs fuel todo seen
    | _ =>
      let (work,known) := (successors ws gs c).foldl (fun (state : List (Cursor × List (Fin 80)) × Std.HashSet Cursor) x =>
        if state.2.contains x then state else ((x,x.1::path)::state.1,state.2.insert x)) (todo,seen)
      findPath ws gs fuel work known

private inductive Pending where
  | none | shutdown | mask
  deriving DecidableEq

/-- A recorded basic-block entrance precedes the register load in that block.
    Matching the two projected lists separately would also accept a PC stream
    placed entirely before the first guard. -/
private def blockGuards (idle : List Row) : Bool := Id.run do
  let mut pending := Pending.none
  for r in idle do
    match r.op with
    | .pc pc =>
      if pending != .none then return false
      pending := match SplitSchedulerMir.idle pc with
        | .shutdown .. => .shutdown | .mask .. => .mask | _ => .none
    | .shutdown _ =>
      if pending != .shutdown then return false
      pending := .none
    | .load .. =>
      if pending != .mask then return false
      pending := .none
    | _ => pure ()
  return pending == .none

/-- Even an inferred path is checked against the MIR CFG. If PCs are recorded,
    use those exact block entries, starting at zero, with their original guards. -/
private def pathFor (ws : List Write) (gs : List (SplitSchedulerMir.Guard ws))
    (pcs : List (Fin 80)) : Option (List (Fin 80)) :=
  match pcs with
  | [] =>
    let fast := shortPath ws gs
    if OwnerMaskGuards.follows SplitSchedulerMir.idle ws fast 0 gs then some fast
    else findPath ws gs.toArray (80*(gs.length+1)) [((0,0),[])] ((∅ : Std.HashSet Cursor).insert (0,0))
  | 0 :: tail => some tail
  | _ => none

theorem path_of_return {gs : List (SplitSchedulerMir.Guard ws)}
    (run : SplitSchedulerMir.Returned SplitSchedulerMir.idle ws 0 gs) :
    ∃ path, pathFor ws gs (0::path) = some path ∧ OwnerMaskGuards.follows SplitSchedulerMir.idle ws path 0 gs := by
  obtain ⟨p,ok⟩ := OwnerMaskGuards.follows_complete run
  exact ⟨p,rfl,ok⟩

abbrev writes (o : Worker) (rs : List Row) := rs.filterMap (Row.write o)
abbrev operands (rs : List Row) := rs.filterMap Row.guardRow

structure Observed (o : Worker) (rs : List Row) extends Decoded o rs where
  read : Read (writes o rs)
  acquired : read.acquire = true
  zero : ((writes o rs).get read.source).value = 0
  afterMark : SequencedBefore ((writes o rs).get mask.mark).event read.event
  nextClear : ∀ k : Fin (writes o rs).length, k.val = mask.mark.val + mask.actions.length + 1 →
    SequencedBefore read.event ((writes o rs).get k).event
  operand : OwnerMaskGuards.Operand
  member : operand ∈ operands rs
  loaded : operand.guard (writes o rs) = some (.mask read.source)
  site : operand.site = read.event
  log : List (SplitSchedulerMir.Guard (writes o rs))
  returned : SplitSchedulerMir.Returned SplitSchedulerMir.idle (writes o rs) 0 log
  live : OwnerMaskGuards.poolOpen log
  finalGuard : log.getLast? = some (.mask read.source)
  exit : Event
  beforeExit : SequencedBefore read.event exit

/-- Decode interleaved idle-loop reads for this bank, delimited by its actual
    clear/mark, returning call, and next clear. All mask modifications, including
    redundant resets later than the selected owner observation, stay in the MO. -/
def observe (owner : Worker) (rs : List Row) (mark : Event) : Option (Observed owner rs) := do
  let d ← raw owner rs mark
  let ws := writes owner rs
  let idle := d.span.filter idleRow
  let exit ← idle.getLast?
  guard (exit.op == .returned && idle.dropLast.all (fun r => r.op != .returned))
  guard ((mark :: idle.map Row.event).Pairwise (fun a b => a.worker = owner.val ∧ b.worker = owner.val ∧ a.step < b.step))
  let reads := idle.filterMap Row.guardRow
  let gs ← reads.mapM (fun r => r.guard ws)
  guard (gs.all fun
    | .shutdown _ => true
    | .mask src => d.mask.mark.val ≤ src.val && src.val ≤ d.mask.mark.val+d.mask.actions.length)
  let pcs := idle.filterMap Row.pc?
  guard (pcs.isEmpty || blockGuards idle)
  let path ← pathFor ws gs pcs
  if traced : OwnerMaskGuards.follows SplitSchedulerMir.idle ws path 0 gs then
    if opened : OwnerMaskGuards.poolOpen gs then
      let last ← (idle.filter (fun r => r.guardRow.isSome)).getLast?
      if mem : last ∈ rs then
        let src ← match last.op with
          | .load stamp 0 true => OwnerMaskGuards.source ws stamp
          | _ => none
        match field : last.guardRow with
        | none => none
        | some row =>
          if decode : row.guard ws = some (.mask src) then
            if final : gs.getLast? = some (.mask src) then
              if word : (ws.get src).value = 0 then
                if markBefore : (ws.get d.mask.mark).event.worker = row.site.worker ∧
                    (ws.get d.mask.mark).event.step < row.site.step then
                  if returned : row.site.worker = exit.event.worker ∧ row.site.step < exit.event.step then
                    if before : ∀ k : Fin ws.length, k.val = d.mask.mark.val + d.mask.actions.length + 1 →
                        row.site.worker = (ws.get k).event.worker ∧ exit.event.step < (ws.get k).event.step then
                      return { toDecoded := d, read := ⟨row.site,src,true⟩, acquired := rfl, zero := word
                               afterMark := markBefore, nextClear := fun k next =>
                                 ⟨(before k next).1, Nat.lt_trans returned.2 (before k next).2⟩
                               operand := row, member := List.mem_filterMap.mpr ⟨last,mem,field⟩
                               loaded := decode, site := rfl, log := gs, returned := OwnerMaskGuards.follows_correct traced
                               live := opened, finalGuard := final, exit := exit.event, beforeExit := returned }
          none
  none

/-- Native per-object coherence and release-sequence synchronization, interpreted
    on the full physical modification order, before projecting any redundant RMW. -/
structure Memory (h : History) (owner : Worker) (rs : List Row) : Prop where
  coherence : OwnerMaskGuards.Coherent h (writes owner rs) (operands rs)
  sync : ∀ row ∈ operands rs, ∀ src, row.guard (writes owner rs) = some (.mask src) →
    ∀ j, InReleaseSequence (writes owner rs) j src → h.otherSync ((writes owner rs).get j).event row.site

private def sourceRank (ws : List Write) (op : OwnerMaskGuards.Operand) : Option (Fin ws.length) :=
  match op.guard ws with
  | some (.mask i) => some i
  | _ => none

private theorem sourceRank_eq (ws : List Write) (op : OwnerMaskGuards.Operand) (i : Fin ws.length) :
    sourceRank ws op = some i ↔ op.guard ws = some (.mask i) := by
  unfold sourceRank
  cases op.guard ws with
  | none => simp
  | some g => cases g <;> simp

private instance {ws : List Write} {i j : Fin ws.length} : Decidable (InReleaseSequence ws i j) :=
  inferInstanceAs (Decidable ((ws.get i).release = true ∧ i.val ≤ j.val ∧
    ∀ k : Fin ws.length, i.val < k.val → k.val ≤ j.val → (ws.get k).rmw = true))

/-- The physical mask contributes one synchronization edge for each release
    sequence head read by each acquire, including equal-valued releases. -/
def releaseEdges (o : Worker) (rs : List Row) : List (Event × Event) :=
  let ws := writes o rs
  (operands rs).flatMap fun op => (sourceRank ws op).toList.flatMap fun src =>
    ((List.finRange ws.length).filter (fun j => decide (InReleaseSequence ws j src))).map
      fun j => ((ws.get j).event,op.site)

def ReleaseEdgeContract := ∀ (o : Worker) (rs : List Row) (edge : Event × Event),
  let ws := writes o rs
  edge ∈ releaseEdges o rs ↔ ∃ op ∈ operands rs, ∃ src, op.guard ws = some (.mask src) ∧
    ∃ j, InReleaseSequence ws j src ∧ ((ws.get j).event,op.site) = edge

theorem physicalEdges : ReleaseEdgeContract := by
  intro o rs edge
  simp only [releaseEdges,List.mem_flatMap,Option.mem_toList,sourceRank_eq,List.mem_map,
    List.mem_filter,List.mem_finRange,true_and,decide_eq_true_eq]

/-- Check every decoded mask acquire against this object's entire modification
    order. An independent write ordered earlier in a clock need not happen-before the read. -/
def memoryOK (c : MaskClock.Capture h) (o : Worker) (rs : List Row) : Bool :=
  let ws := writes o rs
  let transfers := c.transfers
  (operands rs).all fun op => (sourceRank ws op).all fun i =>
    decide (∀ j : Fin ws.length,
      ((FlagSegment.published transfers (ws.get j).event op.site).isSome = true → j.val ≤ i.val) ∧
      ((FlagSegment.published transfers op.site (ws.get j).event).isSome = true → i.val < j.val) ∧
      (InReleaseSequence ws j i → ((ws.get j).event,op.site) ∈ c.sync.edges))

def MemoryContract := ∀ (h : History) (c : MaskClock.Capture h) (o : Worker) (rs : List Row),
  memoryOK c o rs = true ↔ Memory h o rs

theorem checkedMemory : MemoryContract := by
  intro h c o rs
  unfold memoryOK
  simp only [List.all_eq_true,Option.all_eq_true,decide_eq_true_eq]
  constructor
  · intro ok
    constructor
    · intro op mem i found
      have atRead := ok op mem i ((sourceRank_eq ..).mpr found)
      exact ⟨fun j hb => (atRead j).1 ((MaskClock.published h c ..).mp hb),
        fun j hb => (atRead j).2.1 ((MaskClock.published h c ..).mp hb)⟩
    · intro op mem i found j seq
      exact (c.sync.exact ..).mp ((ok op mem i ((sourceRank_eq ..).mpr found) j).2.2 seq)
  · intro m op mem i rank j
    have found := (sourceRank_eq ..).mp rank
    have coh := m.coherence op mem i found
    exact ⟨fun hb => coh.writeRead _ ((MaskClock.published h c ..).mpr hb),
      fun hb => coh.readWrite _ ((MaskClock.published h c ..).mpr hb),
      fun seq => (c.sync.exact ..).mpr (m.sync op mem i found j seq)⟩

def Observed.generation (o : Observed owner rs) (bank : Nat) (memory : Memory h owner rs) :
    NestedRetention.Generation h := by
  have co := memory.coherence o.operand o.member o.read.source o.loaded
  rw [o.site] at co
  let window : AtomicWord.ReadWindow o.read (HappensBefore h) o.mask.mark (o.mask.mark.val+o.mask.actions.length) :=
    ⟨.sequenced o.afterMark,fun k next => .sequenced (o.nextClear k next),co.writeRead,co.readWrite⟩
  exact ⟨(owner,bank),_,o.mask,o.closed,NestedRetention.retireRead o.closed o.read o.zero o.acquired o.afterMark window
    (fun j source => o.site ▸ memory.sync _ o.member _ o.loaded j source),o.actors⟩

@[simp] theorem Observed.wait (o : Observed owner rs) (m : Memory h owner rs) :
    (o.generation bank m).wait = o.read.event := rfl

def ofTape (h : History) (es : Joint.Tape Operand) (owner : Worker) (bank : Nat) (mark : Event)
    (memory : Memory h owner (rows owner bank es)) : Option (NestedRetention.Generation h) :=
  (observe owner (rows owner bank es) mark).map (fun o => o.generation bank memory)

/-- Resolve the bank from the original owner instruction; the same slot can
    supply a fresh generation later with a different mark. -/
def locate (address : TaskRetention.Address) (es : Joint.Tape Operand) (g : TaskAncestry.Generation) :
    Option (Worker × Nat) := do
  if h : g.mark.worker < 64 then
    match es.filterMap fun row => match row with
      | .splitWord w t (.mark b) =>
        if g.mark == ⟨w,t⟩ && address w b == some g.slot then some b else none
      | _ => none with
    | [b] => return (⟨g.mark.worker,h⟩,b)
    | _ => none
  else none

def input (c : MaskClock.Capture h) (es : Joint.Tape Operand) (address : TaskRetention.Address) :
    TaskRetention.Input h where
  address := address
  clock := c.clock
  generation key := do
    let (owner,bank) ← locate address es key
    if ok : memoryOK c owner (rows owner bank es) then
      ofTape h es owner bank key.mark ((checkedMemory ..).mp ok)
    else none

namespace Regression

private abbrev owner : Worker := 0
private abbrev maskOp (w t : Nat) (op : Joint.MaskWord) : Joint.Event Unit := .maskWord w t op
private abbrev splitOp (w t : Nat) (op : Joint.SplitWord) : Joint.Event Unit := .splitWord w t op
private def mark : Event := ⟨0,20⟩
private def firstZero : Event := ⟨1,70⟩
private def repeatedZero : Event := ⟨1,110⟩
private def reuse : Event := ⟨0,120⟩

private def sample (rf : Event) (acquire : Bool) (busy : BitVec 64) (extra : Joint.Tape Unit) :
    Joint.Tape Unit :=
  [maskOp 0 10 (.clear 0),splitOp 0 20 (.mark 0),splitOp 0 30 (.maskRmw 0 0 1 true 1 3),
   maskOp 0 40 (.shutdown 0 false),maskOp 0 50 (.load 0 ⟨0,30⟩ busy true),
   splitOp 0 60 (.maskRmw 0 0 0 false 3 2),splitOp 1 70 (.maskRmw 0 0 1 false 2 0)] ++
  extra ++ [maskOp 0 80 (.shutdown 0 false),maskOp 0 90 (.load 0 rf 0 acquire),
   maskOp 0 100 (.returned 0),splitOp 1 110 (.maskRmw 0 0 1 false 0 0),maskOp 0 120 (.clear 0)]

private def es := sample firstZero true 3 []
private abbrev rs := rows owner 0 es
private abbrev ws := writes owner rs
private def result (es : Joint.Tape Unit) :=
  (observe owner (rows owner 0 es) mark).map fun o =>
    (o.read.source.val,o.read.event,o.exit,o.mask.actions.length)

theorem two_equal_releases :
    result es = some (4,⟨0,90⟩,⟨0,100⟩,4) ∧
    result (sample repeatedZero true 3 []) = some (5,⟨0,90⟩,⟨0,100⟩,4) ∧
    result (sample reuse true 3 []) = none ∧ result (sample firstZero false 3 []) = none ∧
    result (sample firstZero true 1 []) = none ∧
    result (sample firstZero true 3 [splitOp 0 75 (.maskRmw 0 0 2 true 0 4),
      splitOp 2 76 (.maskRmw 0 0 2 false 4 0)]) = none := by
  decide +kernel

private def edges (x y : Event) := y = ⟨0,90⟩ ∧ (x = ⟨0,60⟩ ∨ x = ⟨1,70⟩)
private def history : History := ⟨[],fun _ => [],[],[],edges⟩

private theorem earlier (hb : HappensBefore history x y) : x.step < y.step := by
  induction hb with
  | sequenced sb => exact sb.2
  | synchronized i => exact Fin.elim0 i
  | other edge =>
    obtain ⟨rfl,left | right⟩ := edge <;> subst_vars <;> decide
  | trans _ _ fst snd => exact Nat.lt_trans fst snd

private def rank (op : OwnerMaskGuards.Operand) := (op.guard ws).bind fun
  | .mask src => some src.val | _ => none

private theorem memory : Memory history owner rs := by
  constructor
  · intro row inReads i found
    obtain ⟨r,rfl⟩ := List.get_of_mem inReads
    have regs : ∀ r : Fin (operands rs).length, ∀ i j : Fin ws.length, rank ((operands rs).get r) = some i.val →
        (((ws.get j).event.step < ((operands rs).get r).site.step → j.val ≤ i.val) ∧
        (((operands rs).get r).site.step < (ws.get j).event.step → i.val < j.val)) := by decide +kernel
    have got : rank ((operands rs).get r) = some i.val := by unfold rank; rw [found]; rfl
    exact ⟨fun j hb => (regs r i j got).1 (earlier hb),fun j hb => (regs r i j got).2 (earlier hb)⟩
  · intro row inReads i found j seq
    obtain ⟨r,rfl⟩ := List.get_of_mem inReads
    have edgesAt : ∀ r : Fin (operands rs).length, ∀ i j : Fin ws.length, rank ((operands rs).get r) = some i.val →
        (ws.get j).release=true → j.val ≤ i.val → edges (ws.get j).event ((operands rs).get r).site := by
      unfold edges; decide +kernel
    exact edgesAt r i j (by unfold rank; rw [found]; rfl) seq.1 seq.2.1

theorem generated_wait :
    (ofTape history es owner 0 mark memory).map NestedRetention.Generation.wait = some ⟨0,90⟩ := by
  decide +kernel

end Regression

end NeuralReversi.PvsParallel.Interleaving.MaskGeneration
