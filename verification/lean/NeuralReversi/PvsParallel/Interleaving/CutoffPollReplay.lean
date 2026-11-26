import NeuralReversi.PvsParallel.Interleaving.ActivePointer
import NeuralReversi.PvsParallel.Interleaving.FlagSegment
import NeuralReversi.PvsParallel.Interleaving.MaskGeneration
import NeuralReversi.PvsParallel.Interleaving.OwnerTape
import NeuralReversi.PvsParallel.Interleaving.SchedulerTrace

namespace NeuralReversi.PvsParallel.Interleaving.CutoffPollReplay

open Threading
open AtomicWord (Event SequencedBefore)
open CutoffVisibility (History FlagRead)
open TaskAncestry (Journal View Generation)

private instance : Decidable (SequencedBefore a b) :=
  inferInstanceAs (Decidable (a.worker = b.worker ∧ a.step < b.step))

private instance : Decidable (NativeStopReturns.InWorker p e) :=
  inferInstanceAs (Decidable (e.worker = p.worker ∧ e.step ≤ p.step))

private def atEvent (rows : List α) (event : α → Event) (site : Event) : Option {r // r ∈ rows} :=
  match rows.attach.filter fun x => event x.val == site with
  | [r] => some r
  | _ => none

structure Scan (h : History) (at_ : Event) (chain : List Nat) where
  reads : List (FlagRead h.flags)
  walk : CutoffVisibility.Walk h.flags chain reads
  logged : ∀ r ∈ reads, r ∈ h.flagReads
  later : ∀ r ∈ reads, SequencedBefore at_ r.event

private def scan (h : History) (j : Journal) (v : View) (at_ : Event) : (chain : List Nat) →
    List Event → Option (Scan h at_ chain)
  | [],[] => some ⟨[],.empty,by simp,by simp⟩
  | addr :: tail,id :: ids => do
    let ⟨r,mem⟩ ← atEvent h.flagReads (·.event) id
    let _ ← TaskAncestry.source j v r
    if ok : r.slot = addr ∧ SequencedBefore at_ r.event then
      if set : r.value = true then
        guard ids.isEmpty
        return ⟨[r],ok.1 ▸ .found r tail set,by simpa using mem,by simpa using ok.2⟩
      else
        let rest ← scan h j v at_ tail ids
        return ⟨r :: rest.reads,ok.1 ▸ .descend r (Bool.eq_false_iff.mpr set) rest.walk,
          by simpa using And.intro mem rest.logged,by simpa using And.intro ok.2 rest.later⟩
    else none
  | _,_ => none

/-- Reconstruct the short-circuiting scan from the recorded relaxed loads, its
    boolean return, the current task path, and the flag read-from events. -/
def scanAt (h : History) (j : Journal) (v : View) (site : Event) : Option (Scan h site (v.path.map Generation.slot)) := do
  let walked ← j.walks site
  guard (walked.view.path = v.path ∧ walked.view.reset = v.reset)
  let out ← scan h j v site (v.path.map Generation.slot) walked.reverseReads.reverse
  guard ((out.reads.map FlagRead.value).any id = walked.found)
  return out

def ownerScan (h : History) (j : Journal) (site : Event) : Option (SplitReturn.Observation.AncestorScan h) := do
  let (_,v) ← j.after site
  let s ← scanAt h j v site
  return ⟨site,_,s.reads,s.walk,fun r mem => ⟨s.logged r mem,s.later r mem⟩⟩

private structure CacheState (h : History) (chain : List Nat) where
  cache : SplitCutoff.Cache
  seenSource : Option (Fin h.epochs.length)
  reads : List (FlagRead h.flags)
  hits : List (AtomicWord.Read h.epochs × Fin h.epochs.length)
  ran : CutoffVisibility.StopSource.Execution h chain cache reads
  hitExact : ∀ x ∈ hits, x.1 ∈ h.epochReads ∧ x.1.acquire = true ∧ x.1.source = x.2
  latch : Option Event := none
  checkedAt : Option Event := none

private inductive Op where
  | reset (site load src : Event) (value : BitVec 64)
  | latch (site : Event) (bit : Bool)
  | checked (site : Event) (opening : Nat)
  | used
  | epoch (site load src : Event) (value seen : BitVec 64)
  | miss (start : Event)
  | after (start : Event)
  | end_ (bit : Bool)

/-- Select completions on the poll's worker since its most recent cache reset.
    The reset and epoch instructions retain the local load PC and its RF source. -/
private def pick (site : Event) (first : Option Event) : Joint.Event Operand → Option Op
  | .taskWord w pc op =>
    if w != site.worker || pc > site.step || first.any (fun e => pc < e.step) then none else
    match op with
    | .resetCache load src value => some (.reset ⟨w,pc⟩ load src value)
    | .latch bit => some (.latch ⟨w,pc⟩ bit)
    | .poll => some .used
    | .checkStop opening => some (.checked ⟨w,pc⟩ opening)
    | .epoch load src value seen => some (.epoch ⟨w,pc⟩ load src value seen)
    | .beginWalk _ => some (.miss ⟨w,pc⟩)
    | .afterSplit _ => some (.after ⟨w,pc⟩)
    | .endWalk bit => some (.end_ bit)
    | _ => none
  | _ => none

private def epochLoad (h : History) (site load src : Event) (value : BitVec 64) (acq : Bool) :
    Option {r : AtomicWord.Read h.epochs // r ∈ h.epochReads ∧ r.acquire = acq ∧
      (h.epochs.get r.source).value = value} := do
  let ⟨r,logged⟩ ← atEvent h.epochReads (·.event) load
  guard (load.worker = site.worker ∧ load.step ≤ site.step ∧ (h.epochs.get r.source).event = src)
  if ok : r.acquire = acq ∧ (h.epochs.get r.source).value = value then
    return ⟨r,logged,ok⟩
  else none

private def initialEpoch (h : History) : Option (Fin h.epochs.length) :=
  if h0 : 0 < h.epochs.length then
    if (h.epochs.get ⟨0,h0⟩).value == 0 then some ⟨0,h0⟩ else none
  else none

private def run (h : History) (j : Journal) (v : View) (s : CacheState h (v.path.map Generation.slot)) :
    (ops : List Op) → Option (CacheState h (v.path.map Generation.slot))
  | [] => some s
  | .used :: rest => run h j v {s with latch := none, checkedAt := none} rest
  | .checked site opening :: rest => do
    guard (s.latch = some ⟨site.worker,opening⟩)
    run h j v {s with latch := none, checkedAt := some site} rest
  | .latch first true :: rest => do
    if hit : s.cache.latched = true then
      run h j v {s with ran := .cached s.ran hit, latch := some first, checkedAt := none} rest
    else none
  | .latch first false :: .epoch site load src word seen :: rest => do
    let ⟨r,logged,mode,_⟩ ← epochLoad h site load src word true
    guard (seen = s.cache.seen ∧ SequencedBefore first load)
    if clear : s.cache.latched = false then
      if same : (h.epochs.get r.source).value = s.cache.seen then
        let some prev := s.seenSource | none
        if exactSrc : r.source = prev then
          let hitOk : ∀ x ∈ (r,prev) :: s.hits, x.1 ∈ h.epochReads ∧ x.1.acquire = true ∧ x.1.source = x.2 := by
            intro x hx
            rcases List.mem_cons.mp hx with rfl | old
            · exact ⟨logged,mode,exactSrc⟩
            · exact s.hitExact x old
          run h j v ⟨s.cache,s.seenSource,s.reads,(r,prev) :: s.hits,
            .hit s.ran clear r logged mode same,hitOk,some first,none⟩ rest
        else none
      else
        match rest with
        | .miss event :: .end_ bit :: tail => do
          let walk ← j.walks event
          guard (walk.view.reset = v.reset)
          let scanned ← scanAt h j walk.view event
          guard (bit = (scanned.reads.map FlagRead.value).any id)
          if after : SequencedBefore r.event event then
            let flags : CutoffVisibility.FlagReads h r :=
              ⟨_,scanned.reads,fun x mem => ⟨scanned.logged x mem,
                after.1.trans (scanned.later x mem).1,Nat.lt_trans after.2 (scanned.later x mem).2⟩⟩
            let a : CutoffVisibility.PollRead h s.cache := ⟨r,logged,mode,flags,fun _ => scanned.walk⟩
            if path : walk.view.path = v.path then
              run h j v ⟨_,some r.source,_,s.hits,.miss s.ran clear a same (congrArg (List.map Generation.slot) path),
                s.hitExact,some first,none⟩ tail
            else if quiet : (scanned.reads.map FlagRead.value).any id = false then
              run h j v ⟨_,some r.source,_,s.hits,.otherClear s.ran clear a same quiet,s.hitExact,some first,none⟩ tail
            else none
          else none
        | _ => none
    else none
  | .after event :: .end_ bit :: rest => do
    let walk ← j.walks event
    guard (walk.view.reset = v.reset)
    let scanned ← scanAt h j walk.view event
    guard (bit = (scanned.reads.map FlagRead.value).any id)
    if path : walk.view.path = v.path then
      run h j v ⟨_,s.seenSource,_,s.hits,.afterSplit s.ran (path ▸ scanned.walk) scanned.logged,s.hitExact,none,none⟩ rest
    else if !bit && !s.cache.latched then
      run h j v {s with latch := none, checkedAt := none} rest
    else none
  | _ => none
termination_by structural ops => ops

structure Recovered (m : NativeStopReturns.Memory) (j : Journal) (v : View) (poll : Event)
    extends StopSchedule.CacheAt m poll (v.path.map Generation.slot) where
  origins : ∀ r ∈ reads, (TaskAncestry.source j v r).isSome
  seenSource : Option (Fin m.history.epochs.length)
  hits : List (AtomicWord.Read m.history.epochs × Fin m.history.epochs.length)
  hitExact : ∀ x ∈ hits, x.1 ∈ m.history.epochReads ∧ x.1.acquire = true ∧ x.1.source = x.2

/-- No cache latch, previous scan, cache execution proof, or choice of parent
    links is an input. The initial thread fields are (seen=0, latched=false).
    An epoch cache hit additionally verifies exact write-source identity rather
    than matching only the 64-bit numeric counter value. -/
def cacheAt (m : NativeStopReturns.Memory) (j : Journal) (es : Joint.Tape Operand) (poll : Event) (v : View) :
    Option (Recovered m j v poll) := do
  let ops := es.filterMap (pick poll v.reset)
  let final_ ← match v.reset,ops with
    | some at_, .reset first load src word :: tail => do
      guard (at_ = first)
      let _ ← epochLoad m.history first load src word false
      run m.history j v ⟨_,none,[],[],.reset _ word,by simp,none,none⟩ tail
    | none,_ => run m.history j v ⟨⟨0,false⟩,initialEpoch m.history,[],[],.reset _ 1,by simp,none,none⟩ ops
    | _,_ => none
  guard (final_.checkedAt = some poll)
  if times : ∀ r ∈ final_.reads, NativeStopReturns.InWorker poll r.event then
    if all_ : final_.reads.all (fun r => (TaskAncestry.source j v r).isSome) then
      return ⟨⟨final_.cache,final_.reads,final_.ran,times⟩,List.all_eq_true.mp all_,
        final_.seenSource,final_.hits,final_.hitExact⟩
    else none
  else none

/-- Every epoch cache hit accepted by `cacheAt` matches the exact physical
    `cutoff_epoch` write that populated the worker's local cache, rejecting
    reused 64-bit values from other writes. -/
def EpochHitContract : Prop :=
  ∀ {Operand : Type} (m : NativeStopReturns.Memory) (j : Journal) (es : Joint.Tape Operand)
    (poll : Event) (v : View) (c : Recovered m j v poll), cacheAt m j es poll v = some c →
    ∀ x ∈ c.hits, x.1 ∈ m.history.epochReads ∧ x.1.acquire = true ∧ x.1.source = x.2

theorem epochHitSource : EpochHitContract := by
  intro _ _ _ _ _ _ c _ x mem
  exact c.hitExact x mem

def LatchContract := ∀ {Operand : Type} (m : NativeStopReturns.Memory) (j : Journal) (es : Joint.Tape Operand)
    (poll : Event) (v : View) (c : Recovered m j v poll), cacheAt m j es poll v = some c →
    c.cache.latched = true → ∃ r ∈ c.reads, r ∈ m.history.flagReads ∧ r.value = true ∧
      ∃ raw g, j.loads r.event = some raw ∧ raw.view.path = v.path ∧ raw.view.reset = v.reset ∧
        g ∈ v.path ∧ g.slot = r.slot ∧ j.stores ((m.history.flags r.slot).get r.source).event = some (g,true)

theorem latchedSource : LatchContract := by
  intro _ m j es poll v c _ yes
  obtain ⟨r,member,logged,_,set⟩ := CutoffVisibility.StopSource.positive c.assignment yes
  let src := (TaskAncestry.source j v r).get (c.origins r member)
  exact ⟨r,member,logged,set,src.load,src.generation,src.readAt,src.assignment.1,src.assignment.2,
    src.field.2.2,src.field.2.1,by rw [←src.register.1,←set]; exact src.storeAt⟩

def sampled (j : Journal) (es : Joint.Tape Operand) (g : Generation) (m : NativeStopReturns.Memory)
    (check : SplitReturn.Observation.OwnerCheck m.history) (wait : Event) (ps : List Tickets.Post) (poll : Event)
    (context : (rs : List (FlagRead m.history.flags)) → Option (StopSchedule.PollContext m check wait g.slot ps poll rs)) :
    Option (StopSchedule.Samples m check wait g.slot ps poll) := do
  let v ← j.polls poll
  let recovered ← cacheAt m j es poll v
  TaskAncestry.samples j g m check wait ps poll fun chain => do
    if eq : v.path.map Generation.slot = chain then
      let c : StopSchedule.CacheAt m poll chain := eq ▸ recovered.toCacheAt
      let ctx ← context c.reads
      if positive : NativeStopReturns.poll (.cached chain c.cache c.reads c.assignment c.localClock) ctx.loads = some true then
        return ⟨c,ctx,positive⟩
    none

private abbrev waitAt {t : NativeSplit.Tickets rc pv beta} (b : AlphaOrigin.SetupOperands t) :=
  b.owner.exit.clock.atStep (3+b.owner.exit.log.length)

def generation {t : NativeSplit.Tickets rc pv beta} (b : AlphaOrigin.SetupOperands t) : Generation :=
  ⟨b.flagAddress (b.generation.owner,b.bankIndex),(b.mask.get b.generation.mark).event⟩

def slots {t : NativeSplit.Tickets rc pv beta} (b : AlphaOrigin.SetupOperands t) (worker bank : Nat) :=
  if fits : worker < 64 then some (b.flagAddress (⟨worker,fits⟩,bank)) else none

private def maskMark (g : NestedRetention.Generation h) := (g.writes.get g.mask.mark).event

/-- Physical mask histories and ordering of clear operations. The task path,
    dispatch retention, protected flag segment and positive poll are reconstructed. -/
structure FlagMemory (h : History) : Prop where
  coherent : CutoffVisibility.Coherent h
  readRead : ∀ a ∈ h.flagReads, ∀ b ∈ h.flagReads, a.slot = b.slot →
    CutoffVisibility.HappensBefore h a.event b.event → a.source.val ≤ b.source.val

def flagMemoryOK (c : MaskClock.Capture h) : Bool :=
  let transfers := c.transfers
  let before (a b : Event) := (FlagSegment.published transfers a b).isSome = true
  (h.epochReads.all fun a => h.epochReads.all fun b =>
    decide (before a.event b.event → a.source.val ≤ b.source.val)) &&
  (h.flagReads.all fun r =>
    decide (∀ w : Fin (h.flags r.slot).length,
      (before ((h.flags r.slot).get w).event r.event → w.val ≤ r.source.val) ∧
      (before r.event ((h.flags r.slot).get w).event → r.source.val < w.val)) &&
    h.flagReads.all fun b => decide (r.slot = b.slot → before r.event b.event → r.source.val ≤ b.source.val))

def FlagMemoryContract := ∀ (h : History) (c : MaskClock.Capture h),
  flagMemoryOK c = true ↔ FlagMemory h

theorem checkedFlags : FlagMemoryContract := by
  intro h c
  simp only [flagMemoryOK,Bool.and_eq_true,List.all_eq_true,decide_eq_true_eq,← MaskClock.published h c]
  constructor
  · rintro ⟨epoch,flag⟩
    exact ⟨⟨epoch,fun r hr w => ((flag r hr).1 w).1,fun r hr w => ((flag r hr).1 w).2⟩,
      fun r hr => (flag r hr).2⟩
  · intro m
    exact ⟨m.coherent.epochReadRead,fun r hr => ⟨fun w =>
      ⟨m.coherent.flagWriteRead r hr w,m.coherent.flagReadWrite r hr w⟩,m.readRead r hr⟩⟩

/-- Full write-read, read-write, and read-read coherence for the 64-bit cutoff_epoch word. -/
structure EpochMemory (h : History) : Prop where
  writeRead : ∀ r ∈ h.epochReads, ∀ w : Fin h.epochs.length,
    CutoffVisibility.HappensBefore h (h.epochs.get w).event r.event → w.val ≤ r.source.val
  readWrite : ∀ r ∈ h.epochReads, ∀ w : Fin h.epochs.length,
    CutoffVisibility.HappensBefore h r.event (h.epochs.get w).event → r.source.val < w.val
  readRead : ∀ a ∈ h.epochReads, ∀ b ∈ h.epochReads,
    CutoffVisibility.HappensBefore h a.event b.event → a.source.val ≤ b.source.val

def epochMemoryOK (c : MaskClock.Capture h) : Bool :=
  let transfers := c.transfers
  let before (a b : Event) := (FlagSegment.published transfers a b).isSome = true
  (h.epochReads.all fun r =>
    decide (∀ w : Fin h.epochs.length,
      (before (h.epochs.get w).event r.event → w.val ≤ r.source.val) ∧
      (before r.event (h.epochs.get w).event → r.source.val < w.val)) &&
    h.epochReads.all fun b => decide (before r.event b.event → r.source.val ≤ b.source.val))

def EpochMemoryContract := ∀ (h : History) (c : MaskClock.Capture h),
  epochMemoryOK c = true ↔ EpochMemory h

theorem checkedEpochs : EpochMemoryContract := by
  intro h c
  simp only [epochMemoryOK,List.all_eq_true,Bool.and_eq_true,decide_eq_true_eq,← MaskClock.published h c]
  constructor
  · intro ok
    exact ⟨fun r hr w => ((ok r hr).1 w).1,fun r hr w => ((ok r hr).1 w).2,fun a ha b hb => (ok a ha).2 b hb⟩
  · intro m r hr
    exact ⟨fun w => ⟨m.writeRead r hr w,m.readWrite r hr w⟩,fun b hb => m.readRead r hr b hb⟩

structure Context {t : NativeSplit.Tickets rc pv beta} (b : AlphaOrigin.SetupOperands t) (es : Joint.Tape Operand) where
  chronology : MaskClock.Capture b.memory.history
  lockInit : AlphaReplay.Initialization
  memory : (rank : Nat) → (ps : List Tickets.Post) → (poll : Event) → (rs : List (FlagRead b.memory.history.flags)) →
    Option (StopSchedule.PollMemory b.memory b.owner.check (waitAt b) (generation b).slot ps poll rs)

private theorem logged {h : History} {rs : List (FlagRead h.flags)} (run : CutoffVisibility.StopSource.Execution h chain cache rs) :
    ∀ r ∈ rs, r ∈ h.flagReads := by
  induction run with
  | reset => simp
  | cached _ _ ih | hit _ _ _ _ _ _ ih => exact ih
  | miss _ _ op _ _ ih =>
    intro r mem
    exact (List.mem_append.mp mem).elim (ih r) (fun seen => (op.flags.afterLoad r seen).1)
  | otherClear _ _ _ _ _ ih => exact ih
  | afterSplit _ _ added ih =>
    intro r mem
    exact (List.mem_append.mp mem).elim (ih r) (added r)

structure Enclosing (h : History) (scan : SplitReturn.Observation.AncestorScan h) (r : FlagRead h.flags) where
  parent : StopSchedule.Parent h scan r
  flags : FlagMemory h

/-- The flag branch returns from should_stop before loading the abort token.
    A current local flag also requires a posted cutoff in the preceding lock history. -/
def flagged (m : NativeStopReturns.Memory) (j : Journal) (es : Joint.Tape Operand) (g : Generation)
    (check : SplitReturn.Observation.OwnerCheck m.history) (wait : Event) (ps : List Tickets.Post) (poll : Event)
    (parent : (r : FlagRead m.history.flags) → Option (Enclosing m.history check.ancestors r)) :
    Option (StopSchedule.Evidence m check wait g.slot ps poll) := do
  let w ← TaskAncestry.window j g poll check.ancestors.event
  guard (TaskAncestry.checkedScan j w.owner check.ancestors)
  let c ← cacheAt m j es poll w.view
  if shape : w.view.path.map Generation.slot = g.slot :: check.ancestors.chain then
    let cache : StopSchedule.CacheAt m poll (g.slot :: check.ancestors.chain) := shape ▸ c.toCacheAt
    if on : cache.cache.latched = true then
      cache.reads.attach.findSome? fun ⟨r,mem⟩ => do
        if yes : r.value = true then
          let cause : StopSchedule.FlagCause m.history check wait g.slot ps r ← if same : r.slot = g.slot then do
              if cut : ps.any Tickets.Post.cut then pure (.here same (List.any_eq_true.mp cut)) else none
            else do
              if target : r.slot ∈ check.ancestors.chain then
                if later : SequencedBefore wait check.ancestors.event then
                  let data ← parent r
                  pure (.ancestor target data.parent data.flags.coherent data.flags.readRead later)
                else none
              else none
          return .flag cache r on mem (logged cache.assignment r mem) yes cause
        else none
    else none
  else none

private def eras {t : NativeSplit.Tickets rc pv beta} {b : AlphaOrigin.SetupOperands t} (ctx : Context b es)
    (j : Journal) (r : FlagRead b.memory.history.flags) (g : NestedRetention.Generation b.memory.history) :
    Option (TaskRetention.Era b.memory.history r g) := do
  let era ← FlagSegment.recover b.memory.history j ⟨r.slot,maskMark g⟩
  let initialized ← FlagSegment.published ctx.chronology.transfers era.clear r.event
  if next : ∀ x ∈ era.future, x.value = false → SequencedBefore g.wait x.event then
    return ⟨era.toFlags,initialized.down,next⟩
  none

def stops {t : NativeSplit.Tickets rc pv beta} (data : AlphaOrigin.Operands t) (es : Joint.Tape Operand)
    (j : Journal) (ctx : Context data.toOwnerOperands.setup es) : SchedulerReplay.Stops data := fun rank ps poll =>
  let b := data.toOwnerOperands.setup
  let input := MaskGeneration.input ctx.chronology es (slots b)
  let scan := b.owner.check.ancestors
  flagged b.memory j es (generation b) b.owner.check (waitAt b) ps poll
    (fun r => do
      if ok : flagMemoryOK ctx.chronology then
        return ⟨← TaskRetention.parent input es (generation b) poll scan r (eras ctx j r) ctx.chronology.before,
          (checkedFlags ..).mp ok⟩
      none) <|>
      StopSchedule.Evidence.full <$> sampled j es (generation b) b.memory b.owner.check (waitAt b) ps poll (fun rs => do
        let m ← ctx.memory rank ps poll rs
        let parents ← TaskRetention.parentEra input es (generation b) poll scan rs (eras ctx j) ctx.chronology.before
        return ⟨m,parents⟩)

def fromTape {t : NativeSplit.Tickets rc pv beta} (b : AlphaOrigin.SetupOperands t)
    (root cut : Bool) (depth : Nat) (es : Joint.Tape Operand) (ctx : Context b es)
    (before : Option (Event → Event → Bool) := none) := do
  guard (t.dispatches.all fun d =>
    (ActivePointer.splitDispatch ctx.chronology (slots b) es (generation b) d.entry).isSome)
  let owner := OwnerTape.physical b (← OwnerTape.sources b.owner ctx.chronology es)
  let regs ← AlphaReplay.addressed owner ctx.chronology es ctx.lockInit
  let data : AlphaOrigin.Operands t := ⟨owner,regs⟩
  let j := TaskAncestry.capture (slots b) es
  SchedulerTrace.fromTape data root cut depth (fun rank ps poll => j.bind fun log => stops data es log ctx rank ps poll) es before

/-- Every helper or owner dispatch replayed through `fromTape` (both root and nested
    paths) resolves its cloned active pointer to a unique visible store belonging to
    this exact split generation, with worker-lock release preceding search entry. -/
def DispatchProvenanceContract : Prop :=
  ∀ {Operand : Type} {rc : List Move.Move} {pv : Bool} {beta : Int}
    (t : NativeSplit.Tickets rc pv beta) (b : AlphaOrigin.SetupOperands t)
    (root cut : Bool) (depth : Nat) (es : Joint.Tape Operand) (ctx : Context b es)
    (before : Option (Event → Event → Bool)) (cap : AlphaOrigin.Capture t root cut depth (fun _ _ _ _ _ => True)),
    fromTape b root cut depth es ctx before = some cap →
    ∀ d ∈ t.dispatches, ∃ sd : ActivePointer.SplitDispatch b.memory.history (slots b) es (generation b) d.entry,
      ActivePointer.splitDispatch ctx.chronology (slots b) es (generation b) d.entry = some sd ∧
      sd.origin.generation = generation b ∧
      ActivePointer.Visible b.memory.history (ActivePointer.stores d.worker es) sd.origin.read (some sd.origin.store) ∧
      SequencedBefore sd.origin.drop d.entry

theorem fromTape_dispatches : DispatchProvenanceContract := by
  intro _ _ _ _ t b root cut depth es ctx before cap ok d mem
  unfold fromTape at ok
  by_cases allOk : (t.dispatches.all fun d =>
      (ActivePointer.splitDispatch ctx.chronology (slots b) es (generation b) d.entry).isSome) = true
  · have someD := List.all_eq_true.mp allOk d mem
    obtain ⟨sd,found⟩ := Option.isSome_iff_exists.mp someD
    exact ⟨sd,found,sd.sameGeneration,sd.visible,sd.afterUnlock⟩
  · simp [allOk,guard,failure] at ok

end NeuralReversi.PvsParallel.Interleaving.CutoffPollReplay
