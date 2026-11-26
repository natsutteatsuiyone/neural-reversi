import NeuralReversi.PvsParallel.Interleaving.Join
import NeuralReversi.PvsParallel.SlotFrameMir.RawGuards
import NeuralReversi.PvsParallel.SplitRebookingMir.Ordering

namespace NeuralReversi.PvsParallel.Interleaving.Mask

open Threading
open AtomicWord (Event Read Write SequencedBefore InReleaseSequence)
open CutoffVisibility (History)
open SplitCompletion (Worker Action MaskState)
open Retirement (ExecutionWith roster)
open SplitSchedulerMir (Clock Guard)

/-- Local instruction numbers can have arbitrary gaps between the init and idle-loop sites. -/
structure Generation where
  tick : Nat → Nat
  advances : ∀ i j, i < j → tick i < tick j
  actions : List Action

def Generation.clock (g : Generation) (owner : Worker) : Clock := ⟨owner.val,g.tick,g.advances⟩

def writes (bits : BitVec 64) : List Action → List Write
  | [] => []
  | a :: tail =>
    let next := a.updateBits bits
    ⟨a.event,next,a.release,true⟩ :: writes next tail

@[simp] private theorem writes_length (actions : List Action) (bits : BitVec 64) :
    (writes bits actions).length = actions.length := by
  induction actions generalizing bits <;> simp [writes, *]

def Generation.mark (g : Generation) (owner : Worker) : Write :=
  ⟨(g.clock owner).atStep 1,Bitboard.squareMask owner,false,true⟩

def Generation.words (g : Generation) (owner : Worker) : List Write :=
  ⟨(g.clock owner).atStep 0,0,false,false⟩ :: g.mark owner :: writes (Bitboard.squareMask owner) g.actions

@[simp] theorem Generation.length (g : Generation) (owner : Worker) :
    (g.words owner).length = g.actions.length+2 := by simp [Generation.words]

/-- Previous uses of this physical slot stay in modification order. Reusing the slot appends
    a new relaxed clear and owner RMW, even when a previous generation also ended in zero. -/
def tape (prior : List Write) (owner : Worker) (g : Generation) (next : List Generation) :=
  prior ++ (g.words owner ++ next.flatMap (fun gen => gen.words owner))

private theorem atWord (prior : List Write) (owner : Worker) (g : Generation) (next : List Generation)
    (n : Nat) (within : n < (g.words owner).length)
    (fits : prior.length+n < (tape prior owner g next).length) :
    (tape prior owner g next)[prior.length+n] = (g.words owner)[n] := by
  simp only [tape,List.getElem_append_right (Nat.le_add_right ..),Nat.add_sub_cancel_left]
  exact List.getElem_append_left within

private theorem adjacency (a : List Action) (first : Write) (n : Nat) (within : n < a.length) :
    let prev := (first :: writes first.value a)[n]'(by simp; omega)
    let curr := (first :: writes first.value a)[n+1]'(by simp; omega)
    curr.event = a[n].event ∧ curr.release = a[n].release ∧ curr.rmw = true ∧ curr.value = a[n].updateBits prev.value := by
  induction a generalizing first n with
  | nil => simp at within
  | cons x xs ih =>
    cases n with
    | zero => exact ⟨rfl,rfl,rfl,rfl⟩
    | succ n =>
      have lt : n < xs.length := by simp at within; omega
      exact ih ⟨x.event,x.updateBits first.value,x.release,true⟩ n lt

def raw (prior : List Write) (owner : Worker) (g : Generation) (next : List Generation) :
    SlotFrameMir.RawMask (tape prior owner g next) where
  owner := owner
  actions := g.actions
  mark := ⟨prior.length+1,by simp [tape]; omega⟩
  bounded := by simp [tape]; omega
  initial := by
    have ok : prior.length+1 < (tape prior owner g next).length := by simp [tape]; omega
    change ((tape prior owner g next)[prior.length+1]'ok).value = _
    rw [atWord prior owner g next 1 (by simp)]
    rfl
  rmw i := by
    change _ ∧ _ ∧ _ ∧ _
    have h := adjacency g.actions (g.mark owner) i.val i.isLt
    have a := atWord prior owner g next (1+i.val) (by simp; omega) (by simp [tape]; omega)
    have b := atWord prior owner g next (1+(i.val+1)) (by simp; omega) (by simp [tape]; omega)
    simp only [Nat.add_assoc] at *
    simp only [List.get_eq_getElem]
    rw [a,b]
    simpa only [Generation.words,Nat.one_add,List.getElem_cons_succ,Generation.mark,List.get_eq_getElem] using h

private theorem snoc (s : MaskState) (as : List Action) (a : Action) :
    SplitCompletion.run s (as++[a]) = SplitCompletion.step (SplitCompletion.run s as) a := by
  induction as generalizing s <;> simp [SplitCompletion.run, *]

private theorem legalSnoc (s : MaskState) (as : List Action) (a : Action)
    (before : SplitCompletion.Legal s as) (last : SplitCompletion.Allowed (SplitCompletion.run s as) a) :
    SplitCompletion.Legal s (as++[a]) := by
  induction as generalizing s with
  | nil => exact ⟨last,trivial⟩
  | cons _ _ ih => exact ⟨before.1,ih _ before.2 last⟩

private structure Linked (owner : Worker) (actions : List Action) (s : Tickets.State w) : Prop where
  legal : SplitCompletion.Legal (SplitCompletion.initial owner) actions
  bits : ∀ j : Worker, (SplitCompletion.run (SplitCompletion.initial owner) actions).has j = true ↔ j.val ∈ roster s
  gate : s.booking = true → (SplitCompletion.run (SplitCompletion.initial owner) actions).accepting = true
  distinct : (roster s).Nodup

private theorem linked (path : Retirement.ExecutionWith out check len start owner actions s) : Linked owner actions s := by
  induction path with
  | initial =>
    exact ⟨trivial,fun j => by change (Bitboard.squareMask owner).getLsbD j.val = true ↔ j.val ∈ [owner.val]; rw [Bitboard.squareMask_bit]; simp,fun _ => rfl,by simp [roster,Tickets.initial]⟩
  | work _ op silent _ _ ih =>
    have perm := Retirement.roster_step op
    rw [silent] at perm
    exact ⟨ih.legal,fun j => (ih.bits j).trans perm.mem_iff.symm,fun h => ih.gate (Retirement.booking_gate op h),perm.nodup_iff.mpr ih.distinct⟩
  | book i e _ op ih =>
    have perm := Retirement.roster_step op
    change (roster _).Perm (i.val :: roster _) at perm
    obtain ⟨open_,absent⟩ := Retirement.opened op
    refine ⟨legalSnoc _ _ _ ih.legal (ih.gate open_),?_,?_,perm.nodup_iff.mpr (List.nodup_cons.mpr ⟨absent,ih.distinct⟩)⟩
    · intro j
      rw [snoc,SplitCompletion.book_has,perm.mem_iff,List.mem_cons]
      simp [ih.bits,Fin.ext_iff,or_comm]
    · intro _
      simpa [snoc,SplitCompletion.step,Action.release] using ih.gate open_
  | finish i e _ op ih =>
    have perm := Retirement.roster_step op
    change (roster _).Perm (i.val :: roster _) at perm
    obtain ⟨gone,distinct⟩ := List.nodup_cons.mp (perm.nodup_iff.mp ih.distinct)
    refine ⟨legalSnoc _ _ _ ih.legal ((ih.bits i).2 (perm.mem_iff.mpr List.mem_cons_self)),?_,?_,distinct⟩
    · intro j
      simp only [snoc,SplitCompletion.finish_has,Bool.and_eq_true,Bool.not_eq_true',decide_eq_false_iff_not,
        ih.bits,perm.mem_iff,List.mem_cons,Fin.ext_iff]
      by_cases same : j.val = i.val
      · simp [same,gone]
      · simp [same]
    · intro h
      simp [Retirement.closed op] at h

def activation (prior : List Write) (next : List Generation) (g : Generation)
    (path : Retirement.ExecutionWith out check len start owner g.actions s) : SplitCompletion.Activation (tape prior owner g next) :=
  let m := raw prior owner g next
  ⟨owner,m.mark,g.actions,m.bounded,(linked path).legal,m.initial,m.rmw⟩


private def guardRead (g : Generation) (owner : Worker) (n : Nat) (src : Fin ws.length) : Read ws :=
  ⟨(g.clock owner).atStep (4+n),src,true⟩

/-- Primitive mask observations of one real idle-loop invocation. The instruction clock
    orders the next physical clear after this call; coherence chooses a source from the
    entire slot history, including previous and following generations. -/
structure Observed (history : History) (prior : List Write) (owner : Worker) (g : Generation) (following : List Generation) where
  log : List (Guard (tape prior owner g following))
  control : SplitSchedulerMir.Returned SplitSchedulerMir.idle (tape prior owner g following) 0 log
  poolOpen : Guard.shutdown true ∉ log
  reuse : ∀ next ∈ following.head?, g.tick (4+log.length) ≤ next.tick 0
  coherent : ∀ pre src rest, log = pre ++ .mask src :: rest →
    SplitRebookingMir.Coherence history (tape prior owner g following) (guardRead g owner pre.length src)
  synchronize : ∀ pre src rest, log = pre ++ .mask src :: rest → ∀ i,
    InReleaseSequence (tape prior owner g following) i src →
    history.otherSync ((tape prior owner g following).get i).event (guardRead g owner pre.length src).event

def Observed.waitAt (o : Observed hist prior owner g next) :=
  (g.clock owner).atStep (3+o.log.length)

private theorem nextClear (prior : List Write) (owner : Worker) (g next : Generation) (tail : List Generation)
    (i : Fin (tape prior owner g (next :: tail)).length)
    (boundary : i.val = prior.length+(g.words owner).length) :
    ((tape prior owner g (next :: tail)).get i).event = (next.clock owner).atStep 0 := by
  change ((tape prior owner g (next :: tail))[i.val]'i.isLt).event = _
  simp only [boundary,tape,List.flatMap_cons,List.getElem_append_right (Nat.le_add_right ..),
    Nat.add_sub_cancel_left,List.getElem_append_right (Nat.le_refl _),Nat.sub_self]
  rfl

private theorem reuseOrder (prior : List Write) (owner : Worker) (g : Generation) (following : List Generation) (e : Event)
    (later : ∀ head ∈ following.head?, SequencedBefore e ((head.clock owner).atStep 0)) :
    ∀ i : Fin (tape prior owner g following).length, i.val = prior.length+(g.words owner).length →
      SequencedBefore e ((tape prior owner g following).get i).event := by
  cases following with
  | nil =>
    intro i h
    have bound := i.isLt
    simp [tape] at bound h
    omega
  | cons head tail =>
    intro i h
    rw [nextClear prior owner g head tail i h]
    exact later head (by simp)

private theorem Observed.window (o : Observed hist prior owner g next)
    (pre : List (Guard (tape prior owner g next))) (src : Fin (tape prior owner g next).length) (rest)
    (site : o.log = pre ++ .mask src :: rest) :
    AtomicWord.ReadWindow (guardRead g owner pre.length src) (CutoffVisibility.HappensBefore hist)
      (raw prior owner g next).mark (prior.length+1+g.actions.length) := by
  have co := o.coherent pre src rest site
  refine ⟨.sequenced ?_,fun i eq => .sequenced ?_,co.writeRead,co.readWrite⟩
  · change SequencedBefore ((tape prior owner g next)[prior.length + 1]'_).event _
    rw [atWord prior owner g next 1 (by simp)]
    exact ⟨rfl,g.advances 1 _ (by omega)⟩
  · apply reuseOrder prior owner g next _ _ i (by simp; omega)
    intro head member
    refine ⟨rfl,Nat.lt_of_lt_of_le (g.advances _ _ ?_) (o.reuse head member)⟩
    simp [site]

/-- The raw-frame analysis gets each coherent guard window from these primitive
    observations, including its bounds on earlier and later uses. -/
def Observed.frame (o : Observed hist prior owner g next) (timeline : Clock)
    (sites : SplitSchedulerMir.OwnerFields.maskClock timeline = g.clock owner) :
    SlotFrameMir.OwnerExit hist (raw prior owner g next) where
  timeline := timeline
  log := o.log
  returned := o.control
  poolOpen := o.poolOpen
  sampled pre src rest site := by
    rw [sites]
    exact o.window pre src rest site

theorem Observed.wait (o : Observed hist prior owner g next)
    (path : Retirement.ExecutionWith out check len start owner g.actions s) :
    ∃ r : SplitCompletion.Retirement hist (tape prior owner g next) (activation prior next g path),
      r.read.event = o.waitAt ∧ r.read.source.val = prior.length+1+g.actions.length := by
  let a := activation prior next g path
  obtain ⟨pre,src,rest,control,zero⟩ := SplitSchedulerMir.idle_acquired_zero o.control o.poolOpen
  have last := SplitSchedulerMir.after_zero o.control control zero
  subst rest
  let read := guardRead g owner pre.length src
  have co := o.coherent pre src [] control
  have size : o.log.length = pre.length+1 := by simp [control]
  have after : SequencedBefore ((tape prior owner g next).get a.start).event read.event := by
    have eq := atWord prior owner g next 1 (by simp) (by simp [tape]; omega)
    change SequencedBefore ((tape prior owner g next)[prior.length + 1]'_).event _
    rw [eq]
    exact ⟨rfl,g.advances 1 _ (by omega)⟩
  have reuse : ∀ i : Fin (tape prior owner g next).length, i.val = a.start.val+a.actions.length+1 →
      SequencedBefore read.event ((tape prior owner g next).get i).event := by
    intro i eq
    change i.val = prior.length+1+g.actions.length+1 at eq
    apply reuseOrder prior owner g next read.event _ i (by simp; omega)
    intro head member
    refine ⟨rfl,Nat.lt_of_lt_of_le (g.advances _ _ ?_) (o.reuse head member)⟩
    omega
  let r : SplitCompletion.Retirement hist _ a :=
    ⟨read,rfl,zero,after,reuse,co.writeRead,co.readWrite,fun i _ seq => o.synchronize pre src [] control i seq⟩
  refine ⟨r,?_,(SplitCompletion.emptyReadContract_proven _ _ a r).1⟩
  change (g.clock owner).atStep (4+pre.length) = (g.clock owner).atStep (3+o.log.length)
  rw [size]
  congr 1
  omega

/-- Construct an owner join from the physical mask operations and the actual idle-loop
    guard reads. Neither a selected activation nor a completed mask is an input. -/
theorem join (o : Observed hist prior owner g next) (stop : SplitReturn.Observation.OwnerCheck hist)
    (path : Schedule.RunWith (SplitReturn.Observation.ExternalStop hist stop o.waitAt)
      t done_ len start owner g.actions s) :
    ∃ j : Join t done_ len start, j.state = s ∧ j.wait.read.event = o.waitAt ∧
      j.wait.read.source.val = prior.length+1+g.actions.length ∧ j.activation.actions = g.actions ∧
      j.answer = SplitReturn.checked stop.stopped (t.locked s.reports).best := by
  obtain ⟨r,event,index⟩ := o.wait path
  let j : Join t done_ len start := ⟨hist,tape prior owner g next,activation prior next g path,r,stop,s,by
    change Schedule.RunWith (SplitReturn.Observation.ExternalStop hist stop r.read.event) t done_ len start owner g.actions s
    rw [event]
    exact path⟩
  exact ⟨j,rfl,event,index,rfl,rfl⟩

/-- A guard on the original modification history can read a redundant reset that did
    not change the bit set. Transport that exact reads-from choice to the effective
    roster consumed by ticket steps. The physical source remains visible in the result. -/
theorem join_raw {ws : List Write} (g : SlotFrameMir.RawMask ws)
    (exit : SlotFrameMir.OwnerExit hist g) (gates : SlotFrameMir.GateHistory hist g)
    (mark : (ws.get g.mark).event = exit.clock.atStep 1)
    (sync : ∀ pre src rest, exit.log = pre ++ .mask src :: rest → ∀ i, InReleaseSequence ws i src →
      hist.otherSync (ws.get i).event (exit.clock.atStep (4+pre.length)))
    (score : AtomicWord.Read scoreWords)
    (scoreSite : score.event = exit.clock.atStep (exit.log.length+5))
    (coherent : SplitRebookingMir.Coherence hist scoreWords score)
    (stop : SplitReturn.Observation.OwnerCheck hist)
    (path : Schedule.RunWith (SplitReturn.Observation.ExternalStop hist stop (exit.clock.atStep (3+exit.log.length)))
      t done_ len start g.owner (SlotFrameMir.RawProjection.actions g) s) :
    ∃ (j : Join t done_ len start) (pre : List (Guard ws)) (src : Fin ws.length),
      exit.log = pre ++ [.mask src] ∧ (ws.get src).value = 0 ∧
      g.mark.val ≤ src.val ∧ src.val ≤ g.mark.val+g.actions.length ∧
      j.state = s ∧ j.wait.read.event = exit.clock.atStep (3+exit.log.length) ∧
      j.wait.read.source.val = (SlotFrameMir.RawProjection.rf g src).val ∧
      j.answer = SplitReturn.checked stop.stopped (t.locked s.reports).best := by
  let a := SlotFrameMir.RawProjection.activation g gates.closed
  have mem := SlotFrameMir.RawProjection.memory exit gates.closed mark sync scoreWords score.source
    ⟨fun i hb => coherent.writeRead i (by simpa only [scoreSite,SplitSchedulerMir.scoreLoad] using hb),
     fun i hb => coherent.readWrite i (by simpa only [scoreSite,SplitSchedulerMir.scoreLoad] using hb)⟩
  obtain ⟨pre,src,rest,site,zero⟩ := SplitSchedulerMir.idle_acquired_zero exit.returned exit.poolOpen
  have last := SplitSchedulerMir.after_zero exit.returned site zero
  subst rest
  let projected := SlotFrameMir.RawProjection.rf g src
  let mappedPre := pre.map (SlotFrameMir.RawProjection.guard g)
  have eval : exit.log.map (SlotFrameMir.RawProjection.guard g) = mappedPre ++ [.mask projected] := by
    simp [site,mappedPre,projected,SlotFrameMir.RawProjection.guard]
  have bounds := (exit.sampled pre src [] site).bounds
  let read : Read (SlotFrameMir.RawProjection.words g) := ⟨exit.clock.atStep (4+mappedPre.length),projected,true⟩
  let r : SplitCompletion.Retirement hist _ a := {
    read := read
    acquire := rfl
    zero := (SlotFrameMir.RawProjection.rf_value g src bounds).trans zero
    afterOwnerBit := by rw [mem.ownerBit]; exact ⟨rfl,exit.clock.advances 1 _ (by omega)⟩
    beforeReuse := mem.maskBeforeReuse mappedPre projected [] eval
    writeRead := mem.maskWR mappedPre projected [] eval
    readWrite := mem.maskRW mappedPre projected [] eval
    synchronize := fun i _ seq => mem.sync mappedPre projected [] eval i seq }
  have same : read.event = exit.clock.atStep (3+exit.log.length) := by
    apply congrArg exit.clock.atStep
    simp [mappedPre,site]
    omega
  let j : Join t done_ len start := ⟨hist,_,a,r,stop,s,by
    change Schedule.RunWith (SplitReturn.Observation.ExternalStop hist stop read.event) t done_ len start g.owner _ s
    rw [same]
    exact path⟩
  exact ⟨j,pre,src,site,zero,bounds.1,bounds.2,rfl,same,rfl,rfl⟩

/-- A later use of the same slot begins after every possible zero source for this return. -/
theorem before_reuse {hist : History} (prior : List Write) (next : List Generation) (g : Generation)
    (path : Retirement.ExecutionWith out check len start owner g.actions s)
    (r : SplitCompletion.Retirement hist (tape prior owner g next) (activation prior next g path))
    (middle : List Write) : r.read.source.val < (prior ++ g.words owner ++ middle).length+1 := by
  have bound := (SplitCompletion.emptyReadContract_proven _ _ _ r).1
  change r.read.source.val = prior.length+1+g.actions.length at bound
  simp
  omega

/-- Any two uses separated by arbitrarily many other generations share the same
    physical tape and have disjoint reads-from indices for every idle-loop mask load. -/
theorem Observed.disjoint_reads (middle tail : List Generation)
    (first : Observed hist prior owner g (middle ++ later :: tail))
    (second : Observed hist (prior ++ g.words owner ++ middle.flatMap (·.words owner)) owner later tail) :
    tape prior owner g (middle ++ later :: tail) =
      tape (prior ++ g.words owner ++ middle.flatMap (·.words owner)) owner later tail ∧
    ∀ src, Guard.mask src ∈ first.log → ∀ dst, Guard.mask dst ∈ second.log → src.val < dst.val := by
  refine ⟨by simp [tape,List.flatMap_append,List.append_assoc],?_⟩
  intro src read dst laterRead
  obtain ⟨a,b,here⟩ := List.append_of_mem read
  obtain ⟨c,d,there⟩ := List.append_of_mem laterRead
  have upper := (first.window a src b here).bounds.2
  have lower := (second.window c dst d there).bounds.1
  change src.val ≤ prior.length+1+g.actions.length at upper
  change (prior ++ g.words owner ++ middle.flatMap (·.words owner)).length+1 ≤ dst.val at lower
  simp only [List.length_append,Generation.length] at lower
  omega

/-- Consecutive invocations at one slot have different physical acquire events, including
    when their returned mask words and search scores are identical. -/
theorem Observed.later_wait (first : Observed hist prior owner g (other :: tail))
    (second : Observed hist (prior ++ g.words owner) owner other tail) :
    SequencedBefore first.waitAt second.waitAt := by
  refine ⟨rfl,Nat.lt_of_lt_of_le (g.advances _ _ ?_) (Nat.le_trans (first.reuse other (by simp)) ?_)⟩
  · omega
  · exact Nat.le_of_lt (other.advances 0 _ (by omega))

end NeuralReversi.PvsParallel.Interleaving.Mask
