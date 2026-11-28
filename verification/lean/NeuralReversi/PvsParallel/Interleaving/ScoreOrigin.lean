module
public import NeuralReversi.PvsParallel.Interleaving.ScoreReuse
public import NeuralReversi.PvsParallel.IteratorFields

@[expose] public section

namespace NeuralReversi.PvsParallel.Interleaving.ScoreOrigin

open Threading
open AtomicWord (Write Read Event SequencedBefore)
open SlotFrameMir.RawProjection (PhysicalOwner)
open SlotFrameMir (WorkerStore RawMask)
open SplitWorkerMir
open SplitSchedulerMir
open TranspositionTable.Packing (encodeBestMove)

def improvements (t : NativeSplit.Tickets rc pv beta) : List Write :=
  t.calls.flatMap (fun call => call.ops.filterMap FieldStores.Physical.best?)

def scores {g : RawMask mask} {t : NativeSplit.Tickets rc pv beta} (p : PhysicalOwner g hist time slot t) : ScoreReuse.Scores :=
  ⟨p.bPre,t.initial.best,improvements t,p.later⟩

/-- The raw seed is Rust's relaxed AtomicI32 store. Its value and source PC come
    from the field replay and the physical owner initializer, respectively. -/
structure Seed (w : Write) : Prop where
  i32 : w.value = (w.value.setWidth 32).zeroExtend 64
  relaxed : w.release = false
  store : w.rmw = false

private theorem sameWord {a b : Write} (e : a.event = b.event) (v : a.value = b.value)
    (rel : a.release = b.release) (rmw : a.rmw = b.rmw) : a = b := by
  cases a; cases b; simp_all

theorem current {g : RawMask mask} {t : NativeSplit.Tickets rc pv beta}
    (p : PhysicalOwner g hist time slot t) (seed : Seed t.b) :
    (scores p).current p.exit.timeline = t.final.best := by
  have orders := (FieldStores.modification_orders t.calls (IteratorFields.field_replay t.replay)).2.1
  change t.final.best = [t.b] ++ improvements t at orders
  rw [orders]
  change (scores p).initial p.exit.timeline :: improvements t = t.b :: improvements t
  apply congrArg (fun w => w :: improvements t)
  apply sameWord p.bestSeed.symm _ seed.relaxed.symm seed.store.symm
  change t.initial.best.zeroExtend 64 = t.b.value
  rw [←t.seedBest]
  exact seed.i32.symm

theorem publication {g : RawMask mask} {t : NativeSplit.Tickets rc pv beta}
    (p : PhysicalOwner g hist time slot t) (w : Write) (member : w ∈ (scores p).improvements) :
    Nonempty (WorkerStore hist time g slot w.event) := by
  obtain ⟨call,here,stored⟩ := List.mem_flatMap.mp member
  obtain ⟨op,inside,selected⟩ := List.mem_filterMap.mp stored
  have wrote : op = .writeBest w := by cases op <;> simp_all [FieldStores.Physical.best?]
  obtain ⟨c,_,rfl⟩ := List.mem_map.mp here
  change op ∈ t.traffic c.order at inside
  obtain ⟨obs,addr,found,fields⟩ := WorkerFields.traffic_source inside
  have located : op ∈ addr.traffic := fields ▸ inside
  obtain ⟨_,reset,_,_,same⟩ := p.reset found (addr.needs_reset_of_access located)
  have first := addr.store_before_unlock located (Or.inr (Or.inl wrote))
  have second := addr.unlock_before_reset
  rw [←same] at second
  exact ⟨⟨reset,first.1.trans second.1,Nat.lt_trans first.2 second.2⟩⟩

theorem stores {g : RawMask mask} {t : NativeSplit.Tickets rc pv beta}
    (p : PhysicalOwner g hist time slot t) (seed : Seed t.b) :
    SplitWindow.BestStores t.initial.best.toInt (t.final.reports.map SplitResult.Report.score)
      ((scores p).current p.exit.timeline) := by
  rw [current p seed]
  exact (FieldStores.replay_preserves t.calls
    (FieldStores.seed t.initial t.a t.b t.m t.seedAlpha t.seedBest t.seedMove)
    (IteratorFields.field_replay t.replay)).bestWrites


private def emit (packet : FieldStores.Raw) : Option SplitResult.Report :=
  if packet.stopped then none else some (Worker.report packet.input)

private theorem field_report (hit : packet.decode fields pv beta encode atIndex = some r) :
    r.report = emit packet := by
  unfold FieldStores.Raw.decode at hit
  split at hit <;> try contradiction
  rename_i body ran
  dsimp only at hit
  split at hit <;> try contradiction
  cases hit
  simp only [FieldStores.Record.report,emit,Worker.report,issued_of_search ran]

private theorem field_reports (xs : List FieldStores.Raw)
    (ok : FieldStores.replay pv beta encode atIndex initial xs = some final) :
    final.reports = initial.reports ++ xs.filterMap emit := by
  induction xs generalizing initial with
  | nil => cases ok; simp
  | cons packet tail ih =>
    obtain ⟨r,one,after⟩ := Option.bind_eq_some_iff.mp ok
    rw [ih after]
    simp only [FieldStores.Log.append,field_report one]
    cases post : emit packet <;> simp [post]

deriving instance DecidableEq for Input

/-- The fresh alpha and wide child word exist only on the re-search path. Both
    representations may use different placeholders when that branch is skipped. -/
def operands (i : Input) := if retry i then i else {i with fresh := 0,wide := 0}

private theorem operands_report (i : Input) : Worker.report (operands i) = Worker.report i := by
  cases again : retry i with
  | true => simp [operands,again]
  | false =>
    have other : retry {i with fresh := 0,wide := 0} = false := again
    simp only [operands,again,Bool.false_eq_true,ite_false,Worker.report,issued,other,selected]

def actor (t : NativeSplit.Tickets rc pv beta) (index : Nat) : Option Nat :=
  (t.state.writes.find? (fun op => op.old.toNat == index)).map (·.worker)

def poll (t : NativeSplit.Tickets rc pv beta) (s : Tickets.State t.width) : Tickets.Action → Bool
  | .take worker => actor t s.current.toNat == some worker
  | .empty .. => true
  | _ => false

structure Source (hist : CutoffVisibility.History) (alphas : List Write) (task : Schedule.Task) where
  Observation : List Tickets.Post → Type
  commit : ∀ ps, Observation ps → RawSchedule.Commit hist alphas task ps

def Source.raw (hist : CutoffVisibility.History) (alphas : List Write) (task : Schedule.Task) : Source hist alphas task :=
  ⟨RawSchedule.Commit hist alphas task,fun _ c => c⟩

/-- The same operation both advances the worker schedule and consumes its physical
    lock packet. The packet provides raw query operands and stop bit; its score row
    is computed by the worker code. A stopped packet advances the journal alone. -/
inductive RunWith {rc : List Move.Move} (t : NativeSplit.Tickets rc pv beta) (m : NativeStopReturns.Memory)
    (check : SplitReturn.Observation.OwnerCheck m.history) (wait : Event) (slot : Nat) (alphas : List Write)
    (task : Schedule.Task) (child : PvsMoveLoop.ChildRelation Nat) (owner : SplitCompletion.Worker)
    (complete : List SplitCompletion.Action) (source : Source m.history alphas task) :
    List SplitCompletion.Action → List FieldStores.Raw → Tickets.State t.width → Prop where
  | initial : RunWith t m check wait slot alphas task child owner complete source [] [] (Tickets.initial t.start owner.val)
  | poll : RunWith t m check wait slot alphas task child owner complete source mask locks s →
      Tickets.Step rc.length s event next → poll t s event = true →
      RunWith t m check wait slot alphas task child owner complete source mask locks next
  | book (w : SplitCompletion.Worker) (e : Event) :
      RunWith t m check wait slot alphas task child owner complete source mask locks s →
      Tickets.Step rc.length s (.book w.val) next →
      RunWith t m check wait slot alphas task child owner complete source (mask ++ [.book w e]) locks next
  | depart (w : SplitCompletion.Worker) (e : Event) :
      RunWith t m check wait slot alphas task child owner complete source mask locks s →
      Tickets.Step rc.length s (.leave w.val) next →
      RunWith t m check wait slot alphas task child owner complete source (mask ++ [.finish w e]) locks next
  | report (packet : FieldStores.Raw) (obs : source.Observation s.reports) :
      RunWith t m check wait slot alphas task child owner complete source mask locks s →
      actor t (source.commit _ obs).index = some w → operands packet.input = operands (source.commit _ obs).input → packet.stopped = false → (source.commit _ obs).calls child →
      Tickets.Step rc.length s (.report w (source.commit _ obs).post) next →
      RunWith t m check wait slot alphas task child owner complete source mask (locks ++ [packet]) next
  | stop (packet : FieldStores.Raw) (stamp : Event) :
      RunWith t m check wait slot alphas task child owner complete source mask locks s →
      actor t packet.input.index = some stamp.worker → packet.stopped = true →
      s.pending = left ++ (stamp.worker,packet.input.index) :: right →
      StopSchedule.Evidence m check wait slot s.reports stamp → StopSchedule.Release complete stamp →
      RunWith t m check wait slot alphas task child owner complete source mask (locks ++ [packet])
        (StopSchedule.discard s left right stamp.worker packet.input.index)

abbrev Run (t : NativeSplit.Tickets rc pv beta) (m : NativeStopReturns.Memory)
    (check : SplitReturn.Observation.OwnerCheck m.history) (wait : Event) (slot : Nat) (alphas : List Write)
    (task : Schedule.Task) (child : PvsMoveLoop.ChildRelation Nat) (owner : SplitCompletion.Worker)
    (complete : List SplitCompletion.Action) := RunWith t m check wait slot alphas task child owner complete (.raw ..)

/-- Each report is identified by its position in the lock log and the worker
    who acquired this iterator index. Stopped packets have no child obligation. -/
def ReportCalls (t : NativeSplit.Tickets rc pv beta) (source : Source hist alphas task)
    (child : PvsMoveLoop.ChildRelation Nat) (packets : List FieldStores.Raw) : Prop :=
  ∀ pre packet post, packets = pre ++ packet :: post →
    ∀ ps (obs : source.Observation ps) worker,
    actor t (source.commit ps obs).index = some worker →
    operands packet.input = operands (source.commit ps obs).input → packet.stopped = false →
    (source.commit ps obs).calls child

theorem ReportCalls.prefix (all : ReportCalls t src child (packets ++ more)) :
    ReportCalls t src child packets := by
  intro pre packet post cut
  apply all pre packet (post ++ more)
  simp [cut]

theorem RunWith.withReports (h : RunWith t m check wait slot alphas task old owner complete src mask locks s)
    (reports : ReportCalls t src child locks) :
    RunWith t m check wait slot alphas task child owner complete src mask locks s := by
  induction h with
  | initial => exact .initial
  | poll _ step rule ih => exact .poll (ih reports) step rule
  | book w e _ step ih => exact .book w e (ih reports) step
  | depart w e _ step ih => exact .depart w e (ih reports) step
  | report packet obs _ acquired same live _ step ih =>
    exact .report packet obs (ih reports.prefix) acquired same live (reports _ packet [] rfl _ obs _ acquired same live) step
  | stop packet e _ who stopped held sample released ih =>
    exact .stop packet e (ih reports.prefix) who stopped held sample released

theorem RunWith.toRun (h : RunWith t m check wait slot alphas task child owner complete source mask locks s) :
    Run t m check wait slot alphas task child owner complete mask locks s := by
  induction h with
  | initial => exact .initial
  | poll _ step rule ih => exact .poll ih step rule
  | book w e _ step ih => exact .book w e ih step
  | depart w e _ step ih => exact .depart w e ih step
  | report packet obs _ ticket input live calls step ih =>
    exact .report packet (source.commit _ obs) ih ticket input live calls step
  | stop packet e _ owner stopped held sample rel ih => exact .stop packet e ih owner stopped held sample rel

theorem Run.scored {t : NativeSplit.Tickets rc pv beta} {s : Tickets.State t.width} (history : Run t m check wait slot alphas task child owner complete mask locks s) :
    StopSchedule.Run m check wait slot (fun before post => ∃ c : RawSchedule.Commit m.history alphas task before.reports,
      post = c.post ∧ c.calls child) rc.length t.start owner complete mask s := by
  induction history with
  | initial => exact .initial
  | @poll _ _ _ e _ _ step quiet ih =>
    cases e <;> first | contradiction | exact .work ih step rfl (by simp [Tickets.Action.posts])
  | book w e _ step ih => exact .book w e ih step
  | depart w e _ step ih => exact .finish w e ih step
  | report packet c _ _ _ _ calls step ih =>
    exact .work ih step rfl (by simp only [Tickets.Action.posts,List.mem_singleton]; rintro p rfl; exact ⟨c,rfl,calls⟩)
  | stop packet stamp _ _ _ held sample release ih => exact .stop stamp _ ih held sample release

private theorem Run.rows (history : Run t m check wait slot alphas task child owner complete mask locks s) :
    Schedule.reports s = locks.filterMap emit := by
  induction history with
  | initial => rfl
  | @poll _ _ _ e _ _ step quiet ih =>
    unfold Schedule.reports
    rw [step.reports_eq]
    cases e <;> first | contradiction | simpa [Schedule.reports,Tickets.Action.posts] using ih
  | book _ _ _ step ih | depart _ _ _ step ih =>
    simpa [Schedule.reports,step.reports_eq,Tickets.Action.posts] using ih
  | report packet c _ _ input stop _ step ih =>
    unfold Schedule.reports at ih ⊢
    rw [step.reports_eq]
    have selected : Worker.report packet.input = Worker.report c.input :=
      (operands_report _).symm.trans ((congrArg Worker.report input).trans (operands_report _))
    simp [Tickets.Action.posts,ih,emit,selected,stop,RawSchedule.Commit.post,Source.raw]
  | stop packet _ _ _ stopped _ _ _ ih => simpa [Schedule.reports,StopSchedule.discard,emit,stopped] using ih

theorem Run.reports {t : NativeSplit.Tickets rc pv beta} {s : Tickets.State t.width}
    (history : Run t m check wait slot alphas task child owner complete mask t.calls s) :
    Schedule.reports s = t.final.reports := by
  have order := field_reports t.calls (IteratorFields.field_replay t.replay)
  simpa [FieldStores.initialLog,history.rows] using order.symm

def task (t : NativeSplit.Tickets rc pv beta) (root cut : Bool) (depth : Nat) : Schedule.Task :=
  ⟨pv,root,cut,depth,beta,t.initial.state⟩

abbrev original {t : NativeSplit.Tickets rc pv beta} {g : RawMask mask}
    (p : PhysicalOwner g hist time slot t) :=
  p.bPre ++ t.final.best ++ SlotFrameMir.RawProjection.subsequent p.exit slot .score p.later

theorem words {t : NativeSplit.Tickets rc pv beta} {g : RawMask mask}
    (p : PhysicalOwner g hist time slot t) (seed : Seed t.b) :
    (scores p).words p.exit.timeline slot.2 p.exit.log.length = original p := by
  unfold ScoreReuse.Scores.words
  rw [current p seed]
  rfl

def copyRead {xs ys : List Write} (eq : xs = ys) (r : Read xs) : Read ys :=
  ⟨r.event,⟨r.source.val,eq ▸ r.source.isLt⟩,r.acquire⟩

theorem copy_coherent {xs ys : List Write} (eq : xs = ys) (r : Read xs)
    (mem : SplitRebookingMir.Coherence h xs r) : SplitRebookingMir.Coherence h ys (copyRead eq r) := by
  subst ys
  exact mem

private theorem copy_score {xs ys : List Write} (eq : xs = ys) (r : Read xs) :
    AtomicWord.signed32At ys (copyRead eq r).source = AtomicWord.signed32At xs r.source := by
  subst ys
  rfl

structure Capture (t : NativeSplit.Tickets rc pv beta) (root cut : Bool) (depth : Nat) (child : PvsMoveLoop.ChildRelation Nat) where
  memory : NativeStopReturns.Memory
  mask : List Write
  generation : RawMask mask
  clock : Event → Nat
  bankIndex : Nat
  flagAddress : SlotFrameMir.Slot → Nat
  owner : PhysicalOwner generation memory.history clock (generation.owner,bankIndex) t
  seed : Seed t.b
  alphas : List Write
  state : Tickets.State t.width
  run : Run t memory owner.check (owner.exit.clock.atStep (3+owner.exit.log.length))
    (flagAddress (generation.owner,bankIndex)) alphas (task t root cut depth) child generation.owner
    (SlotFrameMir.RawProjection.actions generation) (SlotFrameMir.RawProjection.actions generation) t.calls state

def Capture.frame {t : NativeSplit.Tickets rc pv beta} (c : Capture t root cut depth child) :
    ScoreReuse.Frame (task t root cut depth) child rc.length t.start where
  memory := c.memory
  mask := c.mask
  generation := c.generation
  owner := c.owner.exit
  gates := c.owner.gates
  mark := c.owner.mark
  sync := c.owner.sync
  stop := c.owner.check
  bankIndex := c.bankIndex
  flagAddress := c.flagAddress
  alphas := c.alphas
  state := c.state
  run := c.run.scored
  scores := scores c.owner
  load := copyRead (words c.owner c.seed).symm (scoreLoad _ c.owner.source c.owner.exit.clock c.owner.exit.log)
  site := rfl
  stores := by rw [c.run.reports]; exact ScoreOrigin.stores c.owner c.seed
  time := c.clock
  causal := c.owner.causal
  publication := publication c.owner
  coherent := copy_coherent _ _ c.owner.coherence

/-- The physical owner returns the specific seed or last protected best store. All
    numeric rows, store origins, and the shared interleaving come from the same lock packets. -/
def Contract : Prop := ∀ {rc pv beta} (t : NativeSplit.Tickets rc pv beta)
    (root cut : Bool) (depth : Nat) (child : PvsMoveLoop.ChildRelation Nat) (c : Capture t root cut depth child),
    Schedule.reports c.state = t.final.reports ∧
    c.owner.source.val = c.owner.bPre.length + (improvements t).length ∧
    AtomicWord.signed32At (original c.owner) c.owner.source = ((task t root cut depth).locked c.state.reports).best ∧
    ∀ w ∈ improvements t, Nonempty (WorkerStore c.memory.history c.clock c.generation (c.generation.owner,c.bankIndex) w.event)

theorem contract_proven : Contract := by
  intro rc pv beta t root cut depth child c
  obtain ⟨idx,value⟩ := ScoreReuse.contract_proven (task t root cut depth) child rc.length t.width t.start c.frame
  refine ⟨c.run.reports,idx,?_,publication c.owner⟩
  exact (copy_score (words c.owner c.seed).symm (scoreLoad _ c.owner.source c.owner.exit.clock c.owner.exit.log)).symm.trans value

end NeuralReversi.PvsParallel.Interleaving.ScoreOrigin
