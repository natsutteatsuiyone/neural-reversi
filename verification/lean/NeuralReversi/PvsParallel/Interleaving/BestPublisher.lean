module
public import NeuralReversi.PvsParallel.Interleaving.ScoreOrigin

@[expose] public section

namespace NeuralReversi.PvsParallel.Interleaving.BestPublisher

open Threading.AtomicWord (Event Write)
open SplitWorkerMir
open FieldStores (Physical Raw)

structure Store where
  event : Event
  word : Word
  deriving DecidableEq

inductive Packet where
  | stopped
  | live (worker index : Nat) (previous chosen : Word) (stored : Option Store)
  deriving DecidableEq

abbrev Winner := Store × Option Nat

/-- The load of the incumbent best and the optional store occur under the same
    split-point lock. A later report of the same score keeps the earlier writer. -/
def step (old : Winner) : Packet → Option Winner
  | .stopped => some old
  | .live worker index prev chosen stored =>
    if old.1.word ≠ prev then none
    else if prev.toInt < chosen.toInt then
      stored.bind fun s =>
        if s.event.worker = worker ∧ s.word = chosen then some (s,some index) else none
    else if stored.isNone then some old else none

def replay (seed : Store) (rows : List Packet) : Option Winner :=
  rows.foldlM step (seed,none)

def ofWrite (w : Write) : Store := ⟨w.event,FieldStores.low32 w⟩

def emission : Packet → Option Winner
  | .live _ index _ _ (some s) => some (s,some index)
  | _ => none

private theorem step_last {s t : Winner} {p : Packet} (ok : step s p = some t) :
    t = (emission p).getD s := by
  cases p with
  | stopped => simpa [step,emission] using ok.symm
  | live w i prev cand out =>
    cases out <;> by_cases same : s.1.word = prev <;> by_cases strict : prev.toInt < cand.toInt <;>
      simp_all [step,emission] <;> split at ok <;> simp_all

theorem replay_source {rows : List Packet} {s t : Winner} (ok : rows.foldlM step s = some t) :
    t = (rows.filterMap emission).getLast?.getD s := by
  induction rows generalizing s with
  | nil => simpa using ok.symm
  | cons p ps ih =>
    obtain ⟨next,one,more⟩ := Option.bind_eq_some_iff.mp ok
    rw [ih more,step_last one]
    cases h : emission p <;> simp [h,List.getLast?_cons,Option.getD]

def readBest : Physical → Option Word
  | .readBest w => some w
  | _ => none

/-- Project the loads and stores of a single protected helper report. The candidate
    is the instruction-selected scout or wide return. The caller supplies the worker
    which acquired the lock, including for a ticket that made no write. -/
def packet (worker : Nat) (p : Raw) : Option Packet := do
  let stores := (p.ops.filterMap Physical.best?).map ofWrite
  if p.stopped then
    guard (stores.isEmpty && (p.ops.filterMap readBest).isEmpty)
    return .stopped
  let [old] := p.ops.filterMap readBest | none
  guard (stores.length ≤ 1)
  return .live worker p.input.index old (issued p.input).chosen stores.head?

private theorem reads (ops : List Physical) :
    ops.filterMap readBest = (ops.map Physical.value).filterMap (fun | .readBest w => some w | _ => none) := by
  rw [List.filterMap_map]
  congr 1
  funext op
  cases op <;> rfl

private theorem program_reads (i : Input) (s : Locked) (chosen : Word)
    (encode : Option (Fin 64) → BitVec 8) (atIndex : Nat → Option (Fin 64)) :
    ((FieldStores.accesses i s chosen).map (FieldStores.Access.value encode atIndex)).filterMap
        (fun | .readBest w => some w | _ => none) = if s.stopped then [] else [s.best] := by
  cases stop : s.stopped <;> cases root : i.root <;> cases pv : i.pv <;>
    by_cases better : s.best.toInt < chosen.toInt <;> by_cases raise : s.alpha.toInt < chosen.toInt <;>
      by_cases low : chosen.toInt < i.beta.toInt <;>
      simp [FieldStores.accesses,FieldStores.Access.value,stop,root,pv,better,raise,low]

private def candidate (p : Raw) := (issued p.input).chosen

private def writeIf (p : Raw) (old : Word) : Option Word :=
  if p.stopped then none else if old.toInt < (candidate p).toInt then some (candidate p) else none

private theorem effect (i : Input) (s : Locked) (v : Word) :
    (commit i s v).best = if s.stopped then none else if s.best.toInt < v.toInt then some v else none := by
  rw [commit_eq]
  cases st : s.stopped <;> cases pv : i.pv <;>
    by_cases a : s.alpha.toInt < v.toInt <;> by_cases b : s.best.toInt < v.toInt <;>
      by_cases z : v.toInt < i.beta.toInt <;> simp [*]

private theorem operands {p : Raw} (ok : p.decode s pv beta enc atIndex = some r) :
    p.ops.filterMap readBest = (if p.stopped then [] else [s.best]) ∧
    r.update.best = writeIf p s.best := by
  unfold FieldStores.Raw.decode at ok
  split at ok <;> try contradiction
  rename_i body ran
  dsimp only at ok
  split at ok <;> try contradiction
  rename_i ops
  cases ok
  constructor
  · rw [reads,ops.2.2,program_reads]
  · change (commit p.input ⟨p.stopped,s.alpha,s.best⟩ body.chosen).best = writeIf p s.best
    rw [effect,issued_of_search ran]
    rfl

private def latest (old : Winner) (index : Nat) (write : Option Write) : Winner :=
  (write.map (fun w => (ofWrite w,some index))).getD old

private def issuedPacket (worker : Nat) (p : Raw) (best : Word) (write : Option Write) : Packet :=
  if p.stopped then .stopped else .live worker p.input.index best (candidate p) (write.map ofWrite)

/-- Each accepted critical section provides the previous best word exactly when its
    load executes. Its best-score store, if any, identifies a single physical event. -/
private theorem projection {p : Raw} (ok : p.decode s pv beta enc atIndex = some r) :
    packet worker p = some (issuedPacket worker p s.best r.writes.best) := by
  obtain ⟨read,update⟩ := operands ok
  have writes := (p.stores ok).2.1
  have word := (p.observed ok).bestWrite.trans update
  cases stop : p.stopped
  · cases h : r.writes.best <;>
      simp [packet,issuedPacket,stop,read,←writes,h,candidate,guard]
  · have nothing : r.writes.best = none := by
      simpa [writeIf,stop] using word
    simp [packet,issuedPacket,stop,read,←writes,nothing,guard]

private theorem packet_step {p : Raw} (ok : p.decode s pv beta enc atIndex = some r)
    (incumbent : win.1.word = s.best)
    (actor : ∀ w ∈ p.ops.filterMap Physical.best?, w.event.worker = worker) :
    step win (issuedPacket worker p s.best r.writes.best) = some (latest win p.input.index r.writes.best) ∧
      (latest win p.input.index r.writes.best).1.word = (s.after r atIndex).best := by
  have update := (operands ok).2
  have word := (p.observed ok).bestWrite.trans update
  have where_ := (p.stores ok).2.1
  cases h : r.writes.best with
  | none =>
    cases st : p.stopped <;> by_cases gains : s.best.toInt < (candidate p).toInt <;>
      simp_all [latest,step,issuedPacket,FieldStores.Fields.after,writeIf]
  | some w =>
    have who := actor w (by rw [←where_,h]; simp)
    cases st : p.stopped <;> by_cases gains : s.best.toInt < (candidate p).toInt <;>
      simp_all [latest,step,issuedPacket,FieldStores.Fields.after,writeIf,ofWrite]

def stores (rows : List (Nat × Raw)) : List Winner :=
  rows.flatMap fun (_,p) => (p.ops.filterMap Physical.best?).map fun w => (ofWrite w,some p.input.index)

def packets (rows : List (Nat × Raw)) : Option (List Packet) :=
  rows.mapM fun (w,p) => packet w p

private theorem end_append (a b : List Winner) (seed : Winner) :
    (a++b).getLast?.getD seed = b.getLast?.getD (a.getLast?.getD seed) := by
  simp only [List.getLast?_append]
  cases b.getLast? <;> rfl

private theorem accepts (rows : List (Nat × Raw))
    (actors : ∀ row ∈ rows, ∀ w ∈ row.2.ops.filterMap Physical.best?, w.event.worker = row.1)
    (ok : FieldStores.replay pv beta enc atIndex l (rows.map Prod.snd) = some final)
    {held : Winner} (old : held.1.word = l.fields.best) :
    ∃ ps, packets rows = some ps ∧
      ps.foldlM step held = some ((stores rows).getLast?.getD held) := by
  induction rows generalizing l held with
  | nil => exact ⟨[],rfl,rfl⟩
  | cons row rows ih =>
    rcases row with ⟨worker,p⟩
    obtain ⟨r,one,more⟩ := Option.bind_eq_some_iff.mp ok
    have take := packet_step one old (actors (worker,p) (by simp))
    obtain ⟨rest,found,result⟩ := ih (fun row mem => actors row (by simp [mem])) more take.2
    let first := issuedPacket worker p l.fields.best r.writes.best
    refine ⟨first :: rest,?_,?_⟩
    · change (List.mapM _ ((worker,p)::rows)) = _
      rw [List.mapM_cons]
      simp only [projection one]
      change ((packets rows).bind fun tail => some (first :: tail)) = _
      rw [found]
      rfl
    · rw [List.foldlM_cons]
      change (step held first).bind _ = _
      rw [take.1]
      change rest.foldlM step _ = _
      rw [result]
      apply congrArg some
      change _ = ((_ ++ stores rows).getLast?.getD held)
      rw [end_append]
      congr 1
      change latest held p.input.index r.writes.best =
        ((p.ops.filterMap Physical.best?).map (fun w => (ofWrite w,some p.input.index))).getLast?.getD held
      rw [←(p.stores one).2.1]
      cases r.writes.best <;> rfl

open SplitSchedulerMir.WorkerFields (locate located)

def lockWorker (t : NativeSplit.Tickets rc pv beta) (rank : Nat) :=
  (t.journal[rank]?.map (·.worker)).getD 0

def rows (t : NativeSplit.Tickets rc pv beta) : List (Nat × Raw) :=
  (IteratorFields.chronological t.state.writes t.journal t.stream t.workers).map fun c =>
    (lockWorker t c.order,IteratorFields.raw t.traffic c)

private theorem actors (t : NativeSplit.Tickets rc pv beta) :
    ∀ r ∈ rows t, ∀ w ∈ r.2.ops.filterMap Physical.best?, w.event.worker = r.1 := by
  intro row here w written
  obtain ⟨c,_,rfl⟩ := List.mem_map.mp here
  obtain ⟨op,mem,acceptedOp⟩ := List.mem_filterMap.mp written
  have isBest : op = .writeBest w := by cases op <;> simp_all [Physical.best?]
  change op ∈ t.traffic c.order at mem
  change w.event.worker = lockWorker t c.order
  unfold NativeSplit.Tickets.traffic SplitSchedulerMir.WorkerFields.traffic at mem
  cases j : t.journal[c.order]? with
  | none => simp [j] at mem
  | some obs =>
    cases a : locate t.dispatches obs with
    | none => simp [j,a] at mem
    | some addr =>
      have here : op ∈ addr.traffic := by simpa [j,a] using mem
      obtain ⟨_,who,valid⟩ := located a
      have actual := (addr.recorded_store valid here (Or.inr (Or.inl isBest))).1
      change addr.dispatch.worker = obs.worker at who
      exact actual.trans (by simpa [lockWorker,j] using who)

private theorem projected (t : NativeSplit.Tickets rc pv beta) : (rows t).map Prod.snd = t.calls := by
  simp [rows,NativeSplit.Tickets.calls,List.map_map,Function.comp_def]

theorem native (t : NativeSplit.Tickets rc pv beta) :
    ∃ ps, packets (rows t) = some ps ∧ replay (ofWrite t.b) ps =
      some ((stores (rows t)).getLast?.getD (ofWrite t.b,none)) := by
  have ok := IteratorFields.field_replay t.replay
  change FieldStores.replay _ _ _ _ _ t.calls = some t.final at ok
  rw [←projected] at ok
  exact accepts _ (actors t) ok t.seedBest

private theorem words (t : NativeSplit.Tickets rc pv beta) :
    (stores (rows t)).map Prod.fst = (ScoreOrigin.improvements t).map ofWrite := by
  simp [stores,rows,ScoreOrigin.improvements,NativeSplit.Tickets.calls,List.map_flatMap,
    List.flatMap_map,List.map_map,Function.comp_def]

private theorem lastAt (pre updates post : List α) (seed : α) :
    (pre ++ ((seed :: updates) ++ post))[pre.length+updates.length]? =
      some (updates.getLast?.getD seed) := by
  rw [List.getElem?_append_right (Nat.le_add_right ..),Nat.add_sub_cancel_left,
    List.getElem?_append_left (by simp)]
  have last := List.getLast?_eq_getElem? (l := seed :: updates)
  simp only [List.length_cons,Nat.add_sub_cancel] at last
  rw [←last,List.getLast?_cons]

def scoreWords {t : NativeSplit.Tickets rc pv beta} (c : ScoreOrigin.Capture t root cut depth child) :=
  c.owner.bPre ++ t.final.best ++ SlotFrameMir.RawProjection.subsequent c.owner.exit (c.generation.owner,c.bankIndex) .score c.owner.later

/-- A native ticket lock log determines the publisher's worker, instruction and move
    index (or the owner seed). The owner load's genuine reads-from source is that event,
    including when the slot's earlier or later occupancies wrote the same score. -/
def Contract := ∀ {rc pv beta} (t : NativeSplit.Tickets rc pv beta) (root cut : Bool) (depth : Nat)
    (child : PvsMoveLoop.ChildRelation Nat) (c : ScoreOrigin.Capture t root cut depth child),
    ∃ ps k, packets (rows t) = some ps ∧
      replay (ofWrite t.b) ps = some (ofWrite ((scoreWords c).get c.owner.source),k) ∧
      (ofWrite ((scoreWords c).get c.owner.source),k) = (stores (rows t)).getLast?.getD (ofWrite t.b,none)

theorem identifies : Contract := by
  intro rc pv beta t root cut depth child c
  obtain ⟨ps,generated,ran⟩ := native t
  let winner := (stores (rows t)).getLast?.getD (ofWrite t.b,none)
  have index := (ScoreOrigin.contract_proven t root cut depth child c).2.1
  have modification := (FieldStores.modification_orders t.calls (IteratorFields.field_replay t.replay)).2.1
  change t.final.best = [t.b] ++ ScoreOrigin.improvements t at modification
  have atSource : (scoreWords c)[c.owner.source.val]? = some ((ScoreOrigin.improvements t).getLast?.getD t.b) := by
    rw [index]
    simpa only [scoreWords,modification,List.append_assoc,List.singleton_append] using
      lastAt c.owner.bPre (ScoreOrigin.improvements t) _ t.b
  have physical : (scoreWords c).get c.owner.source = (ScoreOrigin.improvements t).getLast?.getD t.b :=
    Option.some.inj ((List.getElem?_eq_getElem c.owner.source.isLt).symm.trans atSource)
  have fields := congrArg List.getLast? (words t)
  simp only [List.getLast?_map] at fields
  have same : winner.1 = ofWrite ((scoreWords c).get c.owner.source) := by
    apply Eq.trans _ (congrArg ofWrite physical).symm
    unfold winner
    cases src : (stores (rows t)).getLast? <;> cases dst : (ScoreOrigin.improvements t).getLast? <;>
      simp_all
  refine ⟨ps,winner.2,generated,?_,?_⟩
  · rw [←same]
    exact ran
  · rw [←same]

def publisher (t : NativeSplit.Tickets rc pv beta) : Option Winner := do
  let ps ← packets (rows t)
  replay (ofWrite t.b) ps

theorem returnWord {t : NativeSplit.Tickets rc pv beta} (c : ScoreOrigin.Capture t root cut depth child) :
    (publisher t).map (fun result => result.1.word.toInt) =
      some (Threading.AtomicWord.signed32At c.frame.physical.scores c.frame.physical.load.source) := by
  obtain ⟨ps,k,decoded,selected,_⟩ := identifies t root cut depth child c
  have physical := (ScoreOrigin.contract_proven t root cut depth child c).2.2.1
  have frame := (ScoreReuse.contract_proven _ _ _ _ _ c.frame).2
  rw [publisher,decoded]
  change (replay (ofWrite t.b) ps).map _ = _
  rw [selected]
  exact congrArg some (physical.trans frame.symm)

end NeuralReversi.PvsParallel.Interleaving.BestPublisher
