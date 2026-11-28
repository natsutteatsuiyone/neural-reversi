module
public import NeuralReversi.PvsParallel.IteratorFrames
public import NeuralReversi.PvsParallel.SplitWorkerMir.FieldTrace

@[expose] public section

namespace NeuralReversi.PvsParallel.IteratorFields

open MoveList.ConcurrentTrace (Write State forWorker)
open SplitWorkerMir
open SplitWorkerMir.Iterator (LockedCall WorkerResult)
open Threading.SplitCompletion (Worker)
open Threading.SplitResult (Report)
open IteratorFrames (Observation source ranked Posted)
open NativeFrames (Record)

private def emit (c : LockedCall) : Option (Nat × Report) := do
  let b ← search c.ticket.input
  let r ← recordLocked c.ticket.input b c.ticket.locked
  return (c.order,r)

private theorem emit_rank (h : emit c = some p) : p.1 = c.order := by
  obtain ⟨_,_,rest⟩ := Option.bind_eq_some_iff.mp h
  obtain ⟨_,_,same⟩ := Option.bind_eq_some_iff.mp rest
  cases same
  rfl

private theorem retired (calls : List LockedCall) (ok : Iterator.retire calls = some result) :
    calls.filterMap emit = result.published := by
  induction calls generalizing result with
  | nil => cases ok; rfl
  | cons c tail ih =>
    unfold Iterator.retire at ok
    obtain ⟨b,body,remaining⟩ := Option.bind_eq_some_iff.mp ok
    dsimp only at remaining
    cases exit : (commit c.ticket.input c.ticket.locked b.chosen).exit with
    | stopped | cutoff =>
      cases tail <;> simp [exit] at remaining
      subst result
      simp [emit,body,recordLocked,exit]
    | again =>
      rw [exit] at remaining
      obtain ⟨next,ran,equal⟩ := Option.bind_eq_some_iff.mp remaining
      cases equal
      simp [emit,body,recordLocked,exit,ih ran]

def claims (ops : List (Write w)) (journal : List Observation) (stream : List Record) (i : Worker) :=
  ((forWorker ops i.val).mapM (Iterator.acquired (source journal stream))).getD []

/-- Each worker contributes its retired tickets, including a ticket cancelled under the lock. -/
def chronological (ops : List (Write w)) (journal : List Observation) (stream : List Record)
    (workers : List Worker) : List LockedCall :=
  (workers.flatMap (claims ops journal stream)).mergeSort (fun a b => decide (a.order ≤ b.order))

theorem worker_calls (ok : Iterator.worker (source journal stream) ops i.val = some result) (here : i ∈ workers) :
    ∃ cs, Iterator.retire cs = some result ∧ ∀ c ∈ cs, c ∈ chronological ops journal stream workers := by
  obtain ⟨cs,mapped,returned⟩ := Option.bind_eq_some_iff.mp ok
  refine ⟨cs,returned,fun c member => (List.mergeSort_perm _ _).mem_iff.mpr ?_⟩
  exact List.mem_flatMap.mpr ⟨i,here,by simpa [claims,mapped] using member⟩

private theorem reported_perm (ops : List (Write w)) (workers : List Worker) (out : Worker → WorkerResult)
    (ran : ∀ i ∈ workers, Iterator.worker (source journal stream) ops i.val = some (out i)) :
    ((chronological ops journal stream workers).filterMap emit).Perm (ranked workers out) := by
  have per : ∀ i ∈ workers, (claims ops journal stream i).filterMap emit = (out i).published := by
    intro i inside
    obtain ⟨calls,mapped,done⟩ := Option.bind_eq_some_iff.mp (ran i inside)
    simpa [claims,mapped] using retired calls done
  have grouped : (workers.flatMap (claims ops journal stream)).filterMap emit =
      workers.flatMap (fun i => (out i).published) := by
    rw [List.filterMap_flatMap]
    simp only [List.flatMap_def]
    rw [List.map_congr_left per]
  exact ((List.mergeSort_perm _ _).filterMap emit).trans (grouped ▸ (List.mergeSort_perm _ _).symm)

private def byRank (a b : Nat × Report) := decide (a.1 ≤ b.1)

private theorem ordered_reports (xs : List LockedCall)
    (strict : xs.Pairwise (fun a b => a.order < b.order)) :
    (xs.filterMap emit).Pairwise (fun a b => a.1 < b.1) := by
  exact strict.filterMap emit fun _ _ lt _ ah _ bh => (emit_rank ah).symm ▸ (emit_rank bh).symm ▸ lt

private theorem one_rank {xs : List (Nat × Report)} (strict : xs.Pairwise (fun a b => a.1 < b.1))
    (ha : a ∈ xs) (hb : b ∈ xs) (same : a.1 = b.1) : a = b := by
  induction xs with
  | nil => contradiction
  | cons head tail ih =>
    rcases List.mem_cons.mp ha with eq | rest
    · subst a
      rcases List.mem_cons.mp hb with eq | later
      · exact eq.symm
      · have lt := List.rel_of_pairwise_cons strict later
        omega
    · rcases List.mem_cons.mp hb with eq | later
      · subst b
        have lt := List.rel_of_pairwise_cons strict rest
        omega
      · exact ih strict.tail rest later

private theorem strict_covered (xs : List LockedCall) (keys : xs.map LockedCall.order = List.range n) :
    xs.Pairwise (fun a b => a.order < b.order) := by
  apply List.pairwise_map.mp
  rw [keys]
  exact List.pairwise_lt_range

private theorem both_ordered (ops : List (Write w)) (workers : List Worker) (out : Worker → WorkerResult)
    (ran : ∀ i ∈ workers, Iterator.worker (source journal stream) ops i.val = some (out i))
    (keys : (chronological ops journal stream workers).map LockedCall.order = List.range journal.length) :
    (chronological ops journal stream workers).filterMap emit = ranked workers out := by
  have perm := reported_perm ops workers out ran
  have strong := ordered_reports _ (strict_covered _ keys)
  apply perm.eq_of_pairwise (le := fun a b => a.1 ≤ b.1)
  · intro a b am bm ab ba
    exact one_rank strong am (perm.symm.subset bm) (Nat.le_antisymm ab ba)
  · exact strong.imp Nat.le_of_lt
  · have sorted := List.pairwise_mergeSort (le := byRank)
          (by intro a b c x y; simpa [byRank] using Nat.le_trans (of_decide_eq_true x) (of_decide_eq_true y))
          (by intro a b; simp [byRank,Nat.le_total]) (workers.flatMap (fun i => (out i).published))
    change ((workers.flatMap (fun i => (out i).published)).mergeSort byRank).Pairwise _
    exact sorted.imp (fun h => of_decide_eq_true h)

def atIndex (rc : List Move.Move) (i : Nat) : Option (Fin 64) := (rc[i]?).map Move.Move.sq

def raw (traffic : Nat → List FieldStores.Physical) (c : LockedCall) : FieldStores.Raw :=
  ⟨c.ticket.input,c.ticket.locked.stopped,traffic c.order⟩

def operands (traffic : Nat → List FieldStores.Physical) (encode : Option (Fin 64) → BitVec 8)
    (rc : List Move.Move) (c : LockedCall) : Bool :=
  match search c.ticket.input with
  | none => false
  | some b => decide ((traffic c.order).map FieldStores.Physical.value =
      (FieldStores.accesses c.ticket.input c.ticket.locked b.chosen).map (FieldStores.Access.value encode (atIndex rc)))

/-- Check the physical critical-section ordinals before replaying. All operand packets are
    indexed by that same ordinal; a stopped section remains in the stream. -/
def replay (rc : List Move.Move) (ops : List (Write w)) (journal : List Observation) (stream : List Record)
    (workers : List Worker) (traffic : Nat → List FieldStores.Physical) (pv : Bool) (beta : Int)
    (encode : Option (Fin 64) → BitVec 8) (initial : FieldStores.Log) : Option FieldStores.Log :=
  let calls := chronological ops journal stream workers
  let retired := workers.all (fun i => (Iterator.worker (source journal stream) ops i.val).isSome)
  if retired ∧ calls.map LockedCall.order = List.range journal.length ∧ calls.all (operands traffic encode rc) then
    FieldStores.replay pv beta encode (atIndex rc) initial (calls.map (raw traffic))
  else none

def results (ops : List (Write w)) (journal : List Observation) (stream : List Record) (i : Worker) : WorkerResult :=
  (Iterator.worker (source journal stream) ops i.val).getD ⟨⟨[],[],emptyExit⟩,[]⟩

theorem worker_returns (ok : replay rc ops journal stream workers traffic pv beta encode initial = some final)
    (i : Worker) (part : i ∈ workers) :
    Iterator.worker (source journal stream) ops i.val = some (results ops journal stream i) := by
  unfold replay at ok
  dsimp only at ok
  split at ok <;> try contradiction
  rename_i good
  have existsCall := List.all_eq_true.mp good.1 i part
  cases got : Iterator.worker (source journal stream) ops i.val <;> simp_all [results]

private theorem accepted (ok : replay rc ops journal stream workers traffic pv beta encode initial = some final) :
    let calls := chronological ops journal stream workers
    calls.map LockedCall.order = List.range journal.length ∧ calls.all (operands traffic encode rc) ∧
      FieldStores.replay pv beta encode (atIndex rc) initial (calls.map (raw traffic)) = some final := by
  unfold replay at ok
  dsimp only at ok
  split at ok
  next h => exact ⟨h.2.1,h.2.2,ok⟩
  next => contradiction

theorem ranks (ok : replay rc ops journal stream workers traffic pv beta encode initial = some final) :
    (chronological ops journal stream workers).map LockedCall.order = List.range journal.length :=
  (accepted ok).1

theorem field_replay (ok : replay rc ops journal stream workers traffic pv beta encode initial = some final) :
    FieldStores.replay pv beta encode (atIndex rc) initial
      ((chronological ops journal stream workers).map (raw traffic)) = some final :=
  (accepted ok).2.2

theorem traffic_at (ok : replay rc ops journal stream workers traffic pv beta encode initial = some final)
    (here : c ∈ chronological ops journal stream workers) (body : search c.ticket.input = some b) :
    (traffic c.order).map FieldStores.Physical.value = (FieldStores.accesses c.ticket.input c.ticket.locked b.chosen).map
      (FieldStores.Access.value encode (atIndex rc)) := by
  simpa [operands,body] using List.all_eq_true.mp (accepted ok).2.1 c here

private theorem mapped_at (xs : List A) (f : A → Option B) (ok : xs.mapM f = some ys)
    (mem : x ∈ xs) (value : f x = some y) : y ∈ ys := by
  induction xs generalizing ys with
  | nil => contradiction
  | cons a xs ih =>
    rw [List.mapM_cons] at ok
    obtain ⟨v, first, rest⟩ := Option.bind_eq_some_iff.mp ok
    obtain ⟨vs, tail, eq⟩ := Option.bind_eq_some_iff.mp rest
    cases eq
    rcases List.mem_cons.mp mem with rfl | later
    · exact List.mem_cons.mpr (Or.inl (Option.some.inj (value.symm.trans first)))
    · exact List.mem_cons_of_mem _ (ih tail later)

/-- The participating worker's successful iterator call supplies the report's
    own inputs, including its node type and upper bound. -/
theorem posted_call {s : State w} {out : Worker → WorkerResult} (workers : List Worker)
    (covers : ∀ op ∈ s.writes, op.worker ∈ workers.map Fin.val)
    (ran : ∀ i ∈ workers, Iterator.worker (source journal stream) s.writes i.val = some (out i))
    (x : Posted rc s journal stream earlier r seq) :
    (⟨⟨x.input,x.lock.locked⟩,x.rank⟩ : LockedCall) ∈ chronological s.writes journal stream workers := by
  obtain ⟨worker, member, actor⟩ := List.mem_map.mp (covers x.op x.issued)
  have issued : x.op ∈ forWorker s.writes worker.val := List.mem_filter.mpr ⟨x.issued, by simp [actor]⟩
  obtain ⟨cs, mapped, _⟩ := Option.bind_eq_some_iff.mp (ran worker member)
  apply (List.mergeSort_perm _ _).mem_iff.mpr
  apply List.mem_flatMap.mpr
  refine ⟨worker, member, ?_⟩
  simp only [claims, mapped, Option.getD_some]
  apply mapped_at _ _ mapped issued
  exact x.acquired

theorem posted_window {s : State w} {out : Worker → WorkerResult}
    (covers : ∀ op ∈ s.writes, op.worker ∈ workers.map Fin.val)
    (ran : ∀ i ∈ workers, Iterator.worker (source journal stream) s.writes i.val = some (out i))
    (ok : replay rc s.writes journal stream workers traffic pv beta encode initial = some final)
    (x : Posted rc s journal stream earlier r seq) :
    x.input.pv = pv ∧ x.input.beta.toInt = beta :=
  FieldStores.replay_window _ (field_replay ok)
    (List.mem_map_of_mem (f := raw traffic) (posted_call workers covers ran x))

private theorem decoded (read : (raw traffic c).decode s pv beta encode cells = some record) :
    record.report = (emit c).map Prod.snd := by
  generalize h : raw traffic c = packet at read
  unfold FieldStores.Raw.decode at read
  split at read <;> try contradiction
  rename_i b body
  dsimp only at read
  split at read <;> try contradiction
  cases read
  subst packet
  change search c.ticket.input = some b at body
  cases stopped : c.ticket.locked.stopped <;> simp [FieldStores.Record.report,raw,emit,body,recordLocked_eq,stopped]

private theorem replayed (xs : List LockedCall)
    (ok : FieldStores.replay pv beta encode cells initial (xs.map (raw traffic)) = some final) :
    final.reports = initial.reports ++ (xs.filterMap emit).map Prod.snd := by
  induction xs generalizing initial with
  | nil => cases ok; simp
  | cons x xs ih =>
    obtain ⟨r,read,more⟩ := Option.bind_eq_some_iff.mp ok
    rw [ih more]
    simp only [FieldStores.Log.append,decoded read]
    cases post : emit x <;> simp [post]

theorem reports (rc : List Move.Move) (ops : List (Write w)) (workers : List Worker) (out : Worker → WorkerResult)
    (ran : ∀ i ∈ workers, Iterator.worker (source journal stream) ops i.val = some (out i))
    (ok : replay rc ops journal stream workers traffic pv beta encode initial = some final) :
    final.reports = initial.reports ++ Iterator.lockReports out workers := by
  obtain ⟨keys,_,read⟩ := accepted ok
  rw [replayed _ read,both_ordered ops workers out ran keys]
  rfl

private theorem earlier_filter (key : α → Nat) (xs : List α) (p : xs.Pairwise (fun a b => key a < key b))
    (atRow : xs[i]? = some x) : xs.filter (fun y => decide (key y < key x)) = xs.take i := by
  induction xs generalizing i with
  | nil => simp at atRow
  | cons h t ih =>
    cases i with
    | zero =>
      simp only [List.getElem?_cons_zero,Option.some.injEq] at atRow
      subst x
      simp only [List.filter_cons,Nat.lt_irrefl,decide_false,List.take_zero]
      apply List.filter_eq_nil_iff.mpr
      intro y mem
      have lt := List.rel_of_pairwise_cons p mem
      simp only [decide_eq_true_eq]
      omega
    | succ i =>
      have item : t[i]? = some x := atRow
      have lt := List.rel_of_pairwise_cons p (List.mem_of_getElem? item)
      simp [lt,ih p.tail item]

private theorem filter_emitted (xs : List LockedCall) (k : Nat) :
    (xs.filter (fun c => decide (c.order < k))).filterMap emit =
      (xs.filterMap emit).filter (fun p => decide (p.1 < k)) := by
  rw [List.filter_filterMap,List.filterMap_filter]
  congr 1
  funext c
  cases out : emit c with
  | none => simp
  | some pair =>
    have same := emit_rank out
    by_cases lt : c.order < k <;> simp [Option.filter,same,lt]

private theorem split_replay (a b : List FieldStores.Raw)
    (ok : FieldStores.replay pv beta encode cells initial (a ++ b) = some final) :
    ∃ stage, FieldStores.replay pv beta encode cells initial a = some stage ∧
      FieldStores.replay pv beta encode cells stage b = some final := by
  induction a generalizing initial with
  | nil => exact ⟨initial,rfl,ok⟩
  | cons x xs ih =>
    obtain ⟨r,read,next⟩ := Option.bind_eq_some_iff.mp ok
    obtain ⟨mid,cut,rest⟩ := ih next
    exact ⟨mid,by simp [FieldStores.replay,read,cut],rest⟩

theorem before_posted (rc : List Move.Move) {s : State w} (workers : List Worker) (out : Worker → WorkerResult)
    (ran : ∀ i ∈ workers, Iterator.worker (source journal stream) s.writes i.val = some (out i))
    (ok : replay rc s.writes journal stream workers traffic pv beta encode initial = some final)
    (x : Posted rc s journal stream earlier r (ranked workers out))
    (part : Iterator.lockReports out workers = earlier ++ r :: later) :
    let calls := chronological s.writes journal stream workers
    ∃ mid, FieldStores.replay pv beta encode (atIndex rc) initial ((calls.take x.rank).map (raw traffic)) = some mid ∧
      mid.reports = initial.reports ++ earlier ∧
      FieldStores.replay pv beta encode (atIndex rc) mid ((calls.drop x.rank).map (raw traffic)) = some final := by
  let calls := chronological s.writes journal stream workers
  obtain ⟨keys,_,read⟩ := accepted ok
  have ord := strict_covered calls keys
  have ranking := both_ordered s.writes workers out ran keys
  have klt := (List.getElem?_eq_some_iff.mp x.atLock).1
  have stamp : (calls[x.rank]?).map LockedCall.order = some x.rank := by
    rw [← List.getElem?_map]
    exact keys ▸ List.getElem?_range klt
  obtain ⟨c,present,which⟩ := Option.map_eq_some_iff.mp stamp
  have beforeC := earlier_filter LockedCall.order calls ord present
  rw [which] at beforeC
  change calls.filterMap emit = ranked workers out at ranking
  have seen : (calls.filterMap emit)[earlier.length]? = some (x.rank,r) := by rw [ranking]; exact x.sorted
  have beforeR := earlier_filter Prod.fst (calls.filterMap emit) (ordered_reports calls ord) seen
  have pieces : (calls.take x.rank).filterMap emit = (ranked workers out).take earlier.length := by
    rw [← beforeC,filter_emitted,beforeR,ranking]
  have combined : (calls.take x.rank).map (raw traffic) ++ (calls.drop x.rank).map (raw traffic) =
      calls.map (raw traffic) := by rw [← List.map_append,List.take_append_drop]
  obtain ⟨mid,prefixReplay,after⟩ := split_replay _ _ (combined.symm ▸ read)
  refine ⟨mid,prefixReplay,?_,after⟩
  rw [replayed _ prefixReplay,pieces,List.map_take]
  change initial.reports ++ (Iterator.lockReports out workers).take earlier.length = _
  simp [part]

theorem stores (rc : List Move.Move) (ops : List (Write w)) (workers : List Worker) (out : Worker → WorkerResult)
    (ran : ∀ i ∈ workers, Iterator.worker (source journal stream) ops i.val = some (out i))
    (a : FieldStores.low32 aStore = initial.alpha) (b : FieldStores.low32 bStore = initial.best)
    (m : FieldStores.low8 mStore = encode initial.picked)
    (ok : replay rc ops journal stream workers traffic pv beta encode
      (FieldStores.initialLog initial aStore bStore mStore) = some final) :
    let scores := (Iterator.lockReports out workers).map Report.score
    Threading.SplitWindow.AlphaStores pv beta initial.state scores final.alpha ∧
    Threading.SplitWindow.BestStores initial.best.toInt scores final.best ∧
    Threading.SplitWindow.MoveStores encode (atIndex rc) pv beta initial.state initial.picked
      (Iterator.lockReports out workers) final.fields.picked final.moves := by
  have same : final.reports = Iterator.lockReports out workers := reports rc ops workers out ran ok
  have inv := FieldStores.replay_preserves _ (FieldStores.seed initial _ _ _ a b m) (accepted ok).2.2
  exact same ▸ ⟨inv.alphaWrites,inv.bestWrites,inv.moveWrites⟩

theorem scout_stores (rc : List Move.Move) {s : State w} (workers : List Worker) (out : Worker → WorkerResult)
    (ran : ∀ i ∈ workers, Iterator.worker (source journal stream) s.writes i.val = some (out i))
    (a : FieldStores.low32 aStore = initial.alpha) (b : FieldStores.low32 bStore = initial.best)
    (m : FieldStores.low8 mStore = encode initial.picked)
    (ok : replay rc s.writes journal stream workers traffic pv beta encode
      (FieldStores.initialLog initial aStore bStore mStore) = some final)
    (x : Posted rc s journal stream earlier r (ranked workers out))
    (part : Iterator.lockReports out workers = earlier ++ r :: later) :
    ∃ mid, FieldStores.replay pv beta encode (atIndex rc) (FieldStores.initialLog initial aStore bStore mStore)
      (((chronological s.writes journal stream workers).take x.rank).map (raw traffic)) = some mid ∧
      final.alpha = mid.alpha ++ ((chronological s.writes journal stream workers).drop x.rank).flatMap
        (fun c => (traffic c.order).filterMap FieldStores.Physical.alpha?) ∧
      Threading.SplitWindow.AlphaStores pv beta initial.state (earlier.map Report.score) mid.alpha := by
  obtain ⟨mid,pre,rs,tail⟩ := before_posted rc workers out ran ok x part
  have same : mid.reports = earlier := rs
  have inv := FieldStores.replay_preserves _ (FieldStores.seed initial _ _ _ a b m) pre
  exact ⟨mid,pre,by simpa only [List.flatMap_map, Function.comp_def, raw] using (FieldStores.modification_orders _ tail).1,same ▸ inv.alphaWrites⟩

private theorem from_mapM (xs : List A) (f : A → Option B) (ok : xs.mapM f = some ys) (found : y ∈ ys) :
    ∃ x ∈ xs, f x = some y := by
  induction xs generalizing ys with
  | nil => cases ok; contradiction
  | cons x xs ih =>
    rw [List.mapM_cons] at ok
    obtain ⟨head,read,next⟩ := Option.bind_eq_some_iff.mp ok
    obtain ⟨tail,more,finish⟩ := Option.bind_eq_some_iff.mp next
    cases finish
    rcases List.mem_cons.mp found with eq | later
    · exact ⟨x,by simp,eq ▸ read⟩
    · obtain ⟨p,mem,same⟩ := ih more later
      exact ⟨p,List.mem_cons_of_mem _ mem,same⟩

theorem acquired_at {s : State w} (workers : List Worker) (out : Worker → WorkerResult)
    (ran : ∀ i ∈ workers, Iterator.worker (source journal stream) s.writes i.val = some (out i))
    (atCall : c ∈ chronological s.writes journal stream workers) :
    ∃ op ∈ s.writes, Iterator.acquired (source journal stream) op = some c := by
  have inGrouped := (List.mergeSort_perm _ _).mem_iff.mp atCall
  obtain ⟨i,participant,member⟩ := List.mem_flatMap.mp inGrouped
  obtain ⟨cs,mapped,_⟩ := Option.bind_eq_some_iff.mp (ran i participant)
  change c ∈ claims s.writes journal stream i at member
  simp only [claims,mapped,Option.getD_some] at member
  obtain ⟨op,memberOp,acq⟩ := from_mapM _ _ mapped member
  have issued := (List.mem_filter.mp memberOp).1
  exact ⟨op,issued,acq⟩

/-- The lock ordinal, move operand and PVS return words for each field section come
    from the same successful iterator ticket, including a stopped final ticket. -/
theorem origin (rc : List Move.Move) {s : State w} (run : MoveList.ConcurrentTrace.Execution rc.length start s)
    (fits : rc.length < 2 ^ w) (workers : List Worker) (out : Worker → WorkerResult)
    (ran : ∀ i ∈ workers, Iterator.worker (source journal stream) s.writes i.val = some (out i))
    (atCall : c ∈ chronological s.writes journal stream workers)
    (ok : replay rc s.writes journal stream workers traffic pv beta encode initial = some final) :
    ∃ (op : Write w) (lock : Observation) (results : List NativeWorkerFeed.Return) (mv : Move.Move) (body : Searched),
      op ∈ s.writes ∧ journal[c.order]? = some lock ∧
      lock.worker = op.worker ∧ lock.opening = op.opening ∧
      NativeWorkerFeed.returns op.worker stream = some results ∧
      c.ticket = ⟨NativeWorkerFeed.received lock.task lock.prefix results op.old lock.alpha lock.fresh,lock.locked⟩ ∧
      NativeWorkerFeed.ticket rc op.old = some (mv,op.stored.toNat) ∧
      atIndex rc c.ticket.input.index = some mv.sq ∧
      search c.ticket.input = some body ∧ (traffic c.order).map FieldStores.Physical.value =
        (FieldStores.accesses c.ticket.input lock.locked body.chosen).map (FieldStores.Access.value encode (atIndex rc)) := by
  obtain ⟨op,issued,acq⟩ := acquired_at workers out ran atCall
  obtain ⟨mv,cell,ticket,_⟩ := IteratorFrames.issued_move rc run issued fits
  unfold Iterator.acquired at acq
  cases item : IteratorFrames.findLock journal op.worker op.opening with
  | none => simp [source,item] at acq
  | some pair =>
    cases words : NativeWorkerFeed.returns op.worker stream with
    | none => simp [source,item,words] at acq
    | some results =>
      cases control : NativeWorkerFeed.completeControl pair.2.prefix pair.2.task results pair.2.alpha
          (fun wide => pair.2.stamp (.load wide)) pair.2.ends pair.2.acquired pair.2.locked.stopped <;>
        simp [source,item,words,control] at acq
      have matched := IteratorFrames.lock_at (k := pair.1) (r := pair.2) item
      have phys := List.all_eq_true.mp (accepted ok).2.1 c atCall
      cases h : search c.ticket.input with
      | none => simp [operands,h] at phys
      | some body =>
        have packet := of_decide_eq_true (by simpa only [operands,h] using phys)
        subst c
        exact ⟨op,pair.2,results,mv,body,issued,matched.1,matched.2.1,matched.2.2,words,rfl,ticket,
          by simp [atIndex,cell],rfl,packet⟩

end NeuralReversi.PvsParallel.IteratorFields
