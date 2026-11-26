import NeuralReversi.PvsParallel.Interleaving.Retirement

namespace NeuralReversi.PvsParallel.Interleaving.Drive

open Tickets (Post State Step)

inductive Event where
  | book (worker : Nat)
  | take (worker index : Nat)
  | empty (worker : Nat)
  | report (worker : Nat) (post : Post)
  | discard (worker index : Nat)
  | leave (worker : Nat)
  | externalStop

def Event.action : Event → Tickets.Action
  | .book i => .book i
  | .take i _ => .take i
  | .empty i => .empty i
  | .report i post => .report i post
  | .discard i _ => .discard i
  | .leave i => .leave i
  | .externalStop => .externalStop

def remove [DecidableEq α] (needle : α) : (xs : List α) →
    Option {pair : List α × List α // xs = pair.1 ++ needle :: pair.2}
  | [] => none
  | x :: xs => do
    if eq : x = needle then return ⟨([],xs),by simp [eq]⟩
    let ⟨(left,right),split⟩ ← remove needle xs
    return ⟨(x :: left,right),by simpa using congrArg (List.cons x) split⟩

/-- A known occurrence is the one removed when its prefix contains no earlier copy. -/
theorem remove_at [DecidableEq α] {needle : α} (left right : List α) (first : needle ∉ left) :
    remove needle (left ++ needle :: right) = some ⟨(left,right),rfl⟩ := by
  induction left with
  | nil => simp [remove]
  | cons x xs ih =>
    simp only [List.mem_cons,not_or] at first
    obtain ⟨other,rest⟩ := first
    simp [remove,Ne.symm other,ih rest]

private theorem remove_unique [DecidableEq α] {needle : α} {xs left right : List α}
    (unique : xs.Nodup) (split : xs = left ++ needle :: right) :
    remove needle xs = some ⟨(left,right),split⟩ := by
  subst xs
  exact remove_at left right (fun prior => (List.nodup_append.mp unique).2.2
    _ prior _ List.mem_cons_self rfl)

/-- The two ticket indices erased by `Event.action` come from the physical claim and the held ticket. -/
def Event.indices (e : Event) (s : State w) : Prop :=
  match e with
  | .take _ index => index = s.current.toNat
  | .discard worker index => (worker,index) ∈ s.pending
  | _ => True

private def checked (len : Nat) (s : State w) : (e : Event) →
    Option {next : State w // Step len s e.action next ∧ e.indices s}
  | .book i => do
    if gate : s.booking = true then
      if clear : s.reports.all (fun p => !p.cut) then
        if free : i ∉ s.pollers ∧ i ∉ s.pending.map Prod.fst ∧ i ∉ s.departing then
          return ⟨_,.book gate (by simpa using List.all_eq_true.mp clear) free,trivial⟩
    none
  | .take i index => do
    let ⟨(left,right),workerAt⟩ ← remove i s.pollers
    if atIndex : index = s.current.toNat then
      match h : MoveList.Concurrent.advance len s.current with
      | some _ => return ⟨_,.take (left := left) (right := right) workerAt h,atIndex⟩
      | none => none
    else none
  | .empty i => do
    let ⟨(left,right),workerAt⟩ ← remove i s.pollers
    if stop : MoveList.Concurrent.advance len s.current = none then
      return ⟨_,.empty (left := left) (right := right) workerAt stop,trivial⟩
    none
  | .report i post => do
    let ⟨(left,right),held⟩ ← remove (i,post.report.index) s.pending
    return ⟨_,.report (left := left) (right := right) held,trivial⟩
  | .discard i index => do
    let ⟨(left,right),held⟩ ← remove (i,index) s.pending
    have add (h : s.stopped) := (⟨_,Step.discard (left := left) (right := right) held h,
        held ▸ List.mem_append_right _ List.mem_cons_self⟩ :
      {next : State w // Step len s (.discard i) next ∧ (i,index) ∈ s.pending})
    if out : s.outside = true then return add (Or.inl out)
    if hit : s.reports.any Post.cut then return add (Or.inr (List.any_eq_true.mp hit))
    none
  | .leave i => do
    let ⟨(left,right),ending⟩ ← remove i s.departing
    return ⟨_,.leave (left := left) (right := right) ending,trivial⟩
  | .externalStop => do return ⟨_,.externalStop,trivial⟩

def step (len : Nat) (s : State w) (e : Event) : Option (State w) :=
  (checked len s e).map Subtype.val

theorem step_sound (hit : step len s e = some next) : Step len s e.action next := by
  obtain ⟨result,_,rfl⟩ := Option.map_eq_some_iff.mp hit
  exact result.property.1

/-- One worker has at most one position among the polling, pending and retiring workers. -/
theorem roster_nodup (run : Tickets.Run len (Tickets.initial start owner) s) :
    (Retirement.roster s).Nodup := by
  induction run with
  | refl => simp [Retirement.roster,Tickets.initial]
  | @step _ event _ _ edge ih =>
    have perm := Retirement.roster_step edge
    cases event with
    | book worker =>
      exact perm.nodup_iff.mpr (List.nodup_cons.mpr ⟨(Retirement.opened edge).2,ih⟩)
    | leave worker => exact (List.nodup_cons.mp (perm.nodup_iff.mp ih)).2
    | _ => exact perm.nodup_iff.mpr ih

/-- A pending worker's concrete list split can be recovered without a caller choosing an occurrence. -/
theorem remove_pending (run : Tickets.Run len (Tickets.initial start owner) s)
    (held : s.pending = left ++ (worker,index) :: right) :
    remove (worker,index) s.pending = some ⟨(left,right),held⟩ := by
  have workers := (List.nodup_append.mp (List.nodup_append.mp (roster_nodup run)).2.1).1
  exact remove_unique ((List.pairwise_map.mp workers).imp
    (fun h eq => h (congrArg Prod.fst eq))) held

private theorem held_index {xs : List (Nat × Nat)} (unique : (xs.map Prod.fst).Nodup)
    (atWorker : xs = left ++ (worker,i) :: right) (mem : (worker,j) ∈ xs) : j = i := by
  subst xs
  simp only [List.map_append,List.map_cons] at unique
  obtain ⟨_,after,apart⟩ := List.nodup_append.mp unique
  rcases List.mem_append.mp mem with before | rest
  · exact False.elim (apart _ (List.mem_map_of_mem before) _ List.mem_cons_self rfl)
  · rcases List.mem_cons.mp rest with same | later
    · exact congrArg Prod.snd same
    · exact False.elim ((List.nodup_cons.mp after).1 (List.mem_map_of_mem later))

theorem step_complete {e : Event} (run : Tickets.Run len (Tickets.initial start owner) s)
    (edge : Step len s e.action next) (indices : e.indices s) : step len s e = some next := by
  obtain ⟨pollers,pendingEnds,_⟩ := List.nodup_append.mp (roster_nodup run)
  obtain ⟨workers,departing,_⟩ := List.nodup_append.mp pendingEnds
  cases e with
  | book i =>
    cases edge with
    | book gate cuts free =>
      have clear : s.reports.all (fun p => !p.cut) = true := List.all_eq_true.mpr (by simpa using cuts)
      simp [step,checked,gate,clear,Tickets.State.available] at free ⊢
      simp [free]
  | take i index =>
    cases edge with
    | take atWorker claimed =>
      simp [step,checked,remove_unique pollers atWorker,show index = _ from indices]
      split <;> simp_all
  | empty i =>
    cases edge with
    | empty atWorker exhausted => simp [step,checked,remove_unique pollers atWorker,exhausted]
  | report i post =>
    cases edge with
    | report atWorker => simp [step,checked,remove_pending run atWorker]
  | discard i index =>
    cases edge with
    | discard atWorker stopped =>
      obtain rfl := held_index workers atWorker indices
      rcases stopped with outside | cut
      · simp [step,checked,remove_pending run atWorker,outside]
        rfl
      · have anyCut : s.reports.any Post.cut = true := List.any_eq_true.mpr cut
        by_cases outside : s.outside = true <;> simp [step,checked,remove_pending run atWorker,outside,anyCut] <;> rfl
  | leave i =>
    cases edge with
    | leave atWorker => simp [step,checked,remove_unique departing atWorker]
  | externalStop => cases edge; rfl

theorem step_iff (run : Tickets.Run len (Tickets.initial start owner) s) :
    step len s e = some next ↔ Step len s e.action next ∧ e.indices s := by
  constructor
  · intro hit
    obtain ⟨result,_,rfl⟩ := Option.map_eq_some_iff.mp hit
    exact result.property
  · rintro ⟨edge,indices⟩
    exact step_complete run edge indices

def replay (len : Nat) (s : State w) : List Event → Option (State w)
  | [] => some s
  | e :: es => (step len s e).bind (fun next => replay len next es)

theorem replay_sound (es : List Event) (history : Tickets.Run len origin s)
    (hit : replay len s es = some next) : Tickets.Run len origin next := by
  induction es generalizing s with
  | nil => cases hit; exact history
  | cons e es ih =>
    obtain ⟨_,edge,rest⟩ := Option.bind_eq_some_iff.mp hit
    exact ih (history.step (step_sound edge)) rest

/-- Ticket transitions carrying the two physical index operands erased by `Event.action`. -/
inductive Steps (len : Nat) : State w → List Event → State w → Prop where
  | nil : Steps len s [] s
  | cons : Step len s e.action t → e.indices s → Steps len t es u → Steps len s (e :: es) u

theorem replay_iff (run : Tickets.Run len (Tickets.initial start owner) s) :
    replay len s es = some next ↔ Steps len s es next := by
  induction es generalizing s with
  | nil =>
    constructor
    · intro hit; cases hit; exact .nil
    · intro path; cases path; rfl
  | cons e es ih =>
    constructor
    · intro hit
      obtain ⟨_,one,more⟩ := Option.bind_eq_some_iff.mp hit
      obtain ⟨edge,indices⟩ := (step_iff run).mp one
      exact .cons edge indices ((ih (run.step edge)).mp more)
    · intro path
      cases path with
      | cons edge indices more =>
        exact Option.bind_eq_some_iff.mpr ⟨_,step_complete run edge indices,(ih (run.step edge)).mpr more⟩

private theorem physical_event (edge : Step len s a t) : ∃ e : Event, e.action = a ∧ e.indices s := by
  cases edge with
  | book => exact ⟨.book _,rfl,trivial⟩
  | take => exact ⟨.take _ s.current.toNat,rfl,rfl⟩
  | empty => exact ⟨.empty _,rfl,trivial⟩
  | report => exact ⟨.report _ _,rfl,trivial⟩
  | discard held _ => exact ⟨.discard _ _,rfl,held ▸ List.mem_append_right _ List.mem_cons_self⟩
  | leave => exact ⟨.leave _,rfl,trivial⟩
  | externalStop => exact ⟨.externalStop,rfl,trivial⟩

private theorem replay_append (es more : List Event) (s : State w) :
    replay len s (es ++ more) = (replay len s es).bind (fun t => replay len t more) := by
  induction es generalizing s with
  | nil => rfl
  | cons e es ih => simp [replay,Option.bind_assoc,ih]

theorem reachable_iff : Tickets.Run len (Tickets.initial start owner) s ↔
    ∃ es, replay len (Tickets.initial start owner) es = some s := by
  constructor
  · intro run
    induction run with
    | refl => exact ⟨[],rfl⟩
    | step prior edge ih =>
      obtain ⟨es,trace⟩ := ih
      obtain ⟨e,rfl,indices⟩ := physical_event edge
      exact ⟨es ++ [e],by rw [replay_append,trace]; simp [replay,step_complete prior edge indices]⟩
  · rintro ⟨es,hit⟩
    exact replay_sound es .refl hit

-- An intervening booking cannot duplicate a polling, working, or departing worker.
def Contract := ∀ (w len : Nat) (start : BitVec w) (owner : Nat) (s : State w),
  Tickets.Run len (Tickets.initial start owner) s ↔ ∃ es, replay len (Tickets.initial start owner) es = some s

theorem complete : Contract := fun _ _ _ _ _ => reachable_iff

example : replay 1 (Tickets.initial (0 : BitVec 8) 0) [.book 1,.book 1] = none := rfl
example : replay 1 (Tickets.initial (0 : BitVec 8) 0) [.book 1,.take 1 0,.book 1] = none := rfl
example : replay 0 (Tickets.initial (0 : BitVec 8) 0) [.book 1,.empty 1,.book 1] = none := rfl

-- Both workers hold a ticket when the stop arrives. Only its owner can discard each index.
example : (replay 2 (Tickets.initial (0 : BitVec 8) 0)
    [.book 1,.take 0 0,.take 1 1,.externalStop,.discard 0 0,.discard 1 1]).isSome = true := rfl
example : replay 2 (Tickets.initial (0 : BitVec 8) 0)
    [.book 1,.take 0 0,.take 1 1,.externalStop,.discard 0 1] = none := rfl
example : replay 2 (Tickets.initial (0 : BitVec 8) 0) [.take 0 1] = none := rfl

end NeuralReversi.PvsParallel.Interleaving.Drive
