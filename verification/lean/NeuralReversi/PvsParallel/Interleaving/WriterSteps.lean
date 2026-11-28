module
public import NeuralReversi.PvsParallel.Interleaving.Joint

@[expose] public section

namespace NeuralReversi.PvsParallel.Interleaving.WriterSteps

open NativeTableWrites.Journal
open TranspositionTable
open CallJournal (State Writer)
open FrameTableReads (Address)

variable {α : Type} {es : List (CallJournal.Event α)} {s t : State α}

private theorem fetch_index {a : Attempt} (hit : a.fetch endian n = some pc) : pc.index = n := by
  have bound : n < (a.program endian).1.length := by
    obtain ⟨_,found,_⟩ := Option.map_eq_some_iff.mp hit
    exact (List.getElem?_eq_some_iff.mp found).1
  cases h : a.success <;> simp [Attempt.program,WriterMir.writeStable,h] at bound
  all_goals
    have ix : n = 0 ∨ n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 5 := by omega
    rcases ix with rfl | rfl | rfl | rfl | rfl | rfl <;> simp [Attempt.fetch,Attempt.program,WriterMir.writeStable,h] at hit <;> subst pc <;> rfl

private theorem pc_ext {a b : Pc} (h : a.index = b.index) : a = b := by
  cases a with
  | field f => cases f <;> cases b with
    | field g => cases g <;> first | rfl | contradiction
    | _ => contradiction
  | _ => cases b with
    | field g => cases g <;> contradiction
    | _ => first | rfl | contradiction

private def atWord (es : List (CallJournal.Event α)) (site : Site) (worker : Nat) (c : Calls) :=
  es[site.event c]? = some (.atomic worker site.invocation)

private theorem extendAt (h : atWord es site w c) (ev : CallJournal.Event α) :
    atWord (es ++ [ev]) site w c := by
  exact (List.getElem?_append_left (List.getElem?_eq_some_iff.mp h).1).trans h

private def clock (c : Calls) (keep : Site → Bool) : Record → Option Nat
  | .localCall .. => none
  | .writer site => if keep site then some (site.event c) else none

private structure History (c : Calls) (es : List (CallJournal.Event α)) (rs : List Record) : Prop where
  origin : ∀ site, .writer site ∈ rs → ∃ worker, atWord es site worker c ∧
    ∀ pc : Pc, pc.index < site.pc.index →
      .writer ⟨site.invocation,pc⟩ ∈ rs ∧ atWord es ⟨site.invocation,pc⟩ worker c ∧
        (c site.invocation).event pc.index < site.event c
  unique : ∀ keep, (rs.filterMap (clock c keep)).Nodup

private structure Reached (c : Calls) (es : List (CallJournal.Event α)) (s : State α) : Prop extends History c es s.records where
  active : ∀ worker (w : Writer α), s.writing worker = some w → ∀ pc : Pc, pc.index < w.next →
    .writer ⟨w.invocation,pc⟩ ∈ s.records ∧ atWord es ⟨w.invocation,pc⟩ worker c

private theorem persist (h : History c es rs) (ev : CallJournal.Event α) (extra : List Record)
    (locals : ∀ a, Record.writer a ∉ extra)
    (zero : ∀ keep, extra.filterMap (clock c keep) = []) : History c (es ++ [ev]) (rs ++ extra) where
  origin s mem := by
    rcases List.mem_append.mp mem with m | m
    · obtain ⟨w,hit,prior⟩ := h.origin s m
      refine ⟨w, extendAt hit ev, fun p lt => ?_⟩
      obtain ⟨old,atOld,time⟩ := prior p lt
      exact ⟨List.mem_append_left _ old,extendAt atOld ev,time⟩
    · exact False.elim (locals s m)
  unique keep := by simpa [List.filterMap_append,zero] using h.unique keep

private theorem preserve (old : Reached c es s) (ev : CallJournal.Event α)
    (same : t.writing = s.writing) (extra : List Record) (records : t.records = s.records ++ extra)
    (empty : ∀ site, Record.writer site ∉ extra) (zero : ∀ keep, extra.filterMap (clock c keep) = []) :
    Reached c (es ++ [ev]) t where
  toHistory := records ▸ persist old.toHistory ev extra empty zero
  active w d held p lt := by
    obtain ⟨mem,hit⟩ := old.active w d (same ▸ held) p lt
    exact ⟨records ▸ List.mem_append_left _ mem,extendAt hit ev⟩

private theorem writerHistory (h : Reached c es s) {worker : Nat} {w : Writer α}
    (held : s.writing worker = some w) (idx : pc.index = w.next)
    (time : (c w.invocation).event pc.index = es.length) :
    History c (es ++ [.atomic worker w.invocation]) (s.records ++ [.writer ⟨w.invocation,pc⟩]) where
  origin site member := by
    rcases List.mem_append.mp member with m | m
    · obtain ⟨who,hit,prior⟩ := h.origin _ m
      refine ⟨who,extendAt hit _,fun p lt => ?_⟩
      obtain ⟨old,atOld,lt⟩ := prior p lt
      exact ⟨List.mem_append_left _ old,extendAt atOld _,lt⟩
    · simp only [List.mem_singleton,Record.writer.injEq] at m
      subst site
      refine ⟨worker,?_,fun p lt => ?_⟩
      · simp [atWord,Site.event,time]
      · obtain ⟨past,hit⟩ := h.active _ _ held p (idx ▸ lt)
        exact ⟨List.mem_append_left _ past,extendAt hit _,by simpa [Site.event,time] using (List.getElem?_eq_some_iff.mp hit).1⟩
  unique keep := by
    by_cases selected : keep ⟨w.invocation,pc⟩ = true
    · simp [List.filterMap_append,clock,selected,List.nodup_append, h.unique]
      intro n r old hit same
      subst n
      cases r with
      | localCall => contradiction
      | writer site =>
        change (if keep site then some (site.event c) else none) = some ((c w.invocation).event pc.index) at hit
        split at hit <;> try contradiction
        obtain ⟨who,found,_⟩ := h.origin _ old
        have before := (List.getElem?_eq_some_iff.mp found).1
        simp only [Option.some.injEq] at hit
        change site.event c = (c w.invocation).event pc.index at hit
        omega
    · simpa [List.filterMap_append,clock,selected] using h.unique keep

private theorem advance (old : Reached c es s) (ev : CallJournal.Event α)
    (ran : CallJournal.step mc fc c endian s es.length ev = some t) : Reached c (es ++ [ev]) t := by
  cases ev with
  | other w =>
    cases ran
    exact preserve old _ rfl [] (by simp) (by simp) (by simp)
  | localCall w e args =>
    obtain ⟨same,issued⟩ := CallJournal.localWriter ran
    exact preserve old _ same [.localCall w e] (by simp [State.records,issued,CallJournal.Issued.record])
      (by simp) (by simp [clock])
  | boundaryScore =>
    obtain ⟨same,issued⟩ := CallJournal.boundaryWriter ran
    exact preserve old _ same [] (by simp [State.records,issued]) (by simp) (by simp)
  | beginWrite who k raw =>
    simp only [CallJournal.step,guard,Bind.bind,Option.bind,Pure.pure] at ran
    split at ran <;> try contradiction
    cases ran
    constructor
    · change History _ _ s.records
      simpa using persist old.toHistory (.beginWrite who k raw) [] (by simp) (by simp)
    · intro w d held p lt
      change (if who = w then some _ else s.writing w) = some d at held
      split at held
      · cases held; contradiction
      · obtain ⟨mem,hit⟩ := old.active _ _ held p lt
        exact ⟨mem,extendAt hit _⟩
  | atomic who k =>
    unfold CallJournal.step at ran
    obtain ⟨w,held,step⟩ := Option.bind_eq_some_iff.mp ran
    simp only [guard] at step
    split at step <;> try contradiction
    rename_i same
    have same := beq_iff_eq.mp same
    subst k
    change (Option.bind ((c w.invocation).fetch endian w.next) _) = some t at step
    obtain ⟨pc,hit,step⟩ := Option.bind_eq_some_iff.mp step
    split at step <;> try contradiction
    rename_i stamp
    have time := beq_iff_eq.mp stamp
    cases step
    have idx := fetch_index hit
    have earlier := writerHistory old held idx time
    constructor
    · simpa [State.records,CallJournal.Issued.record] using earlier
    · intro other d stored p lt
      change (if who = other then (if _ then some {w with next := w.next+1} else none)
        else s.writing other) = some d at stored
      have inherited {u : Nat} {e : Writer α} (running : s.writing u = some e) {q : Pc} (less : q.index < e.next) :
          Record.writer ⟨e.invocation,q⟩ ∈ (s.issued ++ [CallJournal.Issued.atomic w pc]).map CallJournal.Issued.record ∧
          atWord (es ++ [.atomic who w.invocation]) ⟨e.invocation,q⟩ u c := by
        obtain ⟨mem,atOld⟩ := old.active _ _ running q less
        exact ⟨by simpa [State.records,CallJournal.Issued.record] using
          List.mem_append_left [Record.writer ⟨w.invocation,pc⟩] mem,extendAt atOld _⟩
      split at stored
      · rename_i same
        subst other
        split at stored <;> try contradiction
        cases stored
        change p.index < w.next+1 at lt
        by_cases less : p.index < w.next
        · exact inherited held less
        · obtain rfl : p = pc := pc_ext (by omega)
          refine ⟨by simp [State.records,CallJournal.Issued.record],?_⟩
          simp [atWord,Site.event,time]
      · exact inherited stored lt

private theorem reached {es : List (CallJournal.Event α)} (ran : CallJournal.run mc fc c endian es = some t) :
    Reached c es t := by
  have walk (before after : List (CallJournal.Event α)) (s : State α) (old : Reached c before s)
      (ok : (after.zipIdx before.length).foldlM (fun z ev => CallJournal.step mc fc c endian z ev.2 ev.1) s = some t) :
      Reached c (before ++ after) t := by
    induction after generalizing before s with
    | nil =>
      cases ok
      simpa using old
    | cons a rest ih =>
      change (CallJournal.step mc fc c endian s before.length a >>= _) = some t at ok
      obtain ⟨s',issued,later⟩ := Option.bind_eq_some_iff.mp ok
      simpa [List.append_assoc] using ih (before ++ [a]) s' (advance old a issued) (by simpa using later)
  exact walk [] es CallJournal.start ⟨⟨by simp [State.records,CallJournal.start],by simp [State.records,CallJournal.start]⟩,
    by simp [CallJournal.start]⟩ ran

/-- A writer's global event number points to the atomic instruction on its own
    worker. Instructions preceding its MIR PC have already executed there. -/
structure Facts (c : Calls) (es : Joint.Tape α) (rs : List Record) : Prop where
  sites : ∀ site, .writer site ∈ rs → ∃ worker,
    es[site.event c]? = some (.engine (.atomic worker site.invocation)) ∧
    ∀ pc : Pc, pc.index < site.pc.index →
      .writer ⟨site.invocation,pc⟩ ∈ rs ∧
      es[(c site.invocation).event pc.index]? = some (.engine (.atomic worker site.invocation)) ∧
      (c site.invocation).event pc.index < site.event c
  distinct : ∀ a loc, ((mods c rs a loc).map (Site.event c)).Nodup

private theorem jointAt {es : Joint.Tape α} {i worker k : Nat}
    (found : (es.map Joint.Event.engineEvent)[i]? = some (.atomic worker k)) :
    es[i]? = some (.engine (.atomic worker k)) := by
  obtain ⟨ev,present,tag⟩ := Option.map_eq_some_iff.mp (List.getElem?_map ▸ found)
  cases ev <;> cases tag
  exact present

theorem run {es : Joint.Tape α} (ran : CallJournal.run mc fc c endian (es.map Joint.Event.engineEvent) = some t) :
    Facts c es t.records := by
  have h := reached ran
  constructor
  · intro site mem
    obtain ⟨w,hit,prior⟩ := h.origin site mem
    refine ⟨w,jointAt hit, fun pc lt => ?_⟩
    obtain ⟨mem,hit,earlier⟩ := prior pc lt
    exact ⟨mem,jointAt hit,earlier⟩
  · intro a loc
    let keep := fun s : Site => decide ((c s.invocation).address = a ∧
      (c s.invocation).success = true ∧ s.pc.location = some loc)
    have same : ((mods c t.records a loc).map (Site.event c)) = t.records.filterMap (clock c keep) := by
      rw [mods,List.map_filterMap]
      congr 1
      funext r
      cases r with
      | localCall => rfl
      | writer s =>
        simp only [select,clock,keep,decide_eq_true_eq]
        split <;> rfl
    exact same ▸ h.unique keep

theorem decoded {es : Joint.Tape (CallJournal.Operand c endian seen)} {t : State (CallJournal.Operand c endian seen)}
    (ok : CallJournal.decode mc fc c endian seen (es.map Joint.Event.engineEvent) = some t) :
    Facts c es t.records := by
  obtain ⟨s,ran,accepted⟩ := Option.bind_eq_some_iff.mp ok
  split at accepted <;> try contradiction
  cases accepted
  exact run ran

variable {tape : Joint.Tape α} {rs : List Record}

theorem Facts.atSite (h : Facts c tape rs) (m : Record.writer site ∈ rs) :
    ∃ worker, tape[site.event c]? = some (.engine (.atomic worker site.invocation)) := by
  obtain ⟨w,atWord,_⟩ := h.sites site m
  exact ⟨w,atWord⟩

theorem Facts.bound (h : Facts c tape rs) (m : Record.writer site ∈ rs) : site.event c < tape.length :=
  (List.getElem?_eq_some_iff.mp (h.atSite m).choose_spec).1

theorem Facts.injective (h : Facts c tape rs) (m : Record.writer site ∈ rs) (n : Record.writer other ∈ rs)
    (same : site.event c = other.event c) : site = other := by
  obtain ⟨w,found,above⟩ := h.sites site m
  obtain ⟨v,otherAt,below⟩ := h.sites other n
  rw [same,otherAt] at found
  obtain ⟨_,inv⟩ := CallJournal.Event.atomic.inj (Joint.Event.engine.inj (Option.some.inj found))
  obtain ⟨i,p⟩ := site
  obtain ⟨j,q⟩ := other
  change j = i at inv
  subst j
  simp only [Site.event] at same
  rcases Nat.lt_trichotomy p.index q.index with less | eq | more
  · have impossible := (below p less).2.2
    change (c i).event p.index < (c i).event q.index at impossible
    omega
  · exact congrArg (Site.mk i) (pc_ext eq)
  · have impossible := (above q more).2.2
    change (c i).event q.index < (c i).event p.index at impossible
    omega

/-- Read-from, modification order, and the ordering mode are the primitive
    atomic observations for each address. -/
structure Access (c : Calls) (endian : Bool) (rs : List Record) where
  memory : Address → SeqLock.Memory Nat
  source : ∀ a loc i r, (memory a).rf ((memory a).write loc i) r → i ≤ (mods c rs a loc).length
  modified : ∀ a loc i s, (mods c rs a loc)[i]? = some s →
    (memory a).write loc (i+1) = s.event c ∧
      writes (c s.invocation).success ((c s.invocation).program endian).1[s.pc.index]? =
        some (loc,(memory a).writeValue loc (i+1))
  rmw : ∀ a i k, (mods c rs a .sequence)[i]? = some ⟨k,.cas⟩ →
    (memory a).rf ((memory a).write .sequence i) ((c k).event 0)
  ordering : ∀ s, .writer s ∈ rs →
    mode (memory (c s.invocation).address) (s.event c) (c s.invocation).success
      ((c s.invocation).program endian).1[s.pc.index]?

theorem Facts.prior (h : Facts c tape rs) (present : Record.writer site ∈ rs) (pc : Pc)
    (lt : pc.index < site.pc.index) : Record.writer ⟨site.invocation,pc⟩ ∈ rs ∧
    (c site.invocation).event pc.index < site.event c ∧
    Joint.worker tape ((c site.invocation).event pc.index) = Joint.worker tape (site.event c) := by
  obtain ⟨who,hit,earlier⟩ := h.sites site present
  obtain ⟨old,previous,before⟩ := earlier pc lt
  exact ⟨old,before,by simp [Joint.worker,Joint.Event.worker,hit,previous]⟩

def Facts.execution (h : Facts c tape rs) (a : Access c endian rs)
    (sb : ∀ addr i j, i < j → j < tape.length → Joint.worker tape i = Joint.worker tape j →
      (a.memory addr).sb i j) : NativeTableWrites.Journal.Execution c endian rs where
  memory := a.memory
  distinct := h.distinct
  source := a.source
  modified := a.modified
  rmw := a.rmw
  ordering := a.ordering
  earlier site present pc lt := by
    obtain ⟨mem,time,worker⟩ := h.prior present pc lt
    exact ⟨mem,sb _ _ _ time (h.bound present) worker⟩

namespace Regression

private def attempts (fence : Nat) : Calls := fun k =>
  ⟨⟨0,0⟩,⟨0,rfl⟩,.clear,k == 1,fun pc => if k = 0 then 2 else if pc = 0 then 3 else fence⟩

/-- Worker 6 loses its CAS. Worker 9 acquires the same slot and the tape ends at
    its release fence, before any payload store or publication. -/
private def sample (second : Nat) : Joint.Tape Unit :=
  [.engine (.beginWrite 6 0 ()),.engine (.beginWrite 9 1 ()),.engine (.atomic 6 0),
   .engine (.atomic 9 1),.engine (.atomic second 1)]

private def replay mc fc (es : Joint.Tape Unit) (fence : Nat) :=
  CallJournal.run mc fc (attempts fence) false (es.map Joint.Event.engineEvent)

example mc fc : ((replay mc fc (sample 9) 4).map fun s =>
    (s.records.length, (mods (attempts 4) s.records ⟨0,0⟩ .sequence).map (Site.event (attempts 4)))) =
    some (3,[3]) := rfl
example mc fc : (replay mc fc (sample 6) 4).isNone = true := rfl
example mc fc : (replay mc fc (sample 9) 3).isNone = true := rfl
example mc fc : (replay mc fc ((sample 9) ++ [.engine (.atomic 6 0)]) 4).isNone = true := rfl

end Regression

end NeuralReversi.PvsParallel.Interleaving.WriterSteps
