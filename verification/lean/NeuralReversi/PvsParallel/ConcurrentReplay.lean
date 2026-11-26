import NeuralReversi.PvsParallel.CausalReplay
import NeuralReversi.MoveList.Ordering

namespace NeuralReversi.PvsParallel.ConcurrentReplay

open CausalReplay (ScoreResult Recorded PrefixExitFrom PrefixWriteFrom Run)
open Cutoffs (History FlipConfig RawSample)
open PvsBoardTree (Position)
open Bitboard.Moves (X86Config)
open TranspositionTable

private theorem minimal {before : α → α → Prop}
    (irrefl : ∀ a, ¬before a a) (trans : ∀ {a b c}, before a b → before b c → before a c)
    (events : List α) (nonempty : events ≠ []) :
    ∃ event ∈ events, ∀ other ∈ events, ¬before other event := by
  induction events with
  | nil => contradiction
  | cons head tail ih =>
    by_cases empty : tail = []
    · subst tail
      exact ⟨head, by simp, by simpa using irrefl head⟩
    · obtain ⟨chosen, present, earliest⟩ := ih empty
      by_cases earlier : before head chosen
      · refine ⟨head, by simp, ?_⟩
        intro other member impossible
        rcases List.mem_cons.mp member with equal | inTail
        · exact irrefl head (equal ▸ impossible)
        · exact earliest other inTail (trans impossible earlier)
      · refine ⟨chosen, List.mem_cons_of_mem _ present, ?_⟩
        intro other member
        rcases List.mem_cons.mp member with equal | inTail
        · exact equal.symm ▸ earlier
        · exact earliest other inTail

private theorem splitMember {events : List α} {event : α} (member : event ∈ events) :
    ∃ left right, events = left ++ event :: right := by
  induction events with
  | nil => simp at member
  | cons head tail ih =>
    rcases List.mem_cons.mp member with equal | inTail
    · exact ⟨[], tail, by rw [equal]; rfl⟩
    · obtain ⟨left, right, same⟩ := ih inTail
      exact ⟨head :: left, right, congrArg (List.cons head) same⟩

private theorem linearize {before : α → α → Prop}
    (irrefl : ∀ a, ¬before a a) (trans : ∀ {a b c}, before a b → before b c → before a c)
    (events : List α) :
    ∃ ordered, events.Perm ordered ∧ ordered.Pairwise (fun earlier later => ¬before later earlier) := by
  by_cases empty : events = []
  · subst events
    exact ⟨[], .nil, .nil⟩
  · obtain ⟨event, present, earliest⟩ := minimal irrefl trans events empty
    obtain ⟨left, right, shape⟩ := splitMember present
    obtain ⟨later, permuted, ordered⟩ := linearize irrefl trans (left ++ right)
    have extract : events.Perm (event :: (left ++ right)) := shape ▸ List.perm_middle
    refine ⟨event :: later, extract.trans (.cons event permuted), List.pairwise_cons.mpr ⟨?_, ordered⟩⟩
    intro other member
    have contained : other ∈ event :: later := List.mem_cons_of_mem _ member
    exact earliest other ((extract.trans (.cons event permuted)).mem_iff.mpr contained)
termination_by events.length
decreasing_by simp [shape]

inductive Action where
  | scored (frame : Nat) (result : ScoreResult)
  | returned (frame : Nat) (result : ScoreResult)
  | cleared (call : WriterTrace.Call Nat)
  | stored (frame : Option Nat) (call : WriterTrace.Call Nat) (board : Position) (entry : EntryData)

structure Event where
  id : Nat
  action : Action

def Event.score (event : Event) : Option ScoreResult :=
  match event.action with
  | .scored _ result => some result
  | _ => none

def Event.call (event : Event) : Option (WriterTrace.Call Nat) :=
  match event.action with
  | .cleared call | .stored _ call _ _ => some call
  | .scored _ _ | .returned _ _ => none

def Event.entry (event : Event) : Option (Position × EntryData) :=
  match event.action with
  | .stored _ _ board entry => some (board, entry)
  | _ => none

def scores (events : List Event) : List ScoreResult := events.reverse.filterMap Event.score
def calls (events : List Event) : List (WriterTrace.Call Nat) := events.filterMap Event.call
def history (events : List Event) : History := events.filterMap Event.entry

def Consumed (before : Nat → Nat → Prop) (events : List Event) (consumer : Nat) : Search :=
  fun tree pv alpha beta score =>
    ∃ event ∈ events, before event.id consumer ∧
      ∃ frame, event.action = .returned frame ⟨tree, pv, alpha, beta, score⟩

def UsedChild (identify : Threading.AtomicWord.Event → Nat)
    (before : Nat → Nat → Prop) (events : List Event) (consumer : Nat) : ChildResults
  | .serial => Consumed before events consumer
  | .report use => fun tree pv alpha beta score =>
    ∃ event ∈ events, ∃ (returned : Threading.AtomicWord.Event) (frame : Nat),
      event.id = identify returned ∧
      event.action = .returned frame ⟨tree, pv, alpha, beta, score⟩ ∧
      Threading.AtomicWord.SequencedBefore use.load returned ∧
      Threading.AtomicWord.SequencedBefore returned use.next ∧
      (∀ a b, Threading.CutoffVisibility.HappensBefore use.history a b → before (identify a) (identify b)) ∧
      before (identify use.ownerRead) consumer

theorem UsedChild.consumed (children : UsedChild identify before events consumer stage tree pv alpha beta score)
    (trans : ∀ {a b c}, before a b → before b c → before a c) :
    Consumed before events consumer tree pv alpha beta score := by
  cases stage with
  | serial => exact children
  | report use =>
    obtain ⟨event, member, returned, frame, id, action, _, beforeNext, agrees, afterRead⟩ := children
    have beforeRead := agrees returned use.ownerRead (.trans (.sequenced beforeNext) use.handoff)
    exact ⟨event, member, id.symm ▸ trans beforeRead afterRead, frame, action⟩

def Determined (owner : Option Nat) (past : List Event) : Search :=
  fun tree pv alpha beta score =>
    ∃ frame, owner = some frame ∧
      ∃ event ∈ past, event.action = .scored frame ⟨tree, pv, alpha, beta, score⟩

private theorem Determined.recorded (known : Determined owner past tree pv alpha beta score) :
    Recorded (scores past) tree pv alpha beta score := by
  obtain ⟨_, _, event, present, tag⟩ := known
  exact List.mem_filterMap.mpr ⟨event, List.mem_reverse.mpr present, by simp [Event.score, tag]⟩

def Action.Valid (mc : X86Config) (fc : FlipConfig) (endian : Bool)
    (children : ChildResults) (reads : RawSample → Prop) (past : List Event) : Action → Prop
  | .scored _ result =>
      ExecutionWithUses (PrefixExitFrom mc fc reads) children
        result.tree result.pv result.alpha result.beta result.score
  | .returned frame result => ∃ event ∈ past, event.action = .scored frame result
  | .cleared _ => True
  | .stored frame _ board entry => PrefixWriteFrom mc fc endian (Determined frame past) reads board entry

def Action.Words (endian : Bool) : Action → Prop
  | .scored _ _ | .returned _ _ => True
  | .cleared call => call.payload = ⟨0, 0, 0⟩
  | .stored _ call board entry => call.payload = SlotJournal.packed endian board entry

private theorem filterMapIncluded (f : Event → Option α) {earlier later : List Event}
    (included : ∀ e ∈ earlier, e ∈ later) : ∀ x ∈ earlier.filterMap f, x ∈ later.filterMap f := by
  intro x present
  obtain ⟨event, member, mapped⟩ := List.mem_filterMap.mp present
  exact List.mem_filterMap.mpr ⟨event, included _ member, mapped⟩

private theorem scoresIncluded {earlier later : List Event}
    (included : ∀ e ∈ earlier, e ∈ later) : ∀ r ∈ scores earlier, r ∈ scores later :=
  filterMapIncluded Event.score fun e member => by simpa using included e (by simpa using member)

private theorem Action.Valid.weaken (valid : Action.Valid mc fc endian children reads earlier action)
    (included : ∀ e ∈ earlier, e ∈ later)
    (childBefore : ∀ stage tree pv a b s, children stage tree pv a b s → Recorded (scores later) tree pv a b s)
    (readBefore : ∀ sample, reads sample → sample.publication (history later)) :
    Action.Valid mc fc endian (fun _ => Recorded (scores later)) (·.publication (history later)) later action := by
  cases action with
  | scored _ result =>
    exact (show ExecutionWithUses _ _ _ _ _ _ _ from valid).weaken
      childBefore (fun _ _ _ _ _ _ _ exit => exit.mapEvidence readBefore)
  | returned frame result =>
    obtain ⟨event, member, ready⟩ := valid
    exact ⟨event, included event member, ready⟩
  | cleared => exact valid
  | stored _ _ _ _ =>
    refine PrefixWriteFrom.mapEvidence (show PrefixWriteFrom mc fc endian _ _ _ _ from valid) ?_ readBefore
    rintro _ _ _ _ _ ⟨frame, owner, event, present, tag⟩
    exact ⟨frame, owner, event, included event present, tag⟩

private theorem replayEvent {event : Event} (run : Run mc fc endian (scores past) (calls past) (history past))
    (words : event.action.Words endian)
    (valid : event.action.Valid mc fc endian (fun _ => Recorded (scores past)) (·.publication (history past)) past) :
    Run mc fc endian (scores (past ++ [event])) (calls (past ++ [event])) (history (past ++ [event])) := by
  obtain ⟨id, action⟩ := event
  cases action with
  | scored _ result =>
    simpa [scores, calls, history, Event.score, Event.call, Event.entry, List.filterMap] using
      Run.scored result run valid
  | returned _ _ =>
    simpa [scores, calls, history, Event.score, Event.call, Event.entry, List.filterMap] using run
  | cleared call =>
    simpa [scores, calls, history, Event.score, Event.call, Event.entry, List.filterMap] using
      Run.clear call run words
  | stored _ call board entry =>
    have checked := PrefixWriteFrom.mapEvidence (show PrefixWriteFrom mc fc endian _ _ _ _ from valid)
      (fun _ _ _ _ _ known => Determined.recorded known) (fun _ seen => seen)
    simpa [scores, calls, history, Event.score, Event.call, Event.entry, List.filterMap] using
      Run.stored call board entry run checked words

noncomputable def predecessors (before : Nat → Nat → Prop) (events : List Event) (consumer : Nat) :
    List Event := by
  classical
  exact events.filter (fun other => decide (before other.id consumer))

theorem mem_predecessors :
    other ∈ predecessors before events consumer ↔ other ∈ events ∧ before other.id consumer := by
  classical
  simp [predecessors]

theorem scorePredecessor {earlier : Event} (result : ScoreResult)
    (present : earlier ∈ events) (edge : before earlier.id consumer)
    (scored : earlier.action = .scored frame result) :
    Recorded (scores (predecessors before events consumer)) result.tree result.pv result.alpha
      result.beta result.score := by
  apply List.mem_filterMap.mpr
  exact ⟨earlier, List.mem_reverse.mpr (mem_predecessors.mpr ⟨present, edge⟩),
    by simp [Event.score, scored]⟩

theorem publicationPredecessor (raw : RawSample)
    (even : raw.first &&& 1 = 0) (positive : 0 < raw.firstIndex)
    (beforeReturn : raw.memory.hb raw.reader.first consumer)
    (agrees : ∀ a b, raw.memory.hb a b → before a b)
    (tracked : ∀ call, raw.slot.history[raw.firstIndex - 1]? = some (.publish call) → call ∈ calls events)
    (physical : ∀ event ∈ events, ∀ call, event.call = some call → event.id = call.finish) :
    ∃ call, raw.slot.payloadAt (raw.firstIndex / 2) = call.payload ∧
      call ∈ calls (predecessors before events consumer) := by
  obtain ⟨call, index, payload, prior⟩ := raw.publishingCall even positive
  obtain ⟨publication, member, stored⟩ := List.mem_filterMap.mp (tracked call index)
  have edge : before publication.id consumer := by
    rw [physical publication member call stored]
    exact agrees _ _ (raw.memory.hb_trans prior beforeReturn)
  exact ⟨call, payload, List.mem_filterMap.mpr ⟨publication,
    mem_predecessors.mpr ⟨member, edge⟩, stored⟩⟩

private theorem knownWords {source : Event} (words : source.action.Words endian)
    (present : source ∈ past) (matched : source.call = some call) :
    SlotJournal.KnownPayload (history past) endian call.payload := by
  cases kind : source.action with
  | scored => simp [Event.call, kind] at matched
  | returned => simp [Event.call, kind] at matched
  | cleared written =>
    have same : written = call := by simpa [Event.call, kind] using matched
    have zero : written.payload = ⟨0, 0, 0⟩ := by simpa only [kind, Action.Words] using words
    rw [← same, zero]
    exact SlotJournal.KnownPayload.cleared _ _
  | stored _ written board entry =>
    have same : written = call := by simpa [Event.call, kind] using matched
    have packed : written.payload = SlotJournal.packed endian board entry := by
      simpa only [kind, Action.Words] using words
    rw [← same, packed]
    intro occupied
    have seen : (board, entry) ∈ history past :=
      List.mem_filterMap.mpr ⟨source, present, by simp [Event.entry, kind]⟩
    exact ⟨board, by simpa [SlotJournal.packed, Packing.unpack_pack], rfl, rfl⟩

inductive SlotOrigin (before : Nat → Nat → Prop) (events : List Event) (raw : RawSample) : Prop
  | empty : (Packing.unpack raw.bigEndian (raw.slot.payloadAt 0).metadata).bound = 0 →
      SlotOrigin before events raw
  | copied (event : Event) (call : WriterTrace.Call Nat) :
      event ∈ events → event.call = some call → before event.id raw.reader.first →
      raw.slot.payloadAt 0 = call.payload → SlotOrigin before events raw

structure ReadAt (before : Nat → Nat → Prop) (events : List Event) (consumer : Nat) (endian : Bool)
    (raw : RawSample) : Prop where
  byteOrder : raw.bigEndian = endian
  even : raw.first &&& 1 = 0
  origin : raw.firstIndex = 0 → SlotOrigin before events raw
  afterValidation : raw.memory.sb raw.reader.last consumer
  agrees : ∀ a b, raw.memory.hb a b → before a b
  tracked : ∀ call, raw.slot.history[raw.firstIndex - 1]? = some (.publish call) → call ∈ calls events

def SampleContract : Prop :=
  ∀ (before : Nat → Nat → Prop) (events : List Event) (consumer : Nat) (endian : Bool) (raw : RawSample),
    ReadAt before events consumer endian raw →
    (∀ {a b c}, before a b → before b c → before a c) →
    (∀ event ∈ events, ∀ call, event.call = some call → event.id = call.finish) →
    (∀ source ∈ events, source.action.Words endian) →
    raw.publication (history (predecessors before events consumer))

theorem sampleContract_proven : SampleContract := by
  intro before events consumer endian raw read trans physical words
  have packed : ∀ source ∈ events, source.action.Words raw.bigEndian := by rwa [read.byteOrder]
  have beforeUse := raw.memory.hb_trans (raw.memory.sb_hb raw.reader.first_before_fence)
    (raw.memory.hb_trans (raw.memory.sb_hb raw.reader.fence_before_last)
      (raw.memory.sb_hb read.afterValidation))
  change SlotJournal.KnownPayload _ raw.bigEndian (raw.slot.payloadAt (raw.firstIndex / 2))
  by_cases zero : raw.firstIndex = 0
  · rw [zero]
    cases read.origin zero with
    | empty vacant => exact fun occupied => (occupied vacant).elim
    | copied source call present matched earlier copied =>
      rw [copied]
      exact knownWords (packed source present) (mem_predecessors.mpr
        ⟨present, trans earlier (read.agrees _ _ beforeUse)⟩) matched
  · obtain ⟨call, payload, member⟩ :=
      publicationPredecessor raw read.even (Nat.pos_of_ne_zero zero) beforeUse read.agrees read.tracked physical
    obtain ⟨source, prior, recorded⟩ := List.mem_filterMap.mp member
    rw [payload]
    exact knownWords (packed source (mem_predecessors.mp prior).1) prior recorded

structure Observations (mc : X86Config) (fc : FlipConfig) (endian : Bool) (events : List Event) where
  identify : Threading.AtomicWord.Event → Nat
  before : Nat → Nat → Prop
  irrefl : ∀ a, ¬before a a
  trans : ∀ {a b c}, before a b → before b c → before a c
  words : ∀ event ∈ events, event.action.Words endian
  tablePublications : ∀ event ∈ events, ∀ call, event.call = some call → event.id = call.finish
  evidence : ∀ event ∈ events,
    event.action.Valid mc fc endian (UsedChild identify before events event.id) (ReadAt before events event.id endian)
      (predecessors before events event.id)

def Observations.ofPriorReturns (identify : Threading.AtomicWord.Event → Nat) (before : Nat → Nat → Prop)
    (irrefl : ∀ i, ¬before i i) (trans : ∀ {a b c}, before a b → before b c → before a c)
    (words : ∀ e ∈ events, e.action.Words endian)
    (publications : ∀ e ∈ events, ∀ call, e.call = some call → e.id = call.finish)
    (frame : ∀ e ∈ events, TableReads.LegalHints (history (predecessors before events e.id)) →
      (∀ stage tree pv a z v, UsedChild identify before events e.id stage tree pv a z v → Replayed tree pv a z v) →
      e.action.Valid mc fc endian (UsedChild identify before events e.id) (ReadAt before events e.id endian)
        (predecessors before events e.id)) : Observations mc fc endian events := by
  refine ⟨identify, before, irrefl, trans, words, publications, ?_⟩
  let valid (e : Event) := e.action.Valid mc fc endian (UsedChild identify before events e.id)
    (ReadAt before events e.id endian) (predecessors before events e.id)
  obtain ⟨ordered, perm, sorted⟩ := linearize (before := fun (a b : Event) => before a.id b.id)
    (fun e => irrefl e.id) (fun x y => trans x y) events
  have visit (past remaining : List Event) (allEvents : (past ++ remaining).Perm events)
      (sorted : remaining.Pairwise (fun a b => ¬before b.id a.id)) (known : ∀ e ∈ past, valid e)
      (run : Run mc fc endian (scores past) (calls past) (history past)) : ∀ e ∈ remaining, valid e := by
    induction remaining generalizing past with
    | nil => simp
    | cons e rest ih =>
      obtain ⟨after, orderedRest⟩ := List.pairwise_cons.mp sorted
      have included : ∀ old ∈ predecessors before events e.id, old ∈ past := by
        intro old predecessor
        obtain ⟨member, precedes⟩ := mem_predecessors.mp predecessor
        rcases List.mem_append.mp (allEvents.mem_iff.mpr member) with prev | later
        · exact prev
        · rcases List.mem_cons.mp later with rfl | later
          · exact (irrefl _ precedes).elim
          · exact (after old later precedes).elim
      have recorded : ∀ stage tree pv a z v, UsedChild identify before events e.id stage tree pv a z v →
          Recorded (scores past) tree pv a z v := by
        intro stage tree pv a z v child
        obtain ⟨ret, member, used, frameId, tag⟩ := child.consumed trans
        have retValid := known ret (included _ (mem_predecessors.mpr ⟨member, used⟩))
        change ret.action.Valid .. at retValid
        rw [tag] at retValid
        obtain ⟨ev, earlier, computed⟩ := retValid
        obtain ⟨seen, prior⟩ := mem_predecessors.mp earlier
        exact scoresIncluded included _ (scorePredecessor _ seen (trans prior used) computed)
      have member := allEvents.mem_iff.mp (show e ∈ past ++ e :: rest by simp)
      have now := frame e member
        (fun b entry old => run.legalHints b entry (filterMapIncluded Event.entry included _ old))
        (fun stage tree pv a z v child => ⟨_, (CausalReplay.contract_proven mc fc endian _ _ _ run).1
          tree pv a z v (recorded stage tree pv a z v child)⟩)
      have next := replayEvent run (words e member) (now.weaken included recorded (fun raw sample occupied => by
        obtain ⟨b, found, player, opponent⟩ := sampleContract_proven before events e.id endian raw sample
          trans publications words occupied
        exact ⟨b, filterMapIncluded Event.entry included _ found, player, opponent⟩))
      have restKnown := ih (past ++ [e]) (by simpa using allEvents) orderedRest (by
        intro old member
        rcases List.mem_append.mp member with prev | current
        · exact known old prev
        · simpa using (List.mem_singleton.mp current) ▸ now) next
      intro old member
      rcases List.mem_cons.mp member with rfl | later
      · exact now
      · exact restKnown old later
  have checked := visit [] ordered (by simpa using perm.symm) sorted (by simp) .start
  exact fun e mem => checked e (perm.mem_iff.mp mem)

/-- A frame that uses a TT move may depend on the legality of earlier published moves. -/
def Observations.ofPriorHints (identify : Threading.AtomicWord.Event → Nat) (before : Nat → Nat → Prop)
    (irrefl : ∀ i, ¬before i i) (trans : ∀ {a b c}, before a b → before b c → before a c)
    (words : ∀ e ∈ events, e.action.Words endian)
    (publications : ∀ e ∈ events, ∀ call, e.call = some call → e.id = call.finish)
    (frame : ∀ e ∈ events, TableReads.LegalHints (history (predecessors before events e.id)) →
      e.action.Valid mc fc endian (UsedChild identify before events e.id) (ReadAt before events e.id endian)
        (predecessors before events e.id)) : Observations mc fc endian events :=
  Observations.ofPriorReturns identify before irrefl trans words publications (fun e h prior _ => frame e h prior)

theorem Observations.consumedScore (observed : Observations mc fc endian events)
    (completed : Consumed observed.before events consumer tree pv alpha beta score) :
    Recorded (scores (predecessors observed.before events consumer)) tree pv alpha beta score := by
  obtain ⟨event, member, used, frame, tag⟩ := completed
  have returned := observed.evidence event member
  rw [tag] at returned
  obtain ⟨known, preceding, same⟩ := returned
  obtain ⟨present, edge⟩ := mem_predecessors.mp preceding
  exact scorePredecessor _ present (observed.trans edge used) same

theorem Observations.usedScore (observed : Observations mc fc endian events)
    (completed : UsedChild observed.identify observed.before events consumer stage tree pv alpha beta score) :
    Recorded (scores (predecessors observed.before events consumer)) tree pv alpha beta score :=
  observed.consumedScore (completed.consumed observed.trans)

private theorem replayOrdered (observed : Observations mc fc endian events)
    (past remaining : List Event)
    (same : (past ++ remaining).Perm events)
    (ordered : (past ++ remaining).Pairwise (fun earlier later => ¬observed.before later.id earlier.id))
    (run : Run mc fc endian (scores past) (calls past) (history past)) :
    Run mc fc endian (scores (past ++ remaining)) (calls (past ++ remaining)) (history (past ++ remaining)) := by
  induction remaining generalizing past with
  | nil => simpa using run
  | cons event rest ih =>
    have eventPresent : event ∈ events := same.mem_iff.mp (by simp)
    have follows := List.pairwise_cons.mp (List.pairwise_append.mp ordered).2.1
    have included : ∀ dep ∈ predecessors observed.before events event.id, dep ∈ past := by
      intro dep evidence
      obtain ⟨present, edge⟩ := mem_predecessors.mp evidence
      rcases List.mem_append.mp (same.mem_iff.mpr present) with earlier | later
      · exact earlier
      · rcases List.mem_cons.mp later with equal | future
        · exact False.elim (observed.irrefl event.id (equal ▸ edge))
        · exact False.elim (follows.1 _ future edge)
    have valid := (observed.evidence event eventPresent).weaken included
      (fun _ _ _ _ _ _ child => scoresIncluded included _ (observed.usedScore child)) (by
        intro raw read occupied
        have checked := sampleContract_proven observed.before events event.id endian raw read observed.trans
          observed.tablePublications observed.words
        obtain ⟨board, found, player, opponent⟩ := checked occupied
        exact ⟨board, filterMapIncluded Event.entry included _ found, player, opponent⟩)
    simpa only [List.append_assoc, List.singleton_append] using
      ih (past ++ [event])
        (by simpa only [List.append_assoc, List.singleton_append] using same)
        (by simpa only [List.append_assoc, List.singleton_append] using ordered)
        (replayEvent run (observed.words event eventPresent) valid)

def Contract : Prop :=
  ∀ mc fc endian events (observed : Observations mc fc endian events),
    ∃ ordered, events.Perm ordered ∧
      ordered.Pairwise (fun earlier later => ¬observed.before later.id earlier.id) ∧
      Run mc fc endian (scores ordered) (calls ordered) (history ordered)

theorem contract_proven : Contract := by
  intro mc fc endian events observed
  obtain ⟨ordered, same, sorted⟩ := linearize (before := fun a b : Event => observed.before a.id b.id)
    (fun a => observed.irrefl a.id) (fun a b => observed.trans a b) events
  exact ⟨ordered, same, sorted, replayOrdered observed [] ordered same.symm sorted .start⟩

theorem Observations.legalHints (observed : Observations mc fc endian events) :
    TableReads.LegalHints (history events) := by
  obtain ⟨ordered, order, _, run⟩ := contract_proven mc fc endian events observed
  intro board entry member
  obtain ⟨event, seen, action⟩ := List.mem_filterMap.mp member
  exact run.legalHints board entry (List.mem_filterMap.mpr ⟨event, order.mem_iff.mp seen, action⟩)

def PresearchHintContract : Prop :=
  ∀ mc fc endian events (observed : Observations mc fc endian events) (consumer : Nat) (board : Position)
    (attempts : List ProbeRetry.RawAttempt) (slot : Fin 2) (entry : EntryData) (sq : Bitboard.Square)
    (moves ordered : List (PvsBoardTree.GeneratedSquare mc board.val))
    (heuristic : PvsBoardTree.GeneratedSquare mc board.val → Int),
    TableReads.ProbeReadsFrom (ReadAt observed.before events consumer endian) attempts →
    ProbeRetry.probe (attempts.map (ProbeRetry.observe (Cutoffs.key board))) = .hit slot entry →
    Packing.decodeBestMove entry.bestMove = some sq →
    moves.Perm (PvsBoardTree.legalSquares mc board.val) →
    (∀ move ∈ moves, move.val ≠ sq → heuristic move < MoveList.Ordering.ttMoveValue) →
    MoveList.Ordering.TtPresearchOrder Subtype.val heuristic sq moves ordered →
    ∃ (first : PvsBoardTree.GeneratedSquare mc board.val)
      (rest : List (PvsBoardTree.GeneratedSquare mc board.val)), first.val = sq ∧
      ordered = first :: rest ∧
      (first :: rest).Perm (PvsBoardTree.legalSquares mc board.val)

theorem presearchHintContract_proven : PresearchHintContract := by
  intro mc fc endian events observed consumer board attempts slot entry sq moves ordered heuristic
    observedProbe hit decoded order below orderedByHint
  have reads : TableReads.ProbeReads (history events) attempts := observedProbe.mapEvidence (by
    intro raw read occupied
    have sampled := sampleContract_proven observed.before events consumer endian raw read observed.trans
      observed.tablePublications observed.words
    obtain ⟨b, member, player, opponent⟩ := sampled occupied
    obtain ⟨event, previous, publication⟩ := List.mem_filterMap.mp member
    exact ⟨b, List.mem_filterMap.mpr ⟨event, (mem_predecessors.mp previous).1, publication⟩, player, opponent⟩)
  obtain ⟨chosen, present, value⟩ := (reads.legalHint observed.legalHints hit).generated (mc := mc) decoded
  have front := MoveList.Ordering.presearchedOrderContract_proven _ _ Subtype.val moves ordered sq heuristic
    ⟨chosen, order.mem_iff.mpr present, value⟩ below orderedByHint
  cases ordered with
  | nil => simp at front
  | cons first rest => exact ⟨first, rest, by simpa using front.2, rfl, front.1.trans order⟩

def RootAttempt (mc : X86Config) (fc : FlipConfig) (endian : Bool) (board : Position)
    (roots : List Bitboard.Square) (idx : Nat) (alpha beta score : Int)
    (payload : Bitboard.Square → Int) : Prop :=
  ∃ (events : List Event) (observed : Observations mc fc endian events) (completion : Nat),
    RootEvidence.AttemptWithUses (UsedChild observed.identify observed.before events completion)
      mc fc board roots idx alpha beta score payload

private theorem RootAttempt.causal (attempt : RootAttempt mc fc endian board roots idx alpha beta score payload) :
    ∃ previous writes table, Run mc fc endian previous writes table ∧
      RootEvidence.AttemptWith (Recorded previous) mc fc board roots idx alpha beta score payload := by
  obtain ⟨events, observed, _, root⟩ := attempt
  obtain ⟨ordered, same, _, run⟩ := contract_proven mc fc endian events observed
  exact ⟨scores ordered, calls ordered, history ordered, run, root.weaken
    (fun _ _ _ _ _ _ child => scoresIncluded (fun _ member => same.mem_iff.mp (mem_predecessors.mp member).1)
      _ (observed.usedScore child))⟩

theorem RootAttempt.sound (attempt : RootAttempt mc fc endian board roots idx alpha beta score payload)
    (covers : Root.Covers mc board roots) (inside : idx < roots.length)
    (window : Aspiration.Window alpha beta) :
    RootOrder.SoundAttempt mc fc board roots idx alpha beta score payload := by
  obtain ⟨previous, writes, table, run, root⟩ := attempt.causal
  have verified := (CausalReplay.contract_proven mc fc endian previous writes table run).1
  have replay : RootEvidence.Attempt mc fc board roots idx alpha beta score payload :=
    root.weaken (fun _ _ _ _ _ child => ⟨previous.length, verified _ _ _ _ _ child⟩)
  exact replay.sound covers inside window

def RootChecked (mc : X86Config) (fc : FlipConfig) (endian : Bool) (board : Position)
    (roots : List Bitboard.Square) (idx : Nat) (search : RootAspiration.Search) : Prop :=
  ∀ attempt alpha beta, Aspiration.Window alpha beta → (search attempt alpha beta).aborted = false →
    RootAttempt mc fc endian board roots idx alpha beta
      (search attempt alpha beta).score (search attempt alpha beta).payload

def RootContract : Prop :=
  ∀ mc fc endian board roots idx search, RootChecked mc fc endian board roots idx search →
    CausalReplay.RootChecked mc fc endian board roots idx search

theorem rootContract_proven : RootContract := by
  intro mc fc endian board roots idx search checked attempt alpha beta window running
  exact (checked attempt alpha beta window running).causal

def SingleReturnedContract : Prop :=
  ∀ mc fc endian board roots search fuel alpha beta accepted
    (pv : Bitboard.Square → List Bitboard.Square) (depth nodes : Nat),
    Root.Covers mc board roots → 0 < roots.length → Aspiration.Window alpha beta →
    Aspiration.runWith search fuel 0 alpha beta 256 = some accepted →
    RootAttempt mc fc endian board roots 0 accepted.alpha accepted.beta accepted.trial.score
      accepted.trial.payload →
    let rows := roots.map (fun sq => (⟨sq, accepted.trial.payload sq, pv sq⟩ : SearchPublication.Row))
    ∃ result,
      SearchFallback.fromSnapshot (SearchPublication.capture rows ⟨depth, 5⟩ true) nodes = some result ∧
      result.best.score = PvsBoardTree.minimax mc fc board

theorem singleReturnedContract_proven : SingleReturnedContract := by
  intro mc fc endian board roots search fuel alpha beta accepted pv depth nodes covers present window done attempt
  have witness := Aspiration.runWith_witness search window (by decide) done
  exact ExactEndgameScore.singleResult_of_accepted covers present witness
    (attempt.sound covers present witness.window) pv depth nodes

def AllMovesReturnedContract : Prop :=
  ∀ mc fc endian board roots (observe : RootMultiPV.Observations) (seed : RootMultiPV.Seed)
    (pv : Bitboard.Square → List Bitboard.Square) (depth nodes : Nat),
    Root.Covers mc board roots → 0 < roots.length →
    RootMultiPV.LineCalls observe seed (fun order idx _ =>
      let (alpha, beta) := seed order idx
      ∃ accepted, Aspiration.Window alpha beta ∧
        Aspiration.runWith (observe order idx) 11 0 alpha beta 256 = some accepted ∧
        RootAttempt mc fc endian board order idx accepted.alpha accepted.beta accepted.trial.score
          accepted.trial.payload) roots.length 0 roots →
    ∃ batch result,
      RootMultiPV.run observe seed roots.length 0 roots = some batch ∧
      let rows := batch.picks.map (fun pick => (⟨pick.square, pick.score, pv pick.square⟩ : SearchPublication.Row))
      SearchFallback.fromSnapshot (SearchPublication.capture rows ⟨depth, 5⟩ true) nodes = some result ∧
        result.best.score = PvsBoardTree.minimax mc fc board ∧ result.endgame = true

theorem allMovesReturnedContract_proven : AllMovesReturnedContract := by
  intro mc fc endian board roots observe seed pv depth nodes covers positive calls
  obtain ⟨batch, found, length, _, _, _, ranked⟩ := calls.ranked (by
    intro order idx line legal inside ran checked
    obtain ⟨accepted, window, chosen, attempt⟩ := checked
    have witness := Aspiration.runWith_witness (observe order idx) window (by decide) chosen
    obtain ⟨expected, finished, _, _, valid⟩ := RootAspiration.line_of_accepted inside witness
      (attempt.sound legal inside witness.window)
    have equal : expected = line := Option.some.inj (finished.symm.trans
      (by simpa [RootAspiration.runLine, chosen] using ran))
    exact equal ▸ valid) covers (by simp)
  have present : batch.picks ≠ [] := by
    intro empty
    simp [empty] at length
    omega
  obtain ⟨result, saved, score, endgame, _⟩ :=
    ExactEndgameScore.rootResult_of_ranked covers batch.picks present ranked pv depth nodes [] (by simp)
  exact ⟨batch, result, found, by simpa using saved, score, endgame⟩

end NeuralReversi.PvsParallel.ConcurrentReplay
