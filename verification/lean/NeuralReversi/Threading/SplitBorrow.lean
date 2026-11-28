module
public import NeuralReversi.Threading.SplitShutdown

@[expose] public section

namespace NeuralReversi.Threading.SplitBorrow

open AtomicWord (Event SequencedBefore Write Read)
open CutoffVisibility (History HappensBefore)
open SplitCompletion (RetainedThrough)
open SplitPublication (PublishedThrough)

inductive Resource where
  | table
  | evaluator
  | rootMoves
  deriving DecidableEq

structure ResourceScope where
  object : Nat
  ready : Event
  release : Event

structure RootBorrow where
  capture : Event
  scope : Resource → ResourceScope

structure Task where
  initialized : Event
  objects : Resource → Nat

inductive TaskLineage (history : History) (root : RootBorrow) : Task → Prop
  | first (initialized : Event) :
      SequencedBefore root.capture initialized →
      TaskLineage history root ⟨initialized, fun kind => (root.scope kind).object⟩
  | nested {parent : Task} (capture initialized : Event) :
      TaskLineage history root parent →
      PublishedThrough history parent.initialized capture →
      SequencedBefore capture initialized →
      TaskLineage history root ⟨initialized, parent.objects⟩

def BorrowOriginContract : Prop :=
  ∀ (history : History) (root : RootBorrow) (task : Task),
    TaskLineage history root task →
    (∀ kind, task.objects kind = (root.scope kind).object) ∧
      HappensBefore history root.capture task.initialized

theorem borrowOriginContract_proven : BorrowOriginContract := by
  intro history root task lineage
  induction lineage with
  | first _ before => exact ⟨fun _ => rfl, .sequenced before⟩
  | nested capture initialized _ published copied ih =>
    refine ⟨ih.1, .trans ih.2 (.trans ?_ (.sequenced copied))⟩
    exact SplitPublication.initializationContract_proven history _ capture published

inductive ResourceReady (history : History) : Resource → Event → Event → Prop
  | stack : SequencedBefore ready capture → ResourceReady history .rootMoves ready capture
  | message {kind : Resource} (shared : kind ≠ .rootMoves)
      {writes : List Write} (sent : Fin writes.length) (received : Read writes) :
      HappensBefore history ready (writes.get sent).event →
      SplitShutdown.MessageRead history sent received → SequencedBefore received.event capture →
      ResourceReady history kind ready capture

private theorem ResourceReady.orders (ready : ResourceReady history kind initialized capture) :
    HappensBefore history initialized capture := by
  cases ready with
  | stack before => exact .sequenced before
  | message _ sent received before handoff after =>
    have delivered := (SplitShutdown.messageHandoffContract_proven history _ sent received handoff).2
    exact .trans before (.trans delivered (.sequenced after))

inductive ResourceEnd (history : History) : Resource → Event → Event → Prop
  | stack : ResourceEnd history .rootMoves destroy destroy
  | arc {kind : Resource} (shared : kind ≠ .rootMoves)
      {writes : List Write} {strong : ArcDrop.StrongHistory writes}
      (last : ArcDrop.LastRelease strong) (held : Fin writes.length) :
      strong.publication.val < held.val → (writes.get held).release = true →
      ArcDrop.DestructorAcquire history last.read destroy →
      ResourceEnd history kind (writes.get held).event destroy

private theorem ResourceEnd.orders (ends : ResourceEnd history kind release destroy) :
    release = destroy ∨ HappensBefore history release destroy := by
  cases ends with
  | stack => exact Or.inl rfl
  | arc _ last held published released acquire =>
    exact Or.inr (ArcDrop.releaseToDestructorContract_proven _ _ _ last _ held published released acquire)

private theorem retained_trans (first : RetainedThrough history access localWait)
    (rest : RetainedThrough history localWait rootWait) : RetainedThrough history access rootWait := by
  induction rest with
  | here => exact first
  | wait retirement index released _ before ih =>
    exact .wait retirement index released ih before

def BorrowValidityContract : Prop :=
  ∀ (history : History) (root : RootBorrow) (task : Task) (kind : Resource)
    (access slotWait clear rootWait destroy : Event),
    TaskLineage history root task →
    ResourceReady history kind (root.scope kind).ready root.capture →
    PublishedThrough history task.initialized access →
    RetainedThrough history access slotWait → SequencedBefore slotWait clear →
    RetainedThrough history slotWait rootWait →
    SequencedBefore rootWait (root.scope kind).release →
    ResourceEnd history kind (root.scope kind).release destroy →
    task.objects kind = (root.scope kind).object ∧
      HappensBefore history task.initialized access ∧ HappensBefore history access clear ∧
      HappensBefore history (root.scope kind).ready access ∧
      HappensBefore history access (root.scope kind).release ∧ HappensBefore history access destroy

theorem borrowValidityContract_proven : BorrowValidityContract := by
  intro history root task kind access slotWait clear rootWait destroy
    lineage ready published held clearing enclosing release ends
  obtain ⟨objects, capture⟩ := borrowOriginContract_proven history root task lineage
  obtain ⟨initialized, slotAlive⟩ := SplitPublication.taskAccessContract_proven
    history task.initialized access slotWait clear published held clearing
  have keep := retained_trans held enclosing
  have beforeRelease := SplitCompletion.retainedAccessContract_proven
    history access rootWait (root.scope kind).release keep release
  refine ⟨objects kind, initialized, slotAlive, .trans ready.orders (.trans capture initialized),
    beforeRelease, ?_⟩
  rcases ends.orders with same | later
  · exact same ▸ beforeRelease
  · exact .trans beforeRelease later

def FinalizerTaskReadContract : Prop :=
  ∀ (history : History) (initialized wait read clear : Event),
    SequencedBefore initialized wait → SequencedBefore wait read → SequencedBefore read clear →
    HappensBefore history initialized read ∧ HappensBefore history read clear

theorem finalizerTaskReadContract_proven : FinalizerTaskReadContract := by
  intro history initialized wait read clear published completed beforeClear
  exact ⟨.trans (.sequenced published) (.sequenced completed), .sequenced beforeClear⟩

end NeuralReversi.Threading.SplitBorrow
