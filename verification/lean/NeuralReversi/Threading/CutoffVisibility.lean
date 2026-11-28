module
public import NeuralReversi.Threading.SplitCutoff
public import NeuralReversi.Threading.AtomicWord

@[expose] public section

namespace NeuralReversi.Threading.CutoffVisibility

open SplitCutoff (Epoch Cache Observation poll reset)

export AtomicWord (Event SequencedBefore)

abbrev EpochWrite := AtomicWord.Write
abbrev EpochRead := AtomicWord.Read

structure FlagWrite where
  event : Event
  value : Bool

structure FlagRead (flags : Nat → List FlagWrite) where
  slot : Nat
  event : Event
  source : Fin (flags slot).length

def FlagRead.value (read : FlagRead flags) : Bool :=
  (flags read.slot).get read.source |>.value

structure History where
  epochs : List EpochWrite
  flags : Nat → List FlagWrite
  epochReads : List (EpochRead epochs)
  flagReads : List (FlagRead flags)
  otherSync : Event → Event → Prop

def InReleaseSequence (history : History) (head source : Fin history.epochs.length) : Prop :=
  AtomicWord.InReleaseSequence history.epochs head source

inductive HappensBefore (history : History) : Event → Event → Prop
  | sequenced : SequencedBefore a b → HappensBefore history a b
  | synchronized (head : Fin history.epochs.length) (read : EpochRead history.epochs) :
      read ∈ history.epochReads → read.acquire = true →
      InReleaseSequence history head read.source →
      HappensBefore history (history.epochs.get head).event read.event
  | other : history.otherSync a b → HappensBefore history a b
  | trans : HappensBefore history a b → HappensBefore history b c → HappensBefore history a c

structure Coherent (history : History) : Prop where
  epochReadRead : ∀ a ∈ history.epochReads, ∀ b ∈ history.epochReads,
    HappensBefore history a.event b.event → a.source.val ≤ b.source.val
  flagWriteRead : ∀ r ∈ history.flagReads, ∀ w : Fin (history.flags r.slot).length,
    HappensBefore history ((history.flags r.slot).get w).event r.event → w.val ≤ r.source.val
  flagReadWrite : ∀ r ∈ history.flagReads, ∀ w : Fin (history.flags r.slot).length,
    HappensBefore history r.event ((history.flags r.slot).get w).event → r.source.val < w.val

def predecessor (i : Fin length) : Fin length :=
  ⟨i.val - 1, Nat.lt_of_le_of_lt (Nat.sub_le _ _) i.isLt⟩

structure Counter (history : History) : Prop where
  initial : ∀ i : Fin history.epochs.length, i.val = 0 → (history.epochs.get i).value = 0
  increment : ∀ i : Fin history.epochs.length, 0 < i.val →
    (history.epochs.get i).release = true ∧ (history.epochs.get i).rmw = true ∧
    (history.epochs.get i).value = (history.epochs.get (predecessor i)).value + 1

theorem epoch_value (history : History) (counter : Counter history)
    (n : Nat) (present : n < history.epochs.length) :
    (history.epochs.get ⟨n, present⟩).value = BitVec.ofNat 64 n := by
  cases n with
  | zero => exact counter.initial _ rfl
  | succ n =>
    have step := (counter.increment ⟨n + 1, present⟩ (Nat.zero_lt_succ n)).2.2
    rw [step]
    change (history.epochs.get ⟨n, by omega⟩).value + 1 = BitVec.ofNat 64 (n + 1)
    rw [epoch_value history counter n (by omega), BitVec.ofNat_add]
    rfl
termination_by n

structure Publication (history : History) where
  slot : Nat
  mark : Fin (history.flags slot).length
  epoch : Fin history.epochs.length
  increment : 0 < epoch.val
  set : ((history.flags slot).get mark).value = true
  before : SequencedBefore ((history.flags slot).get mark).event (history.epochs.get epoch).event

private theorem acquire_mark (history : History) (counter : Counter history)
    (publication : Publication history) (load : EpochRead history.epochs)
    (present : load ∈ history.epochReads) (acquire : load.acquire = true)
    (includes : publication.epoch.val ≤ load.source.val) :
    HappensBefore history ((history.flags publication.slot).get publication.mark).event load.event := by
  have release := (counter.increment publication.epoch publication.increment).1
  have later := fun (i : Fin history.epochs.length) (after : publication.epoch.val < i.val)
      (_ : i.val ≤ load.source.val) => (counter.increment i (by omega)).2.1
  exact .trans (.sequenced publication.before)
    (.synchronized publication.epoch load present acquire ⟨release, includes, later⟩)

inductive Walk (flags : Nat → List FlagWrite) : List Nat → List (FlagRead flags) → Prop
  | empty : Walk flags [] []
  | found (read : FlagRead flags) (tail : List Nat) :
      read.value = true → Walk flags (read.slot :: tail) [read]
  | descend (read : FlagRead flags) :
      read.value = false → Walk flags tail rest → Walk flags (read.slot :: tail) (read :: rest)

theorem walk_finds (walk : Walk flags chain reads) (target : Nat) (member : target ∈ chain)
    (visible : ∀ r ∈ reads, r.slot = target → r.value = true) :
    true ∈ reads.map FlagRead.value := by
  induction walk with
  | empty => simp at member
  | found read _ set => simp [set]
  | descend read _ tail ih =>
    cases List.mem_cons.mp member with
    | inl same => exact List.mem_map.mpr ⟨read, by simp, visible read (by simp) same.symm⟩
    | inr later =>
      exact List.mem_cons_of_mem _ (ih later (fun r seen => visible r (List.mem_cons_of_mem _ seen)))

structure FlagReads (history : History) (load : EpochRead history.epochs) where
  chain : List Nat
  reads : List (FlagRead history.flags)
  afterLoad : ∀ r ∈ reads, r ∈ history.flagReads ∧ SequencedBefore load.event r.event

def FlagReads.observation {history : History} {load : EpochRead history.epochs}
    (scan : FlagReads history load) : Observation :=
  ⟨(history.epochs.get load.source).value, scan.reads.map FlagRead.value⟩

def Uncleared {history : History} (publication : Publication history)
    {load : EpochRead history.epochs} (scan : FlagReads history load) : Prop :=
  ∀ r ∈ scan.reads, r.slot = publication.slot →
    ∀ i : Fin (history.flags publication.slot).length,
      publication.mark.val < i.val → i.val ≤ r.source.val →
      ((history.flags publication.slot).get i).value = true

def OrdersClearsAfter {history : History} (publication : Publication history)
    {load : EpochRead history.epochs} (scan : FlagReads history load) : Prop :=
  ∀ r ∈ scan.reads, r.slot = publication.slot →
    ∀ i : Fin (history.flags r.slot).length, publication.mark.val < i.val →
      ((history.flags r.slot).get i).value = false →
      HappensBefore history r.event ((history.flags r.slot).get i).event

def ClearOrderContract : Prop :=
  ∀ (history : History) (_ : Coherent history) (publication : Publication history)
    (load : EpochRead history.epochs) (scan : FlagReads history load),
    OrdersClearsAfter publication scan → Uncleared publication scan

theorem clearOrderContract_proven : ClearOrderContract := by
  intro history coherent publication load scan retired read member sameSlot stamp after observed
  cases read with
  | mk slot event source =>
    change slot = publication.slot at sameSlot
    subst slot
    cases value : ((history.flags publication.slot).get stamp).value
    · have ordered := retired _ member rfl stamp after value
      have earliest := coherent.flagReadWrite _ (scan.afterLoad _ member).1 stamp ordered
      change stamp.val ≤ source.val at observed
      change source.val < stamp.val at earliest
      omega
    · rfl

def ScanContract : Prop :=
  ∀ (history : History) (_ : Counter history) (_ : Coherent history)
    (publication : Publication history) (load : EpochRead history.epochs)
    (scan : FlagReads history load),
    load ∈ history.epochReads → load.acquire = true → publication.slot ∈ scan.chain →
    Walk history.flags scan.chain scan.reads →
    OrdersClearsAfter publication scan → publication.epoch.val ≤ load.source.val →
    true ∈ scan.observation.chain

theorem scanContract_proven : ScanContract := by
  intro history counter coherent publication load scan present acquire member walked clears includes
  have uncleared := clearOrderContract_proven history coherent publication load scan clears
  apply walk_finds walked publication.slot member
  intro read readIn sameSlot
  obtain ⟨inHistory, after⟩ := scan.afterLoad read readIn
  have hb := HappensBefore.trans (acquire_mark history counter publication load present acquire includes)
    (.sequenced after)
  cases read with
  | mk slot event source =>
    change slot = publication.slot at sameSlot
    subst slot
    have order := coherent.flagWriteRead _ inHistory publication.mark hb
    change publication.mark.val ≤ source.val at order
    change ((history.flags publication.slot).get source).value = true
    by_cases exactMark : publication.mark.val = source.val
    · rw [← Fin.ext exactMark]
      exact publication.set
    · exact uncleared _ readIn rfl source (by omega) (Nat.le_refl _)

structure PollRead (history : History) (cache : Cache) where
  load : EpochRead history.epochs
  present : load ∈ history.epochReads
  acquire : load.acquire = true
  flags : FlagReads history load
  walkOnMiss : flags.observation.epoch ≠ cache.seen → Walk history.flags flags.chain flags.reads

private theorem load_progress (history : History) (coherent : Coherent history)
    (before after : EpochRead history.epochs)
    (oldRead : before ∈ history.epochReads) (newRead : after ∈ history.epochReads)
    (order : SequencedBefore before.event after.event) : before.source.val ≤ after.source.val :=
  coherent.epochReadRead before oldRead after newRead (.sequenced order)

theorem relativeEpoch (history : History) (counter : Counter history) (coherent : Coherent history)
    (before after : EpochRead history.epochs)
    (oldRead : before ∈ history.epochReads) (newRead : after ∈ history.epochReads)
    (order : SequencedBefore before.event after.event) :
    (history.epochs.get after.source).value =
      (history.epochs.get before.source).value + BitVec.ofNat 64 (after.source.val - before.source.val) := by
  have serial := load_progress history coherent before after oldRead newRead order
  rw [epoch_value history counter, epoch_value history counter _ before.source.isLt,
    ← BitVec.ofNat_add, Nat.add_sub_of_le serial]

private theorem visible_on_miss {history : History} {cache : Cache}
    (counter : Counter history) (coherent : Coherent history) (mark : Publication history)
    (attempt : PollRead history cache) (member : mark.slot ∈ attempt.flags.chain)
    (clears : OrdersClearsAfter mark attempt.flags) (miss : attempt.flags.observation.epoch ≠ cache.seen)
    (includes : mark.epoch.val ≤ attempt.load.source.val) :
    true ∈ attempt.flags.observation.chain :=
  scanContract_proven history counter coherent mark attempt.load attempt.flags
    attempt.present attempt.acquire member (attempt.walkOnMiss miss) clears includes

def ResetPollContract : Prop :=
  ∀ (history : History) (_ : Counter history) (_ : Coherent history)
    (mark : Publication history) (initial : EpochRead history.epochs)
    (attempt : PollRead history (reset (history.epochs.get initial.source).value)),
    initial ∈ history.epochReads → SequencedBefore initial.event attempt.load.event →
    mark.slot ∈ attempt.flags.chain → OrdersClearsAfter mark attempt.flags →
    (attempt.load.source.val - initial.source.val + 1) % 2^64 ≠ 0 →
    let next := poll (reset (history.epochs.get initial.source).value) attempt.flags.observation
    (mark.epoch.val ≤ attempt.load.source.val → next.latched = true) ∧
    (next.latched = false →
      next.seen = (history.epochs.get attempt.load.source).value ∧ attempt.load.source.val < mark.epoch.val)

theorem resetPollContract_proven : ResetPollContract := by
  intro history counter coherent mark initial attempt present ordered member clears noAlias
  let raw := (history.epochs.get initial.source).value
  let read := attempt.flags.observation
  have relative := relativeEpoch history counter coherent initial attempt.load present attempt.present ordered
  obtain ⟨alias, update⟩ := SplitCutoff.resetContract_proven raw _ read relative
  have changed := fun eq => noAlias (alias.mp eq)
  have visible := visible_on_miss counter coherent mark attempt member clears changed
  obtain ⟨seen, latch⟩ := update noAlias
  refine ⟨fun published => latch.mpr (visible published), fun pending => ⟨seen, ?_⟩⟩
  by_cases published : mark.epoch.val ≤ attempt.load.source.val
  · have seenTrue := latch.mpr (visible published)
    change (poll (reset raw) read).latched = false at pending
    simp [pending] at seenTrue
  · omega

def CachedPollContract : Prop :=
  ∀ (history : History) (_ : Counter history) (_ : Coherent history)
    (mark : Publication history) (previous : EpochRead history.epochs)
    (cache : Cache) (attempt : PollRead history cache),
    previous ∈ history.epochReads → SequencedBefore previous.event attempt.load.event →
    cache.seen = (history.epochs.get previous.source).value →
    previous.source.val < mark.epoch.val →
    mark.slot ∈ attempt.flags.chain → OrdersClearsAfter mark attempt.flags →
    attempt.load.source.val - previous.source.val < 2^64 →
    let next := poll cache attempt.flags.observation
    (mark.epoch.val ≤ attempt.load.source.val → next.latched = true) ∧
    (next.latched = false →
      next.seen = (history.epochs.get attempt.load.source).value ∧ attempt.load.source.val < mark.epoch.val)

theorem cachedPollContract_proven : CachedPollContract := by
  intro history counter coherent mark previous cache attempt present ordered seen before
    member clears noWrap
  let read := attempt.flags.observation
  have serialOrder := load_progress history coherent previous attempt.load present attempt.present ordered
  have oldWord := seen.trans (epoch_value history counter _ previous.source.isLt)
  have epochWord : read.epoch = BitVec.ofNat 64 attempt.load.source.val :=
    epoch_value history counter _ attempt.load.source.isLt
  have visible (published : mark.epoch.val ≤ attempt.load.source.val) : true ∈ read.chain := by
    apply visible_on_miss counter coherent mark attempt member clears _ published
    intro sameWord
    have indices := SplitCutoff.epoch_index_eq _ _ serialOrder noWrap
      (epochWord.symm.trans (sameWord.trans oldWord))
    omega
  obtain ⟨latch, miss⟩ := SplitCutoff.visibilityContract_proven cache read
    previous.source.val attempt.load.source.val mark.epoch.val oldWord (fun _ => before)
    epochWord serialOrder noWrap visible
  exact ⟨latch, fun pending => ⟨(miss pending).1.trans epochWord.symm, (miss pending).2⟩⟩

end NeuralReversi.Threading.CutoffVisibility
