import NeuralReversi.PvsParallel.SplitRebookingMir.Correct

namespace NeuralReversi.PvsParallel.SplitRebookingMir

open Threading
open AtomicWord (Event Write Read ReadWindow SequencedBefore InReleaseSequence)
open CutoffVisibility (History HappensBefore)
open SplitCompletion (Worker MaskState Action)

/-- The new reservation accesses the mask under the slot lock. The next mask modification
    is that RMW. A subsequent flag modification is ordered after it by the same lock. -/
inductive AtOrAfter (history : History) (event : Event) : Event → Prop
  | here : AtOrAfter history event event
  | after : HappensBefore history event later → AtOrAfter history event later

structure Coherence (history : History) (words : List Write) (load : Read words) : Prop where
  writeRead : ∀ i : Fin words.length, HappensBefore history (words.get i).event load.event → i.val ≤ load.source.val
  readWrite : ∀ i : Fin words.length, HappensBefore history load.event (words.get i).event → load.source.val < i.val

theorem before_next_booking (history : History) (words : List Write) (load : Read words)
    (p : Prefix words f s acts) (book : Event) (memory : Coherence history words load)
    (published : HappensBefore history (words.get p.first).event load.event)
    (nestedBeforeBook : HappensBefore history load.event book)
    (next : ∀ i : Fin words.length, i.val = p.last.val + 1 → AtOrAfter history book (words.get i).event) :
    ReadWindow load (HappensBefore history) p.first p.last.val := by
  refine ⟨published, ?_, memory.writeRead, memory.readWrite⟩
  intro i eq
  cases next i eq with
  | here => exact nestedBeforeBook
  | after edge => exact .trans nestedBeforeBook edge

/-- `searching` is the worker's own flag. Its original busy store must already be visible
    to the rechecking thread. Subsequent idle stores admitted here come from the worker's
    first nested call or later program order, while the original reservation is outstanding. -/
theorem from_searching_acquire (history : History) (words : List Write) (read : Read words)
    (busy : Fin words.length) (nestedLoad book : Event) (wasBusy : (words.get busy).value = 1)
    (sawIdle : (words.get read.source).value = 0) (acquired : read.acquire = true)
    (busyBefore : HappensBefore history (words.get busy).event read.event)
    (memory : Coherence history words read)
    (laterIdle : ∀ i : Fin words.length, busy.val < i.val → i.val ≤ read.source.val → (words.get i).value = 0 →
      SequencedBefore nestedLoad (words.get i).event ∧ (words.get i).release = true)
    (sync : ∀ i : Fin words.length, read.acquire = true → InReleaseSequence words i read.source →
      history.otherSync (words.get i).event read.event)
    (afterCheck : SequencedBefore read.event book) : HappensBefore history nestedLoad book := by
  have atLeast := memory.writeRead busy busyBefore
  have later : busy.val < read.source.val := by
    by_cases here : busy.val < read.source.val
    · exact here
    · exfalso
      have eq : read.source = busy := Fin.ext (by omega)
      rw [eq,wasBusy] at sawIdle; contradiction
  obtain ⟨next,published⟩ := laterIdle read.source later (Nat.le_refl _) sawIdle
  have sequence : InReleaseSequence words read.source read.source :=
    ⟨published, Nat.le_refl _, fun _ after before => by omega⟩
  exact .trans (.sequenced next) (.trans (.other (sync read.source acquired sequence)) (.sequenced afterCheck))

/-- The two physical paths of the active-parent test. Only the open-flag path executes a
    count load, so the flag and count can have different read-from sources. -/
inductive FirstNested (history : History) (s : MaskState) (acts : List Action) (book : Event) : Prop
  | closed (words : List Write) (flag : Read words) (p : Prefix words .accepting s acts) (pool depth : Nat) :
      nested pool depth false 0 = true → (words.get flag.source).value = 0 →
      Coherence history words flag → HappensBefore history (words.get p.first).event flag.event →
      HappensBefore history flag.event book →
      (∀ i : Fin words.length, i.val = p.last.val + 1 → AtOrAfter history book (words.get i).event) →
      FirstNested history s acts book
  | openFlag (words : List Write) (load : Read words) (p : Prefix words .mask s acts) (pool depth : Nat) :
      nested pool depth true (AtomicBitSet.count (words.get load.source).value) = true →
      Coherence history words load → HappensBefore history (words.get p.first).event load.event →
      HappensBefore history load.event book →
      (∀ i : Fin words.length, i.val = p.last.val + 1 → AtOrAfter history book (words.get i).event) →
      FirstNested history s acts book

theorem FirstNested.sample (first : FirstNested history s acts book) : NestedSample history s acts := by
  cases first with
  | closed words flag pre pool depth _ zero memory pub ahead after =>
    exact .flag words flag pre (before_next_booking history words flag pre book memory pub ahead after) zero
  | openFlag words load pre pool depth granted memory pub ahead after =>
    have full := (nested_needs granted).resolve_left (by decide)
    exact .count words load pre (before_next_booking history words load pre book memory pub ahead after) full

def recheck_from_execution (history : History) (s : MaskState) (acts : List Action)
    (fields : SplitActivationMir.Fields) (w : Worker) (late : Bool) (book : Event)
    (searching : List Write) (load : Read searching)
    (opened : (SplitCompletion.run s acts).accepting = true)
    (hasRoom : room late (AtomicBitSet.count (SplitCompletion.run s acts).bits) = true)
    (canJoin : SplitActivationMir.gate late true (decide ((searching.get load.source).value = 0)) = true)
    (origin : ∀ receipt, fields.assignment w = some receipt →
      (∃ stamp : Fin searching.length, (searching.get stamp).value = 1 ∧
        Nonempty (SplitPublication.LockedFlagRead history load stamp stamp.val)) ∨
      FirstNested history s acts book) : SplitActivationMir.Recheck history fields w := by
  refine ⟨searching,load,late,true,canJoin,?_⟩
  intro receipt eq
  rcases origin receipt eq with busy | inner
  · exact busy
  · exact False.elim (open_and_room_excludes_inner s acts late opened hasRoom inner.sample)

end NeuralReversi.PvsParallel.SplitRebookingMir
