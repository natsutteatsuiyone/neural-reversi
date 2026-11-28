module
public import NeuralReversi.PvsParallel.SplitRebookingMir.Program
public import NeuralReversi.Threading.CutoffVisibility
public import NeuralReversi.Threading.SplitCompletion

@[expose] public section

namespace NeuralReversi.PvsParallel.SplitRebookingMir

open Threading
open SplitCompletion (Action MaskState step initial run)
open AtomicWord (Event Write Read ReadWindow SequencedBefore)
open CutoffVisibility (History HappensBefore)

private theorem count_add (x : BitVec 64) (w : SplitCompletion.Worker) :
    AtomicBitSet.count x ≤ AtomicBitSet.count (x ||| Bitboard.squareMask w) := by
  rw [(AtomicBitSet.maskContract_proven.2 x).1,(AtomicBitSet.maskContract_proven.2 _).1]
  apply List.countP_mono_left
  intro i _ before
  simp [BitVec.getLsbD_or,before]

theorem open_history (m : MaskState) (acts : List Action) (openNow : (run m acts).accepting = true) :
    m.accepting = true ∧ AtomicBitSet.count m.bits ≤ AtomicBitSet.count (run m acts).bits := by
  induction acts generalizing m with
  | nil => exact ⟨openNow,Nat.le_refl _⟩
  | cons a rest ih =>
    have later := ih (step m a) openNow
    cases a with
    | finish w e =>
      have never : (step m (.finish w e)).accepting = false := by simp [step,Action.release]
      simp [never] at later
    | book w e =>
      exact ⟨by simpa [step,Action.release] using later.1,Nat.le_trans (count_add m.bits w) later.2⟩

inductive Field
  | accepting | mask
  deriving DecidableEq

def Field.word : Field → MaskState → BitVec 64
  | .accepting, s => if s.accepting then 1 else 0
  | .mask, s => s.bits

def changes : Field → Action → Bool
  | .accepting, a => a.release
  | .mask, _ => true

def updates (field : Field) (start : MaskState) : List Action → List (BitVec 64)
  | [] => []
  | a :: rest =>
    let next := step start a
    let tail := updates field next rest
    if changes field a then field.word next :: tail else tail

def values (field : Field) (start : MaskState) (actions : List Action) :=
  field.word start :: updates field start actions

private def Permitted (capNow : Nat) (f : Field) (word : BitVec 64) : Prop :=
  match f with
  | .accepting => word = 1
  | .mask => AtomicBitSet.count word ≤ capNow

private theorem bounded_words (f : Field) (m : MaskState) (actions : List Action)
    (openNow : (run m actions).accepting = true) :
    ∀ word ∈ values f m actions, Permitted (AtomicBitSet.count (run m actions).bits) f word := by
  induction actions generalizing m with
  | nil =>
    intro word member
    have eq : word = f.word m := by simpa [values,updates] using member
    subst word; cases f <;> simp [Permitted,Field.word,run] at openNow ⊢ <;> assumption
  | cons a rest ih =>
    intro word member
    rcases List.mem_cons.mp member with here | later
    · subst word
      have past := open_history m (a::rest) openNow
      cases f with
      | accepting => simp [Permitted,Field.word,past.1]
      | mask => exact past.2
    · apply ih (step m a) openNow
      unfold updates at later
      split at later
      · exact later
      · exact List.mem_cons_of_mem _ later

/-- An observed word belongs to the actual sequence of writes from this activation's
    initialization through the candidate's slot-lock recheck. A read-from index can be stale. -/
structure Prefix (writes : List Write) (field : Field) (m : MaskState) (acts : List Action) where
  first : Fin writes.length
  last : Fin writes.length
  before : List (BitVec 64)
  after : List (BitVec 64)
  words : writes.map Write.value = before ++ values field m acts ++ after
  starts : first.val = before.length
  ends : last.val + 1 = before.length + (values field m acts).length

private theorem Prefix.member (p : Prefix ws f m xs) (load : Read ws)
    (range : p.first.val ≤ load.source.val ∧ load.source.val ≤ p.last.val) :
    (ws.get load.source).value ∈ values f m xs := by
  have low := p.starts; have high := p.ends
  let k := load.source.val - p.before.length
  have within : k < (values f m xs).length := by dsimp [k]; omega
  have pos : p.before.length + k = load.source.val := by dsimp [k]; omega
  have onList : (ws.map Write.value)[load.source.val]? = some (ws.get load.source).value := by simp
  rw [p.words,← pos,List.getElem?_append_left (by simp; omega), List.getElem?_append_right (by omega)] at onList
  simp only [Nat.add_sub_cancel_left] at onList
  exact List.mem_of_getElem? onList

theorem nested_needs (h : nested pool depth opened count = true) : opened = false ∨ count = cap := by
  cases opened <;> simp [nested] at h ⊢
  exact h.2.2.1

/-- The flag path and the count path are alternatives. A closed flag short-circuits the
    count load in Rust. Both witnesses refer to the first nested split of the outer job. -/
inductive NestedSample (history : History) (m : MaskState) (acts : List Action) : Prop
  | flag (words : List Write) (load : Read words) (p : Prefix words .accepting m acts) :
      ReadWindow load (HappensBefore history) p.first p.last.val →
      (words.get load.source).value = 0 → NestedSample history m acts
  | count (words : List Write) (load : Read words) (p : Prefix words .mask m acts) :
      ReadWindow load (HappensBefore history) p.first p.last.val →
      AtomicBitSet.count (words.get load.source).value = cap → NestedSample history m acts

theorem no_nested_rebooking (m : MaskState) (acts : List Action) (late : Bool)
    (opened : (run m acts).accepting = true) (nestedJob : NestedSample history m acts) :
    room late (AtomicBitSet.count (run m acts).bits) = false := by
  cases nestedJob with
  | flag words load p window shut =>
    have yes := bounded_words .accepting m acts opened _ (p.member load window.bounds)
    change (words.get load.source).value = 1 at yes
    rw [shut] at yes; contradiction
  | count words load p window full =>
    have grew := bounded_words .mask m acts opened _ (p.member load window.bounds)
    change AtomicBitSet.count (words.get load.source).value ≤ _ at grew
    rw [full] at grew
    cases late <;> simp [room, Nat.not_lt_of_ge grew]

/-- Under the protected recheck a successful booking cannot overlap a first inner split,
    even when the inner guard and booking observed different atomic modifications. -/
theorem open_and_room_excludes_inner (m : MaskState) (acts : List Action) (late : Bool)
    (opened : (run m acts).accepting = true) (capacity : room late (AtomicBitSet.count (run m acts).bits) = true) :
    ¬NestedSample history m acts := by
  intro got
  rw [no_nested_rebooking m acts late opened got] at capacity
  contradiction

end NeuralReversi.PvsParallel.SplitRebookingMir
