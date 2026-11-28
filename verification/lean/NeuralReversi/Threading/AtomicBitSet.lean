module
public import NeuralReversi.Bitboard.Counting
public import NeuralReversi.Bitboard.SquareProperties
public import NeuralReversi.Threading.AtomicWord

@[expose] public section

namespace NeuralReversi.Threading.AtomicBitSet

open AtomicWord (Event Write InReleaseSequence)

abbrev Word := BitVec 64
abbrev Index := Fin 64

inductive Update
  | set (index : Index)
  | reset (index : Index)
  | clear
  deriving DecidableEq

def Update.apply (update : Update) (old : Word) : Word :=
  match update with
  | .set i => old ||| Bitboard.squareMask i
  | .reset i => old &&& ~~~(Bitboard.squareMask i)
  | .clear => 0

def Update.encode (update : Update) (old : Word) (event : Event) : Write :=
  ⟨event, update.apply old, match update with | .reset _ => true | _ => false,
    decide (update ≠ .clear)⟩

def count (value : Word) : Nat := value.cpop.toNat

def test (value : Word) (i : Index) : Bool := ((value >>> i.val) &&& 1) != 0

def none (value : Word) : Bool := value == 0

private theorem test_bit (value : Word) (i : Index) : test value i = value.getLsbD i.val := by
  let low := (value >>> i.val) &&& 1
  change (low != 0) = value.getLsbD i.val
  have lowBit : low.getLsbD 0 = value.getLsbD i.val := by
    simp [low]
  have cleared : ∀ j, j ≠ 0 → low.getLsbD j = false := by
    intro j other
    simp [low, BitVec.getLsbD_and, BitVec.getLsbD_one, other]
  cases bit : value.getLsbD i.val
  · have zero : low = 0 := BitVec.zero_iff_eq_false.mpr fun j => by
      by_cases isLow : j = 0
      · subst j
        exact lowBit.trans bit
      · exact cleared j isLow
    rw [zero]
    rfl
  · have nonzero : low ≠ 0 := by
      intro zero
      rw [zero, bit] at lowBit
      cases lowBit
    exact bne_iff_ne.mpr nonzero

def MaskContract : Prop :=
  (count 0 = 0 ∧ none 0 = true) ∧
  ∀ (value : Word),
    count value = (List.range 64).countP value.getLsbD ∧ count value ≤ 64 ∧
    (none value = true ↔ ∀ i : Index, test value i = false) ∧
    Update.clear.apply value = 0 ∧
    ∀ i j : Index,
      test ((Update.set i).apply value) j = (test value j || decide (j = i)) ∧
      test ((Update.reset i).apply value) j = (test value j && !decide (j = i))

theorem maskContract_proven : MaskContract := by
  refine ⟨⟨rfl, rfl⟩, ?_⟩
  intro value
  refine ⟨by rw [count, BitVec.toNat_cpop, Bitboard.cpop_countP], BitVec.toNat_cpop_le _,
    ?_, rfl, ?_⟩
  · simp only [none, beq_iff_eq, test_bit]
    constructor
    · rintro rfl i
      simp
    · intro absent
      apply BitVec.zero_iff_eq_false.mpr
      intro j
      by_cases within : j < 64
      · exact absent ⟨j, within⟩
      · exact BitVec.getLsbD_of_ge value j (by omega)
  · intro i j
    simp only [Update.apply, test_bit, BitVec.getLsbD_or, BitVec.getLsbD_and, BitVec.getLsbD_not,
      Bitboard.squareMask_bit, j.isLt, decide_true, Bool.true_and, Fin.val_inj, true_and]

structure EncodedUpdates (writes : List Write) where
  operation : Fin writes.length → Update
  before : Fin writes.length → Word
  encoded : ∀ i, writes.get i = (operation i).encode (before i) (writes.get i).event

def ReleaseSequenceContract : Prop :=
  ∀ writes (trace : EncodedUpdates writes) head source,
    InReleaseSequence writes head source ↔
      (∃ i, trace.operation head = .reset i) ∧ head.val ≤ source.val ∧
      ∀ j : Fin writes.length, head.val < j.val → j.val ≤ source.val → trace.operation j ≠ .clear

theorem releaseSequenceContract_proven : ReleaseSequenceContract := by
  intro writes trace head source
  have release : (writes.get head).release = true ↔ ∃ i, trace.operation head = .reset i := by
    rw [trace.encoded head]
    cases trace.operation head <;> simp [Update.encode]
  have rmw (j : Fin writes.length) : (writes.get j).rmw = true ↔ trace.operation j ≠ .clear := by
    rw [trace.encoded j]
    simp [Update.encode]
  simp only [InReleaseSequence, release, rmw]

end NeuralReversi.Threading.AtomicBitSet
