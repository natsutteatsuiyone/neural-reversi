import NeuralReversi.Bitboard.IteratorModel
import NeuralReversi.Bitboard.Counting

namespace NeuralReversi.Bitboard

theorem iteratorNextDirect_zero : iteratorNextDirect 0 = (none, 0) := by
  unfold iteratorNextDirect
  rw [leastSquareContract_proven.2.2.1]

theorem iteratorNextDirect_nonzero (bits : Bits) (nonzero : bits ≠ 0) :
    ∃ square rest, iteratorNextDirect bits = (some square, rest) ∧
      IsLeastSetSquare bits square ∧
      ∀ i, rest.getLsbD i = true ↔ bits.getLsbD i = true ∧ square.val < i := by
  obtain ⟨square, hlsb, _, _, least⟩ := leastSquareContract_proven.2.2.2 bits nonzero
  refine ⟨square, clearLsb bits, ?_, least, ?_⟩
  · simp [iteratorNextDirect, hlsb]
  · intro i
    have hctz : square.val = bits.ctz.toNat := by
      have hlt : bits.ctz.toNat < 64 := by
        have bound := (BitVec.ctz_lt_iff_ne_zero (x := bits)).mpr nonzero
        simpa [BitVec.lt_def] using bound
      simp [lsbSquare, hlt] at hlsb
      exact congrArg Fin.val hlsb.symm
    rw [hctz]
    exact clearLsb_ctz bits nonzero i

def IteratorDirectStepContract : Prop :=
  ∀ bits : Bits,
    (iteratorNextDirect bits = (none, bits) ↔ bits = 0) ∧
    (bits ≠ 0 → ∃ square rest, iteratorNextDirect bits = (some square, rest) ∧
      IsLeastSetSquare bits square ∧
      ∀ i, rest.getLsbD i = true ↔ bits.getLsbD i = true ∧ square.val < i)

theorem iteratorDirectStepContract_proven : IteratorDirectStepContract := by
  intro bits
  refine ⟨?_, iteratorNextDirect_nonzero bits⟩
  constructor
  · intro empty
    by_cases zero : bits = 0
    · exact zero
    · obtain ⟨square, rest, step, _, _⟩ := iteratorNextDirect_nonzero bits zero
      simp [step] at empty
  · intro zero
    simpa [zero] using iteratorNextDirect_zero

theorem iteratorNext_zero (checked : Bool) : iteratorNext checked 0 = .ok (none, 0) := by
  unfold iteratorNext
  rw [iteratorNextDirect_zero]

theorem iteratorNext_nonzero (bits : Bits) (checked : Bool) (nonzero : bits ≠ 0) :
    ∃ square rest, iteratorNext checked bits = .ok (some square, rest) ∧
      IsLeastSetSquare bits square ∧
      ∀ i, rest.getLsbD i = true ↔ bits.getLsbD i = true ∧ square.val < i := by
  obtain ⟨square, rest, step, least, tail⟩ := iteratorNextDirect_nonzero bits nonzero
  exact ⟨square, rest, by simp [iteratorNext, step], least, tail⟩

def IteratorStepContract : Prop :=
  ∀ (bits : Bits) (checked : Bool),
    (iteratorNext checked bits = .ok (none, bits) ↔ bits = 0) ∧
    (bits ≠ 0 → ∃ square rest, iteratorNext checked bits = .ok (some square, rest) ∧
      IsLeastSetSquare bits square ∧
      ∀ i, rest.getLsbD i = true ↔ bits.getLsbD i = true ∧ square.val < i)

theorem iteratorStepContract_proven : IteratorStepContract := by
  intro bits checked
  refine ⟨?_, iteratorNext_nonzero bits checked⟩
  constructor
  · intro empty
    by_cases zero : bits = 0
    · exact zero
    · obtain ⟨square, rest, step, _, _⟩ := iteratorNext_nonzero bits checked zero
      simp [step] at empty
  · intro zero
    simpa [zero] using iteratorNext_zero checked

theorem filter_range_pop (p q : Nat → Bool) (square n : Nat)
    (within : square < n) (present : p square = true)
    (lower : ∀ i < square, p i = false)
    (tail : ∀ i, q i = true ↔ p i = true ∧ square < i) :
    (List.range n).filter p = square :: (List.range n).filter q := by
  induction n with
  | zero => omega
  | succ n ih =>
    by_cases last : square = n
    · subst square
      have emptyPrefix : (List.range n).filter p = [] := by
        apply List.filter_eq_nil_iff.mpr
        intro i member
        simp [lower i (List.mem_range.mp member)]
      have emptyTail : (List.range (n + 1)).filter q = [] := by
        apply List.filter_eq_nil_iff.mpr
        intro i member selected
        have bound := List.mem_range.mp member
        have := (tail i).mp selected
        omega
      rw [emptyTail, List.range_succ, List.filter_append, emptyPrefix]
      simp [present]
    · have before : square < n := by omega
      have same : q n = p n := by
        apply Bool.eq_iff_iff.mpr
        simp [tail n, before]
      rw [List.range_succ, List.filter_append, List.filter_append, ih before]
      simp [List.filter_cons, same]

theorem setIndices_pop (bits rest : Bits) (square : NeuralReversi.Square.Square)
    (least : IsLeastSetSquare bits square)
    (tail : ∀ i, rest.getLsbD i = true ↔ bits.getLsbD i = true ∧ square.val < i) :
    setIndices bits = square.val :: setIndices rest :=
  filter_range_pop bits.getLsbD rest.getLsbD square.val 64 least.1 least.2.1 least.2.2 tail

theorem setIndices_zero : setIndices 0 = [] := by
  apply List.filter_eq_nil_iff.mpr
  intro i _
  simp

theorem setIndices_length_le (bits : Bits) : (setIndices bits).length ≤ 64 := by
  simpa [setIndices] using List.length_filter_le bits.getLsbD (List.range 64)

theorem iteratorTrace_correct (fuel : Nat) (checked : Bool) (bits : Bits)
    (enough : (setIndices bits).length ≤ fuel) :
    iteratorTrace fuel checked bits = .ok (expectedIteratorTrace fuel bits) := by
  induction fuel generalizing bits with
  | zero =>
    have empty : setIndices bits = [] := List.length_eq_zero_iff.mp (by omega)
    simp [iteratorTrace, expectedIteratorTrace, empty]
  | succ fuel ih =>
    by_cases zero : bits = 0
    · subst bits
      have trace := ih 0 (by rw [setIndices_zero]; simp)
      simp only [iteratorTrace, iteratorNext_zero]
      change (iteratorTrace fuel checked 0).map (none :: ·) = _
      rw [trace]
      simp only [Except.map, expectedIteratorTrace, setIndices_zero]
      simp [List.replicate_succ]
    · obtain ⟨square, rest, step, least, tail⟩ := iteratorNext_nonzero bits checked zero
      have decomposition := setIndices_pop bits rest square least tail
      have remaining : (setIndices rest).length ≤ fuel := by
        rw [decomposition, List.length_cons] at enough
        omega
      have trace := ih rest remaining
      simp only [iteratorTrace, step]
      change (iteratorTrace fuel checked rest).map (some square.val :: ·) = _
      rw [trace]
      simp [Except.map, expectedIteratorTrace, decomposition]

def IteratorEnumerationContract : Prop :=
  ∀ (bits : Bits) (checked : Bool),
    (∀ fuel : Nat, (setIndices bits).length ≤ fuel →
      iteratorTrace fuel checked (iteratorNew bits) = .ok (expectedIteratorTrace fuel bits) ∧
      iteratorTrace fuel checked (bitboardIter bits) = .ok (expectedIteratorTrace fuel bits) ∧
      iteratorTrace fuel checked (intoIterator bits) = .ok (expectedIteratorTrace fuel bits)) ∧
    (setIndices bits).Pairwise (· < ·) ∧
    (∀ i : Nat, i ∈ setIndices bits ↔ i < 64 ∧ bits.getLsbD i = true) ∧
    (setIndices bits).length = (count bits).toNat ∧
    (setIndices bits).length ≤ 64

theorem iteratorEnumerationContract_proven : IteratorEnumerationContract := by
  intro bits checked
  refine ⟨?_, ?_, ?_, ?_, setIndices_length_le bits⟩
  · intro fuel enough
    have trace := iteratorTrace_correct fuel checked bits enough
    exact ⟨trace, trace, trace⟩
  · exact List.pairwise_lt_range.filter bits.getLsbD
  · intro i
    simp [setIndices]
  · simpa [setIndices, List.countP_eq_length_filter] using
      (populationCountContract_proven bits).1.symm

theorem iteratorTrace66 (bits : Bits) (checked : Bool) :
    iteratorTrace 66 checked bits = .ok (expectedIteratorTrace 66 bits) := by
  apply iteratorTrace_correct
  have := setIndices_length_le bits
  omega

end NeuralReversi.Bitboard
