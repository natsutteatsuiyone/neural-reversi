module
public import NeuralReversi.Flip.Dispatch.Model
public import NeuralReversi.Flip.Dispatch.Spec
public import NeuralReversi.Flip.Avx2.Properties
public import NeuralReversi.Flip.Avx512.Properties
public import NeuralReversi.Flip.Neon.Properties

@[expose] public section

namespace NeuralReversi.Flip.Dispatch

theorem flip_eq_value (config : X86Config) (checks : Bool) (sq : Square) (player opponent : Word) :
    flip config checks sq player opponent = .ok (resultValue config sq player opponent) := by
  unfold flip resultValue
  split
  · rfl
  · split
    · rfl
    · exact operational_eq_value checks sq player opponent

theorem value_raw_coordinate (config : X86Config) (sq : Square) (player opponent : Word) (i : Nat) :
    (resultValue config sq player opponent).getLsbD i = true ↔ RawCoordinate config sq player opponent i := by
  unfold resultValue RawCoordinate
  split
  · exact Avx512.flip_coordinate sq player opponent i
  · split
    · exact Avx2.publicContract_proven.1 sq player opponent i
    · exact value_coordinate sq player opponent i

theorem value_disjoint_scalar (config : X86Config) (sq : Square) (player opponent : Word)
    (disjoint : player &&& opponent = 0) : resultValue config sq player opponent = value sq player opponent := by
  unfold resultValue
  split
  · exact Avx512.flip_eq_scalarValue sq player opponent
  · split
    · exact Avx2.publicContract_proven.2 sq player opponent disjoint
    · rfl

def X86FlipContract : Prop :=
  (∀ config checks sq player opponent, ∃ bits,
    flip config checks sq player opponent = .ok bits ∧
    ∀ i, bits.getLsbD i = true ↔ RawCoordinate config sq player opponent i) ∧
  (∀ config checks sq player opponent, player &&& opponent = 0 → ∃ bits,
    flip config checks sq player opponent = .ok bits ∧
    ∀ i, bits.getLsbD i = true ↔ CoordinateBit sq player opponent i)

theorem x86FlipContract_proven : X86FlipContract := by
  refine ⟨?_,?_⟩
  · intro config checks sq player opponent
    exact ⟨resultValue config sq player opponent,flip_eq_value config checks sq player opponent,
      value_raw_coordinate config sq player opponent⟩
  · intro config checks sq player opponent disjoint
    refine ⟨value sq player opponent,?_,value_coordinate sq player opponent⟩
    rw [flip_eq_value,value_disjoint_scalar config sq player opponent disjoint]

theorem flipTarget_eq_value_disjoint (config : TargetConfig) (checks : Bool)
    (sq : Square) (player opponent : Word) (disjoint : player &&& opponent = 0) :
    flipTarget config checks sq player opponent = .ok (value sq player opponent) := by
  cases config with
  | x86_64 x86 =>
    dsimp only [flipTarget]
    rw [flip_eq_value, value_disjoint_scalar x86 sq player opponent disjoint]
  | aarch64 neon sha3 =>
    cases neon <;> simp only [flipTarget, Bool.false_eq_true, ↓reduceIte]
    · exact operational_eq_value checks sq player opponent
    · rw [Neon.flip_eq_scalarValue sha3 sq player opponent disjoint]
  | other =>
    dsimp only [flipTarget]
    exact operational_eq_value checks sq player opponent

def TargetFlipContract : Prop :=
  ∀ config checks sq player opponent, player &&& opponent = 0 →
    ∃ bits, flipTarget config checks sq player opponent = .ok bits ∧
      ∀ i, bits.getLsbD i = true ↔ CoordinateBit sq player opponent i

theorem targetFlipContract_proven : TargetFlipContract := by
  intro config checks sq player opponent disjoint
  exact ⟨value sq player opponent,
    flipTarget_eq_value_disjoint config checks sq player opponent disjoint,
    value_coordinate sq player opponent⟩

end NeuralReversi.Flip.Dispatch
