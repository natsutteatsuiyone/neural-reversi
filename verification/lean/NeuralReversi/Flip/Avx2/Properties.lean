module
public import NeuralReversi.Flip.Avx2.Bridge

@[expose] public section

namespace NeuralReversi.Flip.Avx2
open NeuralReversi.Simd.X86

def PreparedContract : Prop :=
  ∀ pp oo (pos : Index64) (valid : pos.toNat < 66) (j : Fin 2) (i : Nat),
    (mmFlipPrepared pp oo pos valid j).getLsbD i = true ↔
      PairBitSpec ⟨pos.toNat,valid⟩ pp oo j i

theorem preparedContract_proven : PreparedContract := by
  intro pp oo pos valid j i
  rw [mmFlipPrepared_eq_derived]
  exact prepared_pair_bit ⟨pos.toNat,valid⟩ pp oo j i

def PublicContract : Prop :=
  (∀ sq player opponent i,
    (flip sq player opponent).getLsbD i = true ↔ RawSpec sq player opponent i) ∧
  (∀ sq player opponent, player &&& opponent = 0 →
    flip sq player opponent = NeuralReversi.Flip.value sq player opponent)

theorem publicContract_proven : PublicContract := by
  constructor
  · intro sq player opponent i
    rw [flip_eq_summary]
    exact summary_raw sq player opponent i
  · intro sq player opponent disjoint
    rw [flip_eq_summary]
    exact disjoint_scalar_equality sq player opponent disjoint

theorem mm_placeholders (pp oo : Vec256) (pos : Index64) (valid : pos.toNat < 66)
    (placeholder : 64 ≤ pos.toNat) : mmFlipPrepared pp oo pos valid = fun _ => 0 := by
  rw [mmFlipPrepared_eq_derived]
  exact prepared_placeholders ⟨pos.toNat,valid⟩ placeholder pp oo

end NeuralReversi.Flip.Avx2
