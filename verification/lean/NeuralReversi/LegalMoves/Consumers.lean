module
public import NeuralReversi.LegalMoves.Encoding
public import NeuralReversi.Eval.MainScratch

@[expose] public section

namespace NeuralReversi.LegalMoves.MoveBound

open Bitboard (Bits)
open Bitboard.Moves (legalMovesTarget)
open Std.Tactic.BVDecide.LRAT (IntAction)

def Check35 (proof : Array IntAction) : Prop :=
  Std.Tactic.BVDecide.LRAT.check proof (dimacsFormula 35) = true

def GeneratedWrapperContract : Prop :=
  ∀ proof, Check35 proof → ∀ config (p o : Bits), p &&& o = 0#64 →
    let moves := legalMovesTarget config p o
    (∀ debug, fromGenerated debug moves = .ok ⟨moves⟩) ∧
      ∀ filters, bounded (applyFilters filters ⟨moves⟩)

theorem generatedWrapperContract_proven : GeneratedWrapperContract := by
  intro proof checked config p o disjoint
  have bound := certifiedBoundContract_proven 35 proof (by decide) checked config p o disjoint
  refine ⟨?_, fun filters => filters_bounded filters _ bound⟩
  intro debug
  cases debug
  · exact release_unconditional _
  · exact (debug_success_iff _ _).mpr ⟨rfl, bound⟩

open Eval.MainScratch

def MobilityByteContract : Prop :=
  ∀ proof, Check35 proof → ∀ config (p o : Bits), p &&& o = 0#64 →
    ∀ old base phase,
      let mobility := LegalMoves.count ⟨legalMovesTarget config p o⟩
      ((prepare old base phase mobility) ⟨256, by decide⟩).toNat = 7 * mobility ∧
        7 * mobility ≤ 245

theorem mobilityByteContract_proven : MobilityByteContract := by
  intro proof checked config p o disjoint old base phase
  have bound := certifiedBoundContract_proven 35 proof (by decide) checked config p o disjoint
  exact ⟨(prepareContract_proven old base phase _).2.2.1 bound, by omega⟩

end NeuralReversi.LegalMoves.MoveBound
