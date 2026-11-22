import NeuralReversi.Board.MovesProperties

namespace NeuralReversi.Board.WrapperWitnesses
open Bitboard.Moves (X86Config BothCoordinateOutputs getMovesAndPotentialX86)
abbrev Bits := Bitboard.Bits

def portable : X86Config := ⟨false,false⟩

def CombinedCandidateCorrect (candidate : X86Config → Bits → Bits → Bits × Bits) : Prop :=
  ∀ config p o i, BothCoordinateOutputs p o i (candidate config p o)

theorem original_correct : CombinedCandidateCorrect getMovesAndPotentialX86 :=
  Bitboard.Moves.getMovesAndPotentialX86_exact

theorem coordinates_determine_words (candidate : X86Config → Bits → Bits → Bits × Bits)
    (correct : CombinedCandidateCorrect candidate) (config : X86Config) (p o : Bits) :
    candidate config p o = getMovesAndPotentialX86 config p o := by
  apply Prod.ext
  · apply BitVec.eq_of_getLsbD_eq
    intro i _
    have hc := correct config p o i
    have hs := Bitboard.Moves.getMovesAndPotentialX86_exact config p o i
    have same := hc.1.trans hs.1.symm
    cases left : (candidate config p o).1.getLsbD i <;>
      cases right : (getMovesAndPotentialX86 config p o).1.getLsbD i <;> simp_all
  · apply BitVec.eq_of_getLsbD_eq
    intro i _
    have hc := correct config p o i
    have hs := Bitboard.Moves.getMovesAndPotentialX86_exact config p o i
    have same := hc.2.trans hs.2.symm
    cases left : (candidate config p o).2.getLsbD i <;>
      cases right : (getMovesAndPotentialX86 config p o).2.getLsbD i <;> simp_all

theorem counterexample_rejects (candidate : X86Config → Bits → Bits → Bits × Bits)
    (config : X86Config) (p o : Bits)
    (wrong : candidate config p o ≠ getMovesAndPotentialX86 config p o) :
    ¬ CombinedCandidateCorrect candidate :=
  fun correct => wrong (coordinates_determine_words candidate correct config p o)

def zeroPair (_ : X86Config) (_ _ : Bits) : Bits × Bits := (0,0)
def identityPair (_ : X86Config) (p o : Bits) : Bits × Bits := (p,o)
def swapped (config : X86Config) (p o : Bits) : Bits × Bits :=
  let result := getMovesAndPotentialX86 config p o
  (result.2,result.1)
def dropPotential (config : X86Config) (p o : Bits) : Bits × Bits :=
  ((getMovesAndPotentialX86 config p o).1,0)
def dropLegal (config : X86Config) (p o : Bits) : Bits × Bits :=
  (0,(getMovesAndPotentialX86 config p o).2)

theorem zero_pair_rejected : ¬ CombinedCandidateCorrect zeroPair :=
  counterexample_rejects zeroPair portable 1 2 (by decide)
theorem identity_pair_rejected : ¬ CombinedCandidateCorrect identityPair :=
  counterexample_rejects identityPair portable 0 2 (by decide)
theorem swapped_words_rejected : ¬ CombinedCandidateCorrect swapped :=
  counterexample_rejects swapped portable 0 2 (by decide)
theorem dropped_potential_rejected : ¬ CombinedCandidateCorrect dropPotential :=
  counterexample_rejects dropPotential portable 0 2 (by decide)
theorem dropped_legal_rejected : ¬ CombinedCandidateCorrect dropLegal :=
  counterexample_rejects dropLegal portable 1 2 (by decide)

theorem constant_false_legality_rejected :
    (Except.ok false : Except String Bool) ≠ isLegalMove portable false false ⟨4,2⟩ ⟨0,by decide⟩ := by
  change (Except.ok false : Except String Bool) ≠ .ok true
  intro h
  have impossible : false = true := Except.ok.inj h
  cases impossible

theorem constant_true_legality_rejected :
    (Except.ok true : Except String Bool) ≠ isLegalMove portable false false ⟨0,0⟩ ⟨0,by decide⟩ := by
  change (Except.ok true : Except String Bool) ≠ .ok false
  intro h
  have impossible : true = false := Except.ok.inj h
  cases impossible

end NeuralReversi.Board.WrapperWitnesses
