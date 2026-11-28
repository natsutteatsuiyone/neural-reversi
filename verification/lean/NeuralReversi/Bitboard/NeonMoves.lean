module
public import NeuralReversi.Bitboard.DispatchProperties
public import NeuralReversi.Flip.Dispatch.Types
public import NeuralReversi.Flip.Neon.Properties

@[expose] public section

namespace NeuralReversi.Bitboard.Moves
set_option maxRecDepth 4096
open NeuralReversi.Flip.Neon (Vec128 vdupqNU64 vandqU64 vorrqU64 vbcaxqU64
  getLsbD_reverse_eq reverse_and)

def signedShift (word : Bits) (count : Int) : Bits :=
  if 0 ≤ count then word <<< count.toNat else word >>> count.natAbs

def vshift (words : Vec128) (counts : Int × Int) : Vec128 :=
  (signedShift words.1 counts.1, signedShift words.2 counts.2)

def zip1 (a b : Vec128) : Vec128 := (a.1, b.1)
def zip2 (a b : Vec128) : Vec128 := (a.2, b.2)

def expandDiagonals (sha3 : Bool) (p h : Bits) : Vec128 :=
  let pv := vdupqNU64 p
  let mv := vdupqNU64 h
  let sh : Int × Int := (7, 9)
  let shNeg : Int × Int := (-7, -9)
  let sh2 : Int × Int := (14, 18)
  let sh2Neg : Int × Int := (-14, -18)
  let pre := vandqU64 mv (vshift mv sh)
  let preNeg := vshift pre shNeg
  let fl := vandqU64 mv (vshift pv sh)
  let fr := vandqU64 mv (vshift pv shNeg)
  let nmv := vdupqNU64 (~~~h)
  let seed := fun f s =>
    if sha3 then vbcaxqU64 f (vshift f s) nmv
    else vorrqU64 f (vandqU64 mv (vshift f s))
  let fl := seed fl sh
  let fr := seed fr shNeg
  let fl := vorrqU64 fl (vandqU64 pre (vshift fl sh2))
  let fl := vorrqU64 fl (vandqU64 pre (vshift fl sh2))
  let fr := vorrqU64 fr (vandqU64 preNeg (vshift fr sh2Neg))
  let fr := vorrqU64 fr (vandqU64 preNeg (vshift fr sh2Neg))
  vorrqU64 (vshift fl sh) (vshift fr shNeg)

def finishLegal (p o h diagonals : Bits) : Bits :=
  let rp := p.reverse
  let rh := h.reverse
  let horizontal := (h + (h &&& (p <<< (1 : Nat)))) |||
    (rh + (rh &&& (rp <<< (1 : Nat)))).reverse
  let vertical := ((expandedLeft o p 8) <<< (8 : Nat)) |||
    ((expandedRight o p 8) >>> (8 : Nat))
  (horizontal ||| vertical ||| diagonals) &&& ~~~(p ||| o)

def legalMovesNeon (sha3 : Bool) (p o : Bits) : Bits :=
  let h := o &&& horizontalMask
  let md := expandDiagonals sha3 p h
  finishLegal p o h (md.1 ||| md.2)

def legalMovesAndPotentialNeon (sha3 : Bool) (p o : Bits) : Bits × Bits :=
  let h := o &&& horizontalMask
  let md := expandDiagonals sha3 p h
  let dd := vdupqNU64 (o &&& diagonalMask)
  let pd := vorrqU64 (vshift dd (7, 9)) (vshift dd (-7, -9))
  let comb := vorrqU64 (zip1 md pd) (zip2 md pd)
  let v := o &&& verticalMask
  let hv := (h <<< (1 : Nat)) ||| (h >>> (1 : Nat)) |||
    (v <<< (8 : Nat)) ||| (v >>> (8 : Nat))
  (finishLegal p o h comb.1, (hv ||| comb.2) &&& ~~~(p ||| o))

private theorem seedXor_eq_or (p h : Bits) (shift : Bits → Bits)
    (disjoint : p &&& h = 0#64)
    (shiftAnd : ∀ a b, shift (a &&& b) = shift a &&& shift b)
    (shiftZero : shift 0#64 = 0#64) :
    let f := h &&& shift p
    f ^^^ (shift f &&& ~~~(~~~h)) = f ||| (h &&& shift f) := by
  dsimp
  rw [BitVec.not_not]
  have zero : shift p &&& shift h = 0#64 := by rw [← shiftAnd, disjoint, shiftZero]
  have excluded : (h &&& shift p) &&& (h &&& shift (h &&& shift p)) = 0#64 := by
    rw [shiftAnd]
    calc
      _ = (shift p &&& shift h) &&& (h &&& h &&& shift (shift p)) := by ac_rfl
      _ = _ := by simp [zero]
  apply BitVec.eq_of_getLsbD_eq
  intro i _
  let f := h &&& shift p
  have bit : (f.getLsbD i && (h &&& shift f).getLsbD i) = false := by
    rw [← BitVec.getLsbD_and, show f &&& (h &&& shift f) = 0#64 from excluded]
    exact BitVec.getLsbD_zero
  change (f ^^^ (shift f &&& h)).getLsbD i = (f ||| (h &&& shift f)).getLsbD i
  rw [BitVec.and_comm (shift f) h, BitVec.getLsbD_xor, BitVec.getLsbD_or]
  cases ha : f.getLsbD i <;> cases hb : (h &&& shift f).getLsbD i <;> try rfl
  rw [ha, hb] at bit
  contradiction

private theorem expanded_lanes (sha3 : Bool) (p h : Bits)
    (disjoint : sha3 = true → p &&& h = 0#64) :
    expandDiagonals sha3 p h =
      (((expandedLeft h p 7) <<< (7 : Nat)) ||| ((expandedRight h p 7) >>> (7 : Nat)),
       ((expandedLeft h p 9) <<< (9 : Nat)) ||| ((expandedRight h p 9) >>> (9 : Nat))) := by
  have leftSeed (d : Nat) (hd : p &&& h = 0#64) :=
    seedXor_eq_or p h (fun w => w <<< d) hd
      (fun a b => BitVec.shiftLeft_and_distrib a b d) (by simp)
  have rightSeed (d : Nat) (hd : p &&& h = 0#64) :=
    seedXor_eq_or p h (fun w => w >>> d) hd
      (fun a b => BitVec.ushiftRight_and_distrib a b d) (by simp)
  have shift7 (w : Bits) : signedShift w 7 = w <<< (7 : Nat) := rfl
  have shift9 (w : Bits) : signedShift w 9 = w <<< (9 : Nat) := rfl
  have shift14 (w : Bits) : signedShift w 14 = w <<< (14 : Nat) := rfl
  have shift18 (w : Bits) : signedShift w 18 = w <<< (18 : Nat) := rfl
  have shiftNeg7 (w : Bits) : signedShift w (-7) = w >>> (7 : Nat) := rfl
  have shiftNeg9 (w : Bits) : signedShift w (-9) = w >>> (9 : Nat) := rfl
  have shiftNeg14 (w : Bits) : signedShift w (-14) = w >>> (14 : Nat) := rfl
  have shiftNeg18 (w : Bits) : signedShift w (-18) = w >>> (18 : Nat) := rfl
  cases sha3 with
  | false =>
    simp only [expandDiagonals, vdupqNU64, vandqU64, vorrqU64, vshift, Bool.false_eq_true,
      ↓reduceIte, shift7, shift9, shift14, shift18, shiftNeg7, shiftNeg9, shiftNeg14, shiftNeg18,
      expandedLeft, expandedRight, shifted_preMask]
  | true =>
    have dh := disjoint rfl
    simp only [expandDiagonals, vdupqNU64, vandqU64, vorrqU64, vbcaxqU64, vshift,
      ↓reduceIte, shift7, shift9, shift14, shift18, shiftNeg7, shiftNeg9, shiftNeg14, shiftNeg18,
      leftSeed _ dh, rightSeed _ dh, expandedLeft, expandedRight, shifted_preMask]

private theorem reverse_shiftLeft (w : Bits) (d : Nat) :
    (w <<< d).reverse = w.reverse >>> d := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  rw [getLsbD_reverse_eq _ _ hi, BitVec.getLsbD_ushiftRight]
  by_cases hb : d + i < 64
  · rw [getLsbD_reverse_eq _ _ hb]
    apply Bool.eq_iff_iff.mpr
    rw [leftShift_true_iff w d (63-i) (by omega)]
    simp [show d ≤ 63-i by omega, show 63-i-d = 63-(d+i) by omega]
  · rw [BitVec.getLsbD_of_ge w.reverse _ (by omega)]
    have miss : ¬ (w <<< d).getLsbD (63-i) = true := by
      rw [leftShift_true_iff w d (63-i) (by omega)]
      omega
    exact Bool.eq_false_iff.mpr miss

private theorem reverse_left_prefix (h p : Bits) (d : Nat) :
    ((expandedLeft h.reverse p.reverse d) <<< d).reverse =
      (expandedRight h p d) >>> d := by
  simp only [expandedLeft, expandedRight, reverse_shiftLeft, reverse_and,
    NeuralReversi.Flip.Neon.reverse_or, BitVec.reverse_reverse_eq]

private theorem carry_eq_expand (p o : Bits) (i : Nat) (hi : i < 64)
    (empty : (p ||| o).getLsbD i = false) :
    let h := o &&& horizontalMask
    (h + (h &&& (p <<< (1 : Nat)))).getLsbD i =
      ((expandedLeft h p 1) <<< (1 : Nat)).getLsbD i := by
  apply Bool.eq_iff_iff.mpr
  rw [portable_east_exact p o i hi empty, eastRay_geometry p o i hi]
  exact (left_direction_exact p o i 1 0 1 hi (by decide) (by decide) (by decide)).symm

private theorem reversed_carry_eq_expand (p o : Bits) (i : Nat) (hi : i < 64)
    (empty : (p ||| o).getLsbD i = false) :
    let h := (o &&& horizontalMask).reverse
    (h + (h &&& (p.reverse <<< (1 : Nat)))).reverse.getLsbD i =
      ((expandedRight (o &&& horizontalMask) p 1) >>> (1 : Nat)).getLsbD i := by
  have mask : horizontalMask.reverse = horizontalMask := by decide
  have emptRev : (p.reverse ||| o.reverse).getLsbD (63-i) = false := by
    rw [← NeuralReversi.Flip.Neon.reverse_or, getLsbD_reverse_eq _ _ (by omega),
      show 63 - (63-i) = i by omega]
    exact empty
  dsimp
  rw [← reverse_left_prefix, getLsbD_reverse_eq _ _ hi, getLsbD_reverse_eq _ _ hi]
  simpa only [reverse_and, mask] using carry_eq_expand p.reverse o.reverse (63-i)
    (by omega) emptRev

theorem legalMovesNeon_eq_portable (sha3 : Bool) (p o : Bits)
    (disjoint : sha3 = true → p &&& o = 0#64) :
    legalMovesNeon sha3 p o = getMovesPortable p o := by
  have dh : sha3 = true → p &&& (o &&& horizontalMask) = 0#64 := by
    intro h
    rw [← BitVec.and_assoc, disjoint h, BitVec.zero_and]
  dsimp only [legalMovesNeon]
  rw [expanded_lanes _ _ _ dh, portable_decomposition]
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  simp only [finishLegal, portableDirections, BitVec.getLsbD_and, BitVec.getLsbD_not,
    hi, decide_true, Bool.true_and]
  cases empty : (p ||| o).getLsbD i
  · simp only [Bool.not_false, Bool.and_true, BitVec.getLsbD_or]
    rw [reversed_carry_eq_expand p o i hi empty]
    ac_rfl
  · simp

theorem legalMovesAndPotentialNeon_eq_portable (sha3 : Bool) (p o : Bits)
    (disjoint : sha3 = true → p &&& o = 0#64) :
    legalMovesAndPotentialNeon sha3 p o = getMovesAndPotentialPortable p o := by
  apply Prod.ext
  · exact legalMovesNeon_eq_portable sha3 p o disjoint
  · change _ = potentialCandidates o &&& ~~~(p ||| o)
    simp only [legalMovesAndPotentialNeon, potentialCandidates, zip1, zip2, vorrqU64,
      vdupqNU64, vshift]
    rw [show signedShift (o &&& diagonalMask) 7 = (o &&& diagonalMask) <<< (7 : Nat) from rfl,
      show signedShift (o &&& diagonalMask) (-7) = (o &&& diagonalMask) >>> (7 : Nat) from rfl,
      show signedShift (o &&& diagonalMask) 9 = (o &&& diagonalMask) <<< (9 : Nat) from rfl,
      show signedShift (o &&& diagonalMask) (-9) = (o &&& diagonalMask) >>> (9 : Nat) from rfl]
    congr 1
    ac_rfl

def NeonMovesContract : Prop :=
  ∀ (sha3 : Bool) (p o : Bits) (target : Nat), (sha3 = true → p &&& o = 0#64) →
    ((legalMovesNeon sha3 p o).getLsbD target = true ↔ LegalMove p o target)

theorem neonMovesContract_proven : NeonMovesContract := by
  intro sha3 p o target h
  rw [legalMovesNeon_eq_portable sha3 p o h]
  exact getMovesPortable_exact p o target

def NeonMovesAndPotentialContract : Prop :=
  ∀ (sha3 : Bool) (p o : Bits) (target : Nat), (sha3 = true → p &&& o = 0#64) →
    let result := legalMovesAndPotentialNeon sha3 p o
    (result.1.getLsbD target = true ↔ LegalMove p o target) ∧
      (result.2.getLsbD target = true ↔ PotentialMove p o target)

theorem neonMovesAndPotentialContract_proven : NeonMovesAndPotentialContract := by
  intro sha3 p o target h
  rw [legalMovesAndPotentialNeon_eq_portable sha3 p o h]
  exact ⟨getMovesPortable_exact p o target,
    potential_output_from_candidates p o (potentialCandidates_bit o) target⟩

theorem sha3_overlapping_words_counterexample :
    (legalMovesNeon true 0x201#64 0x40200#64).getLsbD 27 = false ∧
      (getMovesPortable 0x201#64 0x40200#64).getLsbD 27 = true := by decide

open NeuralReversi.Flip.Dispatch (TargetConfig)

def legalMovesTarget (config : TargetConfig) (p o : Bits) : Bits :=
  match config with
  | .x86_64 x86 => getMovesX86 ⟨x86.avx512vl, x86.avx2⟩ p o
  | .aarch64 neon sha3 => if neon then legalMovesNeon sha3 p o else getMovesPortable p o
  | .wasm32 _ | .other => getMovesPortable p o

def legalMovesAndPotentialTarget (config : TargetConfig) (p o : Bits) : Bits × Bits :=
  match config with
  | .x86_64 x86 => getMovesAndPotentialX86 ⟨x86.avx512vl, x86.avx2⟩ p o
  | .aarch64 neon sha3 =>
    if neon then legalMovesAndPotentialNeon sha3 p o else getMovesAndPotentialPortable p o
  | .wasm32 _ | .other => getMovesAndPotentialPortable p o

def TargetMoveGenerationContract : Prop :=
  ∀ (config : TargetConfig) (p o : Bits) (target : Nat), p &&& o = 0#64 →
    ((legalMovesTarget config p o).getLsbD target = true ↔ LegalMove p o target) ∧
      BothCoordinateOutputs p o target (legalMovesAndPotentialTarget config p o)

theorem targetMoveGenerationContract_proven : TargetMoveGenerationContract := by
  intro config p o target disjoint
  have portable : ((getMovesPortable p o).getLsbD target = true ↔ LegalMove p o target) ∧
      BothCoordinateOutputs p o target (getMovesAndPotentialPortable p o) :=
    ⟨getMovesPortable_exact p o target,
    portableMovesAndPotentialContract_proven p o target⟩
  cases config with
  | other | wasm32 => exact portable
  | x86_64 cfg =>
    exact ⟨getMovesX86_exact ⟨cfg.avx512vl, cfg.avx2⟩ p o target,
      getMovesAndPotentialX86_exact ⟨cfg.avx512vl, cfg.avx2⟩ p o target⟩
  | aarch64 neon sha3 =>
    cases neon
    · exact portable
    · exact ⟨neonMovesContract_proven sha3 p o target (fun _ => disjoint),
        neonMovesAndPotentialContract_proven sha3 p o target (fun _ => disjoint)⟩

end NeuralReversi.Bitboard.Moves
