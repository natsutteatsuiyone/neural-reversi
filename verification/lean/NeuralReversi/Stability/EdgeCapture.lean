module
public import NeuralReversi.Stability.EdgeModel
public import NeuralReversi.Stability.EdgeEncoding
public import NeuralReversi.Stability.LineBracket
public import NeuralReversi.Stability.EdgeTrace

@[expose] public section

namespace NeuralReversi.Stability.Edge
open Semantics

def BracketBits (p o : Byte) (a x : Fin 8) : Prop :=
  ∃ z : Fin 8, p.getLsbD z.val = true ∧
    ((a.val < x.val ∧ x.val < z.val ∧
      ∀ y : Fin 8, a.val < y.val → y.val < z.val → o.getLsbD y.val = true) ∨
     (z.val < x.val ∧ x.val < a.val ∧
      ∀ y : Fin 8, z.val < y.val → y.val < a.val → o.getLsbD y.val = true))

instance (p o : Byte) (a x : Fin 8) : Decidable (BracketBits p o a x) := by
  unfold BracketBits
  infer_instance

def captureSpec (p o : Byte) (a x : Fin 8) : Bool := decide (BracketBits p o a x)

theorem captureSpec_semantics (p o : Byte) (c : Color) (a x : Fin 8)
    (disjoint : p &&& o = 0) :
    captureSpec p o a x = true ↔ LineCaptured (decodeColor p o c) c a x := by
  rw [lineCaptured_bracket]
  simp only [captureSpec, decide_eq_true_eq, BracketBits, LineBracket]
  constructor
  · rintro ⟨z,hz,h⟩
    refine ⟨z,(decodeColor_bits p o c z disjoint).1.mpr hz,?_⟩
    rcases h with ⟨lo,hi,run⟩ | ⟨lo,hi,run⟩
    · exact Or.inl ⟨lo,hi,fun y hy hz => (decodeColor_bits p o c y disjoint).2.mpr (run y hy hz)⟩
    · exact Or.inr ⟨lo,hi,fun y hy hz => (decodeColor_bits p o c y disjoint).2.mpr (run y hy hz)⟩
  · rintro ⟨z,hz,h⟩
    refine ⟨z,(decodeColor_bits p o c z disjoint).1.mp hz,?_⟩
    rcases h with ⟨lo,hi,run⟩ | ⟨lo,hi,run⟩
    · exact Or.inl ⟨lo,hi,fun y hy hz => (decodeColor_bits p o c y disjoint).2.mp (run y hy hz)⟩
    · exact Or.inr ⟨lo,hi,fun y hy hz => (decodeColor_bits p o c y disjoint).2.mp (run y hy hz)⟩

theorem captureSpec_opponent (p o : Byte) (a x : Fin 8)
    (h : captureSpec p o a x = true) : o.getLsbD x.val = true := by
  have h := of_decide_eq_true h
  rcases h with ⟨z,_,⟨lo,hi,run⟩ | ⟨lo,hi,run⟩⟩
  · exact run x lo hi
  · exact run x lo hi

def moveSpec (p o : Byte) (a : Fin 8) : State :=
  let flips := encodeBits (captureSpec p o a)
  (p ||| bit8 a.val ||| flips, o &&& ~~~flips)

theorem bit8_bit (a x : Fin 8) :
    (bit8 a.val).getLsbD x.val = decide (x = a) := by
  by_cases h : x = a
  · subst x
    simp [bit8, a.isLt, -BitVec.getLsbD_eq_getElem]
  · have hn : x.val ≠ a.val := fun e => h (Fin.ext e)
    simp [bit8, h, -BitVec.getLsbD_eq_getElem]
    omega

theorem moveSpec_line (p o : Byte) (c : Color) (a : Fin 8)
    (disjoint : p &&& o = 0) :
    decodeColor (moveSpec p o a).1 (moveSpec p o a).2 c =
      linePlace (decodeColor p o c) c a := by
  classical
  funext x
  have capture := captureSpec_semantics p o c a x disjoint
  by_cases atMove : x = a
  · simp [moveSpec, decodeColor, linePlace, encodeBits_bit, bit8_bit, atMove,
      -BitVec.getLsbD_eq_getElem]
  · by_cases captured : captureSpec p o a x = true
    · have sem := capture.mp captured
      simp [moveSpec, decodeColor, linePlace, encodeBits_bit, bit8_bit,
        captured, sem, atMove, -BitVec.getLsbD_eq_getElem]
    · have noCapture : captureSpec p o a x = false := Bool.eq_false_iff.mpr captured
      have noSem : ¬ LineCaptured (decodeColor p o c) c a x := fun h => captured (capture.mpr h)
      simp [moveSpec, decodeColor, linePlace, encodeBits_bit, bit8_bit,
        noCapture, noSem, atMove, -BitVec.getLsbD_eq_getElem]

theorem emptyAt_bits (p o : Byte) (a : Fin 8) (h : emptyAt p o a = true) :
    p.getLsbD a.val = false ∧ o.getLsbD a.val = false := by
  have eq : (p ||| o) &&& bit8 a.val = 0 := by simpa [emptyAt] using h
  have bits := congrArg (fun v : Byte => v.getLsbD a.val) eq
  simpa [bit8_bit, -BitVec.getLsbD_eq_getElem] using bits

theorem bits_emptyAt (p o : Byte) (a : Fin 8)
    (hp : p.getLsbD a.val = false) (ho : o.getLsbD a.val = false) :
    emptyAt p o a = true := by
  have eq : (p ||| o) &&& bit8 a.val = 0 := by
    apply BitVec.eq_of_getLsbD_eq
    intro i hi
    have hb := bit8_bit a ⟨i,hi⟩
    by_cases same : (⟨i,hi⟩ : Fin 8) = a
    · have e : i = a.val := congrArg Fin.val same
      simp [e, hp, ho, -BitVec.getLsbD_eq_getElem]
    · simp [hb, same, -BitVec.getLsbD_eq_getElem]
  simpa [emptyAt] using eq

theorem moveSpec_disjoint (p o : Byte) (a : Fin 8)
    (disjoint : p &&& o = 0) (empty : emptyAt p o a = true) :
    (moveSpec p o a).1 &&& (moveSpec p o a).2 = 0 := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  let x : Fin 8 := ⟨i,hi⟩
  have old := congrArg (fun v : Byte => v.getLsbD x.val) disjoint
  simp only [BitVec.getLsbD_and] at old
  have emptyBits := emptyAt_bits p o a empty
  change ((moveSpec p o a).1 &&& (moveSpec p o a).2).getLsbD x.val = _
  by_cases same : x = a
  · simp [moveSpec, encodeBits_bit, bit8_bit, same, emptyBits.2,
      -BitVec.getLsbD_eq_getElem]
  · cases hp : p.getLsbD x.val <;> cases ho : o.getLsbD x.val <;>
      cases hc : captureSpec p o a x <;>
      simp_all [moveSpec, encodeBits_bit, bit8_bit, -BitVec.getLsbD_eq_getElem]

end NeuralReversi.Stability.Edge
