module
public import NeuralReversi.Flip.Avx2.Spec
public import NeuralReversi.Flip.Avx2.Assembly

@[expose] public section

namespace NeuralReversi.Flip.Avx2
open NeuralReversi.Flip

def publicIndex (sq : Square) : Index := ⟨sq.val, lookup_safe sq⟩

def mask66 (pos : Index) (d : Fin 8) : Word := (lrmask.get pos).get d

theorem mask66_public (sq : Square) (d : Fin 8) :
    mask66 (publicIndex sq) d = mask sq d := rfl

theorem mask66_typed (pos : Index) (h : pos.val < 64) (d : Fin 8) :
    mask66 pos d = mask ⟨pos.val, h⟩ d := rfl

theorem mask66_placeholders : ∀ pos : Index, 64 ≤ pos.val → ∀ d : Fin 8, mask66 pos d = 0 := by
  decide

theorem mask66_sentinel : ∀ d : Fin 8, mask66 ⟨65, by decide⟩ d = 0 := by decide

theorem laneSpec66_public (sq : Square) (d : Fin 8) (player opponent : Word) (i : Nat) :
    LaneSpec66 (publicIndex sq) d player opponent i ↔ LaneSpec sq d player opponent i := by
  simp only [LaneSpec66, publicIndex, sq.isLt, ↓reduceDIte]

theorem laneSpec66_playable (pos : Index) (d : Fin 8) (player opponent : Word) (i : Nat)
    (h : LaneSpec66 pos d player opponent i) : pos.val < 64 := by
  unfold LaneSpec66 at h
  split at h
  · by_cases hd : d.val < 4
    · simp only [LaneSpec, hd, ↓reduceIte] at h
      exact h.1.1.1
    · simp only [LaneSpec, hd, ↓reduceIte] at h
      exact h.2.1.1
  · exact False.elim h

def directionWord (pos : Index) (d : Fin 8) (player opponent : Word) : Word :=
  if d.val < 4 then leftWord (mask66 pos d) player opponent
  else rightWord (mask66 pos d) player opponent (rightStep d)

theorem left_zero (player opponent : Word) : leftWord 0 player opponent = 0 := by
  simp [leftWord, leftPrefixWord, eraseWithoutAnchor]

theorem right_zero (player opponent : Word) (step : Nat) :
    rightWord 0 player opponent step = 0 := by
  simp [rightWord]

theorem direction_bit (pos : Index) (d : Fin 8) (player opponent : Word) (i : Nat) :
    (directionWord pos d player opponent).getLsbD i = true ↔
      LaneSpec66 pos d player opponent i := by
  by_cases hp : pos.val < 64
  · simp only [LaneSpec66, hp, ↓reduceDIte, directionWord, mask66_typed pos hp]
    exact lane_raw ⟨pos.val, hp⟩ d player opponent i
  · have hge : 64 ≤ pos.val := by omega
    simp only [LaneSpec66, hp, ↓reduceDIte, directionWord, mask66_placeholders pos hge]
    by_cases hd : d.val < 4 <;> simp only [hd, ↓reduceIte, left_zero, right_zero] <;> simp

def leftDirection (l : Fin 4) : Fin 8 := ⟨l.val, by have := l.isLt; omega⟩
def rightDirection (l : Fin 4) : Fin 8 := ⟨4 + l.val, by have := l.isLt; omega⟩


def stride (l : Fin 4) : Nat :=
  match l.val with | 0 => 1 | 1 => 8 | 2 => 9 | _ => 7

theorem rightStep_lane : ∀ l : Fin 4, rightStep (rightDirection l) = stride l := by decide

theorem inputLane_left (l : Fin 4) : inputLane (leftDirection l) = l := by
  apply Fin.ext
  exact Nat.mod_eq_of_lt l.isLt

theorem inputLane_right (l : Fin 4) : inputLane (rightDirection l) = l := by
  apply Fin.ext
  change (4 + l.val) % 4 = l.val
  have := l.isLt
  omega

def derivedLane (pos : Index) (pp oo : Lanes) (l : Fin 4) : Word :=
  leftWord (mask66 pos (leftDirection l)) (pp l) (oo l) |||
    rightWord (mask66 pos (rightDirection l)) (pp l) (oo l) (stride l)

def directionInputWord (pos : Index) (pp oo : Lanes) (d : Fin 8) : Word :=
  directionWord pos d (pp (inputLane d)) (oo (inputLane d))

theorem derivedLane_parts (pos : Index) (pp oo : Lanes) (l : Fin 4) :
    derivedLane pos pp oo l =
      directionInputWord pos pp oo (leftDirection l) |||
        directionInputWord pos pp oo (rightDirection l) := by
  simp only [derivedLane, directionInputWord, inputLane_left, inputLane_right, directionWord]
  have hl : (leftDirection l).val < 4 := l.isLt
  have hr : ¬ (rightDirection l).val < 4 := by simp only [rightDirection]; omega
  rw [ite_eq_left hl, ite_eq_right hr, rightStep_lane]

def lowLane (j : Fin 2) : Fin 4 := ⟨j.val, by have := j.isLt; omega⟩
def highLane (j : Fin 2) : Fin 4 := ⟨j.val + 2, by have := j.isLt; omega⟩

def derivedPrepared (pos : Index) (pp oo : Lanes) : Pair := fun j =>
  derivedLane pos pp oo (lowLane j) ||| derivedLane pos pp oo (highLane j)

theorem direction_routing (j : Fin 2) (P : Fin 8 → Prop) :
    ((P (leftDirection (lowLane j)) ∨ P (rightDirection (lowLane j))) ∨
      (P (leftDirection (highLane j)) ∨ P (rightDirection (highLane j)))) ↔
      ∃ d : Fin 8, d.val % 2 = j.val ∧ P d := by
  constructor
  · rintro ((h | h) | (h | h))
    · exact ⟨leftDirection (lowLane j), by simp only [leftDirection, lowLane]; have := j.isLt; omega, h⟩
    · exact ⟨rightDirection (lowLane j), by simp only [rightDirection, lowLane]; have := j.isLt; omega, h⟩
    · exact ⟨leftDirection (highLane j), by simp only [leftDirection, highLane]; have := j.isLt; omega, h⟩
    · exact ⟨rightDirection (highLane j), by simp only [rightDirection, highLane]; have := j.isLt; omega, h⟩
  · rintro ⟨d, hd, hP⟩
    have hcases : d.val = j.val ∨ d.val = 4 + j.val ∨
        d.val = j.val + 2 ∨ d.val = 4 + (j.val + 2) := by
      have := j.isLt
      have := d.isLt
      omega
    rcases hcases with h | h | h | h
    · have heq : d = leftDirection (lowLane j) := Fin.ext h
      exact .inl (.inl (heq ▸ hP))
    · have heq : d = rightDirection (lowLane j) := Fin.ext h
      exact .inl (.inr (heq ▸ hP))
    · have heq : d = leftDirection (highLane j) := Fin.ext h
      exact .inr (.inl (heq ▸ hP))
    · have heq : d = rightDirection (highLane j) := Fin.ext h
      exact .inr (.inr (heq ▸ hP))



theorem prepared_pair_bit (pos : Index) (pp oo : Lanes) (j : Fin 2) (i : Nat) :
    (derivedPrepared pos pp oo j).getLsbD i = true ↔ PairBitSpec pos pp oo j i := by
  simp only [derivedPrepared, derivedLane_parts, BitVec.getLsbD_or, Bool.or_eq_true,
    directionInputWord, direction_bit]
  exact direction_routing j (fun d => LaneSpec66 pos d (pp (inputLane d)) (oo (inputLane d)) i)

theorem prepared_placeholders (pos : Index) (hpos : 64 ≤ pos.val) (pp oo : Lanes) :
    derivedPrepared pos pp oo = fun _ => 0 := by
  funext j
  apply BitVec.eq_of_getLsbD_eq
  intro i _
  have hn : ¬ (derivedPrepared pos pp oo j).getLsbD i = true := by
    intro h
    obtain ⟨d, _, hd⟩ := (prepared_pair_bit pos pp oo j i).mp h
    have := laneSpec66_playable pos d (pp (inputLane d)) (oo (inputLane d)) i hd
    omega
  simpa using hn

theorem prepared_none (pp oo : Lanes) :
    derivedPrepared ⟨64, by decide⟩ pp oo = fun _ => 0 :=
  prepared_placeholders _ (by decide) pp oo

theorem prepared_sentinel (pp oo : Lanes) :
    derivedPrepared ⟨65, by decide⟩ pp oo = fun _ => 0 :=
  prepared_placeholders _ (by decide) pp oo

def broadcast (word : Word) : Lanes := fun _ => word

theorem broadcast_public_reduction (sq : Square) (player opponent : Word) :
    derivedPrepared (publicIndex sq) (broadcast player) (broadcast opponent) 0 |||
      derivedPrepared (publicIndex sq) (broadcast player) (broadcast opponent) 1 =
      summary sq player opponent := by
  apply BitVec.eq_of_getLsbD_eq
  intro i _
  apply Bool.eq_iff_iff.mpr
  rw [BitVec.getLsbD_or]
  simp only [Bool.or_eq_true, prepared_pair_bit, PairBitSpec, broadcast,
    laneSpec66_public, Fin.val_zero, Fin.val_one]
  rw [summary_raw]
  constructor
  · rintro (⟨d, _, hd⟩ | ⟨d, _, hd⟩) <;> exact ⟨d, hd⟩
  · rintro ⟨d, hd⟩
    rcases Nat.mod_two_eq_zero_or_one d.val with h | h
    · exact .inl ⟨d, h, hd⟩
    · exact .inr ⟨d, h, hd⟩


end NeuralReversi.Flip.Avx2
