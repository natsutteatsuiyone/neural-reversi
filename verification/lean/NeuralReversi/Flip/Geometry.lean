module
public import NeuralReversi.Flip.Spec

@[expose] public section

namespace NeuralReversi.Flip
open NeuralReversi.Bitboard.Moves (OnRay)

def Before (d : Fin 8) (i j : Nat) : Prop :=
  if d.val < 4 then i < j else j < i

theorem onRay_order (sq i j k l : Nat) (d : Fin 8)
    (hi : OnRay sq i k (delta d).1 (delta d).2)
    (hj : OnRay sq j l (delta d).1 (delta d).2) :
    Before d i j ↔ k < l := by
  have hd : d.val = 0 ∨ d.val = 1 ∨ d.val = 2 ∨ d.val = 3 ∨
      d.val = 4 ∨ d.val = 5 ∨ d.val = 6 ∨ d.val = 7 := by omega
  rcases hd with hd | hd | hd | hd | hd | hd | hd | hd
  all_goals simp only [delta, hd, OnRay, Before, Nat.reduceLT, ↓reduceIte,
    Int.mul_zero, Int.mul_one, Int.mul_neg, Int.add_zero, Int.reduceNeg] at hi hj ⊢
  all_goals omega

theorem onRay_unique (sq i j k : Nat) (dr dc : Int)
    (hi : OnRay sq i k dr dc) (hj : OnRay sq j k dr dc) : i = j := by
  rcases hi with ⟨hir, hic⟩
  rcases hj with ⟨hjr, hjc⟩
  omega

theorem delta_range (d : Fin 8) :
    (-1 ≤ (delta d).1 ∧ (delta d).1 ≤ 1) ∧
    (-1 ≤ (delta d).2 ∧ (delta d).2 ≤ 1) := by
  have hd : d.val = 0 ∨ d.val = 1 ∨ d.val = 2 ∨ d.val = 3 ∨
      d.val = 4 ∨ d.val = 5 ∨ d.val = 6 ∨ d.val = 7 := by omega
  rcases hd with hd | hd | hd | hd | hd | hd | hd | hd
  all_goals simp [delta, hd]

theorem onRay_prefix_exists (sq anchor distance k : Nat) (dr dc : Int)
    (hs : sq < 64) (ha : anchor < 64)
    (hdr : -1 ≤ dr ∧ dr ≤ 1) (hdc : -1 ≤ dc ∧ dc ≤ 1)
    (hk : k ≤ distance) (hr : OnRay sq anchor distance dr dc) :
    ∃ i, i < 64 ∧ OnRay sq i k dr dc := by
  rcases hr with ⟨hr, hc⟩
  have rowBound : 0 ≤ (sq / 8 : Int) + (k : Int) * dr ∧
      (sq / 8 : Int) + (k : Int) * dr < 8 := by
    have hd : dr = -1 ∨ dr = 0 ∨ dr = 1 := by omega
    rcases hd with rfl | rfl | rfl
    all_goals simp only [Int.mul_neg, Int.mul_one, Int.mul_zero, Int.add_zero] at hr ⊢
    all_goals omega
  have colBound : 0 ≤ (sq % 8 : Int) + (k : Int) * dc ∧
      (sq % 8 : Int) + (k : Int) * dc < 8 := by
    have hd : dc = -1 ∨ dc = 0 ∨ dc = 1 := by omega
    rcases hd with rfl | rfl | rfl
    all_goals simp only [Int.mul_neg, Int.mul_one, Int.mul_zero, Int.add_zero] at hc ⊢
    all_goals omega
  let r := (sq / 8 : Int) + (k : Int) * dr
  let c := (sq % 8 : Int) + (k : Int) * dc
  have rb : 0 ≤ r ∧ r < 8 := rowBound
  have cb : 0 ≤ c ∧ c < 8 := colBound
  refine ⟨r.toNat * 8 + c.toNat, ?_, ?_⟩
  · omega
  · change ((r.toNat * 8 + c.toNat) / 8 : Int) = r ∧
      ((r.toNat * 8 + c.toNat) % 8 : Int) = c
    omega

end NeuralReversi.Flip
