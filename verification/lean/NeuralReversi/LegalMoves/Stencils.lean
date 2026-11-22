import NeuralReversi.Bitboard.MovesProperties

namespace NeuralReversi.LegalMoves.MoveBound

open Bitboard (Bits)
open Bitboard.Moves (OnRay Ray Bracketed LegalMove)

def directions : List (Int × Int) :=
  [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

abbrev Direction := Fin directions.length

def direction (d : Direction) : Int × Int := directions[d]

def row (target k : Nat) (d : Direction) : Int :=
  (target / 8 : Nat) - (k : Int) * (direction d).1

def col (target k : Nat) (d : Direction) : Int :=
  (target % 8 : Nat) - (k : Int) * (direction d).2

def inside (target k : Nat) (d : Direction) : Bool :=
  decide (0 ≤ row target k d ∧ row target k d < 8 ∧
    0 ≤ col target k d ∧ col target k d < 8)

def rayCell (target k : Nat) (d : Direction) : Nat :=
  8 * (row target k d).toNat + (col target k d).toNat

private theorem inside_geometry (target k : Nat) (d : Direction)
    (h : inside target k d = true) :
    rayCell target k d < 64 ∧
      OnRay (rayCell target k d) target k (direction d).1 (direction d).2 := by
  obtain ⟨rowLow, rowHigh, colLow, colHigh⟩ := of_decide_eq_true h
  have rowCast := Int.toNat_of_nonneg rowLow
  have colCast := Int.toNat_of_nonneg colLow
  unfold rayCell OnRay
  simp only [row, col] at rowCast colCast rowLow rowHigh colLow colHigh ⊢
  omega

private theorem geometry_inside (index target k : Nat) (d : Direction) (ib : index < 64)
    (h : OnRay index target k (direction d).1 (direction d).2) :
    inside target k d = true ∧ rayCell target k d = index := by
  have rw : (index / 8 : Nat) = row target k d := by unfold row; have := h.1; omega
  have cl : (index % 8 : Nat) = col target k d := by unfold col; have := h.2; omega
  constructor
  · apply decide_eq_true
    rw [← rw, ← cl]
    omega
  · unfold rayCell
    rw [← rw, ← cl]
    omega

private theorem rest_of_ray {anchor middle target k distance : Nat} {dr dc : Int}
    (last : OnRay anchor target distance dr dc)
    (part : OnRay anchor middle k dr dc) (within : k ≤ distance) :
    OnRay middle target (distance - k) dr dc := by
  simp only [OnRay, Int.ofNat_sub within, Int.sub_mul] at *
  exact ⟨by omega, by omega⟩

private theorem split_ray {anchor middle target k distance : Nat} {dr dc : Int}
    (last : OnRay anchor target distance dr dc)
    (rest : OnRay middle target k dr dc) (within : k ≤ distance) :
    OnRay anchor middle (distance - k) dr dc := by
  simp only [OnRay, Int.ofNat_sub within, Int.sub_mul] at *
  exact ⟨by omega, by omega⟩

def middleSteps (distance : Nat) : List Nat := List.range' 1 (distance - 1)

theorem mem_middleSteps (distance k : Nat) : k ∈ middleSteps distance ↔ 0 < k ∧ k < distance := by
  simp only [middleSteps, List.mem_range'_1]
  omega

def fits (target distance : Nat) (d : Direction) : Bool :=
  inside target distance d && (middleSteps distance).all (fun k => inside target k d)

def rayMatches (p o : Bits) (target distance : Nat) (d : Direction) : Bool :=
  p.getLsbD (rayCell target distance d) &&
    (middleSteps distance).all (fun k => o.getLsbD (rayCell target k d))

theorem fits_matches_iff (p o : Bits) (target : Nat) (d : Direction) :
    (∃ distance, 2 ≤ distance ∧ distance ≤ 7 ∧
      fits target distance d = true ∧ rayMatches p o target distance d = true) ↔
    Ray p o target (direction d).1 (direction d).2 := by
  constructor
  · rintro ⟨distance, low, high, shape, colors⟩
    obtain ⟨anchorValid, midsValid⟩ := Bool.and_eq_true_iff.mp shape
    obtain ⟨player, opponents⟩ := Bool.and_eq_true_iff.mp colors
    obtain ⟨anchorBelow, last⟩ := inside_geometry target distance d anchorValid
    refine ⟨rayCell target distance d, distance, anchorBelow, low, high, player, last, ?_⟩
    intro k pos within
    have step : distance - k ∈ middleSteps distance := (mem_middleSteps _ _).mpr (by omega)
    obtain ⟨midBelow, rest⟩ := inside_geometry target (distance - k) d
      (List.all_eq_true.mp midsValid _ step)
    have remaining := split_ray last rest (by omega : distance - k ≤ distance)
    rw [show distance - (distance - k) = k by omega] at remaining
    exact ⟨rayCell target (distance - k) d, midBelow, remaining, List.all_eq_true.mp opponents _ step⟩
  · rintro ⟨anchor, distance, anchorBelow, low, high, player, last, opponents⟩
    obtain ⟨anchorValid, atAnchor⟩ := geometry_inside anchor target distance d anchorBelow last
    have all (k : Nat) (mem : k ∈ middleSteps distance) :
        inside target k d = true ∧ o.getLsbD (rayCell target k d) = true := by
      have h := (mem_middleSteps _ _).mp mem
      obtain ⟨index, below, part, opponent⟩ := opponents (distance - k) (by omega) (by omega)
      have rest := rest_of_ray last part (by omega : distance - k ≤ distance)
      rw [show distance - (distance - k) = k by omega] at rest
      obtain ⟨valid, atIndex⟩ := geometry_inside index target k d below rest
      exact ⟨valid, atIndex ▸ opponent⟩
    refine ⟨distance, low, high, Bool.and_eq_true_iff.mpr ⟨anchorValid, ?_⟩,
      Bool.and_eq_true_iff.mpr ⟨?_, ?_⟩⟩
    · exact List.all_eq_true.mpr (fun k mem => (all k mem).1)
    · simpa only [atAnchor] using player
    · exact List.all_eq_true.mpr (fun k mem => (all k mem).2)

theorem bracket_iff (p o : Bits) (target : Nat) :
    (∃ d : Direction, ∃ distance, 2 ≤ distance ∧ distance ≤ 7 ∧
      fits target distance d = true ∧ rayMatches p o target distance d = true) ↔
    Bracketed p o target := by
  unfold Bracketed
  simp only [fits_matches_iff]
  change (∃ d : Fin 8, Ray p o target (direction d).1 (direction d).2) ↔ _
  simp only [Fin.exists_fin_succ, Fin.exists_fin_zero, or_false]
  rfl

def StencilContract : Prop :=
  ∀ p o target,
    LegalMove p o target ↔ target < 64 ∧ p.getLsbD target = false ∧
      o.getLsbD target = false ∧ ∃ d : Direction, ∃ distance, 2 ≤ distance ∧ distance ≤ 7 ∧
        fits target distance d = true ∧ rayMatches p o target distance d = true

theorem stencilContract_proven : StencilContract := by
  intro p o target
  unfold LegalMove
  rw [bracket_iff]

end NeuralReversi.LegalMoves.MoveBound
