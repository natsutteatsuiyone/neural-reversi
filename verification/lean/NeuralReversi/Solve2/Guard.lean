module
public import NeuralReversi.Solve2.Dispatch
public import NeuralReversi.Flip.Properties
public import NeuralReversi.Bitboard.AdjacencyProperties

@[expose] public section

namespace NeuralReversi.Solve2.Guard
open NeuralReversi.Bitboard.Moves (OnRay)
set_option linter.unusedSimpArgs false

theorem delta_nonzero (d : Fin 8) : (Flip.delta d).2 ≠ 0 ∨ (Flip.delta d).1 ≠ 0 := by
  have hd : d.val = 0 ∨ d.val = 1 ∨ d.val = 2 ∨ d.val = 3 ∨
      d.val = 4 ∨ d.val = 5 ∨ d.val = 6 ∨ d.val = 7 := by omega
  rcases hd with hd | hd | hd | hd | hd | hd | hd | hd <;> simp [Flip.delta, hd]

theorem bracketable_of_two_ray_cells (sq near beyond : Nat) (d : Fin 8)
    (hs : sq < 64) (hn : near < 64) (hb : beyond < 64)
    (nearRay : OnRay sq near 1 (Flip.delta d).1 (Flip.delta d).2)
    (beyondRay : OnRay sq beyond 2 (Flip.delta d).1 (Flip.delta d).2) :
    Bitboard.bracketableAdjacent sq near = true := by
  have ranges := Flip.delta_range d
  have nonzero := delta_nonzero d
  simp only [OnRay, Int.natCast_one, Int.cast_ofNat_Int, Int.one_mul] at nearRay beyondRay
  have dx : (near % 8 : Int) - (sq % 8 : Int) = (Flip.delta d).2 := by omega
  have dy : (near / 8 : Int) - (sq / 8 : Int) = (Flip.delta d).1 := by omega
  simp only [Bitboard.bracketableAdjacent, decide_eq_true_eq, Int.natCast_emod, Int.natCast_ediv, Int.cast_ofNat_Int]
  rw [dx,dy]
  refine ⟨hs,hn,ranges.2.1,ranges.2.2,ranges.1.1,ranges.1.2,nonzero,?_,?_,?_,?_⟩
  all_goals omega

theorem scalar_nonzero_hasAdjacent (sq : Flip.Square) (player opponent : Flip.Word)
    (nonzero : Flip.value sq player opponent ≠ 0) : Bitboard.hasAdjacentBit opponent sq = true := by
  have bit := BitVec.getLsbD_true_ctz_of_ne_zero nonzero
  obtain ⟨d, hs, _, anchor, distance, ha, hd, _, _, _, anchorRay, chain, _⟩ :=
    (Flip.value_coordinate sq player opponent _).mp bit
  obtain ⟨near,hn,nearRay,opp⟩ := chain 1 (by omega) (by omega)
  have ranges := Flip.delta_range d
  obtain ⟨beyond,hb,beyondRay⟩ := Flip.onRay_prefix_exists sq.val anchor distance 2
    (Flip.delta d).1 (Flip.delta d).2 hs ha ranges.1 ranges.2 (by omega) anchorRay
  apply (Bitboard.adjacentBitContract_proven opponent sq).2.mpr
  exact ⟨near,hn,opp,bracketable_of_two_ray_cells sq.val near beyond d hs hn hb nearRay beyondRay⟩

theorem scalar_guard_zero (sq : Flip.Square) (player opponent : Flip.Word)
    (absent : Bitboard.hasAdjacentBit opponent sq = false) : Flip.value sq player opponent = 0 := by
  by_cases h : Flip.value sq player opponent = 0
  · exact h
  · have present := scalar_nonzero_hasAdjacent sq player opponent h
    simp [absent] at present

def scalarFlip : Dispatch.Flip := fun sq p o => Flip.value ⟨sq.val,by omega⟩ p o

def adjacent : Dispatch.Adjacent := fun o sq => Bitboard.hasAdjacentBit o ⟨sq.val,by omega⟩

def scalarPair : Dispatch.Flip2 := fun a b p o => (scalarFlip a p o, scalarFlip b p o)

theorem scalar_compatible : Dispatch.Compatible scalarFlip scalarPair adjacent := by
  constructor
  · intro a b p o; rfl
  · intro sq p o h
    exact scalar_guard_zero ⟨sq.val,by omega⟩ p o h

theorem scalar_eager_eq_fallback (oracle : Dispatch.Oracle) (player opponent : Dispatch.Word)
    (alpha : Score) (sq1 sq2 : Dispatch.Square) (terminal : Score) :
    Dispatch.eager oracle scalarPair player opponent alpha sq1 sq2 terminal =
      Dispatch.fallback oracle scalarFlip adjacent player opponent alpha sq1 sq2 terminal := by
  exact Dispatch.eager_eq_fallback oracle scalarFlip scalarPair adjacent player opponent alpha sq1 sq2 terminal
    rfl rfl (scalar_compatible.guard _ _ _) (scalar_compatible.guard _ _ _)
    (scalar_compatible.guard _ _ _) (scalar_compatible.guard _ _ _)

end NeuralReversi.Solve2.Guard
