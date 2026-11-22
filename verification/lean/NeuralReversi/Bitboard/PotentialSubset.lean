import NeuralReversi.Bitboard.MovesSpec
import NeuralReversi.Bitboard.PotentialSpec
namespace NeuralReversi.Bitboard.Moves
set_option maxHeartbeats 2000000

theorem ray_has_bracketable_opponent (p o : Bits) (target : Nat) (dr dc : Int)
    (targetBound : target < 64)
    (drLow : -1 ≤ dr) (drHigh : dr ≤ 1) (dcLow : -1 ≤ dc) (dcHigh : dc ≤ 1)
    (nonzero : dr ≠ 0 ∨ dc ≠ 0) (ray : Ray p o target dr dc) :
    BracketableOpponent o target := by
  obtain ⟨anchor,distance,anchorBound,distLow,_,_,last,chain⟩ := ray
  obtain ⟨neighbor,neighborBound,near,opponent⟩ := chain (distance-1) (by omega) (by omega)
  have beyond : ∃ further, further < 64 ∧ OnRay anchor further (distance-2) dr dc := by
    by_cases two : distance = 2
    · refine ⟨anchor,anchorBound,?_⟩
      simp [two,OnRay]
    · obtain ⟨further,bound,onRay,_⟩ := chain (distance-2) (by omega) (by omega)
      exact ⟨further,bound,onRay⟩
  obtain ⟨further,furtherBound,far⟩ := beyond
  refine ⟨neighbor,neighborBound,opponent,?_⟩
  have dm : ((distance-1 : Nat) : Int) = (distance : Int)-1 := by omega
  have dt : ((distance-2 : Nat) : Int) = (distance : Int)-2 := by omega
  simp only [OnRay] at last near far
  rw [dm,Int.sub_mul,Int.sub_mul,Int.one_mul,Int.one_mul] at near
  rw [dt,Int.sub_mul,Int.sub_mul] at far
  simp only [bracketableAdjacent,decide_eq_true_eq]
  omega

theorem legalMove_implies_potentialMove (p o : Bits) (target : Nat)
    (legal : LegalMove p o target) : PotentialMove p o target := by
  obtain ⟨bound,emptyP,emptyO,ray⟩ := legal
  refine ⟨bound,emptyP,emptyO,?_⟩
  rcases ray with ray|ray|ray|ray|ray|ray|ray|ray
  all_goals apply ray_has_bracketable_opponent p o target _ _ bound (by decide) (by decide) (by decide) (by decide) (by decide) ray

end NeuralReversi.Bitboard.Moves
