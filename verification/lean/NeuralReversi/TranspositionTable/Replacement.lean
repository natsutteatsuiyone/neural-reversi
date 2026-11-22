import NeuralReversi.TranspositionTable.Proofs

namespace NeuralReversi.TranspositionTable.Replacement

def relativeAge (now stored : BitVec 8) : BitVec 32 :=
  ((now - stored) &&& 127#8).setWidth 32

def replacementScore (now : BitVec 8) (entry : EntryData) : BitVec 32 :=
  entry.depth.setWidth 32 - relativeAge now entry.generation * 8

def victim (now : BitVec 8) (first second : EntryData) : Fin 2 :=
  if (replacementScore now second).toInt < (replacementScore now first).toInt then 1 else 0

def ringDistance (now stored : BitVec 8) : Nat :=
  (now.toNat % 128 + 128 - stored.toNat % 128) % 128

private theorem widen (b : BitVec 8) : (b.setWidth 32).toInt = b.toNat := by
  have small : b.toNat < 256 := b.toFin.isLt
  rw [BitVec.toInt_setWidth]
  exact Int.bmod_eq_of_le (by omega) (by omega)

private theorem age_exact (now stored : BitVec 8) :
    (relativeAge now stored).toInt = ringDistance now stored := by
  rw [relativeAge, widen, BitVec.toNat_and, BitVec.toNat_sub]
  have small : stored.toNat < 256 := stored.toFin.isLt
  apply congrArg Int.ofNat
  change (256 - stored.toNat + now.toNat) % 256 &&& (2^7 - 1) = _
  rw [Nat.and_two_pow_sub_one_eq_mod]
  unfold ringDistance
  omega

private theorem score_exact (now : BitVec 8) (e : EntryData) :
    (replacementScore now e).toInt = (e.depth.toNat : Int) - (ringDistance now e.generation : Int) * 8 := by
  have depthBound : e.depth.toNat < 256 := e.depth.toFin.isLt
  have ageBound : ringDistance now e.generation < 128 := Nat.mod_lt _ (by decide)
  rw [replacementScore, BitVec.toInt_sub, BitVec.toInt_mul, widen, age_exact]
  change (((e.depth.toNat : Int) - ((ringDistance now e.generation : Int) * 8).bmod (2^32)).bmod (2^32)) = _
  rw [Int.bmod_eq_of_le (n := (ringDistance now e.generation : Int) * 8) (by omega) (by omega)]
  exact Int.bmod_eq_of_le (by omega) (by omega)

def Contract : Prop :=
  (∀ now stored, (relativeAge now stored).toInt = ringDistance now stored) ∧
  (∀ now e, let score := (replacementScore now e).toInt
    score = (e.depth.toNat : Int) - (ringDistance now e.generation : Int) * 8 ∧
    -1016 ≤ score ∧ score ≤ 255) ∧
  ∀ now first second,
    victim now first second =
      if second.depth.toNat + 8 * ringDistance now first.generation <
          first.depth.toNat + 8 * ringDistance now second.generation then 1 else 0

theorem contract_proven : Contract := by
  refine ⟨age_exact, ?_, ?_⟩
  · intro now e
    have depthBound : e.depth.toNat < 256 := e.depth.toFin.isLt
    have ageBound : ringDistance now e.generation < 128 := Nat.mod_lt _ (by decide)
    dsimp only
    rw [score_exact]
    exact ⟨rfl, by omega, by omega⟩
  · intro now first second
    simp only [victim, score_exact]
    congr 1
    apply propext
    omega

end NeuralReversi.TranspositionTable.Replacement
