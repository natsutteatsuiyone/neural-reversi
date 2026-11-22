import NeuralReversi.Bitboard.MovesCarry
import NeuralReversi.Bitboard.MovesDirectional
import NeuralReversi.Bitboard.MovesKernel
namespace NeuralReversi.Bitboard.Moves
set_option maxRecDepth 4096
set_option maxHeartbeats 3000000

theorem unitStride_geometry (x y : Nat) (dr dc : Int)
    (dcLow : -1 ≤ dc) (dcHigh : dc ≤ 1)
    (stride : (y : Int) = (x : Int) + 8*dr+dc)
    (edge : dc = 0 ∨ (0 < x%8 ∧ x%8 < 7) ∨ (0 < y%8 ∧ y%8 < 7)) :
    OnRay x y 1 dr dc := by
  simp only [OnRay,Int.natCast_one,Int.one_mul]
  omega

theorem indexedPath_geometry (pos : Nat → Nat) (distance : Nat) (dr dc : Int)
    (distanceLow : 2 ≤ distance) (dcLow : -1 ≤ dc) (dcHigh : dc ≤ 1)
    (stride : ∀ k, k < distance → (pos (k+1) : Int) = (pos k : Int)+8*dr+dc)
    (edge : ∀ k, 0 < k → k < distance → dc = 0 ∨ (0 < (pos k)%8 ∧ (pos k)%8 < 7)) :
    ∀ k, k ≤ distance → OnRay (pos 0) (pos k) k dr dc := by
  intro k
  induction k with
  | zero => intro _; simp [OnRay]
  | succ k ih =>
    intro bound
    have previous := ih (by omega)
    have next := unitStride_geometry (pos k) (pos (k+1)) dr dc dcLow dcHigh (stride k (by omega))
    have interior : dc = 0 ∨ (0 < (pos k)%8 ∧ (pos k)%8 < 7) ∨
        (0 < (pos (k+1))%8 ∧ (pos (k+1))%8 < 7) := by
      by_cases zero : k = 0
      · have e := edge (k+1) (by omega) (by omega)
        rcases e with e|e
        · exact .inl e
        · exact .inr (.inr e)
      · have e := edge k (by omega) (by omega)
        rcases e with e|e
        · exact .inl e
        · exact .inr (.inl e)
    have next := next interior
    simp only [OnRay,Int.natCast_add,Int.natCast_one,Int.add_mul,Int.one_mul] at previous next ⊢
    omega

def LeftLinear (h p : Bits) (d i : Nat) : Prop :=
  ∃ distance, 2 ≤ distance ∧ distance ≤ 7 ∧ distance*d ≤ i ∧
    p.getLsbD (i-distance*d) = true ∧
    ∀ k, 0 < k → k < distance → h.getLsbD (i-k*d) = true

def RightLinear (h p : Bits) (d i : Nat) : Prop :=
  ∃ distance, 2 ≤ distance ∧ distance ≤ 7 ∧ p.getLsbD (i+distance*d) = true ∧
    ∀ k, 0 < k → k < distance → h.getLsbD (i+k*d) = true

theorem expandedLeft_shift_bits (h p : Bits) (d i : Nat) (ib : i < 64) :
    (expandedLeft h p d <<< d).getLsbD i = true ↔ LeftLinear h p d i := by
  rw [expandedLeft_exact,leftShift_true_iff _ d i ib,six_bits]
  constructor
  · rintro ⟨di,n,low,high,hn⟩
    obtain ⟨ni,hp,chain⟩ := (powerLeft_bits h p d n (i-d) (by omega)).mp hn
    refine ⟨n+1,by omega,by omega,by rw [Nat.add_mul]; simp; omega,?_,?_⟩
    · have eq : i-(n+1)*d = i-d-n*d := by rw [Nat.add_mul]; simp; omega
      simpa only [eq] using hp
    · intro k kp kn
      have eq : i-k*d = i-d-(k-1)*d := by
        have km : k*d = (k-1)*d+d := by simpa [Nat.add_mul] using congrArg (fun n : Nat => n*d) (show k = k-1+1 by omega)
        omega
      simpa only [eq] using chain (k-1) (by omega)
  · rintro ⟨distance,low,high,di,hp,chain⟩
    have dpos : d ≤ i := by have hm := Nat.mul_le_mul_right d (show 1≤distance by omega); simpa using Nat.le_trans hm di
    refine ⟨dpos,distance-1,by omega,by omega,?_⟩
    apply (powerLeft_bits h p d (distance-1) (i-d) (by omega)).mpr
    have dm : distance*d = (distance-1)*d+d := by simpa [Nat.add_mul] using congrArg (fun n : Nat => n*d) (show distance = distance-1+1 by omega)
    refine ⟨by omega,?_,?_⟩
    · have eq : i-distance*d = i-d-(distance-1)*d := by omega
      simpa only [eq] using hp
    · intro k kd
      have eq : i-(k+1)*d = i-d-k*d := by rw [Nat.add_mul]; simp; omega
      simpa only [eq] using chain (k+1) (by omega) (by omega)

theorem expandedRight_shift_bits (h p : Bits) (d i : Nat) :
    (expandedRight h p d >>> d).getLsbD i = true ↔ RightLinear h p d i := by
  rw [expandedRight_exact,BitVec.getLsbD_ushiftRight,six_bits]
  constructor
  · rintro ⟨n,low,high,hn⟩
    obtain ⟨hp,chain⟩ := (powerRight_bits h p d n (d+i)).mp hn
    refine ⟨n+1,by omega,by omega,?_,?_⟩
    · have eq : i+(n+1)*d = d+i+n*d := by rw [Nat.add_mul]; simp; omega
      simpa only [eq] using hp
    · intro k kp kn
      have eq : i+k*d = d+i+(k-1)*d := by
        have km : k*d = (k-1)*d+d := by simpa [Nat.add_mul] using congrArg (fun n : Nat => n*d) (show k=k-1+1 by omega)
        omega
      simpa only [eq] using chain (k-1) (by omega)
  · rintro ⟨distance,low,high,hp,chain⟩
    refine ⟨distance-1,by omega,by omega,?_⟩
    apply (powerRight_bits h p d (distance-1) (d+i)).mpr
    have dm : distance*d = (distance-1)*d+d := by simpa [Nat.add_mul] using congrArg (fun n : Nat => n*d) (show distance=distance-1+1 by omega)
    refine ⟨by simpa only [dm,Nat.add_assoc,Nat.add_left_comm,Nat.add_comm] using hp,?_⟩
    intro k kd
    have eq : i+(k+1)*d = d+i+k*d := by rw [Nat.add_mul]; simp; omega
    simpa only [eq] using chain (k+1) (by omega) (by omega)

theorem OnRay_index (anchor index k : Nat) (dr dc : Int) (h : OnRay anchor index k dr dc) :
    (index : Int) = (anchor : Int) + (k : Int)*(8*dr+dc) := by
  rcases h with ⟨row,col⟩
  calc
    (index : Int) = 8*(index/8 : Int)+(index%8 : Int) := by omega
    _ = 8*((anchor/8 : Int)+(k : Int)*dr)+((anchor%8 : Int)+(k : Int)*dc) := by rw [row,col]
    _ = (anchor : Int)+(k : Int)*(8*dr+dc) := by
      have hm : 8*((k : Int)*dr) = (k : Int)*(8*dr) := by ac_rfl
      simp only [Int.mul_add]
      rw [hm]
      omega

theorem OnRay_interior_file (anchor target index distance k : Nat) (dr dc : Int)
    (dcLow : -1 ≤ dc) (dcHigh : dc ≤ 1) (dcNonzero : dc ≠ 0)
    (kp : 0 < k) (kd : k < distance)
    (last : OnRay anchor target distance dr dc) (middle : OnRay anchor index k dr dc) :
    0 < index%8 ∧ index%8 < 7 := by
  have hcases : dc = -1 ∨ dc = 1 := by omega
  rcases hcases with rfl|rfl
  all_goals simp only [OnRay,Int.mul_neg,Int.mul_one] at last middle
  all_goals omega

def rayMask (o : Bits) (dc : Int) : Bits := if dc = 0 then o else o &&& horizontalMask

theorem rayMask_bit (o : Bits) (dc : Int) (i : Nat) (ib : i < 64) :
    (rayMask o dc).getLsbD i = true ↔
      o.getLsbD i = true ∧ (dc = 0 ∨ (0 < i%8 ∧ i%8 < 7)) := by
  by_cases z : dc = 0
  · simp [rayMask,z]
  · simp only [rayMask,z,↓reduceIte,horizontalOpponent_bit o i ib,false_or]

theorem subProduct_step (distance k d : Nat) (kd : k < distance) :
    (distance-k)*d = (distance-(k+1))*d+d := by
  simpa only [Nat.add_mul,Nat.one_mul] using
    congrArg (fun n : Nat => n*d) (show distance-k = distance-(k+1)+1 by omega)

theorem leftLinear_geometry (p o : Bits) (i d : Nat) (dr dc : Int)
    (ib : i < 64) (dcLow : -1 ≤ dc) (dcHigh : dc ≤ 1)
    (strideEq : (d : Int) = 8*dr+dc) :
    LeftLinear (rayMask o dc) p d i ↔ Ray p o i dr dc := by
  constructor
  · rintro ⟨distance,low,high,di,hp,chain⟩
    let pos := fun k => i-(distance-k)*d
    have mulBound : ∀ k, (distance-k)*d ≤ i := by
      intro k
      exact Nat.le_trans (Nat.mul_le_mul_right d (Nat.sub_le distance k)) di
    have path : ∀ k, k ≤ distance → OnRay (pos 0) (pos k) k dr dc := by
      apply indexedPath_geometry pos distance dr dc low dcLow dcHigh
      · intro k kd
        have step := subProduct_step distance k d kd
        have m0 := mulBound k
        have m1 := mulBound (k+1)
        dsimp [pos]
        omega
      · intro k kp kd
        have h := chain (distance-k) (by omega) (by omega)
        exact ((rayMask_bit o dc (pos k) (by dsimp [pos]; omega)).mp h).2
    refine ⟨pos 0,distance,by dsimp [pos]; omega,low,high,hp,?_,?_⟩
    · simpa [pos] using path distance (by omega)
    · intro k kp kd
      have h := chain (distance-k) (by omega) (by omega)
      exact ⟨pos k,by dsimp [pos]; omega,path k (by omega),
        ((rayMask_bit o dc (pos k) (by dsimp [pos]; omega)).mp h).1⟩
  · rintro ⟨anchor,distance,ab,low,high,hp,last,chain⟩
    have endpoint := OnRay_index anchor i distance dr dc last
    rw [←strideEq,←Int.natCast_mul] at endpoint
    have ieq : i = anchor+distance*d := by omega
    refine ⟨distance,low,high,by omega,?_,?_⟩
    · have eq : i-distance*d = anchor := by omega
      simpa only [eq] using hp
    · intro k kp kd
      obtain ⟨index,indexBound,middle,ho⟩ := chain (distance-k) (by omega) (by omega)
      have idx := OnRay_index anchor index (distance-k) dr dc middle
      rw [←strideEq,←Int.natCast_mul,Nat.sub_mul] at idx
      have kprod : k*d ≤ distance*d := Nat.mul_le_mul_right d (by omega)
      have eq : i-k*d = index := by omega
      rw [eq]
      apply (rayMask_bit o dc index indexBound).mpr
      refine ⟨ho,?_⟩
      by_cases z : dc = 0
      · exact .inl z
      · exact .inr (OnRay_interior_file anchor i index distance (distance-k) dr dc dcLow dcHigh z (by omega) (by omega) last middle)

theorem rightLinear_geometry (p o : Bits) (i d : Nat) (dr dc : Int)
    (dcLow : -1 ≤ dc) (dcHigh : dc ≤ 1)
    (strideEq : (d : Int) = -(8*dr+dc)) :
    RightLinear (rayMask o dc) p d i ↔ Ray p o i dr dc := by
  constructor
  · rintro ⟨distance,low,high,hp,chain⟩
    have anchorBound := BitVec.lt_of_getLsbD hp
    let pos := fun k => i+(distance-k)*d
    have mulBound : ∀ k, (distance-k)*d ≤ distance*d := by
      intro k
      exact Nat.mul_le_mul_right d (Nat.sub_le distance k)
    have posBound : ∀ k, pos k < 64 := by
      intro k
      have bound := mulBound k
      dsimp [pos]
      omega
    have path : ∀ k, k ≤ distance → OnRay (pos 0) (pos k) k dr dc := by
      apply indexedPath_geometry pos distance dr dc low dcLow dcHigh
      · intro k kd
        have step := subProduct_step distance k d kd
        dsimp [pos]
        omega
      · intro k kp kd
        have h := chain (distance-k) (by omega) (by omega)
        exact ((rayMask_bit o dc (pos k) (posBound k)).mp h).2
    refine ⟨pos 0,distance,posBound 0,low,high,hp,?_,?_⟩
    · simpa [pos] using path distance (by omega)
    · intro k kp kd
      have h := chain (distance-k) (by omega) (by omega)
      exact ⟨pos k,posBound k,path k (by omega),((rayMask_bit o dc (pos k) (posBound k)).mp h).1⟩
  · rintro ⟨anchor,distance,ab,low,high,hp,last,chain⟩
    have negative : 8*dr+dc = -(d : Int) := by omega
    have endpoint := OnRay_index anchor i distance dr dc last
    rw [negative,Int.mul_neg,←Int.natCast_mul] at endpoint
    have ieq : anchor = i+distance*d := by omega
    refine ⟨distance,low,high,?_,?_⟩
    · simpa only [ieq] using hp
    · intro k kp kd
      obtain ⟨index,indexBound,middle,ho⟩ := chain (distance-k) (by omega) (by omega)
      have idx := OnRay_index anchor index (distance-k) dr dc middle
      rw [negative,Int.mul_neg,←Int.natCast_mul,Nat.sub_mul] at idx
      have kprod : k*d ≤ distance*d := Nat.mul_le_mul_right d (by omega)
      have eq : i+k*d = index := by omega
      rw [eq]
      apply (rayMask_bit o dc index indexBound).mpr
      refine ⟨ho,?_⟩
      by_cases z : dc = 0
      · exact .inl z
      · exact .inr (OnRay_interior_file anchor i index distance (distance-k) dr dc dcLow dcHigh z (by omega) (by omega) last middle)

theorem left_direction_exact (p o : Bits) (i d : Nat) (dr dc : Int)
    (ib : i < 64) (dcLow : -1 ≤ dc) (dcHigh : dc ≤ 1)
    (strideEq : (d : Int) = 8*dr+dc) :
    (expandedLeft (rayMask o dc) p d <<< d).getLsbD i = true ↔ Ray p o i dr dc := by
  rw [expandedLeft_shift_bits _ _ d i ib,leftLinear_geometry p o i d dr dc ib dcLow dcHigh strideEq]

theorem right_direction_exact (p o : Bits) (i d : Nat) (dr dc : Int)
    (dcLow : -1 ≤ dc) (dcHigh : dc ≤ 1)
    (strideEq : (d : Int) = -(8*dr+dc)) :
    (expandedRight (rayMask o dc) p d >>> d).getLsbD i = true ↔ Ray p o i dr dc := by
  rw [expandedRight_shift_bits,rightLinear_geometry p o i d dr dc dcLow dcHigh strideEq]

theorem portableDirections_empty_iff (p o : Bits) (i : Nat) (ib : i < 64)
    (empty : (p ||| o).getLsbD i = false) :
    (portableDirections p o).getLsbD i = true ↔ Bracketed p o i := by
  have l7 : (expandedLeft (o &&& horizontalMask) p 7 <<< (7:Nat)).getLsbD i = true ↔ Ray p o i 1 (-1) := by
    simpa only [rayMask,Int.reduceNeg,Int.reduceEq,↓reduceIte] using left_direction_exact p o i 7 1 (-1) ib (by decide) (by decide) (by decide)
  have l9 : (expandedLeft (o &&& horizontalMask) p 9 <<< (9:Nat)).getLsbD i = true ↔ Ray p o i 1 1 := by
    simpa only [rayMask,Int.reduceEq,↓reduceIte] using left_direction_exact p o i 9 1 1 ib (by decide) (by decide) (by decide)
  have l8 : (expandedLeft o p 8 <<< (8:Nat)).getLsbD i = true ↔ Ray p o i 1 0 := by
    simpa only [rayMask,↓reduceIte] using left_direction_exact p o i 8 1 0 ib (by decide) (by decide) (by decide)
  have r7 : (expandedRight (o &&& horizontalMask) p 7 >>> (7:Nat)).getLsbD i = true ↔ Ray p o i (-1) 1 := by
    simpa only [rayMask,Int.reduceEq,↓reduceIte] using right_direction_exact p o i 7 (-1) 1 (by decide) (by decide) (by decide)
  have r9 : (expandedRight (o &&& horizontalMask) p 9 >>> (9:Nat)).getLsbD i = true ↔ Ray p o i (-1) (-1) := by
    simpa only [rayMask,Int.reduceNeg,Int.reduceEq,↓reduceIte] using right_direction_exact p o i 9 (-1) (-1) (by decide) (by decide) (by decide)
  have r8 : (expandedRight o p 8 >>> (8:Nat)).getLsbD i = true ↔ Ray p o i (-1) 0 := by
    simpa only [rayMask,↓reduceIte] using right_direction_exact p o i 8 (-1) 0 (by decide) (by decide) (by decide)
  have r1 : (expandedRight (o &&& horizontalMask) p 1 >>> (1:Nat)).getLsbD i = true ↔ Ray p o i 0 (-1) := by
    simpa only [rayMask,Int.reduceNeg,Int.reduceEq,↓reduceIte] using right_direction_exact p o i 1 0 (-1) (by decide) (by decide) (by decide)
  simp only [portableDirections,BitVec.getLsbD_or,Bool.or_eq_true]
  rw [portable_east_exact p o i ib empty,eastRay_geometry p o i ib,l7,l9,l8,r7,r9,r8,r1]
  simp only [Bracketed,_root_.or_assoc,_root_.or_left_comm,_root_.or_comm]

theorem getMovesPortable_exact (p o : Bits) (i : Nat) :
    (getMovesPortable p o).getLsbD i = true ↔ LegalMove p o i := by
  by_cases ib : i < 64
  · by_cases empty : (p ||| o).getLsbD i = false
    · have zeros := empty
      simp only [BitVec.getLsbD_or,Bool.or_eq_false_iff] at zeros
      rw [portable_decomposition,BitVec.getLsbD_and,BitVec.getLsbD_not]
      simp only [ib,decide_true,empty,Bool.not_false,Bool.and_true]
      rw [portableDirections_empty_iff p o i ib empty]
      simp only [LegalMove,ib,zeros.1,zeros.2,true_and]
    · have full : (p ||| o).getLsbD i = true := by cases h : (p ||| o).getLsbD i <;> simp_all
      rw [portable_decomposition,BitVec.getLsbD_and,BitVec.getLsbD_not]
      simp only [full,Bool.not_true,Bool.and_false,Bool.false_eq_true,false_iff]
      intro legal
      have contradict : (p ||| o).getLsbD i = false := by
        simp only [BitVec.getLsbD_or,legal.2.1,legal.2.2.1,Bool.false_or]
      exact empty contradict
  · have zero := BitVec.getLsbD_of_ge (getMovesPortable p o) i (by omega)
    simp [zero,LegalMove,ib]

end NeuralReversi.Bitboard.Moves
