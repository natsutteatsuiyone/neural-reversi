module
public import NeuralReversi.Bitboard.MovesModel

@[expose] public section

namespace NeuralReversi.Bitboard.Moves
set_option maxHeartbeats 2000000

def stepLeft (h : BitVec w) (d : Nat) (x : BitVec w) : BitVec w := h &&& (x <<< d)
def stepRight (h : BitVec w) (d : Nat) (x : BitVec w) : BitVec w := h &&& (x >>> d)

theorem stepLeft_or (h x y : BitVec w) (d : Nat) :
    stepLeft h d (x ||| y) = stepLeft h d x ||| stepLeft h d y := by
  simp only [stepLeft,BitVec.shiftLeft_or_distrib,BitVec.and_or_distrib_left]

theorem stepRight_or (h x y : BitVec w) (d : Nat) :
    stepRight h d (x ||| y) = stepRight h d x ||| stepRight h d y := by
  simp only [stepRight,BitVec.ushiftRight_or_distrib,BitVec.and_or_distrib_left]

theorem preMaskLeft_two (h x : BitVec w) (d : Nat) :
    (h &&& (h <<< d)) &&& (x <<< (2*d)) = stepLeft h d (stepLeft h d x) := by
  simp only [stepLeft,BitVec.shiftLeft_and_distrib,←BitVec.shiftLeft_add,BitVec.and_assoc]
  rw [show 2*d = d+d by omega]

theorem preMaskRight_two (h x : BitVec w) (d : Nat) :
    (h &&& (h >>> d)) &&& (x >>> (2*d)) = stepRight h d (stepRight h d x) := by
  simp only [stepRight,BitVec.ushiftRight_and_distrib,←BitVec.shiftRight_add,BitVec.and_assoc]
  rw [show 2*d = d+d by omega]

def power (step : BitVec w → BitVec w) (x : BitVec w) : Nat → BitVec w
  | 0 => x
  | n+1 => step (power step x n)

def six (step : BitVec w → BitVec w) (p : BitVec w) : BitVec w :=
  power step p 1 ||| power step p 2 ||| power step p 3 |||
  power step p 4 ||| power step p 5 ||| power step p 6

theorem or_left_comm (a b c : BitVec w) : a ||| (b ||| c) = b ||| (a ||| c) := by
  rw [←BitVec.or_assoc,BitVec.or_comm a b,BitVec.or_assoc]
theorem or_dup (a b : BitVec w) : a ||| (a ||| b) = a ||| b := by
  rw [←BitVec.or_assoc,BitVec.or_self]
theorem two_doublings_exact (step : BitVec w → BitVec w)
    (join : ∀ x y, step (x ||| y) = step x ||| step y) (p : BitVec w) :
    (let first := step p
     let two := first ||| step first
     let four := two ||| step (step two)
     four ||| step (step four)) = six step p := by
  simp only [six,power,join,or_left_comm,BitVec.or_comm,or_dup]

def expandedLeft (h p : BitVec w) (d : Nat) : BitVec w :=
  let first := h &&& (p <<< d)
  let two := first ||| (h &&& (first <<< d))
  let pre := h &&& (h <<< d)
  let four := two ||| (pre &&& (two <<< (2*d)))
  four ||| (pre &&& (four <<< (2*d)))

def expandedRight (h p : BitVec w) (d : Nat) : BitVec w :=
  let first := h &&& (p >>> d)
  let two := first ||| (h &&& (first >>> d))
  let pre := h &&& (h >>> d)
  let four := two ||| (pre &&& (two >>> (2*d)))
  four ||| (pre &&& (four >>> (2*d)))

theorem expandedLeft_exact (h p : BitVec w) (d : Nat) : expandedLeft h p d = six (stepLeft h d) p := by
  simp only [expandedLeft,preMaskLeft_two]
  exact two_doublings_exact (stepLeft h d) (fun x y => stepLeft_or h x y d) p

theorem expandedRight_exact (h p : BitVec w) (d : Nat) : expandedRight h p d = six (stepRight h d) p := by
  simp only [expandedRight,preMaskRight_two]
  exact two_doublings_exact (stepRight h d) (fun x y => stepRight_or h x y d) p

theorem leftShift_true_iff (x : BitVec w) (d i : Nat) (ib : i < w) :
    (x <<< d).getLsbD i = true ↔ d ≤ i ∧ x.getLsbD (i-d) = true := by
  rw [BitVec.getLsbD_shiftLeft]
  by_cases hd : d ≤ i
  · simp only [show decide (i<w) = true by simp [ib],
      show decide (i<d) = false by simp; omega,Bool.not_false,Bool.true_and,hd,true_and]
  · simp only [show decide (i<w) = true by simp [ib],
      show decide (i<d) = true by simp; omega,Bool.not_true,Bool.and_false,Bool.false_and,hd,false_and]
    decide

theorem powerLeft_bits (h p : BitVec w) (d n i : Nat) (ib : i < w) :
    (power (stepLeft h d) p n).getLsbD i = true ↔
      n*d ≤ i ∧ p.getLsbD (i-n*d) = true ∧ ∀ k, k < n → h.getLsbD (i-k*d) = true := by
  induction n generalizing i with
  | zero => simp [power]
  | succ n ih =>
    rw [power,stepLeft,BitVec.getLsbD_and,Bool.and_eq_true,leftShift_true_iff _ d i ib,
      ih (i-d) (by omega)]
    constructor
    · rintro ⟨h0,hd,hn,hp,chain⟩
      refine ⟨by rw [Nat.succ_mul]; omega,?_,?_⟩
      · have eq : i-(n+1)*d = i-d-n*d := by rw [Nat.succ_mul]; omega
        simpa only [eq] using hp
      · intro k hk
        cases k with
        | zero => simpa only [Nat.zero_mul,Nat.sub_zero] using h0
        | succ k =>
          have eq : i-(k+1)*d = i-d-k*d := by rw [Nat.succ_mul]; omega
          simpa only [eq] using chain k (by omega)
    · rintro ⟨hn,hp,chain⟩
      have hd : d ≤ i := by rw [Nat.succ_mul] at hn; omega
      refine ⟨by simpa only [Nat.zero_mul,Nat.sub_zero] using chain 0 (by omega),hd,?_,?_,?_⟩
      · rw [Nat.succ_mul] at hn; omega
      · have eq : i-(n+1)*d = i-d-n*d := by rw [Nat.succ_mul]; omega
        simpa only [eq] using hp
      · intro k hk
        have eq : i-(k+1)*d = i-d-k*d := by rw [Nat.succ_mul]; omega
        simpa only [eq] using chain (k+1) (by omega)

theorem powerRight_bits (h p : BitVec w) (d n i : Nat) :
    (power (stepRight h d) p n).getLsbD i = true ↔
      p.getLsbD (i+n*d) = true ∧ ∀ k, k < n → h.getLsbD (i+k*d) = true := by
  induction n generalizing i with
  | zero => simp [power]
  | succ n ih =>
    rw [power,stepRight,BitVec.getLsbD_and,Bool.and_eq_true,BitVec.getLsbD_ushiftRight,ih]
    constructor
    · rintro ⟨h0,hp,chain⟩
      refine ⟨?_,?_⟩
      · have eq : i+(n+1)*d = d+i+n*d := by rw [Nat.succ_mul]; omega
        simpa only [eq] using hp
      · intro k hk
        cases k with
        | zero => simpa only [Nat.zero_mul,Nat.add_zero] using h0
        | succ k =>
          have eq : i+(k+1)*d = d+i+k*d := by rw [Nat.succ_mul]; omega
          simpa only [eq] using chain k (by omega)
    · rintro ⟨hp,chain⟩
      refine ⟨by simpa only [Nat.zero_mul,Nat.add_zero] using chain 0 (by omega),?_,?_⟩
      · have eq : i+(n+1)*d = d+i+n*d := by rw [Nat.succ_mul]; omega
        simpa only [eq] using hp
      · intro k hk
        have eq : i+(k+1)*d = d+i+k*d := by rw [Nat.succ_mul]; omega
        simpa only [eq] using chain (k+1) (by omega)

theorem six_bits (step : BitVec w → BitVec w) (p : BitVec w) (i : Nat) :
    (six step p).getLsbD i = true ↔
      ∃ n, 1 ≤ n ∧ n ≤ 6 ∧ (power step p n).getLsbD i = true := by
  simp only [six,BitVec.getLsbD_or,Bool.or_eq_true]
  constructor
  · intro h
    rcases h with (((((h | h) | h) | h) | h) | h)
    all_goals first | exact ⟨1,by decide,by decide,h⟩ | exact ⟨2,by decide,by decide,h⟩ |
      exact ⟨3,by decide,by decide,h⟩ | exact ⟨4,by decide,by decide,h⟩ |
      exact ⟨5,by decide,by decide,h⟩ | exact ⟨6,by decide,by decide,h⟩
  · rintro ⟨n,low,high,h⟩
    have cases : n=1 ∨ n=2 ∨ n=3 ∨ n=4 ∨ n=5 ∨ n=6 := by omega
    rcases cases with rfl|rfl|rfl|rfl|rfl|rfl <;> simp [h]

end NeuralReversi.Bitboard.Moves
