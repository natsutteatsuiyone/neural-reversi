import Lean

namespace NeuralReversi.Eval.PatternUpdate

def code (digits : Nat → Int) : List Nat → Int
  | [] => 0
  | q :: rest => 3 ^ rest.length * digits q + code digits rest

private theorem horner_seed (digits : Nat → Int) (squares : List Nat) (seed : Int) :
    squares.foldl (fun a q => 3 * a + digits q) seed =
      3 ^ squares.length * seed + code digits squares := by
  induction squares generalizing seed with
  | nil => simp [code]
  | cons q rest ih =>
    rw [List.foldl_cons, ih]
    simp only [code, List.length_cons, Int.pow_succ, Int.mul_add, Int.mul_assoc]
    omega

def HornerContract : Prop :=
  ∀ digits squares, squares.foldl (fun a q => 3 * a + digits q) 0 = code digits squares

theorem hornerContract_proven : HornerContract := by
  intro digits squares
  simp [horner_seed]

theorem code_congr (squares : List Nat) (f g : Nat → Int)
    (eq : ∀ i ∈ squares, f i = g i) : code f squares = code g squares := by
  induction squares with
  | nil => rfl
  | cons q rest ih =>
    rw [code, code, eq q (by simp), ih (fun i hi => eq i (by simp [hi]))]

theorem code_add (squares : List Nat) (f g : Nat → Int) :
    code (fun i => f i + g i) squares = code f squares + code g squares := by
  induction squares with
  | nil => rfl
  | cons q rest ih =>
    simp only [code, Int.mul_add, ih]
    omega

theorem code_sub (squares : List Nat) (f g : Nat → Int) :
    code (fun i => f i - g i) squares = code f squares - code g squares := by
  induction squares with
  | nil => rfl
  | cons q rest ih =>
    simp only [code, Int.mul_sub, ih]
    omega

theorem code_scale (squares : List Nat) (f : Nat → Int) (factor : Int) :
    code (fun i => factor * f i) squares = factor * code f squares := by
  induction squares with
  | nil => simp [code]
  | cons q rest ih =>
    simp only [code, ih, Int.mul_add]
    rw [Int.mul_left_comm (3 ^ rest.length)]

theorem code_zero (squares : List Nat) : code (fun _ => 0) squares = 0 := by
  induction squares with
  | nil => rfl
  | cons q rest ih => simp [code, ih]

theorem code_bound (digits : Nat → Int) (squares : List Nat)
    (valid : ∀ q ∈ squares, 0 ≤ digits q ∧ digits q ≤ 2) :
    0 ≤ code digits squares ∧ code digits squares < 3 ^ squares.length := by
  induction squares with
  | nil => simp [code]
  | cons q rest ih =>
    obtain ⟨lo, hi⟩ := valid q (by simp)
    obtain ⟨restLo, restHi⟩ := ih (fun i hi => valid i (by simp [hi]))
    have pos : 0 ≤ (3 : Int) ^ rest.length := Int.pow_nonneg (by decide)
    have termLow := Int.mul_nonneg pos lo
    have termHigh := Int.mul_le_mul_of_nonneg_left hi pos
    simp only [code, List.length_cons, Int.pow_succ]
    exact ⟨by omega, by omega⟩

def bit (b : Bool) : Int := if b then 1 else 0
def squareIndicator (square q : Nat) : Int := bit (decide (square = q))

def lookupWeight (square : Nat) : List Nat → Int
  | [] => 0
  | q :: rest => if square = q then 3 ^ rest.length else lookupWeight square rest

def LookupWeightContract : Prop :=
  ∀ squares square, squares.Nodup →
    lookupWeight square squares = code (squareIndicator square) squares

theorem lookupWeightContract_proven : LookupWeightContract := by
  intro squares square unique
  induction squares with
  | nil => rfl
  | cons q rest ih =>
    have distinct : q ∉ rest ∧ rest.Nodup := List.nodup_cons.mp unique
    by_cases h : square = q
    · subst q
      have z : code (squareIndicator square) rest = 0 := by
        rw [code_congr rest _ (fun _ => 0) (by
          intro i hi
          have neq : square ≠ i := fun eq => distinct.1 (by simpa [eq] using hi)
          simp [squareIndicator, bit, neq]), code_zero]
      simp [lookupWeight, code, squareIndicator, bit, z]
    · simp [lookupWeight, code, squareIndicator, bit, h, ih distinct.2]

end NeuralReversi.Eval.PatternUpdate
