module
public import NeuralReversi.Solve34.BoardBridge
public import NeuralReversi.Solve34.BaseBounds
public import NeuralReversi.Solve34.Control
public import NeuralReversi.Solve34.Terminal
public import NeuralReversi.Solve34.ControlBounds

@[expose] public section

namespace NeuralReversi.Solve34

def WindowSafe (alpha : Score) : Prop :=
  -2147483648 < alpha.toInt ∧ alpha.toInt < 2147483647

theorem childAlpha_windowSafe (alpha : Score) (safe : WindowSafe alpha) :
    WindowSafe (Solve2.childAlpha alpha) := by
  rw [WindowSafe, Solve2.childAlpha_safe alpha safe.2]
  obtain ⟨lo, hi⟩ := safe
  constructor <;> omega

def generatedFlip (b : EmptyBoard n) (sq : Square) : Word :=
  Flip.value ⟨sq.val, by omega⟩ b.player b.opponent

theorem generatedFlip_subset (b : EmptyBoard n) (sq : Square) :
    Solve2.FlipSubset (generatedFlip b sq) b.opponent :=
  Solve2.scalar_flip_subset sq b.player b.opponent

def moveChild (b : EmptyBoard n) (sq : {s // s ∈ b.squares}) : EmptyBoard (n-1) :=
  b.play sq.val sq.property (generatedFlip b sq.val) (generatedFlip_subset b sq.val)

def candidateScores (b : EmptyBoard n) (child : EmptyBoard (n-1) → Score → Score)
    (alpha : Score) : List (Option Score) :=
  b.squares.attach.map fun sq =>
    if generatedFlip b sq.val = 0 then none else some (child (moveChild b sq) alpha)

def candidateValues (b : EmptyBoard n) (child : EmptyBoard (n-1) → Int) :
    List (Option Int) :=
  b.squares.attach.map fun sq =>
    if generatedFlip b sq.val = 0 then none else some (child (moveChild b sq))

theorem listRel_map (xs : List α) (f : α → β) (g : α → γ) (R : β → γ → Prop)
    (h : ∀ x, R (f x) (g x)) : ListRel R (xs.map f) (xs.map g) := by
  induction xs with
  | nil => exact .nil
  | cons x xs ih => exact .cons (h x) ih

theorem candidates_contract (b : EmptyBoard n)
    (child : EmptyBoard (n-1) → Score → Score) (value : EmptyBoard (n-1) → Int)
    (alpha : Score)
    (correct : ∀ cb, Solve2.NWS alpha.toInt (value cb) (child cb alpha).toInt) :
    ListRel (Solve2.OptionalRel (Solve2.NWS alpha.toInt))
      (candidateValues b value) ((candidateScores b child alpha).map (Option.map BitVec.toInt)) := by
  unfold candidateValues candidateScores
  rw [List.map_map]
  apply listRel_map
  intro sq
  dsimp
  split
  · trivial
  · exact correct (moveChild b sq)

theorem candidates_bounds (b : EmptyBoard n)
    (child : EmptyBoard (n-1) → Score → Score) (alpha : Score) (lo hi : Int)
    (bounds : ∀ cb, lo ≤ (child cb alpha).toInt ∧ (child cb alpha).toInt ≤ hi) :
    ∀ o ∈ (candidateScores b child alpha).map (Option.map BitVec.toInt),
      BoundedOption lo hi o := by
  intro o ho
  simp only [candidateScores, List.map_map, List.mem_map] at ho
  obtain ⟨sq, _, rfl⟩ := ho
  dsimp
  split
  · intro s hs; contradiction
  · intro s hs
    cases hs
    exact bounds (moveChild b sq)

theorem candidates_within (b : EmptyBoard n)
    (child : EmptyBoard (n-1) → Score → Score) (alpha : Score)
    (bounds : ∀ cb, -114 ≤ (child cb alpha).toInt ∧ (child cb alpha).toInt ≤ 114) :
    AllWithin ((candidateScores b child alpha).map (Option.map BitVec.toInt)) := by
  intro o ho r hr
  have h := candidates_bounds b child alpha (-114) 114 bounds o ho r hr
  dsimp [inf]
  constructor <;> omega

def solveNext (b : EmptyBoard n) (child : EmptyBoard (n-1) → Score → Score)
    (alpha : Score) : Score :=
  solveWord alpha (candidateScores b child (Solve2.childAlpha alpha))
    (candidateScores b.pass child alpha) (terminalScore b)

def nextValue (b : EmptyBoard n) (child : EmptyBoard (n-1) → Int) : Int :=
  value (candidateValues b child) (candidateValues b.pass child)
    (Board.AwardedScore b.asBoard)

theorem solveNext_nws (b : EmptyBoard n)
    (child : EmptyBoard (n-1) → Score → Score) (childValue : EmptyBoard (n-1) → Int)
    (correct : ∀ cb a, WindowSafe a →
      Solve2.NWS a.toInt (childValue cb) (child cb a).toInt)
    (bounds : ∀ cb a, WindowSafe a →
      -114 ≤ (child cb a).toInt ∧ (child cb a).toInt ≤ 114)
    (alpha : Score) (safe : WindowSafe alpha) :
    Solve2.NWS alpha.toInt (nextValue b childValue) (solveNext b child alpha).toInt := by
  unfold solveNext nextValue
  rw [← terminal_awarded b]
  exact solveWord_nws safe.2
    (candidates_contract b child childValue _ (fun cb => correct cb _ (childAlpha_windowSafe alpha safe)))
    (candidates_contract b.pass child childValue _ (fun cb => correct cb _ safe))
    (candidates_within b child _ (fun cb => bounds cb _ (childAlpha_windowSafe alpha safe)))
    (candidates_within b.pass child _ (fun cb => bounds cb _ safe))

theorem solveNext_bounds (b : EmptyBoard n)
    (child : EmptyBoard (n-1) → Score → Score)
    (bounds : ∀ cb a, WindowSafe a →
      -114 ≤ (child cb a).toInt ∧ (child cb a).toInt ≤ 114)
    (alpha : Score) (safe : WindowSafe alpha) :
    -114 ≤ (solveNext b child alpha).toInt ∧ (solveNext b child alpha).toInt ≤ 114 := by
  apply solveWord_bounds alpha _ _ _ safe.2
  · exact candidates_bounds b child _ (-114) 114
      (fun cb => bounds cb _ (childAlpha_windowSafe alpha safe))
  · exact candidates_bounds b.pass child _ (-114) 114 (fun cb => bounds cb _ safe)
  · have h := terminal_bounds b
    constructor <;> omega

def twoSolver (b : EmptyBoard 2) (alpha : Score) : Score :=
  Solve2.solveConcrete b.toTwoEmpty alpha

def twoValue (b : EmptyBoard 2) : Int := Solve2.exactTwoValue b.toTwoEmpty

theorem twoSolver_nws (b : EmptyBoard 2) (alpha : Score) (safe : WindowSafe alpha) :
    Solve2.NWS alpha.toInt (twoValue b) (twoSolver b alpha).toInt :=
  Solve2.solveConcrete_nws b.toTwoEmpty alpha safe.2

theorem twoSolver_bounds (b : EmptyBoard 2) (alpha : Score) (safe : WindowSafe alpha) :
    -114 ≤ (twoSolver b alpha).toInt ∧ (twoSolver b alpha).toInt ≤ 114 :=
  solve2_bounds b.toTwoEmpty alpha safe.2

def solve3Concrete (b : EmptyBoard 3) (alpha : Score) : Score :=
  solveNext b twoSolver alpha

def exactThreeValue (b : EmptyBoard 3) : Int := nextValue b twoValue

theorem solve3Concrete_nws (b : EmptyBoard 3) (alpha : Score) (safe : WindowSafe alpha) :
    Solve2.NWS alpha.toInt (exactThreeValue b) (solve3Concrete b alpha).toInt :=
  solveNext_nws b twoSolver twoValue twoSolver_nws twoSolver_bounds alpha safe

theorem solve3Concrete_bounds (b : EmptyBoard 3) (alpha : Score) (safe : WindowSafe alpha) :
    -114 ≤ (solve3Concrete b alpha).toInt ∧ (solve3Concrete b alpha).toInt ≤ 114 :=
  solveNext_bounds b twoSolver twoSolver_bounds alpha safe

def solve4Concrete (b : EmptyBoard 4) (alpha : Score) : Score :=
  solveNext b solve3Concrete alpha

def exactFourValue (b : EmptyBoard 4) : Int := nextValue b exactThreeValue

theorem solve4Concrete_nws (b : EmptyBoard 4) (alpha : Score) (safe : WindowSafe alpha) :
    Solve2.NWS alpha.toInt (exactFourValue b) (solve4Concrete b alpha).toInt :=
  solveNext_nws b solve3Concrete exactThreeValue solve3Concrete_nws solve3Concrete_bounds alpha safe

theorem solve4Concrete_bounds (b : EmptyBoard 4) (alpha : Score) (safe : WindowSafe alpha) :
    -114 ≤ (solve4Concrete b alpha).toInt ∧ (solve4Concrete b alpha).toInt ≤ 114 :=
  solveNext_bounds b solve3Concrete solve3Concrete_bounds alpha safe

theorem solve3Concrete_classifies (b : EmptyBoard 3) (alpha : Score) (safe : WindowSafe alpha) :
    ((solve3Concrete b alpha).toInt ≤ alpha.toInt ↔ exactThreeValue b ≤ alpha.toInt) ∧
    (alpha.toInt < (solve3Concrete b alpha).toInt ↔ alpha.toInt < exactThreeValue b) :=
  Solve2.nws_classification (solve3Concrete_nws b alpha safe)

theorem solve4Concrete_classifies (b : EmptyBoard 4) (alpha : Score) (safe : WindowSafe alpha) :
    ((solve4Concrete b alpha).toInt ≤ alpha.toInt ↔ exactFourValue b ≤ alpha.toInt) ∧
    (alpha.toInt < (solve4Concrete b alpha).toInt ↔ alpha.toInt < exactFourValue b) :=
  Solve2.nws_classification (solve4Concrete_nws b alpha safe)

end NeuralReversi.Solve34
