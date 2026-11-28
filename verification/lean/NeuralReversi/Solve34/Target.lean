module
public import NeuralReversi.Solve2.Target
public import NeuralReversi.Solve34.TreeBridge

@[expose] public section

namespace NeuralReversi.Solve34

open Flip.Dispatch (TargetConfig Backend backendFor triple_eq quad_eq)

def squareAt (b : EmptyBoard n) (i : Fin n) : Square :=
  b.squares[i.val]'(b.length_eq.symm ▸ i.isLt)

private theorem squareOrder (b : EmptyBoard n) : List.ofFn (squareAt b) = b.squares := by
  apply List.ext_getElem
  · simp [b.length_eq]
  · intro i hi hj
    rw [List.getElem_ofFn hi]
    rfl

private theorem slotOrder (b : EmptyBoard n) : (boardSlots b).map Prod.fst = b.squares := by
  simp [boardSlots, List.map_map, Function.comp_def]

def masks3 (backend : Backend) (b : EmptyBoard 3) (p o : Word) : List Word :=
  let masks := backend.triple (squareAt b 0) (squareAt b 1) (squareAt b 2) p o
  [masks.1, masks.2.1, masks.2.2]

def masks4 (backend : Backend) (b : EmptyBoard 4) (p o : Word) : List Word :=
  let masks := backend.quad (squareAt b 0) (squareAt b 1) (squareAt b 2) (squareAt b 3) p o
  [masks.1, masks.2.1, masks.2.2.1, masks.2.2.2]

private theorem masks3_eq (backend : Backend) (b : EmptyBoard 3) (p o : Word) (wf : p &&& o = 0) :
    masks3 backend b p o = Dispatch.scalarBatch ((boardSlots b).map Prod.fst) p o := by
  rw [masks3, triple_eq _ _ _ _ _ _ wf, slotOrder, ← squareOrder b]
  rfl

private theorem masks4_eq (backend : Backend) (b : EmptyBoard 4) (p o : Word) (wf : p &&& o = 0) :
    masks4 backend b p o = Dispatch.scalarBatch ((boardSlots b).map Prod.fst) p o := by
  rw [masks4, quad_eq _ _ _ _ _ _ _ wf, slotOrder, ← squareOrder b]
  rfl

def runOn (backend : Backend) (charge : Bool) (oracle : Dispatch.Oracle)
    (b : EmptyBoard n) (alpha : Score) (masks : Word → Word → List Word) : Dispatch.Outcome :=
  if backend = .scalar then
    Dispatch.fallback charge oracle Solve2.Guard.scalarFlip Solve2.Guard.adjacent
      b.player b.opponent alpha (boardSlots b) (terminalScore b)
  else
    Dispatch.eager charge oracle (fun _ => masks)
      b.player b.opponent alpha (boardSlots b) (terminalScore b)

private theorem runOn_eq (backend : Backend) (charge : Bool) (oracle : Dispatch.Oracle)
    (b : EmptyBoard n) (alpha : Score) (masks : Word → Word → List Word)
    (own : masks b.player b.opponent = Dispatch.scalarBatch ((boardSlots b).map Prod.fst) b.player b.opponent)
    (other : masks b.opponent b.player = Dispatch.scalarBatch ((boardSlots b).map Prod.fst) b.opponent b.player) :
    runOn backend charge oracle b alpha masks =
      Dispatch.fallback charge oracle Solve2.Guard.scalarFlip Solve2.Guard.adjacent
        b.player b.opponent alpha (boardSlots b) (terminalScore b) := by
  unfold runOn
  split
  · rfl
  · rw [← Dispatch.scalar_eager_eq_fallback]
    simp only [Dispatch.eager, own, other]

def target2Oracle (target : TargetConfig) (bmi2 : Bool) : Dispatch.Oracle :=
  verifiedOracle (fun b a => (Solve2.targetRun target bmi2 b.toTwoEmpty a).score)
    (decodedNodeCost fun b a => (Solve2.targetRun target bmi2 b.toTwoEmpty a).nodeDelta)

private theorem target2Oracle_eq (target : TargetConfig) (bmi2 : Bool) :
    target2Oracle target bmi2 = verifiedOracle twoSolver (decodedNodeCost twoNodeCount) := by
  simp only [target2Oracle, Solve2.targetRun_eq, Solve2.sourceFallback_score]
  rfl

def target3Run (target : TargetConfig) (bmi2 : Bool) (b : EmptyBoard 3) (alpha : Score) : Dispatch.Outcome :=
  let backend := backendFor target
  runOn backend true (target2Oracle target bmi2) b alpha (masks3 backend b)

theorem target3Run_eq (target : TargetConfig) (bmi2 : Bool) (b : EmptyBoard 3) (alpha : Score) :
    target3Run target bmi2 b alpha = source3Run b alpha := by
  have swapped : b.opponent &&& b.player = 0 := by rw [BitVec.and_comm]; exact b.disjoint
  unfold target3Run
  rw [target2Oracle_eq]
  exact runOn_eq _ _ _ _ _ _ (masks3_eq _ b _ _ b.disjoint) (masks3_eq _ b _ _ swapped)

def target3Oracle (target : TargetConfig) (bmi2 : Bool) : Dispatch.Oracle :=
  verifiedOracle (fun b a => (target3Run target bmi2 b a).score)
    (decodedNodeCost fun b a => (target3Run target bmi2 b a).nodes)

private theorem target3Oracle_eq (target : TargetConfig) (bmi2 : Bool) :
    target3Oracle target bmi2 = verifiedOracle solve3Concrete (decodedNodeCost threeNodeCount) := by
  simp only [target3Oracle, target3Run_eq, source3Run_score]
  rfl

def target4Run (target : TargetConfig) (bmi2 : Bool) (b : EmptyBoard 4) (alpha : Score) : Dispatch.Outcome :=
  let backend := backendFor target
  runOn backend false (target3Oracle target bmi2) b alpha (masks4 backend b)

theorem target4Run_eq (target : TargetConfig) (bmi2 : Bool) (b : EmptyBoard 4) (alpha : Score) :
    target4Run target bmi2 b alpha = source4Run b alpha := by
  have swapped : b.opponent &&& b.player = 0 := by rw [BitVec.and_comm]; exact b.disjoint
  unfold target4Run
  rw [target3Oracle_eq]
  exact runOn_eq _ _ _ _ _ _ (masks4_eq _ b _ _ b.disjoint) (masks4_eq _ b _ _ swapped)

def TargetDispatchContract : Prop :=
  ∀ target bmi2 alpha,
    (∀ b : Solve2.TwoEmpty, Solve2.targetRun target bmi2 b alpha = Solve2.sourceFallback b alpha) ∧
    (∀ b : EmptyBoard 3, target3Run target bmi2 b alpha = source3Run b alpha) ∧
    (∀ b : EmptyBoard 4, target4Run target bmi2 b alpha = source4Run b alpha)

theorem targetDispatchContract_proven : TargetDispatchContract := by
  intro target bmi2 alpha
  exact ⟨fun b => Solve2.targetRun_eq target bmi2 b alpha, fun b => target3Run_eq target bmi2 b alpha,
    fun b => target4Run_eq target bmi2 b alpha⟩

open EndgameTreeCache (discValue)
open EndgameCache (SearchRange)

def TargetTreeContract : Prop :=
  ∀ (target : TargetConfig) (bmi2 : Bool) (mc : Bitboard.Moves.X86Config)
    (fc : Flip.Dispatch.X86Config) (alpha : Score), WindowSafe alpha →
    (∀ b : EmptyBoard 2,
      let result := (Solve2.targetRun target bmi2 b.toTwoEmpty alpha).score.toInt
      SearchRange result ∧ Solve2.NWS alpha.toInt (discValue mc fc b.asBoard) result) ∧
    (∀ b : EmptyBoard 3,
      let result := (target3Run target bmi2 b alpha).score.toInt
      SearchRange result ∧ Solve2.NWS alpha.toInt (discValue mc fc b.asBoard) result) ∧
    (∀ b : EmptyBoard 4,
      let result := (target4Run target bmi2 b alpha).score.toInt
      SearchRange result ∧ Solve2.NWS alpha.toInt (discValue mc fc b.asBoard) result)

theorem targetTreeContract_proven : TargetTreeContract := by
  intro target bmi2 mc fc alpha safe
  refine ⟨?_, ?_, ?_⟩
  · intro b
    dsimp only
    rw [Solve2.targetRun_eq]
    have ⟨lo, hi⟩ := solve2_bounds b.toTwoEmpty alpha safe.2
    rw [← Solve2.sourceFallback_score] at lo hi
    have bounds := Solve2.sourceFallback_nws b.toTwoEmpty alpha safe.2
    exact ⟨⟨by omega, by omega⟩, twoValue_eq_tree mc fc b ▸ bounds⟩
  · intro b
    dsimp only
    rw [target3Run_eq]
    have ⟨lo, hi⟩ := solve3Concrete_bounds b alpha safe
    rw [← source3Run_score] at lo hi
    have bounds := source3Run_nws b alpha safe
    have valueEq : exactThreeValue b = discValue mc fc b.asBoard :=
      nextValue_eq_tree mc fc b twoValue fun child => twoValue_eq_tree mc fc child
    exact ⟨⟨by omega, by omega⟩, valueEq ▸ bounds⟩
  · intro b
    dsimp only
    rw [target4Run_eq]
    exact treeBoundContract_proven mc fc b alpha safe

end NeuralReversi.Solve34
