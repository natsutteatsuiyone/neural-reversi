module
public import NeuralReversi.PvsParallel.ContextPathMir
public import NeuralReversi.PvsParallel.GeneralNonPvMir

@[expose] public section

namespace NeuralReversi.PvsParallel.ContextPathMir

open PvsBoardTree (Position)
open SearchContextState (State)
open TranspositionTable (canCut)
open Cutoffs (exactSelectivity)

def Callee.isPv (parent : Bool) : Callee → Bool
  | .move pv => pv
  | .pass => parent

/-- Follow one chain of recursive call sites. Returning from siblings and copying a
    split task keep the current frame. The next worker call consumes a move. -/
inductive FramePath (mc : Bitboard.Moves.X86Config) (fc : Flip.Dispatch.X86Config) (root : Position) :
    Position → ExactDepth.Depth → State → Bool → Prop
  | initial : FramePath mc fc root root (rootDepth (start root)) (start root) true
  | call (parent : FramePath mc fc root b d s pv)
      (site : recursiveSite worker pc = some target)
      (edge : enter mc fc worker pc b d s arg = some (child, cd, t)) :
      FramePath mc fc root child cd t (target.isPv pv)
  | after (parent : FramePath mc fc root b d s pv) (calls : Calls mc fc b s t) :
      FramePath mc fc root b d t pv
  | worker (parent : FramePath mc fc root b d s pv) : FramePath mc fc root b d (splitCopy s) pv

theorem FramePath.context (p : FramePath mc fc root b d s pv) : At mc fc root b d s := by
  induction p with
  | initial => exact .initial
  | call _ _ edge ih => exact ih.entered edge
  | after _ calls ih => exact ih.after calls
  | worker _ ih => exact ih.worker

theorem FramePath.depth_bound (p : FramePath mc fc root b d s pv) :
    d.toNat ≤ (Board.emptyCount root.val).toNat ∧
    (pv = false → d.toNat < (Board.emptyCount root.val).toNat) := by
  induction p with
  | initial => rw [start_eq]; exact ⟨Nat.le_refl _, by simp⟩
  | after _ _ ih | worker _ ih => exact ih
  | @call board depth state parentPv worker pc target arg child cd t parent site edge ih =>
    cases target <;> cases arg <;> simp [enter,childSite,site] at edge
    · rename_i move
      obtain ⟨_,rfl,_⟩ := edge
      have pos := (ExactDepth.move_depth mc fc board depth move parent.context.full).1
      have decrease : (depth-1).toNat < depth.toNat := by
        rw [BitVec.toNat_sub_of_le (by change (1#32).toNat ≤ depth.toNat; exact pos)]
        change depth.toNat - 1 < depth.toNat
        omega
      exact ⟨Nat.le_trans (Nat.le_of_lt decrease) ih.1,fun _ => Nat.lt_of_lt_of_le decrease ih.1⟩
    · obtain ⟨_,rfl,_⟩ := edge
      exact ih

private theorem probe_depth {n : Nat} {a b c d e f : Bool}
    (selected : GeneralNonPvMir.dispatch n a b c d e f = .probe) : 13 ≤ n := by
  apply Classical.byContradiction
  intro h
  simp [GeneralNonPvMir.dispatch,show n ≤ 12 by omega] at selected

private theorem enhanced_depth {n : Nat} {a b c d e f : Bool}
    (selected : GeneralNonPvMir.dispatch n a b c d e f = .enhanced) : 14 ≤ n := by
  grind [GeneralNonPvMir.dispatch]

private theorem boundary_args
    (selected : DeepLeafMir.dispatch n (DeepLeafMir.parallelPool size) present active = some .probe) :
    n = 12 ∧ 1 < size := by
  unfold DeepLeafMir.dispatch at selected
  split at selected
  · split at selected
    next yes => exact ⟨yes.1,of_decide_eq_true yes.2.1⟩
    next => cases active <;> contradiction
  · contradiction

/-- A mismatched key cannot make shallow metadata deep enough for any of the
    three shared-table score cuts. This also holds for arbitrary depth registers. -/
theorem shallow_metadata (entry : TranspositionTable.EntryData) (shallow : entry.depth.toNat < 12)
    (d : ExactDepth.Depth) (checked : Bool) (alpha beta : TranspositionTable.Score) (sq : Bitboard.Square)
    (size : Nat) (active stable mobile other wipes parentCut childCut : Bool) :
    let action := GeneralNonPvMir.dispatch d.toNat stable mobile other wipes parentCut childCut
    ((DeepLeafMir.dispatch d.toNat (DeepLeafMir.parallelPool size) true active = some .probe ∨ action = .probe) →
      canCut checked entry beta d exactSelectivity true = .ok false) ∧
    (action = .enhanced →
      PvsEnhancedCutoff.candidate ⟨d,exactSelectivity,true,alpha⟩ (sq,some entry) = none) := by
  dsimp only
  constructor
  · intro used
    apply TranspositionTable.canCut_ineligible
    intro valid
    have deep := valid.2.1
    rcases used with boundary | parent
    · have h := (boundary_args boundary).1; omega
    · have h := probe_depth parent; omega
  · intro used
    have dep := enhanced_depth used
    have short : ¬ (d-1).toNat ≤ entry.depth.toNat := by
      rw [BitVec.toNat_sub_of_le (by change 1 ≤ d.toNat; omega)]
      change ¬ d.toNat-1 ≤ entry.depth.toNat
      omega
    simp [PvsEnhancedCutoff.candidate]
    intro _ depthFits
    exact (short (by simpa using depthFits)).elim

/-- At these roots, no NonPV descendant can use an arbitrary shared-table hit
    either to obtain a score or to choose a square to play directly. -/
theorem FramePath.small_root (p : FramePath mc fc root b d s false)
    (small : (Board.emptyCount root.val).toNat ≤ 12 ∨
      size = 1 ∧ (Board.emptyCount root.val).toNat ≤ 13) :
    GeneralNonPvMir.dispatch d.toNat stable mobile other wipes parentCut childCut = .shallow ∧
    DeepLeafMir.dispatch d.toNat (DeepLeafMir.parallelPool size) present active ≠ some .probe := by
  have below := p.depth_bound.2 rfl
  have shallow : d.toNat ≤ 12 := by rcases small with _ | ⟨_,_⟩ <;> omega
  refine ⟨by simp [GeneralNonPvMir.dispatch,shallow],?_⟩
  intro selected
  obtain ⟨atBoundary,parallel⟩ := boundary_args selected
  rcases small with _ | ⟨_,_⟩ <;> omega

/-- The accepted metadata must be deep enough even when its keys came from a
    different write. The bounds need no assumption about publication. -/
theorem FramePath.boundary_cut (p : FramePath mc fc root b d s false)
    (selected : DeepLeafMir.dispatch d.toNat (DeepLeafMir.parallelPool size) present active = some .probe)
    (accepted : canCut checked entry beta d exactSelectivity true = .ok true) :
    1 < size ∧ d.toNat = 12 ∧ 13 ≤ (Board.emptyCount root.val).toNat ∧ 12 ≤ entry.depth.toNat := by
  have g := boundary_args selected
  have h := p.depth_bound.2 rfl
  have deep := ((TranspositionTable.canCut_true_iff ..).mp accepted).1.2.1
  exact ⟨g.2,g.1,by omega,by omega⟩

/-- The NonPV presearch that plays a decoded table square directly is reached only
    from this depth onward. The branch has no minimum depth for the hint itself. -/
theorem FramePath.presearch_depth (p : FramePath mc fc root b d s false)
    (selected : GeneralNonPvMir.dispatch d.toNat stable mobile other wipes parentCut childCut = .rest) :
    13 ≤ d.toNat ∧ 14 ≤ (Board.emptyCount root.val).toNat := by
  have deep : 13 ≤ d.toNat := by
    apply Classical.byContradiction
    intro h
    simp [GeneralNonPvMir.dispatch,show d.toNat ≤ 12 by omega] at selected
  have above := p.depth_bound.2 rfl
  exact ⟨deep,by omega⟩

theorem FramePath.parent_cut (p : FramePath mc fc root b d s false)
    (selected : GeneralNonPvMir.dispatch d.toNat stable mobile other wipes parentCut childCut = .probe)
    (accepted : canCut checked entry beta d exactSelectivity true = .ok true) :
    14 ≤ (Board.emptyCount root.val).toNat ∧ 13 ≤ entry.depth.toNat := by
  have dep := probe_depth selected
  have above := p.depth_bound.2 rfl
  have deep := ((TranspositionTable.canCut_true_iff ..).mp accepted).1.2.1
  exact ⟨by omega,by omega⟩

theorem FramePath.child_cut (p : FramePath mc fc root b d s false)
    (selected : GeneralNonPvMir.dispatch d.toNat stable mobile other wipes parentCut childCut = .enhanced)
    (hit : PvsEnhancedCutoff.scan ⟨d,exactSelectivity,true,alpha⟩ reads = some chosen) :
    15 ≤ (Board.emptyCount root.val).toNat ∧
    ∃ before sq entry after, reads = before ++ (sq,some entry) :: after ∧
      13 ≤ entry.depth.toNat ∧ chosen = (sq,-entry.score.toInt) := by
  have dep := enhanced_depth selected
  have above := p.depth_bound.2 rfl
  obtain ⟨before,⟨sq,item⟩,after,order,⟨entry,rfl,eligible,_,_,rfl⟩,_⟩ :=
    ((PvsEnhancedCutoff.contract_proven _ reads).2.1 _).mp hit
  have deep : (d-1).toNat ≤ entry.depth.toNat := eligible.2.1
  rw [BitVec.toNat_sub_of_le (by change 1 ≤ d.toNat; omega)] at deep
  exact ⟨by omega,before,sq,entry,after,order,by change d.toNat-1 ≤ entry.depth.toNat at deep; omega,rfl⟩

end NeuralReversi.PvsParallel.ContextPathMir
