module
public import NeuralReversi.AspirationMir.NativeRootBirth.Program
public import NeuralReversi.PvsParallel.GeneralNonPvMir
public import NeuralReversi.AspirationMir.NativeRowLines

@[expose] public section

namespace NeuralReversi.AspirationMir.NativeRootBirth

open PvsParallel PvsBoardTree
open Bitboard.Moves (X86Config)

def initial (mc : X86Config) (b : Position) : List RootMoveStorage.Row :=
  (legalSquares mc b.val).map (fun m => RootMoveStorage.fresh m.val)

theorem boot_eq (mc : X86Config) (fc : Cutoffs.FlipConfig) (b : Position) (paired : Bool) (mem) :
    boot mc fc b paired mem =
      if (legalSquares mc b.val).length ≤ capacity then some (0,initial mc b) else none := by
  let ls := legalSquares mc b.val
  have reads (i : Fin ls.length) : ((PvEntryMir.generated mc fc b paired mem).slots i.val).map (fun mv => fresh mv.sq) =
      (pure (RootMoveStorage.fresh ls[i].val) : Option _) := by
    rw [GeneralNonPvMir.physicalMoves mc fc b paired mem i.val i.isLt]
    rfl
  simp only [boot,PvEntryMir.generated_count]
  split <;> rename_i bound
  · simp only [bind,pure,Option.bind, bound,ite_true]
    change (do let rs ← (List.finRange ls.length).mapM (fun (i : Fin ls.length) =>
                 ((PvEntryMir.generated mc fc b paired mem).slots i.val).map (fun (mv : Move.Move) => fresh mv.sq))
               return (0,rs)) = _
    rw [funext reads,List.mapM_pure]
    simp only [bind,pure,Option.bind,Option.some.injEq,Prod.mk.injEq,true_and]
    change (List.finRange ls.length).map (fun i => RootMoveStorage.fresh ls[i].val) = _
    let f := fun (m : GeneratedSquare mc b.val) => RootMoveStorage.fresh m.val
    change (List.finRange ls.length).map (fun i => f ls[i]) = ls.map f
    rw [List.finRange,List.map_ofFn]
    exact (List.map_ofFn (f := fun i : Fin ls.length => ls[i]) (g := f)).symm.trans
      (congrArg (List.map f) List.ofFn_getElem)
  · simp [bound]

theorem certified (mc : X86Config) (fc : Cutoffs.FlipConfig) (b : Position) (paired : Bool) (mem)
    (checked : Std.Tactic.BVDecide.LRAT.check certificate (LegalMoves.MoveBound.dimacsFormula 35) = true) :
    boot mc fc b paired mem = some (0,initial mc b) := by
  have h := MoveList.Generation.certifiedCapacityContract_proven certificate checked mc b.val (Board.getMoves mc b.val)
    b.property (fun _ ok => ok)
  have count := (MoveList.Generation.generationContract_proven mc fc b.val (Board.getMoves mc b.val)
    false paired mem b.property (fun _ ok => ok) (by simp)).1
  change (PvEntryMir.generated mc fc b paired mem).len = _ at count
  rw [PvEntryMir.generated_count,Option.some.injEq,MoveList.Generation.records_length] at count
  rw [boot_eq,ite_eq_left (show (legalSquares mc b.val).length ≤ capacity from count ▸ h)]

/-- The direct search returns NoLegalMove when the new root-vector count is zero. -/
def started (mc : X86Config) (fc : Cutoffs.FlipConfig) (b : Position) (paired : Bool) (mem) := do
  let (_,rows) ← boot mc fc b paired mem
  if rows.isEmpty then none else some rows

theorem started_iff (mc : X86Config) (fc : Cutoffs.FlipConfig) (b : Position) (paired : Bool) (mem) :
    started mc fc b paired mem = some rows ↔
      (legalSquares mc b.val).length ≤ capacity ∧ (legalSquares mc b.val) ≠ [] ∧ initial mc b = rows := by
  cases legal : legalSquares mc b.val with
  | nil => simp [started,boot_eq,initial,legal]
  | cons m ms => by_cases bound : ms.length + 1 ≤ capacity <;> simp [started,boot_eq,initial,legal,bound]

theorem no_moves (noMoves : legalSquares mc b.val = []) :
    boot mc fc b paired mem = some (0,[]) ∧ started mc fc b paired mem = none := by
  simp [boot_eq,started,noMoves,initial]

theorem covers (active : started mc fc b paired mem = some rows) :
    Root.Covers mc b (rows.map RootMoveStorage.Row.square) ∧ rows ≠ [] := by
  simp only [started,boot_eq] at active
  split at active <;> simp_all
  obtain ⟨nonempty,rfl⟩ := active
  exact ⟨by simp [Root.Covers,initial,Function.comp_def,RootMoveStorage.fresh],nonempty⟩

/-- The direct endgame entry builds the root vector before any selected line is searched. -/
def run (c : NativeFrames.Capture pool) (log : FrameReturns.Log mc fc) (inputs : Nat → Nat → Option NativeFrames.Args)
    (samples : List NativeRows.Sample) (b : Position) (worker solve object : Nat)
    (supply : NativeRowLines.Supply c log b worker) (paired : Bool) (mem : MoveList.Initialization.Memory Move.Move)
    (stages : List NativeRowLines.Stage) :
    Option (List RootMoveStorage.Row) := do
  let rows ← started mc fc b paired mem
  (NativeRowLines.run c log inputs samples b worker solve object supply solve 0 rows stages).map NativeRowLines.Batch.final

theorem lines {log : FrameReturns.Log mc fc} {c : NativeFrames.Capture pool} {b : Position} {worker : Nat}
    {supply : NativeRowLines.Supply c log b worker}
    (ran : run c log inputs samples b worker solve object supply paired mem stages = some final) :
    ∃ rows, (NativeRowLines.run c log inputs samples b worker solve object supply solve 0 rows stages).map
        NativeRowLines.Batch.final = some final ∧
      Root.Covers mc b (rows.map RootMoveStorage.Row.square) ∧ rows ≠ [] := by
  obtain ⟨rows,launched,ok⟩ := Option.bind_eq_some_iff.mp ran
  exact ⟨rows,ok,covers launched⟩

end NeuralReversi.AspirationMir.NativeRootBirth

