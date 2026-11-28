module
public import NeuralReversi.AspirationMir.Retry
public import NeuralReversi.PvsParallel.RootWipeout

@[expose] public section

namespace NeuralReversi.AspirationMir

open PvsParallel

/-- When move generation finds a wipeout, every aspiration retry before the next
PV line considers the same set of root squares. Its chosen square is the last
wipeout in generation order. -/
def WipeoutResultContract : Prop :=
  ∀ (mc : Bitboard.Moves.X86Config) (fc : Flip.Dispatch.X86Config) (board : PvsBoardTree.Position)
    (roots : List Bitboard.Square) (square : Bitboard.Square) (paired : Bool)
    (mem : MoveList.Initialization.Memory Move.Move) (m : Machine)
    (pv : Bitboard.Square → List Bitboard.Square) (depth nodes : Nat),
    Root.Covers mc board roots → 0 < roots.length →
    (MoveList.Generation.fill false paired fc board.val
      (Root.remainingMask mc board.val roots 0) mem).wipeout = some (some square) →
    m.pc = 0 → Aspiration.Window m.regs.alpha.toInt m.regs.beta.toInt →
    let payload := (rootWipeoutExit square).1.foldl (fun rows (sq, s, visible) =>
      RootOrder.writeScore rows sq (if visible then s else -PvsMoveLoop.scoreInf)) (fun _ => -PvsMoveLoop.scoreInf)
    let rows := roots.map fun sq => (⟨sq, payload sq, pv sq⟩ : SearchPublication.Row)
    ∃ score last, Returns (fun _ _ _ => BitVec.ofInt 32 (rootWipeoutExit square).2)
      (fun _ => false) m score last ∧
      ∃ result, SearchFallback.fromSnapshot (SearchPublication.capture rows ⟨depth, 5⟩ true) nodes = some result ∧
        result.best.score = score.toInt ∧ score.toInt = PvsBoardTree.minimax mc fc board

theorem wipeoutResultContract_proven : WipeoutResultContract := by
  intro mc fc board roots square paired mem m pv depth nodes covers nonempty hit entry initial
  apply completeSingle (endian := false) m entry initial board covers nonempty
    (fun _ _ _ => BitVec.ofInt 32 (rootWipeoutExit square).2) (fun _ _ _ => _) _ pv depth nodes
  intro _ a b _
  exact ConcurrentReplay.rootAttempt_of_generated_wipeout mc fc false board roots 0
    a.toInt b.toInt paired mem square covers hit

end NeuralReversi.AspirationMir
