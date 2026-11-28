module
public import NeuralReversi.AspirationMir.Returned
public import NeuralReversi.AspirationMir.Termination

@[expose] public section

namespace NeuralReversi.AspirationMir

open PvsParallel

theorem completeSingle (m : Machine) (entry : m.pc = 0)
    (initial : Aspiration.Window m.regs.alpha.toInt m.regs.beta.toInt)
    (board : PvsBoardTree.Position) (covers : Root.Covers mc board roots) (moves : 0 < roots.length)
    (query : Query) (payload : Nat → Word → Word → Bitboard.Square → Int)
    (attempts : ∀ i a b, Aspiration.Window a.toInt b.toInt →
      ConcurrentReplay.RootAttempt mc fc endian board roots 0 a.toInt b.toInt
        (query i a b).toInt (payload i a b))
    (pv : Bitboard.Square → List Bitboard.Square) (depth nodes : Nat) :
    ∃ score last, Returns query (fun _ => false) m score last ∧
      let rows := roots.map fun sq =>
        (⟨sq, payload last.index last.alpha last.beta sq, pv sq⟩ : SearchPublication.Row)
      ∃ result, SearchFallback.fromSnapshot (SearchPublication.capture rows ⟨depth, 5⟩ true) nodes = some result ∧
        result.best.score = score.toInt ∧ score.toInt = PvsBoardTree.minimax mc fc board := by
  have finite : FiniteQueries query := by
    intro i a b w
    have h := ((attempts i a b w).sound covers moves w).1
    exact ⟨h.lower, h.upper⟩
  obtain ⟨score, last, run, _, _, window⟩ := completes_in_twelve finite m entry initial
  exact ⟨score, last, run, singleResult run entry board covers moves
    (fun i => payload i last.alpha last.beta) ⟨window.lower, window.upper⟩
    (attempts _ _ _ window) pv depth nodes⟩

end NeuralReversi.AspirationMir
