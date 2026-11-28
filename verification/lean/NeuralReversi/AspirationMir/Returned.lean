module
public import NeuralReversi.AspirationMir.Driver
public import NeuralReversi.PvsParallel.ConcurrentReplay

@[expose] public section

namespace NeuralReversi.AspirationMir

open PvsParallel

/-- The control flow identifies which root invocation supplied the returned score
and the root rows. The invocation's search evidence supplies the game semantics. -/
theorem singleResult (run : Returns query (fun _ => false) m score last) (entry : m.pc = 0)
    (board : PvsBoardTree.Position) (covers : Root.Covers mc board roots) (moves : 0 < roots.length)
    (payload : Nat → Bitboard.Square → Int)
    (endpoints : -32767 ≤ last.alpha.toInt ∧ last.beta.toInt ≤ 32767)
    (attempt : ConcurrentReplay.RootAttempt mc fc endian board roots 0 last.alpha.toInt last.beta.toInt
      (query last.index last.alpha last.beta).toInt (payload last.index))
    (pv : Bitboard.Square → List Bitboard.Square) (depth nodes : Nat) :
    let rows := roots.map fun sq => (⟨sq, payload last.index sq, pv sq⟩ : SearchPublication.Row)
    ∃ result, SearchFallback.fromSnapshot (SearchPublication.capture rows ⟨depth, 5⟩ true) nodes = some result ∧
      result.best.score = score.toInt ∧ score.toInt = PvsBoardTree.minimax mc fc board := by
  obtain ⟨⟨observed, low, high⟩, rfl⟩ := returned_inside run entry
  rw [show query last.index last.alpha last.beta = last.score from observed] at attempt
  have window : Aspiration.Window last.alpha.toInt last.beta.toInt := ⟨endpoints.1, by omega, endpoints.2⟩
  have sound := attempt.sound covers moves window
  have exactScore := (sound.1.exact low high).trans (Root.value_zero covers moves)
  let accepted : Aspiration.Accepted (Bitboard.Square → Int) :=
    ⟨last.index, last.alpha.toInt, last.beta.toInt, ⟨last.score.toInt, payload last.index, false⟩⟩
  obtain ⟨report, saved, best⟩ :=
    ExactEndgameScore.singleResult_of_interior (accepted := accepted) covers moves ⟨low, high⟩ sound pv depth nodes
  exact ⟨report, saved, best.trans exactScore.symm, exactScore⟩

end NeuralReversi.AspirationMir
