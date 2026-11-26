import NeuralReversi.PvsParallel.SerialOwnerCalls
import NeuralReversi.PvsParallel.RootReplay
import NeuralReversi.TranspositionTable.Proofs

namespace NeuralReversi.AspirationMir.SerialTable.PvInterior

open PvsParallel
open PvsMoveLoop (MoveCalls ScanCalls ChildSearch moveScore scan scoreMax)

def InsideRange (a z word : Int) : Prop := a < word → word < z → -scoreMax ≤ word ∧ word ≤ scoreMax

def Child (_ : Move) (pv : Bool) (a z word : Int) := pv = true → InsideRange a z word

private theorem negate {a z w : Int} (child : InsideRange (-z) (-a) w) : InsideRange a z (-w) := by
  intro lo hi
  obtain ⟨ll,hh⟩ := child (by omega) (by omega)
  omega

/-- An improvement inside a PV window necessarily uses a PV reply at the reflected window. -/
theorem move_inside (heard : MoveCalls Child search true first m a z) :
    InsideRange a z (moveScore true first search m a z) := by
  cases first with
  | true => exact negate (heard rfl)
  | false =>
    change Child m false _ _ _ ∧ ((true = true ∧ a < -search m false (-(a+1)) (-a)) → Child m true _ _ _) at heard
    by_cases high : a < -search m false (-(a+1)) (-a)
    · simpa [moveScore,high] using negate (heard.2 ⟨rfl,high⟩ rfl)
    · simp [moveScore,high,InsideRange]

private theorem scan_inside {original z : Int} (search : ChildSearch Move) (moves : List Move) (first : Bool)
    (alpha best : Int) (prior : alpha = max original best) (bounded : InsideRange original z best)
    (heard : ScanCalls Child search true z moves first alpha best) :
    InsideRange original z (scan true search z moves first alpha best) := by
  induction moves generalizing first alpha best with
  | nil => exact bounded
  | cons m rest ih =>
    let value := moveScore true first search m alpha z
    have current : InsideRange alpha z value := move_inside heard.1
    have tail := heard.2
    change (match PvsMoveLoop.nextScanState true z alpha best value with
      | none => True | some (a,s) => ScanCalls Child search true z rest false a s) at tail
    rw [PvsMoveLoop.scan_step]
    change InsideRange original z (match PvsMoveLoop.nextScanState true z alpha best value with
      | none => value | some (a,s) => scan true search z rest false a s)
    rw [PvsMoveLoop.nextScanState] at tail ⊢
    by_cases better : best < value
    · rw [ite_eq_left better] at tail ⊢
      by_cases raised : alpha < value
      · rw [ite_eq_left raised] at tail ⊢
        have keep : InsideRange original z value := fun _ below => current raised below
        by_cases again : true = true ∧ value < z ∧ value < scoreMax
        · rw [ite_eq_left again] at tail ⊢
          exact ih false value value (by omega) keep tail
        · rw [ite_eq_right again]
          exact keep
      · rw [ite_eq_right raised] at tail ⊢
        have smaller : value ≤ original := by omega
        exact ih false alpha value (by omega) (fun above => (Int.not_lt.mpr smaller above).elim) tail
    · rw [ite_eq_right better] at tail ⊢
      exact ih false alpha best prior bounded tail

/-- The chronological root/PV owner accepts only bounded interior PV replies. No numerical
    restriction is placed on scout replies or on PV bounds outside the requested window. -/
theorem owner {square : Move → Bitboard.Square} (unique : moves.Nodup)
    (issued : SerialCalls.Program.Runs (SerialOwnerCalls.run square z moves (RootReplay.start a)) words out)
    (window : Aspiration.Window a z) (child : ∀ t ∈ words, Child t.1.move t.1.pv t.1.alpha t.1.beta t.2) :
    InsideRange a z out.2.state.best := by
  classical
  let key := fun q : SerialCalls.MoveRequest Move => (q.move,q.pv)
  let search : ChildSearch Move := fun m pv _ _ => SerialCalls.Program.returnAt key words (m,pv)
  obtain ⟨same,calls⟩ := SerialOwnerCalls.runs_serial issued search Child (fun t ht => by
    exact ⟨SerialCalls.Program.returnAt_mem key words (SerialOwnerCalls.runs_distinct issued unique) t ht,child t ht⟩)
  rw [same,RootReplay.serial_best]
  change InsideRange a z (scan true search z moves true a (-32767))
  have lo := window.lower
  apply scan_inside search moves true a (-32767) (by omega) _ calls
  exact fun above => (Int.not_lt.mpr lo above).elim

end NeuralReversi.AspirationMir.SerialTable.PvInterior
