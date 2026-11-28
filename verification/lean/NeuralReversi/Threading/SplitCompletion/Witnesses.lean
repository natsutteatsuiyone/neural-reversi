module
public import NeuralReversi.Threading.SplitCompletion

@[expose] public section

namespace NeuralReversi.Threading.SplitCompletion.Witnesses

def completes : List Action :=
  [.book 63 ⟨0, 1⟩, .finish 63 ⟨63, 0⟩, .finish 0 ⟨0, 2⟩]

def lateBooking : List Action :=
  [.finish 0 ⟨0, 1⟩, .book 63 ⟨63, 0⟩]

def MaskWitnessContract : Prop :=
  Legal (initial 0) completes ∧
    (run (initial 0) (completes.take 1)).bits = 0x8000000000000001#64 ∧
    (run (initial 0) (completes.take 2)).bits = 1 ∧
    (run (initial 0) completes).bits = 0 ∧
    (run (initial 0) (lateBooking.take 1)).bits = 0 ∧
    (run (initial 0) lateBooking).bits = 0x8000000000000000#64 ∧
    ¬ Legal (initial 0) lateBooking

theorem maskWitnessContract_proven : MaskWitnessContract := by
  unfold MaskWitnessContract
  simp only [completes, lateBooking, Legal, Allowed]
  decide

end NeuralReversi.Threading.SplitCompletion.Witnesses
