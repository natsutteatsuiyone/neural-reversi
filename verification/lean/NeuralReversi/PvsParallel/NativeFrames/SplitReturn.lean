module
public import NeuralReversi.PvsParallel.NativeFrames.EntryStream

@[expose] public section

namespace NeuralReversi.PvsParallel.NativeFrames

open Returns (localEvents)

def splitReturnAt (step : Nat) : Event → Bool
  | .leave t _ .split none none => t == step
  | _ => false

def scoreAt (step : Nat) (score : Int) : Event → Bool
  | .scored t v => t == step && v.raw.toInt == score
  | _ => false

/-- Match the split's native return followed by the parent score on the same worker. -/
def resumed (finish scored : Nat) (score : Int) : List Event → Bool
  | [] => false
  | e :: es => if splitReturnAt finish e then es.any (scoreAt scored score) else resumed finish scored score es

theorem resume_order {p : EndgameSearch.CacheHistory.Pool} {cap : Capture p}
    (decoded : decode p args rs = some cap) (worker : Nat)
    (observed : resumed finish scored score (localEvents worker rs)) : finish < scored := by
  have order := EntryStream.worker_clock decoded worker
  generalize localEvents worker rs = es at order observed
  induction es with
  | nil => contradiction
  | cons e es ih =>
    obtain ⟨first, tail⟩ := List.pairwise_cons.mp order
    simp only [resumed] at observed
    split at observed
    · rename_i atReturn
      obtain ⟨s, mem, atScore⟩ := List.any_eq_true.mp observed
      have a : e.step = finish := by
        unfold splitReturnAt at atReturn
        split at atReturn <;> simp_all [Event.step]
      have b : s.step = scored := by
        cases s <;> simp_all [scoreAt, Event.step]
      exact a ▸ b ▸ first s mem
    · exact ih tail observed

end NeuralReversi.PvsParallel.NativeFrames
