import NeuralReversi.PvsParallel.SerialCalls.Ready

namespace NeuralReversi.PvsParallel.SerialTable.MidgamePoll

open SerialCalls (Program)

/-- Child calls complete before the stop check. A stopped owner returns the memory left by
    that child and skips its epilogue. A scout which triggers a wide retry has no check yet. -/
def run (reply : Req → M → Int × M) (poll : Req → Int → Bool) (m : M) :
    Program Req α → Option (Option α × M)
  | .fail => none
  | .pure x => some (some x,m)
  | .call q resume =>
    let (v,next) := reply q m
    if poll q v then some (none,next) else run reply poll next (resume v)

theorem returns {word : Req → Int → Prop} {safe : M → Prop} {code : Program Req α}
    (live : Program.Ready allowed code) (reply : Req → M → Int × M) (poll : Req → Int → Bool)
    (child : ∀ q, allowed q → ∀ m, safe m → safe (reply q m).2 ∧ word q (reply q m).1)
    (m : M) (good : safe m) :
    ∃ out n, run reply poll m code = some (out,n) ∧ safe n ∧
      ∀ ans, out = some ans → ∃ calls, Program.Runs code calls ans ∧ ∀ t ∈ calls, word t.1 t.2 := by
  induction live generalizing m with
  | pure => exact ⟨_,m,rfl,good,fun _ h => by cases h; exact ⟨[],.pure,by simp⟩⟩
  | @call q resume ok _ ih =>
    obtain ⟨post,w⟩ := child q ok m good
    cases stop : poll q (reply q m).1 with
    | true => exact ⟨none,(reply q m).2,by simp [run,stop],post,by simp⟩
    | false =>
      obtain ⟨out,n,ran,g,issued⟩ := ih (reply q m).1 _ post
      refine ⟨out,n,by simpa [run,stop] using ran,g,?_⟩
      intro ans h
      obtain ⟨cs,trace,words⟩ := issued ans h
      refine ⟨(q,_)::cs,.call trace,?_⟩
      intro t mem
      rcases List.mem_cons.mp mem with rfl | later
      · exact w
      · exact words t later

def afterMove (pv : Bool) (beta reply : Int) : Bool := pv || decide (beta ≤ reply)

theorem afterMove_wide (beta v : Int) : afterMove true beta v = true := rfl
theorem afterMove_scout (alpha v : Int) : afterMove false (-alpha) v = true ↔ ¬alpha < -v := by
  simp only [afterMove,Bool.false_or,decide_eq_true_eq]
  omega

end NeuralReversi.PvsParallel.SerialTable.MidgamePoll
