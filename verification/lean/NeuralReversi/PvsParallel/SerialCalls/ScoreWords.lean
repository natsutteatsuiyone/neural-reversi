module
public import NeuralReversi.PvsParallel.SerialCalls.Sites
public import NeuralReversi.PvsParallel.Replay

@[expose] public section

namespace NeuralReversi.PvsParallel.SerialCalls.ScoreWords

open ScoreRefresh (Score addInt negate)
open Aspiration (Window)
open PvsBoardTree (Position)

def encode (n : Int) : Score := ⟨BitVec.ofInt 32 n⟩

private theorem represent {n : Int} (lo : -32767 ≤ n) (hi : n ≤ 32767) :
    (encode n).raw.toInt = n :=
  BitVec.toInt_ofInt_eq_self (by decide) (by omega) (by omega)

private theorem exact_word {action : Except String Score} {v : Int}
    (result : ScoreRefresh.observe action = .ok v) : action = .ok (encode v) := by
  cases action with
  | error _ => cases result
  | ok s =>
    have value : s.raw.toInt = v := Except.ok.inj result
    cases s
    simp [← value, encode]

private theorem neg_word (checked : Bool) (s : Score) (lo : -32767 ≤ s.raw.toInt) (hi : s.raw.toInt ≤ 32767) :
    negate checked s = .ok (encode (-s.raw.toInt)) := by
  apply exact_word
  rw [ScoreRefresh.negate_spec]
  exact Aspiration.safe_math checked _ _ (by unfold ScaledScore.outsideI32; omega)

private theorem successor_word (checked : Bool) (s : Score) (w : Window s.raw.toInt beta) :
    addInt checked s 1 = .ok (encode (s.raw.toInt + 1)) := by
  apply exact_word
  rw [addInt, ScoreRefresh.addWords_spec]
  have one : (1 : BitVec 32).toInt = 1 := rfl
  rw [one]
  obtain ⟨lo, mid, hi⟩ := w
  exact Aspiration.safe_math checked _ _ (by unfold ScaledScore.outsideI32; omega)

/-- The recursive ABI uses signed i32 addition and negation for both reflected
    windows. Both checked and wrapping builds agree on every admissible parent window. -/
theorem request_words (checked : Bool) (next : Position) (depth : ExactDepth.Depth) (cut : Bool)
    (a z : Score) (w : Window a.raw.toInt z.raw.toInt) :
    (do
      let succ ← addInt checked a 1
      let lo ← negate checked succ
      let hi ← negate checked a
      pure (⟨next,depth-1,false,!cut,lo.raw.toInt,hi.raw.toInt⟩ : Request)) =
      .ok (Sites.scoutRequest next depth cut a.raw.toInt) ∧
    (do
      let lo ← negate checked z
      let hi ← negate checked a
      pure (⟨next,depth-1,true,false,lo.raw.toInt,hi.raw.toInt⟩ : Request)) =
      .ok (Sites.wideRequest next depth a.raw.toInt z.raw.toInt) ∧
    ∀ pv, (do
      let lo ← negate checked z
      let hi ← negate checked a
      pure (Program.call (⟨next,depth,pv,!pv && !cut,lo.raw.toInt,hi.raw.toInt⟩ : Request)
        fun word => .pure (-word))) =
      .ok (Sites.pass next depth pv cut a.raw.toInt z.raw.toInt) := by
  have ⟨lo, mid, hi⟩ := w
  have sa := successor_word checked a w
  have succ := represent (n := a.raw.toInt+1) (by omega) (by omega)
  have ns := neg_word checked (encode (a.raw.toInt+1)) (by omega) (by omega)
  have na := neg_word checked a lo (by omega)
  have nz := neg_word checked z (by omega) hi
  simp only [sa, ns, succ, na, nz, bind, Except.bind, pure, Except.pure]
  rw [represent (n := -(a.raw.toInt+1)) (by omega) (by omega),
    represent (n := -a.raw.toInt) (by omega) (by omega),
    represent (n := -z.raw.toInt) (by omega) (by omega)]
  exact ⟨rfl,rfl,fun _ => rfl⟩

def Finite (word : Int) : Prop := -32767 < word ∧ word < 32767

theorem finite_of_replay (w : Window a z) (null : pv = false → z = a+1)
    (replay : boundedReplay fuel tree pv a z word) : Finite word :=
  let sound := replayContract_proven tree pv a z word fuel w null replay
  ⟨sound.lower,sound.upper⟩

def onWord (checked : Bool) (req : Req) (next : Int → Score → Program Req A) : Program Req A :=
  .call req fun raw =>
    match negate checked (encode raw) with
    | .ok score => next raw score
    | .error _ => .fail

/-- The same call sites with their return registers represented as Rust Scores.
    The comparison feeding the optional wide re-search is a signed word comparison. -/
def owner (checked pv first : Bool) (alpha : Score) (scout wide : Req)
    (finish : Option Int → Option Int → Int → Program Req A) : Program Req A :=
  if pv && first then onWord checked wide fun w v => finish none (some w) v.raw.toInt
  else onWord checked scout fun s v =>
    if pv && alpha.raw.slt v.raw then onWord checked wide fun w wideValue =>
      finish (some s) (some w) wideValue.raw.toInt
    else finish (some s) none v.raw.toInt

private theorem call_congr (q : Req) (f g : Int → Program Req A)
    (good : ∀ e ∈ events, Finite e.2)
    (next : ∀ w rest, Finite w → (∀ e ∈ rest, Finite e.2) → ((f w).Runs rest out ↔ (g w).Runs rest out)) :
    (Program.call q f).Runs events out ↔ (Program.call q g).Runs events out := by
  constructor
  · intro h
    cases h with
    | call more =>
      exact .call ((next _ _ (good _ List.mem_cons_self) (fun e mem => good e (.tail _ mem))).mp more)
  · intro h
    cases h with
    | call more =>
      exact .call ((next _ _ (good _ List.mem_cons_self) (fun e mem => good e (.tail _ mem))).mpr more)

private theorem onWord_congr (checked : Bool) (q : Req) (f : Int → Score → Program Req A)
    (g : Int → Program Req A) (good : ∀ e ∈ events, Finite e.2)
    (next : ∀ w rest, Finite w → (∀ e ∈ rest, Finite e.2) →
      ((f w (encode (-w))).Runs rest out ↔ (g w).Runs rest out)) :
    (onWord checked q f).Runs events out ↔ (Program.call q g).Runs events out := by
  apply call_congr _ _ _ good
  intro w rest within valid
  obtain ⟨lo,hi⟩ := within
  have raw := represent (n := w) (by omega) (by omega)
  have neg := neg_word checked (encode w) (by omega) (by omega)
  rw [raw] at neg
  simpa only [neg] using next w rest ⟨lo,hi⟩ valid

/-- A pass or presearch returns the exact negation of its one child word. -/
theorem return_words (checked : Bool) (request : Req) (finish : Int → Program Req A)
    (good : ∀ e ∈ events, Finite e.2) :
    (onWord checked request fun _ score => finish score.raw.toInt).Runs events out ↔
      (Sites.presearchWith request finish).Runs events out := by
  apply onWord_congr _ _ _ _ good
  rintro w _ ⟨lo,hi⟩ _
  rw [represent (n := -w) (by omega) (by omega)]

/-- Equality of completions includes the issued call sequence and each returned
    word. The only operands requiring a finite range are returns on that sequence. -/
theorem owner_words (checked pv first : Bool) (alpha : Score) (scout wide : Req)
    (finish : Option Int → Option Int → Int → Program Req A)
    (good : ∀ e ∈ events, Finite e.2) :
    (owner checked pv first alpha scout wide finish).Runs events out ↔
      (Sites.ownerWith pv first alpha.raw.toInt scout wide finish).Runs events out := by
  have endCall (f : Int → Int → Program Req A) (tail : List (Req × Int))
      (valid : ∀ e ∈ tail, Finite e.2) :
      (onWord checked wide fun w v => f w v.raw.toInt).Runs tail out ↔
      (Program.call wide fun w => f w (-w)).Runs tail out := by
    apply onWord_congr _ _ _ _ valid
    rintro w _ ⟨lo,hi⟩ _
    rw [represent (n := -w) (by omega) (by omega)]
  unfold owner Sites.ownerWith
  split
  · exact endCall (fun w v => finish none (some w) v) events good
  · apply onWord_congr _ _ _ _ good
    rintro s rest ⟨lo,hi⟩ valid
    simp only [BitVec.slt_eq_decide, represent (n := -s) (by omega) (by omega)]
    split
    · exact endCall (fun w v => finish (some s) (some w) v) rest valid
    · rfl

end NeuralReversi.PvsParallel.SerialCalls.ScoreWords
