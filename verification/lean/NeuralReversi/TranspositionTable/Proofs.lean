module
public import NeuralReversi.ScaledScore.Model

@[expose] public section

namespace NeuralReversi.TranspositionTable

abbrev Score := ScaledScore.ScaledScore
abbrev Depth := BitVec 32
abbrev Selectivity := Fin 6

inductive Bound where
  | none | lower | upper | exact
  deriving DecidableEq, Repr

def classify (pv : Bool) (best alpha beta : Score) : Bound :=
  if best.raw.toInt >= beta.raw.toInt then .lower
  else if pv && decide (best.raw.toInt > alpha.raw.toInt) then .exact
  else .upper

structure EntryData where
  score : BitVec 16
  bestMove : BitVec 8
  bound : BitVec 8
  depth : BitVec 8
  selectivity : BitVec 8
  generation : BitVec 8
  flags : BitVec 8
  deriving DecidableEq, Repr

def selectivityRank (raw : BitVec 8) : Nat :=
  match raw.toNat with
  | 0 => 0 | 1 => 1 | 2 => 2 | 3 => 3 | 4 => 4 | _ => 5

def isEndgame (entry : EntryData) : Bool :=
  decide (entry.flags &&& 1#8 ≠ 0#8)

def scoreError : String :=
  "assertion failed: raw_value >= -Self::INF.0 && raw_value <= Self::INF.0"

def entryScore (checked : Bool) (entry : EntryData) : Except String Score :=
  ScaledScore.fromRaw checked (entry.score.signExtend 32)

def canCut (checked : Bool) (entry : EntryData) (beta : Score) (depth : Depth)
    (selectivity : Selectivity) (requireEndgame : Bool) : Except String Bool :=
  if requireEndgame && !isEndgame entry then .ok false
  else if decide (entry.depth.toNat < depth.toNat) ||
      decide (selectivityRank entry.selectivity < selectivity.val) then .ok false
  else match entryScore checked entry with
    | .error reason => .error reason
    | .ok score =>
      let required : BitVec 8 := if score.raw.toInt >= beta.raw.toInt then 1 else 2
      .ok (decide (entry.bound &&& required ≠ 0#8))

abbrev Eligible (entry : EntryData) (depth : Depth) (selectivity : Selectivity)
    (requireEndgame : Bool) : Prop :=
  (requireEndgame = false ∨ entry.flags.getLsbD 0 = true) ∧
  depth.toNat ≤ entry.depth.toNat ∧ selectivity.val ≤ min entry.selectivity.toNat 5

abbrev CutWitness (entry : EntryData) (beta : Score) : Prop :=
  (beta.raw.toInt ≤ entry.score.toInt ∧ entry.bound.getLsbD 0 = true) ∨
  (entry.score.toInt < beta.raw.toInt ∧ entry.bound.getLsbD 1 = true)

abbrev BoundSound (entry : EntryData) (truth : Int) : Prop :=
  (entry.bound.getLsbD 0 = true → entry.score.toInt ≤ truth) ∧
  (entry.bound.getLsbD 1 = true → truth ≤ entry.score.toInt)

set_option maxRecDepth 4096 in
theorem byte_masks : ∀ raw : BitVec 8,
    decide (raw &&& 1#8 ≠ 0#8) = raw.getLsbD 0 ∧
    decide (raw &&& 2#8 ≠ 0#8) = raw.getLsbD 1 := by decide

set_option maxRecDepth 4096 in
theorem selectivityRank_eq_min : ∀ raw : BitVec 8,
    selectivityRank raw = min raw.toNat 5 := by decide

theorem depth_widen (raw : BitVec 8) : (raw.setWidth 32).toNat = raw.toNat := by
  have bound : raw.toNat < 256 := raw.toFin.isLt
  simp only [BitVec.toNat_setWidth]
  apply Nat.mod_eq_of_lt
  omega

theorem entryScore_resolved (checked : Bool) (entry : EntryData) :
    entryScore checked entry =
      if checked && decide (entry.score.toInt = -32768) then .error scoreError
      else .ok ⟨entry.score.signExtend 32⟩ := by
  have sign : (entry.score.signExtend 32).toInt = entry.score.toInt :=
    BitVec.toInt_signExtend_of_le (by decide)
  have lo := BitVec.le_toInt entry.score
  have hi := BitVec.toInt_lt (x := entry.score)
  have invalid : (entry.score.toInt < -32767 ∨ entry.score.toInt > 32767) ↔
      entry.score.toInt = -32768 := by omega
  simp [entryScore, ScaledScore.fromRaw, sign, invalid, scoreError]

def CanCutContract : Prop :=
  ∀ checked entry beta depth selectivity requireEndgame,
    canCut checked entry beta depth selectivity requireEndgame =
      if Eligible entry depth selectivity requireEndgame then
        if checked && decide (entry.score.toInt = -32768) then .error scoreError
        else .ok (decide (CutWitness entry beta))
      else .ok false

theorem canCut_contract : CanCutContract := by
  intro checked entry beta depth selectivity requireEndgame
  have sign : (entry.score.signExtend 32).toInt = entry.score.toInt :=
    BitVec.toInt_signExtend_of_le (by decide)
  have bitDecision (bit : Bool) : decide (bit = true) = bit := by
    cases bit <;> rfl
  have decision :
      decide (entry.bound &&& (if beta.raw.toInt ≤ entry.score.toInt then 1 else 2) ≠ 0#8) =
        decide (CutWitness entry beta) := by
    by_cases high : beta.raw.toInt ≤ entry.score.toInt
    · have low : ¬ entry.score.toInt < beta.raw.toInt := by omega
      simp only [high, ↓reduceIte, CutWitness, low, true_and, false_and,
        or_false, bitDecision]
      exact (byte_masks entry.bound).1
    · have low : entry.score.toInt < beta.raw.toInt := by omega
      simp only [high, ↓reduceIte, CutWitness, low, true_and, false_and,
        false_or, bitDecision]
      exact (byte_masks entry.bound).2
  have guardSpec :
      ((requireEndgame && !isEndgame entry) = false ∧
        (decide (entry.depth.toNat < depth.toNat) ||
          decide (selectivityRank entry.selectivity < selectivity.val)) = false) ↔
        Eligible entry depth selectivity requireEndgame := by
    rw [isEndgame, (byte_masks entry.flags).1, selectivityRank_eq_min]
    cases requireEndgame <;> simp [Eligible]
  have gateOrder (first second : Bool) (value : Except String Bool) :
      (if first then .ok false else if second then .ok false else value) =
        (if first = false ∧ second = false then value else .ok false) := by
    cases first <;> cases second <;> rfl
  have scoreCase (flag : Bool) :
      (match (if flag then Except.error scoreError else
          Except.ok (⟨entry.score.signExtend 32⟩ : Score)) with
        | .error reason => Except.error reason
        | .ok score => Except.ok (decide (entry.bound &&&
            (if score.raw.toInt ≥ beta.raw.toInt then 1 else 2) ≠ 0#8))) =
        (if flag then .error scoreError else .ok (decide (CutWitness entry beta))) := by
    cases flag with
    | false =>
      change Except.ok (decide (entry.bound &&&
          (if (entry.score.signExtend 32).toInt ≥ beta.raw.toInt then 1 else 2) ≠ 0#8)) =
        Except.ok (decide (CutWitness entry beta))
      apply congrArg Except.ok
      rw [sign]
      exact decision
    | true => rfl
  unfold canCut
  rw [gateOrder]
  simp only [guardSpec]
  by_cases eligible : Eligible entry depth selectivity requireEndgame
  · simp only [ite_eq_left eligible]
    rw [entryScore_resolved]
    exact scoreCase (checked && decide (entry.score.toInt = -32768))
  · simp only [ite_eq_right eligible]

theorem canCut_true_iff (checked : Bool) (entry : EntryData) (beta : Score) (depth : Depth)
    (selectivity : Selectivity) (requireEndgame : Bool) :
    canCut checked entry beta depth selectivity requireEndgame = .ok true ↔
      Eligible entry depth selectivity requireEndgame ∧
      (checked = false ∨ entry.score.toInt ≠ -32768) ∧ CutWitness entry beta := by
  have rule (P Q B : Prop) [Decidable P] [Decidable Q] [Decidable B]
      (assertions : Bool) :
      ((if P then
          if assertions && decide B then .error scoreError else .ok (decide Q)
        else .ok false) : Except String Bool) = .ok true ↔
          P ∧ (assertions = false ∨ ¬ B) ∧ Q := by
    by_cases hp : P <;> by_cases hq : Q <;> by_cases hb : B <;>
      cases assertions <;> simp [hp, hq, hb]
  rw [canCut_contract]
  exact rule (Eligible entry depth selectivity requireEndgame)
    (CutWitness entry beta) (entry.score.toInt = -32768) checked

theorem canCut_error_iff (checked : Bool) (entry : EntryData) (beta : Score) (depth : Depth)
    (selectivity : Selectivity) (requireEndgame : Bool) :
    canCut checked entry beta depth selectivity requireEndgame = .error scoreError ↔
      Eligible entry depth selectivity requireEndgame ∧ checked = true ∧
        entry.score.toInt = -32768 := by
  have rule (P B : Prop) [Decidable P] [Decidable B] (assertions value : Bool) :
      ((if P then
          if assertions && decide B then .error scoreError else .ok value
        else .ok false) : Except String Bool) = .error scoreError ↔
          P ∧ assertions = true ∧ B := by
    by_cases hp : P <;> by_cases hb : B <;>
      cases assertions <;> simp [hp, hb]
  rw [canCut_contract]
  exact rule (Eligible entry depth selectivity requireEndgame)
    (entry.score.toInt = -32768) checked (decide (CutWitness entry beta))

theorem canCut_ineligible (checked : Bool) (entry : EntryData) (beta : Score) (depth : Depth)
    (selectivity : Selectivity) (requireEndgame : Bool)
    (rejected : ¬ Eligible entry depth selectivity requireEndgame) :
    canCut checked entry beta depth selectivity requireEndgame = .ok false := by
  rw [canCut_contract]
  simp [rejected]

theorem canCut_sound (checked : Bool) (entry : EntryData) (beta : Score) (depth : Depth)
    (selectivity : Selectivity) (requireEndgame : Bool) (truth : Int)
    (sound : BoundSound entry truth)
    (cut : canCut checked entry beta depth selectivity requireEndgame = .ok true) :
    (beta.raw.toInt ≤ entry.score.toInt ↔ beta.raw.toInt ≤ truth) ∧
    (entry.score.toInt < beta.raw.toInt ↔ truth < beta.raw.toInt) := by
  have witness := ((canCut_true_iff checked entry beta depth selectivity requireEndgame).mp cut).2.2
  rcases witness with lower | upper
  · have bound := sound.1 lower.2
    have side := lower.1
    constructor <;> omega
  · have bound := sound.2 upper.2
    have side := upper.1
    constructor <;> omega

theorem noBound_never_cuts (checked : Bool) (entry : EntryData) (beta : Score) (depth : Depth)
    (selectivity : Selectivity) (requireEndgame : Bool) (hnone : entry.bound = 0#8) :
    canCut checked entry beta depth selectivity requireEndgame ≠ .ok true := by
  intro cut
  have witness := ((canCut_true_iff checked entry beta depth selectivity requireEndgame).mp cut).2.2
  simp [CutWitness, hnone] at witness

theorem exactBound_cuts (checked : Bool) (entry : EntryData) (beta : Score) (depth : Depth)
    (selectivity : Selectivity) (requireEndgame : Bool) (hexact : entry.bound = 3#8)
    (eligible : Eligible entry depth selectivity requireEndgame)
    (valid : checked = false ∨ entry.score.toInt ≠ -32768) :
    canCut checked entry beta depth selectivity requireEndgame = .ok true := by
  apply (canCut_true_iff checked entry beta depth selectivity requireEndgame).mpr
  refine ⟨eligible, valid, ?_⟩
  simp only [CutWitness, hexact]
  by_cases high : beta.raw.toInt ≤ entry.score.toInt
  · exact Or.inl ⟨high, by decide⟩
  · exact Or.inr ⟨by omega, by decide⟩

def ClassifyContract : Prop := ∀ pv best alpha beta,
  (classify pv best alpha beta = .lower ↔ beta.raw.toInt ≤ best.raw.toInt) ∧
  (classify pv best alpha beta = .exact ↔
    pv = true ∧ alpha.raw.toInt < best.raw.toInt ∧ best.raw.toInt < beta.raw.toInt) ∧
  (classify pv best alpha beta = .upper ↔
    best.raw.toInt < beta.raw.toInt ∧ (pv = false ∨ best.raw.toInt ≤ alpha.raw.toInt)) ∧
  classify pv best alpha beta ≠ .none

theorem classify_contract : ClassifyContract := by
  intro pv best alpha beta
  cases pv <;> by_cases high : beta.raw.toInt ≤ best.raw.toInt <;>
    by_cases above : alpha.raw.toInt < best.raw.toInt <;>
    simp [classify, high, above] <;> omega

def sameBoardBits (lp lo rp ro : BitVec 64) : Bool :=
  decide (((lp ^^^ rp) ||| (lo ^^^ ro)) = 0#64)

theorem sameBoardBits_exact (lp lo rp ro : BitVec 64) :
    sameBoardBits lp lo rp ro = true ↔ lp = rp ∧ lo = ro := by
  simp only [sameBoardBits, decide_eq_true_eq]
  constructor
  · intro equal
    constructor
    all_goals
      apply BitVec.eq_of_getLsbD_eq
      intro i _
      have bit := congrArg (fun word : BitVec 64 => word.getLsbD i) equal
      cases hp : lp.getLsbD i <;> cases ho : lo.getLsbD i <;>
        cases hq : rp.getLsbD i <;> cases hr : ro.getLsbD i <;>
        simp_all
  · rintro ⟨rfl, rfl⟩
    simp

private def fixture (score : Int) (bound depth selectivity flags : Nat) : EntryData :=
  ⟨BitVec.ofInt 16 score, 255, BitVec.ofNat 8 bound, BitVec.ofNat 8 depth,
    BitVec.ofNat 8 selectivity, 0, BitVec.ofNat 8 flags⟩

example : canCut true (fixture 256 1 12 5 1) ⟨256⟩ 12 ⟨5, by decide⟩ true = .ok true := by rfl
example : canCut true (fixture 256 2 12 5 1) ⟨256⟩ 12 ⟨5, by decide⟩ true = .ok false := by rfl
example : canCut true (fixture 255 2 12 5 1) ⟨256⟩ 12 ⟨5, by decide⟩ true = .ok true := by rfl
example : canCut true (fixture 256 3 11 5 1) ⟨256⟩ 12 ⟨5, by decide⟩ true = .ok false := by rfl
example : canCut true (fixture 256 3 12 4 1) ⟨256⟩ 12 ⟨5, by decide⟩ true = .ok false := by rfl
example : canCut true (fixture 256 3 12 5 2) ⟨256⟩ 12 ⟨5, by decide⟩ true = .ok false := by rfl
example : canCut true (fixture (-32768) 3 12 5 1) ⟨0⟩ 12 ⟨5, by decide⟩ true = .error scoreError := by rfl
example : canCut true (fixture (-32768) 3 11 5 1) ⟨0⟩ 12 ⟨5, by decide⟩ true = .ok false := by rfl
example : canCut false (fixture (-32768) 3 12 255 1) ⟨0⟩ 12 ⟨5, by decide⟩ true = .ok true := by rfl
example : classify false ⟨1⟩ ⟨0⟩ ⟨2⟩ = .upper := by rfl
example : classify true ⟨1⟩ ⟨0⟩ ⟨2⟩ = .exact := by rfl
example : classify true ⟨2⟩ ⟨0⟩ ⟨2⟩ = .lower := by rfl

example : canCut false (fixture 256 1 12 5 1) ⟨256⟩ 12 ⟨5, by decide⟩ true = .ok true ∧
    ¬ BoundSound (fixture 256 1 12 5 1) 0 := by
  constructor
  · rfl
  · simp [BoundSound, fixture]

end NeuralReversi.TranspositionTable
