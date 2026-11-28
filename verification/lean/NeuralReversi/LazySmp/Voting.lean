module

@[expose] public section

namespace NeuralReversi.LazySmp

structure Ballot where
  square : Fin 64
  score : Int
  depth : Nat
  deriving DecidableEq, Repr

def weight (lowest : Int) (b : Ballot) : Int := (b.score - lowest + 256) * b.depth

def addVote (lowest : Int) (votes : Fin 64 → Int) (b : Ballot) (square : Fin 64) : Int :=
  votes square + if square = b.square then weight lowest b else 0

def totals (lowest : Int) (ballots : List Ballot) : Fin 64 → Int :=
  ballots.foldl (addVote lowest) (fun _ => 0)

abbrev Candidate := Nat × Ballot

def better (votes : Fin 64 → Int) (a b : Ballot) : Prop :=
  votes b.square < votes a.square ∨ votes b.square = votes a.square ∧ b.depth < a.depth

instance (votes : Fin 64 → Int) (a b : Ballot) : Decidable (better votes a b) :=
  inferInstanceAs (Decidable (_ ∨ _))

def preferred (votes : Fin 64 → Int) (a b : Candidate) : Prop :=
  better votes a.2 b.2 ∨
    votes a.2.square = votes b.2.square ∧ a.2.depth = b.2.depth ∧ a.1 ≤ b.1

private theorem preferred_refl (votes : Fin 64 → Int) (a : Candidate) :
    preferred votes a a := Or.inr ⟨rfl, rfl, Nat.le_refl _⟩

private theorem preferred_trans {votes : Fin 64 → Int} {a b c : Candidate}
    (ab : preferred votes a b) (bc : preferred votes b c) : preferred votes a c := by
  unfold preferred better at *
  omega

def choose (votes : Fin 64 → Int) (best next : Candidate) : Candidate :=
  if better votes next.2 best.2 then next else best

private theorem choose_preferred (votes : Fin 64 → Int) (best next : Candidate)
    (order : best.1 < next.1) :
    preferred votes (choose votes best next) best ∧ preferred votes (choose votes best next) next := by
  unfold choose
  split
  · rename_i stronger
    exact ⟨Or.inl stronger, preferred_refl votes next⟩
  · rename_i weaker
    refine ⟨preferred_refl votes best, ?_⟩
    unfold preferred better at *
    omega

def scan (votes : Fin 64 → Int) : Candidate → Nat → List Ballot → Candidate
  | best, _, [] => best
  | best, i, b :: bs => scan votes (choose votes best (i, b)) (i + 1) bs

structure Selection (votes : Fin 64 → Int) (best : Candidate) (i : Nat) (bs : List Ballot)
    (out : Candidate) : Prop where
  source : out = best ∨ ∃ j b, bs[j]? = some b ∧ out = (i + j, b)
  first : preferred votes out best
  rest : ∀ j b, bs[j]? = some b → preferred votes out (i + j, b)

private theorem scan_spec (votes : Fin 64 → Int) (best : Candidate) (i : Nat) (bs : List Ballot)
    (before : best.1 < i) : Selection votes best i bs (scan votes best i bs) := by
  induction bs generalizing best i with
  | nil =>
    exact ⟨Or.inl rfl, preferred_refl votes best, fun _ _ h => by simp at h⟩
  | cons b bs ih =>
    let current := choose votes best (i, b)
    have curBefore : current.1 < i + 1 := by
      unfold current choose
      split <;> omega
    obtain ⟨source, first, rest⟩ := ih current (i + 1) curBefore
    obtain ⟨beatBest, beatCurrent⟩ := choose_preferred votes best (i, b) before
    change Selection votes best i (b :: bs) (scan votes current (i + 1) bs)
    constructor
    · rcases source with same | ⟨j, e, h, found⟩
      · by_cases improves : better votes b best.2
        · exact .inr ⟨0, b, rfl, by simpa [current, choose, improves] using same⟩
        · exact .inl (by simpa [current, choose, improves] using same)
      · exact .inr ⟨j + 1, e, h, by simpa [Nat.add_assoc, Nat.add_comm 1 j] using found⟩
    · exact preferred_trans first beatBest
    · intro j e h
      cases j with
      | zero =>
        have he : b = e := Option.some.inj h
        subst e
        exact preferred_trans first beatCurrent
      | succ j =>
        simpa [Nat.add_assoc, Nat.add_comm 1 j] using rest j e h

def winner (ballots : List Ballot) : Option Candidate :=
  match ballots with
  | [] => none
  | first :: rest =>
    let lowest := ((first :: rest).map Ballot.score).min?.getD first.score
    some (scan (totals lowest ballots) (0, first) 1 rest)

def VotingContract : Prop :=
  ∀ (ballots : List Ballot) (out : Candidate), winner ballots = some out →
    ballots[out.1]? = some out.2 ∧
    ∀ i b, ballots[i]? = some b →
      preferred (totals ((ballots.map Ballot.score).min?.getD 0) ballots) out (i, b)

theorem votingContract_proven : VotingContract := by
  intro ballots out found
  cases ballots with
  | nil => cases found
  | cons first rest =>
    let ballots := first :: rest
    let lowest := (ballots.map Ballot.score).min?.getD first.score
    have fallback : lowest = (ballots.map Ballot.score).min?.getD 0 := by
      simp [lowest, ballots]
    let votes := totals lowest ballots
    have eq : scan votes (0, first) 1 rest = out := Option.some.inj found
    obtain ⟨source, firstPlace, others⟩ := scan_spec votes (0, first) 1 rest (Nat.zero_lt_succ _)
    rw [eq] at source firstPlace others
    constructor
    · rcases source with same | ⟨j, b, h, same⟩
      · simp [same]
      · simpa [same, Nat.add_comm 1 j] using h
    · intro i b h
      rw [← fallback]
      cases i with
      | zero =>
        have hb : first = b := Option.some.inj h
        subst b
        exact firstPlace
      | succ i =>
        simpa [Nat.add_comm 1 i] using others i b h

def BoundedBallot (b : Ballot) : Prop :=
  -32767 ≤ b.score ∧ b.score ≤ 32767 ∧ 1 ≤ b.depth ∧ b.depth ≤ 60

private theorem weight_bounds {lowest : Int} {b : Ballot}
    (floor : -32767 ≤ lowest) (least : lowest ≤ b.score) (valid : BoundedBallot b) :
    256 ≤ b.score - lowest + 256 ∧ b.score - lowest + 256 ≤ 65790 ∧
      256 ≤ weight lowest b ∧ weight lowest b ≤ 3947400 := by
  obtain ⟨_, ceiling, one, depth⟩ := valid
  have lo : 256 ≤ b.score - lowest + 256 := by omega
  have hi : b.score - lowest + 256 ≤ 65790 := by omega
  refine ⟨lo, hi, ?_, ?_⟩
  · calc
      256 = 256 * (1 : Int) := rfl
      _ ≤ 256 * (b.depth : Int) := Int.mul_le_mul_of_nonneg_left (by omega) (by decide)
      _ ≤ weight lowest b := Int.mul_le_mul_of_nonneg_right lo (by omega)
  · calc
      weight lowest b ≤ 65790 * (b.depth : Int) := Int.mul_le_mul_of_nonneg_right hi (by omega)
      _ ≤ 65790 * (60 : Int) := Int.mul_le_mul_of_nonneg_left (by omega) (by decide)
      _ = 3947400 := rfl

private theorem fold_bounds {lowest : Int} (bs : List Ballot) (votes : Fin 64 → Int) (square : Fin 64)
    (bounded : ∀ b ∈ bs, 0 ≤ weight lowest b ∧ weight lowest b ≤ 3947400) :
    votes square ≤ (bs.foldl (addVote lowest) votes) square ∧
      (bs.foldl (addVote lowest) votes) square ≤ votes square + 3947400 * bs.length := by
  induction bs generalizing votes with
  | nil => simp
  | cons b bs ih =>
    have wb := bounded b (by simp)
    have hb : votes square ≤ addVote lowest votes b square ∧
        addVote lowest votes b square ≤ votes square + 3947400 := by
      unfold addVote
      split <;> omega
    have bound := ih (addVote lowest votes b) (fun b mem => bounded b (by simp [mem]))
    simp only [List.foldl_cons, List.length_cons]
    omega

def VoteArithmeticContract : Prop :=
  ∀ (ballots visited tail : List Ballot) (lowest : Int), visited ++ tail = ballots →
    ballots.length ≤ 64 → (∀ b ∈ ballots, BoundedBallot b) →
    (ballots.map Ballot.score).min? = some lowest →
    (∀ b ∈ ballots, 256 ≤ b.score - lowest + 256 ∧ b.score - lowest + 256 ≤ 65790 ∧
      256 ≤ weight lowest b ∧ weight lowest b ≤ 3947400) ∧
    ∀ sq, 0 ≤ totals lowest visited sq ∧ totals lowest visited sq ≤ 252633600

theorem voteArithmeticContract_proven : VoteArithmeticContract := by
  intro ballots visited tail lowest joined count valid minimum
  have ⟨source, bound⟩ := List.min?_eq_some_iff.mp minimum
  obtain ⟨ballot, member, eq⟩ := List.mem_map.mp source
  have floor : -32767 ≤ lowest := eq ▸ (valid ballot member).1
  have checked (b : Ballot) (mem : b ∈ ballots) :=
    weight_bounds floor (bound _ (List.mem_map_of_mem mem)) (valid b mem)
  refine ⟨checked, fun sq => ?_⟩
  have size : visited.length ≤ 64 := by rw [← joined, List.length_append] at count; omega
  have safe := fold_bounds visited (fun _ => 0) sq (fun b mem => by
    have bnd := checked b (joined ▸ List.mem_append_left _ mem)
    exact ⟨by have := bnd.2.2.1; omega, bnd.2.2.2⟩)
  change 0 ≤ totals lowest visited sq ∧ totals lowest visited sq ≤ 0 + 3947400 * (visited.length : Int) at safe
  exact ⟨safe.1, by have := safe.2; omega⟩

example : (winner [⟨63, 0, 10⟩, ⟨0, 0, 10⟩]).map Prod.fst = some 0 := by decide

example : (winner [⟨63, 256, 10⟩, ⟨0, 0, 20⟩]).map Prod.fst = some 1 := by decide

example : (winner [⟨63, 0, 10⟩, ⟨0, 0, 6⟩, ⟨0, 0, 6⟩]).map Prod.fst = some 1 := by decide

end NeuralReversi.LazySmp
