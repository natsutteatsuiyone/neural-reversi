module
public import NeuralReversi.Stability.CutoffModel
public import NeuralReversi.Stability.SingleEmpty
meta import NeuralReversi.Stability.CutoffModel
meta import NeuralReversi.Stability.SingleEmpty
meta import NeuralReversi.Stability.EdgeGameBridge

@[expose] public section

namespace NeuralReversi.Stability.Cutoff

open NeuralReversi.Stability.Semantics

def countColor (b : Board) (c : Color) : Nat :=
  (List.finRange 64).countP (fun sq => decide (b (boardSquare sq) = some c))

def countEmpty (b : Board) : Nat :=
  (List.finRange 64).countP (fun sq => decide (b (boardSquare sq) = none))

def cutoffStable (discs other : Word) (alpha : DiscScore) : Word :=
  let stable := getStableDiscs discs other
  let shortfall := discScoreMax.value - 2 * (popcount stable : Int) - alpha.value
  if 0 < shortfall ∧ shortfall ≤ 2 * singleEmptyMaxShortfall then
    singleEmptyStable discs other stable
  else stable

def stabilityCutoff (player opponent : Word) (nEmpties : Fin 64) (alpha : DiscScore) :
    Option DiscScore :=
  if alpha.value ≥ nwsStabilityThreshold nEmpties then
    if !hasEnoughOpponentDiscsForCutoff (popcount opponent) alpha then
      none
    else
      let score : DiscScore :=
        ⟨discScoreMax.value - 2 * (popcount (cutoffStable opponent player alpha) : Int)⟩
      if score.value ≤ alpha.value then
        some score
      else
        none
  else
    none

private def shieldedB5Player : Word := 0x003e4e0812420f00
private def shieldedB5Opponent : Word := 0xffc1b1f7edbdb0bc

#guard discScoreMax.value -
  2 * (popcount (getStableDiscs shieldedB5Opponent shieldedB5Player) : Int) = 12
#guard stabilityCutoff shieldedB5Player shieldedB5Opponent 4 ⟨6⟩ = some ⟨6⟩

theorem stabilityCutoff_some_spec {player opponent : Word} {nEmpties : Fin 64}
    {alpha cutoff : DiscScore}
    (h : stabilityCutoff player opponent nEmpties alpha = some cutoff) :
    cutoff.value = 64 - 2 * (popcount (cutoffStable opponent player alpha) : Int) ∧
    cutoff.value ≤ alpha.value ∧
    alpha.value ≥ nwsStabilityThreshold nEmpties ∧
    hasEnoughOpponentDiscsForCutoff (popcount opponent) alpha = true := by
  unfold stabilityCutoff at h
  split at h
  · rename_i hthresh
    split at h
    · cases h
    · rename_i hnot
      have henough : hasEnoughOpponentDiscsForCutoff (popcount opponent) alpha = true := by
        cases hq : hasEnoughOpponentDiscsForCutoff (popcount opponent) alpha <;> simp_all
      dsimp only at h
      split at h
      · rename_i hle
        injection h with hEq
        subst hEq
        exact ⟨rfl, hle, hthresh, henough⟩
      · cases h
  · cases h

private theorem countP_mono {α : Type} (l : List α) (p q : α → Bool)
    (hpq : ∀ x, p x = true → q x = true) :
    l.countP p ≤ l.countP q := by
  induction l with
  | nil => simp
  | cons x xs ih =>
    simp only [List.countP_cons]
    cases hp : p x with
    | false =>
      cases q x <;> simp <;> omega
    | true =>
      have hq : q x = true := hpq x hp
      simp [hq]
      omega

private theorem countP_partition3 {α : Type} (l : List α) (p1 p2 p3 : α → Bool)
    (hpart : ∀ x,
      (p1 x = true ∧ p2 x = false ∧ p3 x = false) ∨
      (p1 x = false ∧ p2 x = true ∧ p3 x = false) ∨
      (p1 x = false ∧ p2 x = false ∧ p3 x = true)) :
    l.countP p1 + l.countP p2 + l.countP p3 = l.length := by
  induction l with
  | nil => simp
  | cons x xs ih =>
    simp only [List.countP_cons, List.length_cons]
    rcases hpart x with ⟨h1, h2, h3⟩ | ⟨h1, h2, h3⟩ | ⟨h1, h2, h3⟩ <;>
      simp [h1, h2, h3] <;> omega

theorem board_count_sum (b : Board) (c : Color) :
    countColor b c + countColor b (Opposite c) + countEmpty b = 64 := by
  unfold countColor countEmpty
  have h := countP_partition3 (List.finRange 64)
    (fun sq => decide (b (boardSquare sq) = some c))
    (fun sq => decide (b (boardSquare sq) = some (Opposite c)))
    (fun sq => decide (b (boardSquare sq) = none))
    (fun sq => by
      cases hb : b (boardSquare sq) with
      | none =>
        right; right
        simp
      | some d =>
        cases c <;> cases d <;> simp [Opposite])
  simpa [List.length_finRange] using h

theorem represents_swap {s : State} {c : Color} {p o : Word}
    (repr : Represents s c p o) : Represents s (Opposite c) o p := by
  constructor
  · exact repr.2
  · intro x
    have hopp : Opposite (Opposite c) = c := by cases c <;> rfl
    rw [hopp]
    exact repr.1 x

theorem stable_opponent_count_le_reachable {s t : State} {c : Color} {player opponent : Word}
    (repr : Represents s c player opponent) (future : Reach s t) (alpha : DiscScore) :
    popcount (cutoffStable opponent player alpha) ≤ countColor t.board (Opposite c) := by
  have opp := represents_swap repr
  have first : SoundMask s (Opposite c) (getStableDiscs opponent player) :=
    fun sq marked t reached => returned_disc_never_flips opp reached sq marked
  have sound : SoundMask s (Opposite c) (cutoffStable opponent player alpha) := by
    unfold cutoffStable
    dsimp only
    split
    · exact singleEmptyStable_sound opp first
    · exact first
  unfold popcount countColor
  apply countP_mono
  intro sq hmarked
  have hstay := sound sq hmarked t future
  simp [hstay]

def terminalScore (b : Board) (c : Color) : Int :=
  let p : Int := countColor b c
  let o : Int := countColor b (Opposite c)
  let e : Int := countEmpty b
  if p > o then p - o + e
  else if p < o then p - o - e
  else 0

theorem terminalScore_le_bound (b : Board) (c : Color) :
    terminalScore b c ≤ 64 - 2 * (countColor b (Opposite c) : Int) := by
  have hsum := board_count_sum b c
  unfold terminalScore
  dsimp only
  split
  · omega
  · split <;> omega

theorem rawDiscDiff_le_bound (b : Board) (c : Color) :
    (countColor b c : Int) - (countColor b (Opposite c) : Int) ≤
      64 - 2 * (countColor b (Opposite c) : Int) := by
  have hsum := board_count_sum b c
  omega

theorem terminalScore_le_of_stable {s t : State} {c : Color} {player opponent : Word}
    (repr : Represents s c player opponent) (future : Reach s t) (alpha : DiscScore) :
    terminalScore t.board c ≤ 64 - 2 * (popcount (cutoffStable opponent player alpha) : Int) := by
  have hcount := stable_opponent_count_le_reachable repr future alpha
  have hbound := terminalScore_le_bound t.board c
  omega

def Terminal (s : State) : Prop :=
  ∀ t, ¬ Step s t

inductive MinimaxVal (root : Color) : State → Int → Prop where
  | terminal (s : State) (hterm : Terminal s) :
      MinimaxVal root s (terminalScore s.board root)
  | maxNode (s : State) (hturn : s.turn = root) (v : Int)
      (childVal : State → Int)
      (hchild : ∀ t, Step s t → MinimaxVal root t (childVal t))
      (hbound : ∀ t, Step s t → childVal t ≤ v)
      (best : State) (hbestStep : Step s best) (hbestVal : childVal best = v) :
      MinimaxVal root s v
  | minNode (s : State) (hturn : s.turn = Opposite root) (v : Int)
      (childVal : State → Int)
      (hchild : ∀ t, Step s t → MinimaxVal root t (childVal t))
      (hbound : ∀ t, Step s t → v ≤ childVal t)
      (best : State) (hbestStep : Step s best) (hbestVal : childVal best = v) :
      MinimaxVal root s v

theorem minimaxVal_le_of_reachable_terminal_bound {root : Color} {s : State} {v B : Int}
    (hval : MinimaxVal root s v)
    (hbound : ∀ t, Reach s t → Terminal t → terminalScore t.board root ≤ B) :
    v ≤ B := by
  induction hval with
  | terminal s hterm =>
    exact hbound s (.refl s) hterm
  | maxNode s _ v childVal _ _ best hbestStep hbestVal ih =>
    have hle := ih best hbestStep (fun u htu hterm =>
      hbound u (reach_trans (Reach.tail (.refl s) hbestStep) htu) hterm)
    omega
  | minNode s _ v childVal _ _ best hbestStep hbestVal ih =>
    have hle := ih best hbestStep (fun u htu hterm =>
      hbound u (reach_trans (Reach.tail (.refl s) hbestStep) htu) hterm)
    omega

theorem minimaxVal_unique {root : Color} {s : State} {v₁ v₂ : Int}
    (h₁ : MinimaxVal root s v₁) (h₂ : MinimaxVal root s v₂) : v₁ = v₂ := by
  induction h₁ generalizing v₂ with
  | terminal s hterm =>
    cases h₂ with
    | terminal => rfl
    | maxNode _ _ _ _ _ _ best hbestStep _ =>
      exact False.elim (hterm best hbestStep)
    | minNode _ _ _ _ _ _ best hbestStep _ =>
      exact False.elim (hterm best hbestStep)
  | maxNode s hturn v₁ childVal₁ _ hbound₁ best₁ hbestStep₁ hbestVal₁ ih =>
    cases h₂ with
    | terminal _ hterm =>
      exact False.elim (hterm best₁ hbestStep₁)
    | minNode _ hturn₂ _ _ _ _ _ _ _ =>
      rw [hturn] at hturn₂
      cases root <;> simp [Opposite] at hturn₂
    | maxNode _ _ v₂ childVal₂ hchild₂ hbound₂ best₂ hbestStep₂ hbestVal₂ =>
      have heq₁ : childVal₁ best₁ = childVal₂ best₁ := ih best₁ hbestStep₁ (hchild₂ best₁ hbestStep₁)
      have hle₁ : childVal₂ best₁ ≤ v₂ := hbound₂ best₁ hbestStep₁
      have heq₂ : childVal₁ best₂ = childVal₂ best₂ := ih best₂ hbestStep₂ (hchild₂ best₂ hbestStep₂)
      have hle₂ : childVal₁ best₂ ≤ v₁ := hbound₁ best₂ hbestStep₂
      omega
  | minNode s hturn v₁ childVal₁ _ hbound₁ best₁ hbestStep₁ hbestVal₁ ih =>
    cases h₂ with
    | terminal _ hterm =>
      exact False.elim (hterm best₁ hbestStep₁)
    | maxNode _ hturn₂ _ _ _ _ _ _ _ =>
      rw [hturn₂] at hturn
      cases root <;> simp [Opposite] at hturn
    | minNode _ _ v₂ childVal₂ hchild₂ hbound₂ best₂ hbestStep₂ hbestVal₂ =>
      have heq₁ : childVal₁ best₁ = childVal₂ best₁ := ih best₁ hbestStep₁ (hchild₂ best₁ hbestStep₁)
      have hle₁ : v₂ ≤ childVal₂ best₁ := hbound₂ best₁ hbestStep₁
      have heq₂ : childVal₁ best₂ = childVal₂ best₂ := ih best₂ hbestStep₂ (hchild₂ best₂ hbestStep₂)
      have hle₂ : v₁ ≤ childVal₁ best₂ := hbound₁ best₂ hbestStep₂
      omega

theorem stabilityCutoff_sound {s : State} {c : Color} {player opponent : Word}
    {nEmpties : Fin 64} {alpha cutoff : DiscScore} {v : Int}
    (repr : Represents s c player opponent)
    (hcut : stabilityCutoff player opponent nEmpties alpha = some cutoff)
    (hval : MinimaxVal c s v) :
    v ≤ cutoff.value ∧ cutoff.value ≤ alpha.value := by
  obtain ⟨hscore, hle, _, _⟩ := stabilityCutoff_some_spec hcut
  refine ⟨?_, hle⟩
  rw [hscore]
  exact minimaxVal_le_of_reachable_terminal_bound hval
    (fun t future _ => terminalScore_le_of_stable repr future alpha)

theorem stabilityCutoff_le_alpha {s : State} {c : Color} {player opponent : Word}
    {nEmpties : Fin 64} {alpha cutoff : DiscScore} {v : Int}
    (repr : Represents s c player opponent)
    (hcut : stabilityCutoff player opponent nEmpties alpha = some cutoff)
    (hval : MinimaxVal c s v) :
    v ≤ alpha.value := by
  have ⟨h1, h2⟩ := stabilityCutoff_sound repr hcut hval
  omega

def cutoffAtEmptyCount (player opponent : Word) (nEmpties : Nat) (alpha : DiscScore) :
    Option DiscScore :=
  if inTable : nEmpties < 64 then
    stabilityCutoff player opponent ⟨nEmpties, inTable⟩ alpha
  else none

def SourceCutoffContract : Prop :=
  ∀ player opponent nEmpties alpha,
    (64 ≤ nEmpties → cutoffAtEmptyCount player opponent nEmpties alpha = none) ∧
    ∀ cutoff, cutoffAtEmptyCount player opponent nEmpties alpha = some cutoff →
      cutoff.value = 64 - 2 * (popcount (cutoffStable opponent player alpha) : Int) ∧
      cutoff.value ≤ alpha.value ∧
      hasEnoughOpponentDiscsForCutoff (popcount opponent) alpha = true ∧
      ∀ s c value, Represents s c player opponent → MinimaxVal c s value → value ≤ cutoff.value

theorem sourceCutoffContract_proven : SourceCutoffContract := by
  intro player opponent nEmpties alpha
  by_cases inTable : nEmpties < 64
  · refine ⟨fun outside => by omega, ?_⟩
    intro cutoff picked
    have result : stabilityCutoff player opponent ⟨nEmpties, inTable⟩ alpha = some cutoff := by
      simpa [cutoffAtEmptyCount, inTable] using picked
    obtain ⟨score, bound, _, enough⟩ := stabilityCutoff_some_spec result
    exact ⟨score, bound, enough, fun _ _ _ repr value => (stabilityCutoff_sound repr result value).1⟩
  · simp [cutoffAtEmptyCount, inTable]

theorem getStableDiscsExec_eq : getStableDiscsExec = Semantics.getStableDiscs := rfl

theorem singleEmptyStableExec_eq : singleEmptyStableExec = Semantics.singleEmptyStable := rfl

theorem cutoffAtEmptyCountExec_eq : cutoffAtEmptyCountExec = cutoffAtEmptyCount := rfl

end NeuralReversi.Stability.Cutoff
