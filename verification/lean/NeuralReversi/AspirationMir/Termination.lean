import NeuralReversi.AspirationMir.Driver
import NeuralReversi.Aspiration

namespace NeuralReversi.AspirationMir

private def maxWord (a b : Word) : Word := if a.toInt ≤ b.toInt then b else a
private def minWord (a b : Word) : Word := if a.toInt ≤ b.toInt then a else b

private theorem maxWord_toInt (a b : Word) : (maxWord a b).toInt = max a.toInt b.toInt := by
  unfold maxWord
  split <;> omega

private theorem minWord_toInt (a b : Word) : (minWord a b).toInt = min a.toInt b.toInt := by
  unfold minWord
  split <;> omega

private def failLow (r : WindowRegs) (s : Word) : WindowRegs :=
  ⟨maxWord (s - r.delta) (-BitVec.ofInt 32 infinity), r.alpha, r.delta⟩

private def failHigh (r : WindowRegs) (s : Word) : WindowRegs :=
  ⟨maxWord (r.beta - r.delta) r.alpha, minWord (s + r.delta) (BitVec.ofInt 32 infinity), r.delta⟩

private theorem widen_at_low (r : WindowRegs) (s : Word) (low : s.toInt ≤ r.alpha.toInt) :
    applyWiden r s = some (failLow r s, true) := by
  simp [applyWiden, widenCode, Guard.eval, Expr.eval, low, failLow, maxWord]

private theorem widen_at_high (r : WindowRegs) (s : Word) (above : r.alpha.toInt < s.toInt)
    (high : r.beta.toInt ≤ s.toInt) : applyWiden r s = some (failHigh r s, true) := by
  simp [applyWiden, widenCode, Guard.eval, Expr.eval, Int.not_le.mpr above, high, List.find?, failHigh, minWord, maxWord]

theorem widen_inside (r : WindowRegs) (s : Word) (above : r.alpha.toInt < s.toInt)
    (below : s.toInt < r.beta.toInt) : applyWiden r s = some (r, false) := by
  simp [applyWiden, widenCode, Guard.eval, Expr.eval, Int.not_le.mpr above, Int.not_le.mpr below, List.find?]

private def deltaAt (n : Nat) : Int := initialDelta * (2 : Int)^n

/-- Holds for an extracted `initialDelta` from one disc (256) to 32 discs: eight doublings
    span every finite score, and eleven stay far below Int32 overflow. -/
private theorem delta_bounds {n : Nat} (small : n ≤ 11) :
    0 < deltaAt n ∧ deltaAt n ≤ 16777216 ∧ (8 ≤ n → 65536 ≤ deltaAt n) := by
  match n with
  | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 => decide
  | n + 12 => omega

def doubled (r : WindowRegs) : WindowRegs := { r with delta := r.delta + r.delta }

private theorem add_exact {x y : Word} (lo : -2147483648 ≤ x.toInt + y.toInt)
    (hi : x.toInt + y.toInt < 2147483648) : (x + y).toInt = x.toInt + y.toInt := by
  rw [BitVec.toInt_add]
  exact Int.bmod_eq_of_le (by omega) (by omega)

private theorem sub_exact {x y : Word} (lo : -2147483648 ≤ x.toInt - y.toInt)
    (hi : x.toInt - y.toInt < 2147483648) : (x - y).toInt = x.toInt - y.toInt := by
  rw [BitVec.toInt_sub]
  exact Int.bmod_eq_of_le (by omega) (by omega)

structure Ready (n : Nat) (r : WindowRegs) : Prop where
  window : Aspiration.Window r.alpha.toInt r.beta.toInt
  limit : n ≤ 11
  delta : r.delta.toInt = deltaAt n
  boundary : 9 ≤ n → r.alpha.toInt = -32767 ∨ r.beta.toInt = 32767
  floor : 10 ≤ n → r.alpha.toInt = -32767
  full : 11 ≤ n → r.beta.toInt = 32767

private theorem next_delta (ready : Ready n r) : (r.delta + r.delta).toInt = deltaAt (n + 1) := by
  obtain ⟨pos, bound, _⟩ := delta_bounds ready.limit
  have old := ready.delta
  have double : deltaAt (n + 1) = deltaAt n * 2 := by
    unfold deltaAt
    rw [Int.pow_succ, Int.mul_assoc]
  rw [add_exact (by omega) (by omega), old, double]
  omega

private theorem ready_low (ready : Ready n r) (s : Word) (low : s.toInt ≤ r.alpha.toInt)
    (finite : -32767 < s.toInt ∧ s.toInt < 32767) : Ready (n + 1) (doubled (failLow r s)) := by
  obtain ⟨lo, order, hi⟩ := ready.window
  have current := ready.delta
  obtain ⟨positive, upper, large⟩ := delta_bounds ready.limit
  have a : (doubled (failLow r s)).alpha.toInt = max (s.toInt - r.delta.toInt) (-32767) := by
    change (maxWord _ _).toInt = _
    rw [maxWord_toInt, sub_exact (by omega) (by omega)]
    rfl
  have b : (doubled (failLow r s)).beta = r.alpha := rfl
  have later (last : 8 ≤ n) : (doubled (failLow r s)).alpha.toInt = -32767 := by
    rw [a]
    have := large last
    omega
  have stop : n < 10 := by
    apply Nat.lt_of_not_ge
    intro no
    have := ready.floor (by omega)
    omega
  refine ⟨⟨?_, ?_, ?_⟩, by omega, next_delta (r := r) ready, ?_, ?_, ?_⟩
  · rw [a]; omega
  · rw [a, b]; omega
  · rw [b]; omega
  · intro late; exact Or.inl (later (by omega))
  · intro late; exact later (by omega)
  · omega

private theorem ready_high (ready : Ready n r) (s : Word) (high : r.beta.toInt ≤ s.toInt)
    (finite : -32767 < s.toInt ∧ s.toInt < 32767) : Ready (n + 1) (doubled (failHigh r s)) := by
  obtain ⟨lo, order, hi⟩ := ready.window
  have current := ready.delta
  obtain ⟨positive, upper, large⟩ := delta_bounds ready.limit
  have a : (doubled (failHigh r s)).alpha.toInt = max (r.beta.toInt - r.delta.toInt) r.alpha.toInt := by
    change (maxWord _ _).toInt = _
    rw [maxWord_toInt, sub_exact (by omega) (by omega)]
  have b : (doubled (failHigh r s)).beta.toInt = min (s.toInt + r.delta.toInt) 32767 := by
    change (minWord _ _).toInt = _
    rw [minWord_toInt, add_exact (by omega) (by omega)]
    rfl
  have later (last : 8 ≤ n) : (doubled (failHigh r s)).alpha.toInt = r.alpha.toInt ∧
      (doubled (failHigh r s)).beta.toInt = 32767 := by
    have := large last
    rw [a, b]
    omega
  have stop : n < 11 := by
    apply Nat.lt_of_not_ge
    intro no
    have := ready.full (by omega)
    omega
  refine ⟨⟨?_, ?_, ?_⟩, by omega, next_delta (r := r) ready, ?_, ?_, ?_⟩
  · rw [a]; omega
  · rw [a, b]; omega
  · rw [b]; omega
  · intro late; exact Or.inr (later (by omega)).2
  · intro late
    obtain site | finish := ready.boundary (by omega)
    · exact (later (by omega)).1.trans site
    · omega
  · intro late; exact (later (by omega)).2

theorem first_ready (r : WindowRegs) (window : Aspiration.Window r.alpha.toInt r.beta.toInt) :
    Ready 0 {r with delta := BitVec.ofInt 32 initialDelta} :=
  ⟨window, by decide, rfl, by omega, by omega, by omega⟩

theorem ready_widen (ready : Ready n regs) (word : Word) (finite : -32767 < word.toInt ∧ word.toInt < 32767) :
    applyWiden regs word = some (regs, false) ∨
      ∃ after, applyWiden regs word = some (after, true) ∧ Ready (n+1) (doubled after) := by
  by_cases low : word.toInt ≤ regs.alpha.toInt
  · exact .inr ⟨_, widen_at_low _ _ low, ready_low ready _ low finite⟩
  by_cases high : regs.beta.toInt ≤ word.toInt
  · exact .inr ⟨_, widen_at_high _ _ (by omega) high, ready_high ready _ high finite⟩
  exact .inl (widen_inside _ _ (by omega) (by omega))

def currentCall (query : Query) (m : Machine) : Call :=
  ⟨m.issued, m.regs.alpha, m.regs.beta, query m.issued m.regs.alpha m.regs.beta⟩

private def afterPoll (query : Query) (m : Machine) : Machine :=
  { m with pc := 7, issued := m.issued + 1, last := currentCall query m, flag := false }

def nextTrial (query : Query) (m : Machine) (after : WindowRegs) : Machine :=
  { afterPoll query m with pc := 2, regs := doubled after, flag := true }

private theorem reach_widen (entry : m.pc = 2)
    (rest : Returns query (fun _ => false) (afterPoll query m) score last) :
    Returns query (fun _ => false) m score last := by
  obtain ⟨pc,regs,i,old,flag,out⟩ := m
  change pc = 2 at entry
  subst pc
  exact .next rfl (.next rfl (.next rfl (.next rfl rest)))

theorem retry (entry : m.pc = 2)
    (step : applyWiden m.regs (currentCall query m).score = some (after, true))
    (rest : Returns query (fun _ => false) (nextTrial query m after) score last) :
    Returns query (fun _ => false) m score last := by
  apply reach_widen entry
  refine .next (m' := {afterPoll query m with pc := 8, regs := after, flag := true}) ?_ ?_
  · simp [execute, afterPoll, driverCode, step]
  · exact .next rfl (.next rfl (.next rfl (.next rfl rest)))

theorem accept (entry : m.pc = 2)
    (step : applyWiden m.regs (currentCall query m).score = some (m.regs, false)) :
    Returns query (fun _ => false) m (currentCall query m).score (currentCall query m) := by
  apply reach_widen entry
  refine .next (m' := {afterPoll query m with pc := 8, flag := false}) ?_ ?_
  · simp [execute, afterPoll, driverCode, step]
  · exact .next rfl (.next rfl (.done rfl))

/-- Bounds only on values actually callable at legal windows are sufficient. -/
def FiniteQueries (query : Query) : Prop :=
  ∀ i a b, Aspiration.Window a.toInt b.toInt →
    -32767 < (query i a b).toInt ∧ (query i a b).toInt < 32767

private theorem loop_complete (finite : FiniteQueries query) (fuel : Nat) (ready : Ready n m.regs)
    (entry : m.pc = 2) (index : m.issued = start + n) (remaining : n + fuel = 11) :
    ∃ score last, Returns query (fun _ => false) m score last ∧
      start ≤ last.index ∧ last.index ≤ start + 11 ∧ Aspiration.Window last.alpha.toInt last.beta.toInt := by
  induction fuel generalizing n m with
  | zero =>
    have nEq : n = 11 := by omega
    have floor := ready.floor (by omega)
    have ceil := ready.full (by omega)
    have finiteScore := finite m.issued m.regs.alpha m.regs.beta ready.window
    let call := currentCall query m
    have inside : m.regs.alpha.toInt < call.score.toInt ∧ call.score.toInt < m.regs.beta.toInt := by
      dsimp [call, currentCall]
      omega
    exact ⟨call.score, call, accept entry (widen_inside _ _ inside.1 inside.2),
      by change start ≤ m.issued; omega, by change m.issued ≤ start + 11; omega, ready.window⟩
  | succ fuel ih =>
    let call := currentCall query m
    have finiteScore : -32767 < call.score.toInt ∧ call.score.toInt < 32767 :=
      finite _ _ _ ready.window
    have again (after : WindowRegs) (changed : applyWiden m.regs call.score = some (after, true))
        (next : Ready (n + 1) (doubled after)) :
        ∃ score last, Returns query (fun _ => false) m score last ∧
          start ≤ last.index ∧ last.index ≤ start + 11 ∧ Aspiration.Window last.alpha.toInt last.beta.toInt := by
      obtain ⟨score,last,rest,properties⟩ := ih (m := nextTrial query m after) next rfl
        (by change m.issued + 1 = start + (n + 1); omega) (by omega)
      exact ⟨score, last, retry entry changed rest, properties⟩
    by_cases low : call.score.toInt ≤ m.regs.alpha.toInt
    · exact again _ (widen_at_low _ _ low) (ready_low ready _ low finiteScore)
    have above := Int.lt_of_not_ge low
    by_cases high : m.regs.beta.toInt ≤ call.score.toInt
    · exact again _ (widen_at_high _ _ above high) (ready_high ready _ high finiteScore)
    exact ⟨call.score, call, accept entry (widen_inside _ _ above (Int.lt_of_not_ge high)),
      by change start ≤ m.issued; omega, by change m.issued ≤ start + 11; omega, ready.window⟩

/-- At most twelve calls are issued. Every possible sequence of finite trial scores is covered. -/
theorem completes_in_twelve (finite : FiniteQueries query) (m : Machine) (entry : m.pc = 0)
    (initial : Aspiration.Window m.regs.alpha.toInt m.regs.beta.toInt) :
    ∃ score last, Returns query (fun _ => false) m score last ∧
      m.issued ≤ last.index ∧ last.index ≤ m.issued + 11 ∧
      Aspiration.Window last.alpha.toInt last.beta.toInt := by
  obtain ⟨pc,regs,i,old,flag,out⟩ := m
  change pc = 0 at entry
  subst pc
  let m' : Machine := ⟨2, {regs with delta := BitVec.ofInt 32 initialDelta}, i, old, flag, out⟩
  have ready : Ready 0 m'.regs := ⟨initial, by decide, rfl, by omega, by omega, by omega⟩
  obtain ⟨score,last,steps,properties⟩ := loop_complete finite 11 (start := i) ready rfl rfl rfl
  exact ⟨score,last,.next rfl (.next rfl steps),properties⟩

private theorem same_return (one : Returns query abort m score last)
    (two : Returns query abort m other otherLast) : score = other ∧ last = otherLast := by
  induction one with
  | done h =>
    cases two with
    | done k => rw [h] at k; cases k; exact ⟨rfl, rfl⟩
    | next k _ => rw [h] at k; cases k
  | next h _ ih =>
    cases two with
    | done k => rw [h] at k; cases k
    | next k remaining => rw [h] at k; cases k; exact ih remaining

theorem returned_window (run : Returns query (fun _ => false) m score last) (entry : m.pc = 0)
    (initial : Aspiration.Window m.regs.alpha.toInt m.regs.beta.toInt) (finite : FiniteQueries query) :
    m.issued ≤ last.index ∧ last.index ≤ m.issued + 11 ∧
      Aspiration.Window last.alpha.toInt last.beta.toInt := by
  obtain ⟨_, _, other, bounded⟩ := completes_in_twelve finite m entry initial
  rw [(same_return run other).2]
  exact bounded

end NeuralReversi.AspirationMir
