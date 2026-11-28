module
public import NeuralReversi.AspirationMir.Termination

@[expose] public section

namespace NeuralReversi.AspirationMir.Polled

/-- A stopped call still returns the score supplied by that root invocation.
    Only a call whose final abort observation is false must lie inside its window. -/
def Classified (query : Query) (abort : Nat → Bool) (score : Word) (last : Call) : Prop :=
  last.Observed query ∧ score = last.score ∧
    (abort last.index = false → last.alpha.toInt < last.score.toInt ∧ last.score.toInt < last.beta.toInt)

private def CallSafe (query : Query) (abort : Nat → Bool) (last : Call) : Prop :=
  last.Observed query ∧
    (abort last.index = false → last.alpha.toInt < last.score.toInt ∧ last.score.toInt < last.beta.toInt)

private def Fresh (query : Query) (m : Machine) : Prop :=
  m.last.Observed query ∧ m.regs.alpha = m.last.alpha ∧ m.regs.beta = m.last.beta

private def Invariant (query : Query) (abort : Nat → Bool) (m : Machine) : Prop :=
  match m.pc with
  | 0 | 1 | 2 | 9 | 11 => True
  | 3 | 4 => Fresh query m
  | 5 => Fresh query m ∧ m.flag = abort m.last.index
  | 6 => m.last.Observed query ∧ abort m.last.index = true
  | 7 => Fresh query m ∧ abort m.last.index = false
  | 8 => m.last.Observed query ∧ abort m.last.index = false ∧
      (m.flag = false → m.last.alpha.toInt < m.last.score.toInt ∧ m.last.score.toInt < m.last.beta.toInt)
  | 10 => CallSafe query abort m.last
  | 12 => CallSafe query abort m.last ∧ m.out = m.last.score
  | _ => False

private def Safe (query : Query) (abort : Nat → Bool) : Action → Prop
  | .step m => Invariant query abort m
  | .ret score last => Classified query abort score last

private theorem step_safe (m : Machine) (inv : Invariant query abort m)
    (h : execute query abort m = some action) : Safe query abort action := by
  obtain ⟨pc,regs,issued,last,flag,out⟩ := m
  match pc with
  | 0 | 1 | 2 | 3 | 4 | 9 | 10 | 11 =>
    simp [execute,driverCode] at h
    subst action
    simp [Safe,Invariant,Fresh,Call.Observed] at inv ⊢ <;> assumption
  | 5 =>
    obtain ⟨fresh,observed⟩ := inv
    cases flag with
    | false =>
      simp [execute,driverCode] at h
      subst action
      exact ⟨fresh,observed.symm⟩
    | true =>
      simp [execute,driverCode] at h
      subst action
      exact ⟨fresh.1,observed.symm⟩
  | 6 =>
    obtain ⟨observed,stopped⟩ := inv
    simp [execute,driverCode] at h
    subst action
    exact ⟨⟨observed,fun live => Bool.noConfusion (stopped.symm.trans live)⟩,rfl⟩
  | 7 =>
    obtain ⟨fresh,live⟩ := inv
    cases widened : applyWiden regs last.score with
    | none => simp [execute,driverCode,widened] at h
    | some pair =>
      obtain ⟨after,changed⟩ := pair
      simp [execute,driverCode,widened] at h
      subst action
      refine ⟨fresh.1,live,?_⟩
      intro same
      change changed = false at same
      subst changed
      obtain ⟨lo,hi⟩ := widen_false widened
      exact ⟨fresh.2.1 ▸ lo,fresh.2.2 ▸ hi⟩
  | 8 =>
    cases flag <;> simp [execute,driverCode] at h <;> subst action
    · exact ⟨inv.1,fun _ => inv.2.2 rfl⟩
    · trivial
  | 12 =>
    simp [execute,driverCode] at h
    subst action
    exact ⟨inv.1.1,inv.2,inv.1.2⟩
  | _ + 13 => contradiction

/-- Every normal return of the generated MIR is either a polled abort, with no
    numerical promise, or an accepted window. In both cases the returned word is
    exactly the last observed root word. -/
theorem returned_classified (run : Returns query abort m score last) (entry : m.pc = 0) :
    Classified query abort score last := by
  have propagate : Invariant query abort m → Classified query abort score last := by
    clear entry
    induction run with
    | done h => exact fun inv => step_safe _ inv h
    | next h _ ih => exact fun inv => ih (step_safe _ inv h)
  apply propagate
  simp [Invariant,entry]

private def afterPoll (query : Query) (m : Machine) : Machine :=
  {m with pc := 7, issued := m.issued+1, last := currentCall query m, flag := false}

private theorem reach_widen (entry : m.pc = 2) (live : abort m.issued = false)
    (rest : Returns query abort (afterPoll query m) score last) :
    Returns query abort m score last := by
  obtain ⟨pc,regs,issued,old,flag,out⟩ := m
  change pc = 2 at entry
  subst pc
  change abort issued = false at live
  let read : Machine := ⟨5,regs,issued+1,⟨issued,regs.alpha,regs.beta,query issued regs.alpha regs.beta⟩,false,out⟩
  refine .next rfl (.next rfl (.next (m' := read) ?_ (.next rfl rest)))
  simp [execute,driverCode,live,read]

/-- The generated abort branch returns without applying the widener. -/
theorem aborted (entry : m.pc = 2) (stop : abort m.issued = true) :
    Returns query abort m (currentCall query m).score (currentCall query m) := by
  obtain ⟨pc,regs,issued,old,flag,out⟩ := m
  change pc = 2 at entry
  subst pc
  change abort issued = true at stop
  let read : Machine := ⟨5,regs,issued+1,⟨issued,regs.alpha,regs.beta,query issued regs.alpha regs.beta⟩,true,out⟩
  refine .next rfl (.next rfl (.next (m' := read) ?_ (.next rfl (.next rfl (.done rfl)))))
  simp [execute,driverCode,stop,read]

theorem accept (entry : m.pc = 2) (live : abort m.issued = false)
    (step : applyWiden m.regs (currentCall query m).score = some (m.regs,false)) :
    Returns query abort m (currentCall query m).score (currentCall query m) := by
  apply reach_widen entry live
  refine .next (m' := {afterPoll query m with pc := 8, flag := false}) ?_ ?_
  · simp [execute,afterPoll,driverCode,step]
  · exact .next rfl (.next rfl (.done rfl))

theorem retry (entry : m.pc = 2) (live : abort m.issued = false)
    (step : applyWiden m.regs (currentCall query m).score = some (after,true))
    (rest : Returns query abort (nextTrial query m after) score last) :
    Returns query abort m score last := by
  apply reach_widen entry live
  refine .next (m' := {afterPoll query m with pc := 8, regs := after, flag := true}) ?_ ?_
  · simp [execute,afterPoll,driverCode,step]
  · exact .next rfl (.next rfl (.next rfl (.next rfl rest)))

/-- Finiteness is needed only for a root call which passes the abort check.
    Stopped calls may supply any 32-bit score, including an out-of-range sentinel. -/
def FiniteLive (query : Query) (abort : Nat → Bool) : Prop :=
  ∀ i a z, abort i = false → Aspiration.Window a.toInt z.toInt →
    -32767 < (query i a z).toInt ∧ (query i a z).toInt < 32767

private theorem loop_complete (finite : FiniteLive query abort) (fuel : Nat)
    (ready : Ready n m.regs) (entry : m.pc = 2) (index : m.issued = start+n)
    (budget : n+fuel=12) :
    ∃ score last, Returns query abort m score last ∧
      start ≤ last.index ∧ last.index ≤ start+11 ∧ Aspiration.Window last.alpha.toInt last.beta.toInt := by
  induction fuel generalizing n m with
  | zero => have := ready.limit; omega
  | succ fuel ih =>
    let call := currentCall query m
    by_cases stopped : abort m.issued = true
    · exact ⟨call.score,call,aborted entry stopped,by change start ≤ m.issued; omega,
        by change m.issued ≤ start+11; have := ready.limit; omega,ready.window⟩
    have live : abort m.issued = false := Bool.eq_false_iff.mpr stopped
    have bounded : -32767 < call.score.toInt ∧ call.score.toInt < 32767 :=
      finite _ _ _ live ready.window
    rcases ready_widen ready call.score bounded with stop | ⟨after,wide,valid⟩
    · exact ⟨call.score,call,accept entry live stop,by change start ≤ m.issued; omega,
        by change m.issued ≤ start+11; have := ready.limit; omega,ready.window⟩
    · obtain ⟨score,last,rest,bounds⟩ := ih (m := nextTrial query m after) valid rfl
        (by change m.issued+1 = start+(n+1); omega) (by omega)
      exact ⟨score,last,retry entry live wide rest,bounds⟩

/-- Arbitrary stop observations cannot exhaust the twelve-call bound: a stop
    returns immediately, and only finite live replies cause another widening. -/
theorem completes_in_twelve (finite : FiniteLive query abort) (m : Machine) (entry : m.pc = 0)
    (initial : Aspiration.Window m.regs.alpha.toInt m.regs.beta.toInt) :
    ∃ score last, Returns query abort m score last ∧
      m.issued ≤ last.index ∧ last.index ≤ m.issued+11 ∧
      Aspiration.Window last.alpha.toInt last.beta.toInt := by
  obtain ⟨pc,regs,issued,old,flag,out⟩ := m
  change pc = 0 at entry
  subst pc
  let next : Machine := ⟨2,{regs with delta := BitVec.ofInt 32 initialDelta},issued,old,flag,out⟩
  obtain ⟨score,last,ran,bounds⟩ := loop_complete finite 12 (m := next) (start := issued)
    (first_ready regs initial) rfl rfl rfl
  exact ⟨score,last,.next rfl (.next rfl ran),bounds⟩

def Contract : Prop :=
  (∀ query abort m score last, Returns query abort m score last → m.pc = 0 →
    Classified query abort score last) ∧
  (∀ query abort m, FiniteLive query abort → m.pc = 0 →
    Aspiration.Window m.regs.alpha.toInt m.regs.beta.toInt →
    ∃ score last, Returns query abort m score last ∧ Classified query abort score last ∧
      m.issued ≤ last.index ∧ last.index ≤ m.issued+11 ∧
      Aspiration.Window last.alpha.toInt last.beta.toInt)

theorem contract_proven : Contract := by
  constructor
  · intro query abort m score last ran entry
    exact returned_classified ran entry
  · intro query abort m finite entry initial
    obtain ⟨score,last,ran,bounds⟩ := completes_in_twelve finite m entry initial
    exact ⟨score,last,ran,returned_classified ran entry,bounds⟩

end NeuralReversi.AspirationMir.Polled
