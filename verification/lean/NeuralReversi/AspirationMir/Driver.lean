module
public import NeuralReversi.AspirationMir.Program

@[expose] public section

namespace NeuralReversi.AspirationMir

abbrev Word := BitVec 32

structure WindowRegs where
  alpha : Word
  beta : Word
  delta : Word

def Expr.eval (regs : WindowRegs) (s : Word) : Expr → Word
  | .alpha => regs.alpha
  | .beta => regs.beta
  | .delta => regs.delta
  | .score => s
  | .inf => BitVec.ofInt 32 infinity
  | .neg x => -(x.eval regs s)
  | .add x y => x.eval regs s + y.eval regs s
  | .sub x y => x.eval regs s - y.eval regs s
  | .max x y => if (x.eval regs s).toInt ≤ (y.eval regs s).toInt then y.eval regs s else x.eval regs s
  | .min x y => if (x.eval regs s).toInt ≤ (y.eval regs s).toInt then x.eval regs s else y.eval regs s

def Guard.eval (regs : WindowRegs) (s : Word) : Guard → Bool
  | .le x y => decide ((x.eval regs s).toInt ≤ (y.eval regs s).toInt)
  | .ge x y => decide ((y.eval regs s).toInt ≤ (x.eval regs s).toInt)

def applyWiden (regs : WindowRegs) (s : Word) : Option (WindowRegs × Bool) := do
  let (_, a, b, changed) ← widenCode.find? (fun path =>
    path.1.all fun (g, expected) => g.eval regs s == expected)
  return (⟨a.eval regs s, b.eval regs s, regs.delta⟩, changed)

theorem widen_false {regs after : WindowRegs} {s : Word}
    (h : applyWiden regs s = some (after, false)) :
    regs.alpha.toInt < s.toInt ∧ s.toInt < regs.beta.toInt := by
  by_cases low : s.toInt ≤ regs.alpha.toInt <;> by_cases high : regs.beta.toInt ≤ s.toInt <;>
    simp [applyWiden, widenCode, Guard.eval, Expr.eval, low, high, List.find?] at h <;> omega

structure Call where
  index : Nat
  alpha : Word
  beta : Word
  score : Word

abbrev Query := Nat → Word → Word → Word

def Call.Observed (call : Call) (query : Query) : Prop :=
  query call.index call.alpha call.beta = call.score

def Call.Accepted (call : Call) (query : Query) : Prop :=
  call.Observed query ∧ call.alpha.toInt < call.score.toInt ∧ call.score.toInt < call.beta.toInt

structure Machine where
  pc : Nat
  regs : WindowRegs
  issued : Nat
  last : Call
  flag : Bool
  out : Word

inductive Action where
  | step (next : Machine)
  | ret (score : Word) (last : Call)

def execute (query : Query) (abort : Nat → Bool) (m : Machine) : Option Action := do
  match ← driverCode[m.pc]? with
  | .prepare n => return .step { m with pc := n, regs.delta := BitVec.ofInt 32 initialDelta }
  | .go n => return .step { m with pc := n }
  | .query n =>
    let last := ⟨m.issued, m.regs.alpha, m.regs.beta, query m.issued m.regs.alpha m.regs.beta⟩
    return .step { m with pc := n, issued := m.issued + 1, last := last }
  | .poll n => return .step { m with pc := n, flag := abort m.last.index }
  | .branchAbort f t | .branchWiden f t => return .step { m with pc := if m.flag then t else f }
  | .widen n =>
    let (regs, changed) ← applyWiden m.regs m.last.score
    return .step { m with pc := n, regs := regs, flag := changed }
  | .double n => return .step { m with pc := n, regs.delta := m.regs.delta + m.regs.delta }
  | .saveScore n => return .step { m with pc := n, out := m.last.score }
  | .returned => return .ret m.out m.last
  | .unwind => none

inductive Returns (query : Query) (abort : Nat → Bool) : Machine → Word → Call → Prop
  | done : execute query abort m = some (.ret score last) → Returns query abort m score last
  | next : execute query abort m = some (.step m') →
      Returns query abort m' score last → Returns query abort m score last

private def Invariant (query : Query) (m : Machine) : Prop :=
  let fresh := m.last.Observed query ∧ m.regs.alpha = m.last.alpha ∧ m.regs.beta = m.last.beta
  match m.pc with
  | 0 | 1 | 2 | 9 | 11 => True
  | 3 | 4 | 7 => fresh
  | 5 => fresh ∧ m.flag = false
  | 6 => False
  | 8 => m.flag = false → m.last.Accepted query
  | 10 => m.last.Accepted query
  | 12 => m.last.Accepted query ∧ m.out = m.last.score
  | _ => False

private def Safe (query : Query) : Action → Prop
  | .step m => Invariant query m
  | .ret score last => last.Accepted query ∧ score = last.score

private theorem step_safe (m : Machine) (inv : Invariant query m)
    (h : execute query (fun _ => false) m = some action) : Safe query action := by
  obtain ⟨pc, regs, issued, last, flag, out⟩ := m
  match pc with
  | 0 | 1 | 2 | 3 | 4 | 6 | 9 | 10 | 11 | 12 =>
    simp [execute, driverCode] at h
    subst action
    simp [Safe, Invariant, Call.Observed] at inv ⊢ <;> assumption
  | 5 =>
    obtain ⟨fresh, rfl⟩ := inv
    simp [execute, driverCode] at h
    subst action
    exact fresh
  | 7 =>
    cases result : applyWiden regs last.score with
    | none => simp [execute, driverCode, result] at h
    | some p =>
      obtain ⟨after, changed⟩ := p
      simp [execute, driverCode, result] at h
      subst action
      intro same
      change changed = false at same
      subst changed
      obtain ⟨low, high⟩ := widen_false result
      obtain ⟨observed, a, b⟩ := inv
      exact ⟨observed, a ▸ low, b ▸ high⟩
  | 8 =>
    cases flag <;> simp [execute, driverCode] at h <;> subst action
    · exact inv rfl
    · trivial
  | _ + 13 => contradiction

theorem returned_inside (run : Returns query (fun _ => false) m score last) (entry : m.pc = 0) :
    last.Accepted query ∧ score = last.score := by
  have propagate : Invariant query m → last.Accepted query ∧ score = last.score := by
    clear entry
    induction run with
    | done h => exact fun inv => step_safe _ inv h
    | next h _ ih => exact fun inv => ih (step_safe _ inv h)
  apply propagate
  simp [Invariant, entry]

end NeuralReversi.AspirationMir
