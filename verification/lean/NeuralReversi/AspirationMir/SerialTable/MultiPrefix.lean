import NeuralReversi.AspirationMir.SerialTable.MultiOrdinary
import NeuralReversi.PvsParallel.SerialTable.EndgameInterrupted

namespace NeuralReversi.AspirationMir.SerialTable.MultiPrefix

open PvsParallel
open EndgameSearch.CacheHistory (Pool)
open PvsParallel.SerialTable (Memory)
open PvsParallel.SerialTable.EndgameInterrupted (Poll)
open SelectiveLadder (Start)
open Repeated (ReadyMemory)
open RootMoveStorage (Row)
open SerialStack (State Visit)
open PvsMoveLoop (scoreMax)

structure Point where
  line : Nat
  stage : Nat
  retries : Nat
  poll : Poll

private def retries (idx : Nat) (q : Repeated.Call mc) (p : Pool) (c : SerialScout.Config)
    (fc : Cutoffs.FlipConfig) (ops : FullLadder.Inputs mc) (level : Nat) (poll : Poll) :
    Nat → State → Memory l → Option (Memory l)
  | 0, s, m =>
    let f := ops.order level s
    PvsParallel.SerialTable.EndgameInterrupted.trialAt idx p c q.board (f.base (fc := fc)) f.root
      ⟨min level 5,by omega⟩ (ops.screen level s) poll s m
  | n+1, s, m => do
    let (k,ans,after) ← MultiLevel.trial idx p c q.board ops fc level s m
    let v : Visit := ⟨s,k,ans⟩
    let (regs,again) ← applyWiden s.regs v.call.score
    if again then retries idx q p c fc ops level poll n (SerialStack.next v regs) after else return after

private theorem retries_return (idx : Nat) (q : Repeated.Call mc) (p : Pool) (c : SerialScout.Config)
    (fc : Cutoffs.FlipConfig) (ops : FullLadder.Inputs mc) (level : Nat) (poll : Poll)
    (count : Nat) (s : State) (m : Memory l) (addr : c.address = p.address) (ready : ReadyMemory p mc fc m)
    (cover : Root.Covers mc q.board (s.rows.map Row.square)) (index : idx < s.rows.length) (win : Ready n s.regs) :
    ∃ next, retries idx q p c fc ops level poll count s m = some next ∧ ReadyMemory p mc fc next := by
  induction count generalizing s m n with
  | zero =>
    let f := ops.order level s
    exact PvsParallel.SerialTable.EndgameInterrupted.trialAt_returns idx p c q.board (f.base (fc := fc)) f.root
      ⟨min level 5,by omega⟩ (ops.screen level s) poll s m addr ready cover index win.window
  | succ count ih =>
    obtain ⟨k,ans,after,ran,good,⟨lo,hi⟩,keys,_⟩ := MultiLevel.returns idx p c q.board ops fc level s m
      addr ready cover index win.window
    have repr : (BitVec.ofInt 32 ans.value).toInt = ans.value :=
      BitVec.toInt_ofInt_eq_self (by decide) (by omega) (by omega)
    have finite : -32767 < (BitVec.ofInt 32 ans.value).toInt ∧ (BitVec.ofInt 32 ans.value).toInt < 32767 :=
      by rw [repr]; exact ⟨lo,hi⟩
    rcases ready_widen win _ finite with done_ | ⟨regs,wide,nextWin⟩
    · exact ⟨after,by simp [retries,ran,Visit.call,done_],good⟩
    · have size : ans.rows.length = s.rows.length := by simpa using congrArg List.length keys
      obtain ⟨out,rest,g⟩ := ih (SerialStack.next ⟨s,k,ans⟩ regs) after good (keys ▸ cover) (size ▸ index) nextWin
      exact ⟨out,by simpa [retries,ran,Visit.call,wide] using rest,g⟩

/-- Execute complete lines, then complete stages of the requested line. An out-of-range
    position stops at the end of the corresponding loop. -/
def finish (q : Repeated.Call mc) (p : Pool) (c : SerialScout.Config) (fc : Cutoffs.FlipConfig)
    (ops : MultiBatch.Inputs mc) (levels : Nat → List Nat) (base : Word) (rows : List Row)
    (m : Memory l) (point : Point) : Option (Memory l) := do
  let (before,prior) ← MultiBatch.lines (MultiBatch.trials p c q.board ops fc) levels base
    (min point.line rows.length) 0 rows m
  if rows.length ≤ point.line then return prior
  let idx := point.line
  let start : Start _ := ⟨before,MultiLine.seed idx base before,prior⟩
  let schedule := levels idx
  let active ← (schedule.take point.stage).foldlM (MultiLevel.advance idx (MultiBatch.trials p c q.board ops fc idx)) start
  match schedule[point.stage]? with
  | none => return active.memory
  | some level => retries idx q p c fc (ops idx) level point.poll point.retries (SerialStack.first active.machine active.rows) active.memory


theorem finish_returns (q : Repeated.Call mc) (p : Pool) (c : SerialScout.Config) (fc : Cutoffs.FlipConfig)
    (ops : MultiBatch.Inputs mc) (levels : Nat → List Nat) (base : Word) (rows : List Row)
    (m : Memory l) (point : Point) (addr : c.address = p.address) (ready : ReadyMemory p mc fc m)
    (cover : Root.Covers mc q.board (rows.map Row.square)) (range : -scoreMax ≤ base.toInt ∧ base.toInt ≤ scoreMax)
    (nonempty : ∀ i, i < rows.length → levels i ≠ []) :
    ∃ out, finish q p c fc ops levels base rows m point = some out ∧ ReadyMemory p mc fc out := by
  let count := min point.line rows.length
  obtain ⟨before,prior,ran,safe,covers,size,prev⟩ := MultiOrdinary.lines_returns p c q.board ops fc levels base
    count 0 rows m addr ready cover (by dsimp [count]; omega) range (by simp [MultiLine.previous])
    (fun i _ small => nonempty i (by dsimp [count] at small; omega))
  dsimp only [count] at ran
  by_cases done_ : rows.length ≤ point.line
  · exact ⟨prior,by simp only [finish,ran,bind,Option.bind,ite_eq_left done_,pure,Pure.pure],safe⟩
  · have inside : point.line < before.length := by omega
    have previous : ∀ v ∈ MultiLine.previous point.line before, -32767 ≤ v := by
      simpa [count,Nat.min_eq_left (Nat.le_of_lt (Nat.lt_of_not_ge done_))] using prev
    let start : Start _ := ⟨before,MultiLine.seed point.line base before,prior⟩
    have valid : MultiLevel.Valid mc q.board point.line (ReadyMemory p mc fc) start :=
      ⟨covers,inside,safe,rfl,MultiOrdinary.seed_window point.line base before range previous⟩
    obtain ⟨active,fold,ok,_⟩ := MultiLine.stages point.line p c q.board (ops point.line) fc
      ((levels point.line).take point.stage) start addr valid
    dsimp only [start] at fold
    cases stage : (levels point.line)[point.stage]? with
    | none => exact ⟨active.memory,by simp only [finish,ran,bind,Option.bind,ite_eq_right done_,MultiBatch.trials,fold,stage,pure,Pure.pure],ok.safe⟩
    | some level =>
      obtain ⟨out,tried,g⟩ := retries_return point.line q p c fc (ops point.line) level point.poll point.retries
        (SerialStack.first active.machine active.rows) active.memory addr ok.safe ok.cover ok.index (first_ready _ ok.window)
      exact ⟨out,by simp only [finish,ran,bind,Option.bind,ite_eq_right done_,MultiBatch.trials,fold,stage,tried],g⟩

end NeuralReversi.AspirationMir.SerialTable.MultiPrefix
