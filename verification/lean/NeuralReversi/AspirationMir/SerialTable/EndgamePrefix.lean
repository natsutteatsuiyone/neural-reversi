module
public import NeuralReversi.AspirationMir.SerialTable.TimedMidgame
public import NeuralReversi.PvsParallel.SerialTable.EndgameInterrupted

@[expose] public section

namespace NeuralReversi.AspirationMir.SerialTable.EndgamePrefix

open PvsParallel
open EndgameSearch.CacheHistory (Pool)
open PvsParallel.SerialTable (Layout Memory)
open PvsParallel.SerialTable.EndgameInterrupted (Poll)
open PvsBoardTree (Position)
open TranspositionTable (Selectivity)
open Repeated (ReadyMemory)
open SelectiveLadder (Start FiniteTrial)
open SerialStack (State Visit)
open RootMoveStorage (Row)

/-- Complete this many levels and failed attempts before reading the supplied stop polls.
    Timing and target selectivity may end the request before the indicated point. -/
structure Point where
  levels : Fin 5
  retries : Nat
  poll : Poll

def Point.level (point : Point) : Selectivity := ⟨point.levels.val + 1, by have := point.levels.isLt; omega⟩

def normal (q : Repeated.Call mc) (p : Pool) (c : SerialScout.Config) (fc : Cutoffs.FlipConfig)
    (level : Selectivity) : SerialCache.Trial (Memory l) :=
  if level.val = 5 then search p c q.board (q.moves.order 5) fc else FullLadder.trial p c q.board q.moves fc level.val

private theorem normal_returns (q : Repeated.Call mc) (p : Pool) (c : SerialScout.Config) (fc : Cutoffs.FlipConfig)
    (point : Point) (addr : c.address = p.address) :
    FiniteTrial mc q.board (ReadyMemory (l := l) p mc fc) (normal q p c fc point.level) := by
  by_cases full : point.level.val = 5
  · intro s m good cover pos win
    obtain ⟨k,ans,after,ran,⟨snap,best,writes,att⟩,post,_⟩ := Repeated.exact_trial p c q.board (q.moves.order 5) fc addr s m good cover pos win
    exact ⟨k,ans,after,by simpa only [normal,full,ite_true] using ran,post,att.1.lower,att.1.upper,
      .of_eq ((writes.scores (PvsParallel.Root.roots_nodup cover)).1)⟩
  · have selected : point.level.val ∈ OuterMir.ladder.take 4 := by
      have bound := point.levels.isLt
      change point.levels.val + 1 ≠ 5 at full
      simp [OuterMir.ladder,Point.level]
      omega
    simpa only [normal,full,ite_false] using
      Repeated.earlier (l := l) p c q.board q.moves fc addr point.level.val selected

def retries (q : Repeated.Call mc) (p : Pool) (c : SerialScout.Config) (fc : Cutoffs.FlipConfig)
    (point : Point) : Nat → State → Memory l → Option (Memory l)
  | 0, s, m =>
    let op := q.moves.order point.level.val s
    PvsParallel.SerialTable.EndgameInterrupted.trial p c q.board (op.base (fc := fc)) op.root point.level
      (q.moves.screen point.level.val s) point.poll s m
  | n+1, s, m => do
    let (k,ans,after) ← normal q p c fc point.level s m
    let visit : Visit := ⟨s,k,ans⟩
    let (regs,again) ← applyWiden s.regs visit.call.score
    if again then retries q p c fc point n (SerialStack.next visit regs) after else return after

private theorem retries_return (q : Repeated.Call mc) (p : Pool) (c : SerialScout.Config) (fc : Cutoffs.FlipConfig)
    (point : Point) (count : Nat) (s : State) (m : Memory l) (addr : c.address = p.address) (good : ReadyMemory p mc fc m)
    (cover : PvsParallel.Root.Covers mc q.board (s.rows.map Row.square)) (pos : 0 < s.rows.length) (window : Ready index s.regs) :
    ∃ after, retries q p c fc point count s m = some after ∧ ReadyMemory p mc fc after := by
  induction count generalizing s m index with
  | zero =>
    let op := q.moves.order point.level.val s
    exact PvsParallel.SerialTable.EndgameInterrupted.trial_returns p c q.board (op.base (fc := fc)) op.root
      point.level (q.moves.screen point.level.val s) point.poll s m addr good cover pos window.window
  | succ count ih =>
    obtain ⟨k,ans,after,ran,safe,lo,hi,squares⟩ := normal_returns q p c fc point addr s m good cover pos window.window
    have signed : (BitVec.ofInt 32 ans.value).toInt = ans.value := BitVec.toInt_ofInt_eq_self (by decide) (by omega) (by omega)
    have finite : -32767 < (BitVec.ofInt 32 ans.value).toInt ∧ (BitVec.ofInt 32 ans.value).toInt < 32767 :=
      by rw [signed]; exact ⟨lo,hi⟩
    rcases ready_widen window _ finite with done_ | ⟨regs,more,valid⟩
    · exact ⟨after,by simp [retries,ran,Visit.call,done_],safe⟩
    · have length : ans.rows.length = s.rows.length := by simpa using squares.length_eq
      obtain ⟨out,next,g⟩ := ih (SerialStack.next ⟨s,k,ans⟩ regs) after safe (squares.trans cover) (length ▸ pos) valid
      exact ⟨out,by simpa [retries,ran,Visit.call,more] using next,g⟩

/-- Keep the very same root rows, TT and local caches through all earlier levels and retries. -/
def finish (q : Repeated.Call mc) (p : Pool) (c : SerialScout.Config) (fc : Cutoffs.FlipConfig)
    (point : Point) (start : Start (Memory l)) : Option (Memory l) := do
  if q.last.val < point.level.val then return (← Repeated.finish p c fc q start).2
  let levels := (OuterMir.ladder.take 4).take point.levels.val
  let active ← levels.foldlM (SelectiveLadder.advance (FullLadder.trial p c q.board q.moves fc)) start
  retries q p c fc point point.retries (SerialStack.first active.machine active.rows) active.memory

theorem finish_returns (q : Repeated.Call mc) (p : Pool) (c : SerialScout.Config) (fc : Cutoffs.FlipConfig)
    (point : Point) (start : Start (Memory l)) (addr : c.address = p.address) (ok : start.valid mc q.board (ReadyMemory p mc fc)) :
    ∃ m, finish q p c fc point start = some m ∧ ReadyMemory p mc fc m := by
  by_cases ended : q.last.val < point.level.val
  · obtain ⟨r,m,ran,post,_⟩ := Repeated.finish_returns p c fc q start addr ok
    exact ⟨m,by simp [finish,ended,ran],post⟩
  · obtain ⟨active,ran,_,cover,pos,valid,_,win⟩ := SelectiveLadder.ladder ((OuterMir.ladder.take 4).take point.levels.val) start ok
      (fun i mem => Repeated.earlier p c q.board q.moves fc addr i (List.mem_of_mem_take mem))
    obtain ⟨out,steps,good⟩ := retries_return q p c fc point point.retries (SerialStack.first active.machine active.rows)
      active.memory addr valid cover pos (first_ready _ win)
    exact ⟨out,by simpa [finish,ended,ran] using steps,good⟩

def direct (q : Repeated.Call mc) (p : Pool) (c : SerialScout.Config) (fc : Cutoffs.FlipConfig)
    (point : Point) (m : Memory l) : Option (Memory l) := do
  let rows ← NativeRootBirth.started mc fc q.board c.paired q.scratch
  let prior := Repeated.increment m
  let base := InitialEstimate.estimate prior.tt.image mc fc q.raw q.board
  finish q p c fc point ⟨rows,SerialEntry.firstMachine base,prior⟩

theorem direct_returns (q : Repeated.Call mc) (p : Pool) (c : SerialScout.Config) (fc : Cutoffs.FlipConfig)
    (point : Point) (m : Memory l) (addr : c.address = p.address) (ok : ReadyMemory p mc fc m)
    (checked : Std.Tactic.BVDecide.LRAT.check certificate (LegalMoves.MoveBound.dimacsFormula 35) = true)
    (mobile : Board.hasLegalMoves mc q.board.val = true) :
    ∃ after, direct q p c fc point m = some after ∧ ReadyMemory p mc fc after := by
  have birth := started_of_certificate mc fc q.board c.paired q.scratch checked mobile
  obtain ⟨cover,pos⟩ := NativeRootBirth.covers birth
  let prior := Repeated.increment m
  have ready : ReadyMemory p mc fc prior := Repeated.begin_valid ok
  have bounds := InitialEstimate.warm_bounded prior.tt mc fc q.raw q.board prior.supported ready.2
  let base := InitialEstimate.estimate prior.tt.image mc fc q.raw q.board
  simpa only [direct,birth,bind,Option.bind] using finish_returns q p c fc point
    ⟨NativeRootBirth.initial mc q.board,SerialEntry.firstMachine base,prior⟩ addr
    ⟨cover,List.length_pos_iff.mpr pos,ready,rfl,SerialEntry.firstMachine_window base bounds.1 bounds.2⟩

def timed (q : Repeated.Call mc) (p : Pool) (c : SerialScout.Config) (fc : Cutoffs.FlipConfig)
    (level : Selectivity) (ops : TimedMidgame.Inputs l mc fc) (requested : Nat)
    (clock : Option (MidgameIterate.Clock (Memory l))) (point : Point) (m : Memory l) : Option (Memory l) := do
  match ← TimedMidgame.prepare q p c fc level ops requested clock m with
  | .done s => return s.memory
  | .endgame h =>
    finish {q with last := TimedMidgame.completedLevel clock} p c fc point ⟨h.state.rows,SerialEntry.firstMachine (BitVec.ofInt 32 h.base.score),h.state.memory⟩

theorem timed_returns (q : Repeated.Call mc) (p : Pool) (c : SerialScout.Config) (fc : Cutoffs.FlipConfig)
    (level : Selectivity) (ops : TimedMidgame.Inputs l mc fc) (requested : Nat)
    (clock : Option (MidgameIterate.Clock (Memory l))) (point : Point) (m : Memory l)
    (addr : c.address = p.address) (ok : ReadyMemory p mc fc m)
    (checked : Std.Tactic.BVDecide.LRAT.check certificate (LegalMoves.MoveBound.dimacsFormula 35) = true)
    (mobile : Board.hasLegalMoves mc q.board.val = true) :
    ∃ after, timed q p c fc level ops requested clock point m = some after ∧ ReadyMemory p mc fc after := by
  obtain ⟨out,ran,post⟩ := TimedMidgame.prepare_returns q p c fc level ops requested clock m ok checked mobile
  cases out with
  | done s => exact ⟨s.memory,by simp [timed,ran],post.2.2.2⟩
  | endgame h =>
    obtain ⟨valid,_,_,_,_,bound⟩ := post
    let word := BitVec.ofInt 32 h.base.score
    have fits : word.toInt = h.base.score := BitVec.toInt_ofInt_eq_self (by decide)
      (by unfold PvsMoveLoop.scoreMax at bound; omega) (by unfold PvsMoveLoop.scoreMax at bound; omega)
    simpa only [timed,ran,bind,Option.bind] using finish_returns {q with last := TimedMidgame.completedLevel clock} p c fc point
      ⟨h.state.rows,SerialEntry.firstMachine word,h.state.memory⟩ addr
      ⟨valid.1,valid.2.1,valid.2.2.2,rfl,SerialEntry.firstMachine_window word (fits ▸ bound.1) (fits ▸ bound.2)⟩

end NeuralReversi.AspirationMir.SerialTable.EndgamePrefix
