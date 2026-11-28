module
public import NeuralReversi.AspirationMir.SerialTable.MultiMidgame

@[expose] public section

namespace NeuralReversi.AspirationMir.SerialTable.MultiMidgamePrefix

open EndgameSearch.CacheHistory (Pool)
open PvsParallel
open PvsParallel.SerialTable (Memory)
open MidgameIterate (State Valid)
open MultiMidgame (Inputs)
open Repeated (ReadyMemory)
open TranspositionTable (Selectivity)
open MidgameAspiration (Attempt)

structure Point where
  depths : Nat
  lines : Nat
  retries : Nat
  rootStop : SerialCalls.Request → Bool

def active (q : Repeated.Call mc) (p : Pool) (c : SerialScout.Config) (fc : Cutoffs.FlipConfig)
    (level : Selectivity) (ops : Inputs l) (point : Point) (depth idx : Nat) (s : State (Memory l)) (w : Attempt) :=
  let f := MidgameSession.frames (mc := mc) (fc := fc) (ops idx) c level q.board true depth s w
  MidgamePolledRoot.runAt idx p q.board (BitVec.ofNat 32 depth) c.paired f.scratch f.order level f.children point.rootStop
    s.rows (BitVec.ofInt 32 w.alpha) (BitVec.ofInt 32 w.beta) s.memory

def retry (q : Repeated.Call mc) (p : Pool) (c : SerialScout.Config) (fc : Cutoffs.FlipConfig)
    (level : Selectivity) (ops : Inputs l) (point : Point) (depth idx : Nat) : Nat → State (Memory l) → Attempt → Option (Memory l)
  | 0, s, w => active q p c fc level ops point depth idx s w
  | n+1, s, w => do
    let (ans,after) ← MultiMidgame.trial (mc := mc) (fc := fc) idx p c q.board level ops depth s w
    match w.retry ans.value with
    | none => return after
    | some next => retry q p c fc level ops point depth idx n ⟨ans.rows,after⟩ next

private theorem retry_returns (q : Repeated.Call mc) (p : Pool) (c : SerialScout.Config) (fc : Cutoffs.FlipConfig)
    (level : Selectivity) (ops : Inputs l) (point : Point) (depth idx n : Nat) (s : State (Memory l)) (w : Attempt)
    (ok : Valid mc q.board (ReadyMemory p mc fc) s) (within : idx < s.rows.length) (win : w.window) (positive : 0 < w.delta) :
    ∃ m, retry q p c fc level ops point depth idx n s w = some m ∧ ReadyMemory p mc fc m := by
  induction n generalizing s w with
  | zero =>
    have ⟨lo,_,hi⟩ := win
    have repr {x : Int} (l : -32767 ≤ x) (h : x ≤ 32767) : (BitVec.ofInt 32 x).toInt = x :=
      BitVec.toInt_ofInt_eq_self (by decide) (by omega) (by omega)
    exact MidgamePolledRoot.returnsAt idx p q.board _ _ _ _ _ _ _ _ _ _ _ ok.2.2.2 ok.1 within
      (by rw [repr lo (by omega),repr (by omega) hi]; exact win)
  | succ n ih =>
    obtain ⟨ans,m,ran,safe,len,_,finite,_⟩ := MultiMidgame.works (mc := mc) (fc := fc) idx p c q.board level ops depth s w ok within win
    cases widen : w.retry ans.value with
    | none => exact ⟨m,by simp [retry,ran,widen],safe.2.2.2⟩
    | some next =>
      obtain ⟨window,width⟩ := MidgameAspiration.retry_window win finite positive widen
      obtain ⟨out,more,good⟩ := ih _ next safe (len ▸ within) window (by rw [width]; unfold MidgameAspiration.grow; omega)
      exact ⟨out,by simpa [retry,ran,widen] using more,good⟩

def lines (q : Repeated.Call mc) (p : Pool) (c : SerialScout.Config) (fc : Cutoffs.FlipConfig)
    (level : Selectivity) (ops : Inputs l) (point : Point) (depth : Nat) : Nat → Nat → State (Memory l) → Option (Memory l)
  | 0, idx, s => if idx < s.rows.length then
      retry q p c fc level ops point depth idx point.retries s (MidgameAspiration.initial depth (s.rows[idx]?.map MidgameRows.stats) 0)
    else some s.memory
  | n+1, idx, s => do
    if s.rows.length ≤ idx then return s.memory
    let next ← MultiMidgame.line (MultiMidgame.trial (mc := mc) (fc := fc) idx p c q.board level ops) depth idx s
    lines q p c fc level ops point depth n (idx+1) next

private theorem lines_return (q : Repeated.Call mc) (p : Pool) (c : SerialScout.Config) (fc : Cutoffs.FlipConfig)
    (level : Selectivity) (ops : Inputs l) (point : Point) (depth count idx : Nat) (s : State (Memory l)) (ok : Valid mc q.board (ReadyMemory p mc fc) s) :
    ∃ m, lines q p c fc level ops point depth count idx s = some m ∧ ReadyMemory p mc fc m := by
  induction count generalizing idx s with
  | zero =>
    by_cases within : idx < s.rows.length
    · let hist := s.rows[idx]?.map MidgameRows.stats
      have init := MidgameAspiration.initialContract_proven depth 0 hist (by
        intro h mem
        obtain ⟨r,atIdx,rfl⟩ := Option.map_eq_some_iff.mp mem
        exact ok.2.2.1 r (List.mem_of_getElem? atIdx))
      simpa only [lines,ite_eq_left within] using retry_returns q p c fc level ops point depth idx point.retries s _ ok within init.1
        (Int.lt_of_lt_of_le (by decide : (0:Int) < 512) init.2.1)
    · exact ⟨s.memory,by simp [lines,within],ok.2.2.2⟩
  | succ n ih =>
    by_cases past : s.rows.length ≤ idx
    · exact ⟨s.memory,by simp [lines,past],ok.2.2.2⟩
    obtain ⟨next,ran,good,_⟩ := MultiMidgame.line_returns (MultiMidgame.works (mc := mc) (fc := fc) idx p c q.board level ops) depth s ok (by omega)
    obtain ⟨m,rest,safe⟩ := ih (idx+1) next good
    exact ⟨m,by simpa [lines,past,ran] using rest,safe⟩

def depths (q : Repeated.Call mc) (p : Pool) (c : SerialScout.Config) (fc : Cutoffs.FlipConfig)
    (level : Selectivity) (ops : Inputs l) (point : Point) (target : Nat) : Nat → Nat → State (Memory l) → Option (Memory l)
  | 0, d, s => if target < d then some s.memory else lines q p c fc level ops point d point.lines 0 s
  | n+1, d, s => do
    if target < d then return s.memory
    let it ← MultiMidgame.iteration (fun idx => MultiMidgame.trial (mc := mc) (fc := fc) idx p c q.board level ops) q.sort level.val d s
    depths q p c fc level ops point target n (MidgameIterate.nextDepth d) it.state

private theorem depths_return (q : Repeated.Call mc) (p : Pool) (c : SerialScout.Config) (fc : Cutoffs.FlipConfig)
    (level : Selectivity) (ops : Inputs l) (point : Point) (target count d : Nat) (s : State (Memory l))
    (ok : Valid mc q.board (ReadyMemory p mc fc) s) :
    ∃ m, depths q p c fc level ops point target count d s = some m ∧ ReadyMemory p mc fc m := by
  induction count generalizing d s with
  | zero =>
    by_cases past : target < d
    · exact ⟨s.memory,by simp [depths,past],ok.2.2.2⟩
    · simpa only [depths,ite_eq_right past] using lines_return q p c fc level ops point d point.lines 0 s ok
  | succ n ih =>
    by_cases past : target < d
    · exact ⟨s.memory,by simp [depths,past],ok.2.2.2⟩
    obtain ⟨out,ran,good⟩ := MultiMidgame.iteration_works
      (fun idx => MultiMidgame.works (mc := mc) (fc := fc) idx p c q.board level ops) q.sort level.val d s ok
    obtain ⟨m,rest,safe⟩ := ih (MidgameIterate.nextDepth d) out.state good.valid
    exact ⟨m,by simpa [depths,past,ran] using rest,safe⟩

/-- Advance completed depths, lines and retries in their native order before polling inside a root attempt. -/
def run (q : Repeated.Call mc) (p : Pool) (c : SerialScout.Config) (fc : Cutoffs.FlipConfig)
    (level : Selectivity) (ops : Inputs l) (requested : Nat) (point : Point) (m : Memory l) : Option (Memory l) := do
  let rows ← NativeRootBirth.started mc fc q.board c.paired q.scratch
  let target := min (max requested 1) (Board.emptyCount q.board.val).toNat
  depths q p c fc level ops point target point.depths (MidgameIterate.startDepth target) ⟨rows,Repeated.increment m⟩

theorem returns (q : Repeated.Call mc) (p : Pool) (c : SerialScout.Config) (fc : Cutoffs.FlipConfig)
    (level : Selectivity) (ops : Inputs l) (requested : Nat) (point : Point) (m : Memory l) (ok : ReadyMemory p mc fc m)
    (checked : Std.Tactic.BVDecide.LRAT.check certificate (LegalMoves.MoveBound.dimacsFormula 35) = true)
    (mobile : Board.hasLegalMoves mc q.board.val = true) :
    ∃ after, run q p c fc level ops requested point m = some after ∧ ReadyMemory p mc fc after := by
  have born := started_of_certificate mc fc q.board c.paired q.scratch checked mobile
  simpa only [run,born,bind,Option.bind] using depths_return q p c fc level ops point _ point.depths _
    ⟨NativeRootBirth.initial mc q.board,Repeated.increment m⟩ (MultiMidgame.initial_valid q p c fc m ok born)

end NeuralReversi.AspirationMir.SerialTable.MultiMidgamePrefix
