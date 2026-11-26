import NeuralReversi.AspirationMir.SerialTable.MidgamePolledRoot
import NeuralReversi.AspirationMir.SerialTable.TimedMidgame

namespace NeuralReversi.AspirationMir.SerialTable.MidgameInterrupted

open EndgameSearch.CacheHistory (Pool)
open PvsParallel
open PvsParallel.SerialTable (Layout Memory)
open PvsBoardTree (Position)
open RootMoveStorage (Row)
open MidgameAspiration (Attempt)
open TranspositionTable (Selectivity)
open MidgameIterate (State Valid)
open Repeated (ReadyMemory)

structure Point where
  depths : Nat
  retries : Nat
  rootStop : SerialCalls.Request → Bool

private def active (p : Pool) (b : Position) (paired : Bool) (level : Selectivity) (ops : TimedMidgame.Inputs l mc fc)
    (point : Point) (d : Nat) (s : State (Memory l)) (w : Attempt) : Option (Memory l) :=
  let f := ops d s w
  MidgamePolledRoot.run p b (BitVec.ofNat 32 d) paired f.scratch f.order level f.children point.rootStop
    s.rows (BitVec.ofInt 32 w.alpha) (BitVec.ofInt 32 w.beta) s.memory

private theorem active_returns (p : Pool) (b : Position) (paired : Bool) (level : Selectivity) (ops : TimedMidgame.Inputs l mc fc)
    (point : Point) (d : Nat) (s : State (Memory l)) (w : Attempt) (ok : Valid mc b (ReadyMemory p mc fc) s)
    (win : w.window) : ∃ m, active p b paired level ops point d s w = some m ∧ ReadyMemory p mc fc m := by
  have ⟨a,az,z⟩ := win
  have signed {x : Int} (lo : -32767 ≤ x) (hi : x ≤ 32767) : (BitVec.ofInt 32 x).toInt = x :=
    BitVec.toInt_ofInt_eq_self (by decide) (by omega) (by omega)
  exact MidgamePolledRoot.returns p b _ _ _ _ _ _ _ _ _ _ _ ok.2.2.2 ok.1 ok.2.1
    (by rw [signed a (by omega),signed (by omega) z]; exact win)

/-- Stop inside the chosen root attempt, or after the root's last store if the retry
    loop has already accepted. Failed attempts thread their unsorted rows and memory. -/
private def retries (p : Pool) (b : Position) (paired : Bool) (level : Selectivity)
    (ops observed : TimedMidgame.Inputs l mc fc) (point : Point) (d : Nat) :
    Nat → State (Memory l) → Attempt → Option (Memory l)
  | 0, s, w => active p b paired level observed point d s w
  | n+1, s, w => do
    let (ans,m) ← TimedMidgame.trial p b paired level ops d s w
    match w.retry ans.value with
    | none => return m
    | some next => retries p b paired level ops observed point d n ⟨ans.rows,m⟩ next

private theorem retries_return (p : Pool) (b : Position) (paired : Bool) (level : Selectivity)
    (ops observed : TimedMidgame.Inputs l mc fc) (point : Point) (d n : Nat) (s : State (Memory l))
    (w : Attempt) (ok : Valid mc b (ReadyMemory p mc fc) s) (win : w.window) (pos : 0 < w.delta) :
    ∃ m, retries p b paired level ops observed point d n s w = some m ∧ ReadyMemory p mc fc m := by
  induction n generalizing s w with
  | zero => exact active_returns p b paired level observed point d s w ok win
  | succ n ih =>
    obtain ⟨ans,m,frame,after,finite,_⟩ := TimedMidgame.trial_works p b paired level ops d s w ok win
    cases wide : w.retry ans.value with
    | none => exact ⟨m,by simp [retries,frame,wide],after.2.2.2⟩
    | some next =>
      obtain ⟨window,width⟩ := MidgameAspiration.retry_window win finite pos wide
      obtain ⟨post,run,ready⟩ := ih _ _ after window (by rw [width]; unfold MidgameAspiration.grow; omega)
      exact ⟨post,by simpa [retries,frame,wide] using run,ready⟩

private def depths (q : Repeated.Call mc) (p : Pool) (paired : Bool) (level : Selectivity)
    (ops observed : TimedMidgame.Inputs l mc fc) (point : Point) (target : Nat) :
    Nat → Nat → State (Memory l) → Option (Memory l)
  | 0, d, s =>
    if target < d then some s.memory else
      retries p q.board paired level ops observed point d point.retries s (MidgameAspiration.initial d (s.rows.head?.map MidgameRows.stats) 0)
  | n+1, d, s => do
    if target < d then return s.memory
    let it ← MidgameIterate.iteration (TimedMidgame.trial p q.board paired level ops) q.sort level.val d s
    depths q p paired level ops observed point target n (MidgameIterate.nextDepth d) it.state

private theorem depths_return (q : Repeated.Call mc) (p : Pool) (paired : Bool) (level : Selectivity)
    (ops observed : TimedMidgame.Inputs l mc fc) (point : Point) (target n d : Nat) (s : State (Memory l))
    (ok : Valid mc q.board (ReadyMemory p mc fc) s) :
    ∃ after, depths q p paired level ops observed point target n d s = some after ∧ ReadyMemory p mc fc after := by
  induction n generalizing s d with
  | zero =>
    by_cases past : target < d
    · exact ⟨s.memory,by simp [depths,past],ok.2.2.2⟩
    have start := MidgameAspiration.initialContract_proven d 0 (s.rows.head?.map MidgameRows.stats) (by
      intro hist member
      obtain ⟨r,head,rfl⟩ := Option.map_eq_some_iff.mp member
      exact ok.2.2.1 r (List.mem_of_mem_head? head))
    simpa only [depths,ite_eq_right past] using retries_return p q.board paired level ops observed point d point.retries s _ ok start.1
      (Int.lt_of_lt_of_le (by decide : (0 : Int) < 512) start.2.1)
  | succ n ih =>
    by_cases past : target < d
    · exact ⟨s.memory,by simp [depths,past],ok.2.2.2⟩
    obtain ⟨it,ran,complete⟩ := MidgameIterate.iteration_returns (TimedMidgame.trial_works p q.board paired level ops)
      s ok q.sort level.val d
    obtain ⟨after,rest,safe⟩ := ih _ it.state complete.valid
    exact ⟨after,by simpa [depths,past,ran] using rest,safe⟩

/-- An earlier search stopped during deepening. Only the stop location and the executed
    child branches are supplied; every completed depth, retry, root row and table state is computed. -/
def run (q : Repeated.Call mc) (p : Pool) (c : SerialScout.Config) (fc : Cutoffs.FlipConfig)
    (level : Selectivity) (ops observed : TimedMidgame.Inputs l mc fc) (requested : Nat) (point : Point)
    (m : Memory l) : Option (Memory l) := do
  let rows ← NativeRootBirth.started mc fc q.board c.paired q.scratch
  if q.board.val = Board.newBoard then return Repeated.increment m
  let target := min (max requested 1) (Board.emptyCount q.board.val).toNat
  depths q p c.paired level ops observed point target point.depths (MidgameIterate.startDepth target) ⟨rows,Repeated.increment m⟩

theorem returns (q : Repeated.Call mc) (p : Pool) (c : SerialScout.Config) (fc : Cutoffs.FlipConfig)
    (level : Selectivity) (ops observed : TimedMidgame.Inputs l mc fc) (requested : Nat) (point : Point) (m : Memory l)
    (ok : ReadyMemory p mc fc m)
    (checked : Std.Tactic.BVDecide.LRAT.check certificate (LegalMoves.MoveBound.dimacsFormula 35) = true)
    (mobile : Board.hasLegalMoves mc q.board.val = true) :
    ∃ after, run q p c fc level ops observed requested point m = some after ∧ ReadyMemory p mc fc after := by
  have born := started_of_certificate mc fc q.board c.paired q.scratch checked mobile
  by_cases opening : q.board.val = Board.newBoard
  · exact ⟨_,by simp [run,born,opening],Repeated.begin_valid ok⟩
  obtain ⟨cover,pos⟩ := NativeRootBirth.covers born
  have hist : MidgameRows.Valid (NativeRootBirth.initial mc q.board) := by
    intro r mem
    obtain ⟨_,_,rfl⟩ := List.mem_map.mp mem
    exact Or.inl rfl
  simpa only [run,born,bind,Option.bind,ite_eq_right opening] using depths_return q p c.paired level ops observed point _ point.depths _
    ⟨NativeRootBirth.initial mc q.board,Repeated.increment m⟩ ⟨cover,List.length_pos_iff.mpr pos,hist,Repeated.begin_valid ok⟩

end NeuralReversi.AspirationMir.SerialTable.MidgameInterrupted
