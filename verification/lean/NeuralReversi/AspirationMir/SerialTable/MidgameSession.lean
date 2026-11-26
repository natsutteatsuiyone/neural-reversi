import NeuralReversi.AspirationMir.SerialTable.EndgamePrefix
import NeuralReversi.AspirationMir.SerialTable.MidgameInterrupted
import NeuralReversi.PvsParallel.SerialTable.MidgameNative
import NeuralReversi.PvsParallel.SerialTable.MidgameHistory

namespace NeuralReversi.AspirationMir.SerialTable.MidgameSession

open PvsParallel
open PvsParallel.SerialTable (Layout Memory)
open EndgameSearch.CacheHistory (Pool)
open TranspositionTable (Selectivity)
open PvsBoardTree (Position)
open Repeated (ReadyMemory)
open MidgameIterate (State Clock)
open PvsParallel.SerialTable.MidgameNative (Heuristics planFromRoot)

structure Frame where
  scratch : MoveList.Initialization.Memory Move.Move
  order : SerialRecursion.RootInput
  heuristic : SerialCalls.Request → Heuristics

abbrev Inputs (l : Layout) := Nat → State (Memory l) → MidgameAspiration.Attempt → Frame

/-- Each child uses the same root board, backend, and midgame selectivity.
    A previous interrupted trial can observe the actual stop flag following each move. -/
def frames (ops : Inputs l) (c : SerialScout.Config) (level : Selectivity) (b : Position) (poll : Bool) :
    TimedMidgame.Inputs l mc fc := fun d s w =>
  let f := ops d s w
  ⟨f.scratch,f.order,fun q =>
    let obs := f.heuristic q
    let hs := if poll then obs else {obs with stop := fun _ _ _ => false}
    planFromRoot {toHeuristics := hs,config := c,level := level} b q⟩

/-- Finite histories of the serial engine. Each constructor uses the returned physical state.
    A duplicated TT can be paired with independently retained worker caches. -/
inductive Generated (p : Pool) (c : SerialScout.Config) (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig)
    (additional : (l : Layout) → Memory l → Memory l → Prop) :
    {l : Layout} → Memory l → Prop where
  | empty {l : Layout} : Generated p c mc fc additional (Memory.empty l 0 p)
  | noLegalMove {m : Memory l} (prior : Generated p c mc fc additional m) (b : Position) (immobile : Board.hasLegalMoves mc b.val = false) :
      Generated p c mc fc additional (Repeated.increment m)
  | direct {m : Memory l} (prior : Generated p c mc fc additional m) (q : Repeated.Call mc) (mobile : Board.hasLegalMoves mc q.board.val = true)
      (ran : Repeated.run p c fc q m = some (r,next)) : Generated p c mc fc additional next
  | midgame {m : Memory l} (prior : Generated p c mc fc additional m) (q : Repeated.Call mc) (mobile : Board.hasLegalMoves mc q.board.val = true)
      (ops : Inputs l) (level : Selectivity) (depth : Nat) (clock : Option (Clock (Memory l)))
      (ran : TimedMidgame.run q p c fc level (frames ops c level q.board false) depth clock m = some (r,next)) :
      Generated p c mc fc additional next
  | interrupted {m : Memory l} (prior : Generated p c mc fc additional m) (q : Repeated.Call mc) (mobile : Board.hasLegalMoves mc q.board.val = true)
      (ops : Inputs l) (level : Selectivity) (depth : Nat) (point : MidgameInterrupted.Point)
      (ran : MidgameInterrupted.run q p c fc level (frames ops c level q.board false) (frames ops c level q.board true)
        depth point m = some next) : Generated p c mc fc additional next
  | directPrefix {m : Memory l} (prior : Generated p c mc fc additional m) (q : Repeated.Call mc)
      (mobile : Board.hasLegalMoves mc q.board.val = true) (point : EndgamePrefix.Point)
      (ran : EndgamePrefix.direct q p c fc point m = some next) : Generated p c mc fc additional next
  | handoffPrefix {m : Memory l} (prior : Generated p c mc fc additional m) (q : Repeated.Call mc)
      (mobile : Board.hasLegalMoves mc q.board.val = true) (ops : Inputs l) (level : Selectivity)
      (depth : Nat) (clock : Clock (Memory l)) (point : EndgamePrefix.Point)
      (ran : EndgamePrefix.timed q p c fc level (frames ops c level q.board false) depth (some clock) point m = some next) :
      Generated p c mc fc additional next
  | clear {m : Memory l} (prior : Generated p c mc fc additional m) (shared : Bool) : Generated p c mc fc additional (PvsParallel.SerialTable.MidgameHistory.clear m shared)
  | reset {m : Memory l} (prior : Generated p c mc fc additional m) (shared : Bool) : Generated p c mc fc additional (PvsParallel.SerialTable.MidgameHistory.reset m shared)
  | attach {a : Memory l} {b : Memory old} (src : Generated p c mc fc additional a) (dst : Generated p c mc fc additional b) :
      Generated p c mc fc additional (PvsParallel.SerialTable.MidgameHistory.attach a b)
  | duplicate {a b : Memory l} (src : Generated p c mc fc additional a) (dst : Generated p c mc fc additional b) (buffers : TranspositionTable.Duplicate.Buffers)
      (size : buffers.clusters = l.clusters.toNat) :
      Generated p c mc fc additional (PvsParallel.SerialTable.MidgameHistory.duplicate a b buffers size)
  | extra {a b : Memory l} (prior : Generated p c mc fc additional a) (step : additional l a b) :
      Generated p c mc fc additional b

abbrev History (p : Pool) (c : SerialScout.Config) (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig)
    {l : Layout} (m : Memory l) := Generated p c mc fc (fun _ _ _ => False) m

theorem Generated.ready (h : Generated p c mc fc additional m) (addr : c.address = p.address)
    (checked : Std.Tactic.BVDecide.LRAT.check certificate (LegalMoves.MoveBound.dimacsFormula 35) = true)
    (extraReady : ∀ {l} (a b : Memory l), additional l a b → ReadyMemory p mc fc a → ReadyMemory p mc fc b) :
    ReadyMemory p mc fc m := by
  induction h with
  | empty => exact ⟨Memory.cold_valid ..,PvsParallel.SerialTable.exactScores_empty⟩
  | noLegalMove _ _ _ ih => exact Repeated.begin_valid ih
  | direct prior q mobile ran ih =>
    obtain ⟨_,_,same,ok,_⟩ := Repeated.returns p c fc q _ addr ih checked mobile
    rw [ran] at same
    cases same
    exact ok
  | midgame prior q mobile ops level depth clock ran ih =>
    obtain ⟨_,_,_,_,same,ok,_⟩ := TimedMidgame.returns q p c fc level _ depth clock _ addr ih checked mobile
    rw [ran] at same
    cases same
    exact ok
  | interrupted prior q mobile ops level depth point ran ih =>
    obtain ⟨_,same,ok⟩ := MidgameInterrupted.returns q p c fc level _ _ depth point _ ih checked mobile
    rw [ran] at same
    cases same
    exact ok
  | directPrefix prior q mobile point ran ih =>
    obtain ⟨_,same,ok⟩ := EndgamePrefix.direct_returns q p c fc point _ addr ih checked mobile
    exact Option.some.inj (same.symm.trans ran) ▸ ok
  | handoffPrefix prior q mobile ops level depth clock point ran ih =>
    obtain ⟨_,same,ok⟩ := EndgamePrefix.timed_returns q p c fc level _ depth (some clock) point _ addr ih checked mobile
    exact Option.some.inj (same.symm.trans ran) ▸ ok
  | clear prior shared ih => exact PvsParallel.SerialTable.MidgameHistory.clear_ready _ shared ih
  | reset prior shared ih => exact PvsParallel.SerialTable.MidgameHistory.reset_ready _ shared ih
  | attach _ _ src dst => exact PvsParallel.SerialTable.MidgameHistory.attach_ready _ _ src dst
  | duplicate a b buffers size left right => exact PvsParallel.SerialTable.MidgameHistory.duplicate_ready _ _ buffers size left right
  | extra _ step ih => exact extraReady _ _ step ih

theorem History.ready (h : History p c mc fc m) (addr : c.address = p.address)
    (checked : Std.Tactic.BVDecide.LRAT.check certificate (LegalMoves.MoveBound.dimacsFormula 35) = true) :
    ReadyMemory p mc fc m :=
  Generated.ready h addr checked (fun _ _ impossible => False.elim impossible)

/-- No midgame score, child program, stored entry, cache visit or candidate row is an input.
    At a handoff the last completed midgame line supplies the endgame window. The result is
    read from the final published snapshot; its state can start another engine request. -/
def Contract : Prop := ∀ (l : Layout) (p : Pool) (c : SerialScout.Config) mc fc (q : Repeated.Call mc)
    (ops : Inputs l) (level : Selectivity) (m : Memory l) (clock : Clock (Memory l)) certificate,
    c.address = p.address → Std.Tactic.BVDecide.LRAT.check certificate (LegalMoves.MoveBound.dimacsFormula 35) = true →
    History p c mc fc m → Board.hasLegalMoves mc q.board.val = true →
    let children := frames (fc := fc) ops c level q.board false
    let depth := (Board.emptyCount q.board.val).toNat
    (∀ h, TimedMidgame.prepare q p c fc level children depth (some clock) m = some (.endgame h) →
      ∃ r next, TimedMidgame.run q p c fc level children depth (some clock) m = some (some r,next) ∧
        PublicationMir.first h.saved = some h.base ∧
        PublicationMir.ExactScore.scoreIfExact r =
          (if (TimedMidgame.completedLevel (some clock)).val = 5 then some (256 * EndgameTreeCache.discValue mc fc q.board.val) else none) ∧
        History p c mc fc next) ∧
    (q.board.val ≠ Board.newBoard → ∃ h, TimedMidgame.prepare q p c fc level children depth (some MidgameIterate.continueToEnd) m = some (.endgame h))

theorem handoff : Contract := by
  intro l p c mc fc q ops level m clock cert addr checked history mobile children depth
  have ready := history.ready addr checked
  constructor
  · intro h crossed
    obtain ⟨out,report,next,start,ran,_,source⟩ := TimedMidgame.returns q p c fc level children depth (some clock) m addr ready checked mobile
    rw [crossed] at start
    cases start
    obtain ⟨saved,r,rfl,score⟩ := source
    exact ⟨r,next,ran,saved,score,.midgame history q mobile ops level depth (some clock) ran⟩
  · exact TimedMidgame.reaches_endgame q p c fc level children depth m ready checked mobile

end NeuralReversi.AspirationMir.SerialTable.MidgameSession
