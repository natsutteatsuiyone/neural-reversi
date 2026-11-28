module
public import NeuralReversi.AspirationMir.SerialTable.MultiBatch
public import NeuralReversi.AspirationMir.SerialTable.MultiClock
public import NeuralReversi.AspirationMir.SerialTable.MultiMidgame

@[expose] public section

namespace NeuralReversi.AspirationMir.SerialTable.MultiPublic

open EndgameSearch.CacheHistory (Pool)
open PvsBoardTree (Position)
open PvsParallel
open PvsParallel.SerialTable (Memory Layout)
open Repeated (ReadyMemory)
open RootMoveStorage (Row)
open PvsMoveLoop (scoreMax)
open PublicationMir (Iteration SearchResult)
open MultiBatch (Inputs)

structure Clock where
  prediction : Option EndgameTiming.Prediction
  elapsed : Nat → EndgameProgress.Clock

def Clock.ordered (clock : Clock) (lastLine : Nat) : Prop :=
  (∀ i, i ≤ lastLine → EndgameProgress.ClockOrder (clock.elapsed i) (clock.elapsed lastLine)) ∧
    ∀ step, clock.elapsed lastLine step / 1000000 < 2^64

def unlimited : Clock := ⟨none,fun _ _ => 0⟩

def solve (q : Repeated.Call mc) (p : Pool) (c : SerialScout.Config) (fc : Cutoffs.FlipConfig)
    (ops : Inputs mc) (cap minimum : Nat) (prior : Option Iteration) (clock : Clock) (base : Word)
    (rows : List Row) (m : Memory l) : Option (SearchResult × Memory l) := do
  let levels := MultiClock.levels cap minimum clock.prediction clock.elapsed
  let (final,next) ← MultiBatch.lines (MultiBatch.trials p c q.board ops fc) levels base rows.length 0 rows m
  let last := (levels (rows.length-1)).getLastD 5
  let result ← OuterMir.finish minimum last q.sort final (Board.emptyCount q.board.val).toNat q.nodes prior q.fallback
  return (result,next)

theorem from_flags (q : Repeated.Call mc) (p : Pool) (c : SerialScout.Config) (fc : Cutoffs.FlipConfig)
    (ops : Inputs mc) (cap minimum : Nat) (prior : Option Iteration) (clock : Clock) (base : Word)
    (rows : List Row) (m : Memory l) (addr : c.address = p.address) (ready : ReadyMemory p mc fc m)
    (cover : Root.Covers mc q.board (rows.map Row.square)) (pos : rows ≠ [])
    (range : -scoreMax ≤ base.toInt ∧ base.toInt ≤ scoreMax) (origin : MultiClock.Origin minimum prior)
    (enabled : 1 ≤ cap) (time : clock.ordered (rows.length-1))
    (returned : solve q p c fc ops cap minimum prior clock base rows m = some (.bestMove report,next))
    (tag : report.endgame = true) (full : report.stamp.selectivity = 5) :
    MultiRows.ReportExact mc fc q.board report ∧ ReadyMemory p mc fc next := by
  obtain ⟨⟨final,after⟩,steps,more⟩ := Option.bind_eq_some_iff.mp returned
  obtain ⟨res,printed,eq⟩ := Option.bind_eq_some_iff.mp more
  cases eq
  let levels := MultiClock.levels cap minimum clock.prediction clock.elapsed
  have all := MultiClock.all_of_report origin enabled time.1 time.2 printed tag full
  obtain ⟨actual,mem,ran,safe,lines⟩ := MultiBatch.exact_lines p c q.board ops fc levels base rows.length 0
    rows m addr ready cover (by simp) range (by simp [MultiLine.previous]) (fun i _ hi => all i (by omega))
  rw [steps] at ran
  cases ran
  obtain ⟨best,pub,correct⟩ := MultiRows.publish lines cover pos q.sort (Board.emptyCount q.board.val).toNat q.nodes q.fallback
  have last := all (rows.length-1) (Nat.le_refl _)
  have threshold : minimum ≤ 5 := by cases origin <;> decide
  change OuterMir.finish minimum ((levels (rows.length-1)).getLastD 5) _ _ _ _ _ _ = _ at printed
  change levels (rows.length-1) = _ at last
  rw [last] at printed
  change OuterMir.finish minimum 5 _ _ _ _ _ _ = _ at printed
  rw [OuterMir.finish,ite_eq_left threshold,pub] at printed
  cases printed
  exact ⟨correct,safe⟩

theorem unlimited_returns (q : Repeated.Call mc) (p : Pool) (c : SerialScout.Config) (fc : Cutoffs.FlipConfig)
    (ops : Inputs mc) (minimum : Nat) (prior : Option Iteration) (base : Word) (rows : List Row) (m : Memory l)
    (addr : c.address = p.address) (ready : ReadyMemory p mc fc m) (cover : Root.Covers mc q.board (rows.map Row.square))
    (pos : rows ≠ []) (range : -scoreMax ≤ base.toInt ∧ base.toInt ≤ scoreMax) (reportable : minimum ≤ 5) :
    ∃ best next, solve q p c fc ops 5 minimum prior unlimited base rows m = some (.bestMove best,next) ∧
      MultiRows.ReportExact mc fc q.board best ∧ ReadyMemory p mc fc next := by
  let levels := MultiClock.levels 5 minimum unlimited.prediction unlimited.elapsed
  obtain ⟨final,next,ran,safe,lines⟩ := MultiBatch.exact_lines p c q.board ops fc levels base rows.length 0
    rows m addr ready cover (by simp) range (by simp [MultiLine.previous]) (fun i _ _ => MultiClock.noClock minimum _ i)
  obtain ⟨best,pub,correct⟩ := MultiRows.publish lines cover pos q.sort (Board.emptyCount q.board.val).toNat q.nodes q.fallback
  refine ⟨best,next,?_,correct,safe⟩
  change (MultiBatch.lines _ levels _ _ _ _ _).bind _ = _
  rw [ran]
  change ((OuterMir.finish minimum 5 _ _ _ _ _ _).bind _) = _
  rw [OuterMir.finish,ite_eq_left reportable,pub]
  rfl

def direct (q : Repeated.Call mc) (p : Pool) (c : SerialScout.Config) (fc : Cutoffs.FlipConfig)
    (ops : Inputs mc) (clock : Clock) (m : Memory l) := do
  let rows ← NativeRootBirth.started mc fc q.board c.paired q.scratch
  let now := Repeated.increment m
  let base := InitialEstimate.estimate now.tt.image mc fc q.raw q.board
  solve q p c fc ops q.last.val 1 none clock base rows now

def timed (q : Repeated.Call mc) (p : Pool) (c : SerialScout.Config) (fc : Cutoffs.FlipConfig)
    (ops : Inputs mc) (level : TranspositionTable.Selectivity) (mid : MultiMidgame.Inputs l)
    (time : MidgameIterate.Clock (Memory l)) (clock : Clock) (m : Memory l) : Option (Option SearchResult × Memory l) := do
  match ← MultiMidgame.prepare q p c fc level mid (some time) m with
  | .done s => return (none,s.memory)
  | .endgame h =>
    let (r,next) ← solve q p c fc ops 5 2 (some h.saved) clock (BitVec.ofInt 32 h.base.score) h.state.rows h.state.memory
    return (some r,next)

end NeuralReversi.AspirationMir.SerialTable.MultiPublic
