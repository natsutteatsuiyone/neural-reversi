import NeuralReversi.OuterMir.Program

namespace NeuralReversi.OuterMir.Polled

/-- Generated from the normal edges of the pinned `solve_root` MIR.
    Separate inputs retain the two independently observed abort values. -/
inductive Cutpoint where
  | entry | line | level | afterLine | returned
deriving DecidableEq, Repr

inductive Effect where
  | beginPhase | prepareEndgame | setPvIndex | setLevel | query | nextWindow
  | pollAbort | recordAbort | sortRows | lineCompleted | clockCheck
  | iterationCompleted | timeCheck | returnResult
deriving DecidableEq, Repr

structure Transition where
  next : Cutpoint
  effects : List Effect
deriving DecidableEq, Repr

def entry : Transition := ⟨.line, [.beginPhase, .prepareEndgame]⟩

def lineTable : List ((Bool × Bool) × Transition) := [
  ((false, false), ⟨.returned, [.returnResult]⟩),
  ((false, true), ⟨.returned, [.iterationCompleted, .returnResult]⟩),
  ((true, false), ⟨.level, [.setPvIndex]⟩),
  ((true, true), ⟨.level, [.setPvIndex]⟩)]

def levelTable : List ((Bool × Bool × Bool × Bool × Bool) × Transition) := [
  ((false, false, false, false, false), ⟨.afterLine, []⟩),
  ((false, false, false, false, true), ⟨.afterLine, []⟩),
  ((false, false, false, true, false), ⟨.afterLine, []⟩),
  ((false, false, false, true, true), ⟨.afterLine, []⟩),
  ((false, false, true, false, false), ⟨.afterLine, []⟩),
  ((false, false, true, false, true), ⟨.afterLine, []⟩),
  ((false, false, true, true, false), ⟨.afterLine, []⟩),
  ((false, false, true, true, true), ⟨.afterLine, []⟩),
  ((false, true, false, false, false), ⟨.afterLine, []⟩),
  ((false, true, false, false, true), ⟨.afterLine, []⟩),
  ((false, true, false, true, false), ⟨.afterLine, []⟩),
  ((false, true, false, true, true), ⟨.afterLine, []⟩),
  ((false, true, true, false, false), ⟨.afterLine, []⟩),
  ((false, true, true, false, true), ⟨.afterLine, []⟩),
  ((false, true, true, true, false), ⟨.afterLine, []⟩),
  ((false, true, true, true, true), ⟨.afterLine, []⟩),
  ((true, false, false, false, false), ⟨.level, [.setLevel, .query, .nextWindow, .pollAbort, .sortRows, .clockCheck]⟩),
  ((true, false, false, false, true), ⟨.afterLine, [.setLevel, .query, .nextWindow, .pollAbort, .sortRows, .clockCheck]⟩),
  ((true, false, false, true, false), ⟨.afterLine, [.setLevel, .query, .nextWindow, .pollAbort, .recordAbort]⟩),
  ((true, false, false, true, true), ⟨.afterLine, [.setLevel, .query, .nextWindow, .pollAbort, .recordAbort]⟩),
  ((true, false, true, false, false), ⟨.level, [.setLevel, .query, .nextWindow, .pollAbort, .sortRows, .lineCompleted, .clockCheck]⟩),
  ((true, false, true, false, true), ⟨.afterLine, [.setLevel, .query, .nextWindow, .pollAbort, .sortRows, .lineCompleted, .clockCheck]⟩),
  ((true, false, true, true, false), ⟨.afterLine, [.setLevel, .query, .nextWindow, .pollAbort, .recordAbort]⟩),
  ((true, false, true, true, true), ⟨.afterLine, [.setLevel, .query, .nextWindow, .pollAbort, .recordAbort]⟩),
  ((true, true, false, false, false), ⟨.afterLine, []⟩),
  ((true, true, false, false, true), ⟨.afterLine, []⟩),
  ((true, true, false, true, false), ⟨.afterLine, []⟩),
  ((true, true, false, true, true), ⟨.afterLine, []⟩),
  ((true, true, true, false, false), ⟨.afterLine, []⟩),
  ((true, true, true, false, true), ⟨.afterLine, []⟩),
  ((true, true, true, true, false), ⟨.afterLine, []⟩),
  ((true, true, true, true, true), ⟨.afterLine, []⟩)]

def afterLineTable : List ((Bool × Bool) × Transition) := [
  ((false, false), ⟨.line, [.pollAbort, .timeCheck]⟩),
  ((false, true), ⟨.returned, [.pollAbort, .timeCheck, .returnResult]⟩),
  ((true, false), ⟨.returned, [.pollAbort, .returnResult]⟩),
  ((true, true), ⟨.returned, [.pollAbort, .returnResult]⟩)]

def lineStep (hasLine reportable : Bool) : Option Transition :=
  lineTable.lookup (hasLine, reportable)

def levelStep (hasLevel aboveCap reportable beforeAbort clockStop : Bool) : Option Transition :=
  levelTable.lookup (hasLevel, aboveCap, reportable, beforeAbort, clockStop)

def afterLineStep (afterAbort timeout : Bool) : Option Transition :=
  afterLineTable.lookup (afterAbort, timeout)

end NeuralReversi.OuterMir.Polled
