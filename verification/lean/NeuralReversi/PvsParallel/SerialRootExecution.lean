import NeuralReversi.PvsParallel.SerialExecution
import NeuralReversi.PvsParallel.SerialRootCalls
import NeuralReversi.AspirationMir.NativeRootBirth

namespace NeuralReversi.PvsParallel.SerialRootExecution

open Bitboard.Moves (X86Config)
open Cutoffs (FlipConfig)
open Aspiration (Window)
open AspirationMir.SmallRoot (Answer)
open PvsBoardTree (Position minimax)
open RootMoveStorage (Row)
open SerialCalls (Machine Program Tree Executes)
open SerialExecution (Frame Inputs ReadFacts body push)

def context (b : Position) (a z : BitVec 32) : Frame :=
  let state := ContextPathMir.start b
  ⟨⟨b, ContextPathMir.rootDepth state, true, false, a.toInt, z.toInt⟩, ContextPathMir.leafState b⟩

def attempt (c : SerialScout.Config) (inputs : Inputs mc fc) (opening : Nat) (b : Position)
    (root : SerialRecursion.RootInput) (initial : List Row) (idx : Nat) (a z : BitVec 32) :
    Program SerialCalls.Request Answer :=
  let ctx := context b a z
  SerialRootCalls.frame mc fc b ctx.request.depth c.paired (inputs opening ctx).memory initial idx
    root.key root.sort root.pvLine a z

/-- Reconstruct the root's physical rows and child returns from a finished stack run. -/
theorem returns {mc : X86Config} {fc : FlipConfig} (c : SerialScout.Config) (inputs : Inputs mc fc)
    (valid : ReadFacts inputs) (opening : Nat) (b : Position) (root : SerialRecursion.RootInput)
    (initial : List Row) (idx : Nat) (a z : BitVec 32)
    (covers : Root.Covers mc b (initial.map Row.square)) (within : idx < initial.length)
    (window : Window a.toInt z.toInt)
    (ran : (Machine.running opening (context b a z)
      ((attempt c inputs opening b root initial idx a z).map Answer.value) []).steps
        (body c inputs) (push mc fc) count = some (.finished last value)) :
    ∃ forest answer snap,
      Executes (body c inputs) (push mc fc) opening (context b a z)
        ((attempt c inputs opening b root initial idx a z).map Answer.value) last value forest ∧
      Program.Runs (attempt c inputs opening b root initial idx a z) (forest.map Tree.direct) answer ∧
      answer.value = value ∧ value = snap.state.best ∧
      RootSuffix.Writes initial idx a.toInt z.toInt answer.rows snap ∧
      RootEvidence.Attempt mc fc b (initial.map Row.square) idx a.toInt z.toInt value snap.scores ∧
      RootOrder.SoundAttempt mc fc b (initial.map Row.square) idx a.toInt z.toInt value snap.scores := by
  obtain ⟨forest, execution⟩ := SerialCalls.record_program_of_steps (body c inputs) (push mc fc) (context b a z) ran
  obtain ⟨answer, owner, valueEq⟩ := Program.runs_map execution.directCalls
  have children := execution.children (SerialExecution.Invariant mc fc) (SerialExecution.localRule c inputs valid)
  have init : (context b a z).empties.matches b ∧ ExactDepth.Full b (context b a z).request.depth := by
    simp only [context, ContextPathMir.start_eq]
    exact ⟨EndgameSearch.emptyState_initial b, rfl⟩
  obtain ⟨snap, reported, writes, checked, sound⟩ := SerialRootCalls.frame_replay owner covers within window (by
    intro t h
    obtain ⟨_, _, _, _, w, width⟩ := SerialRootCalls.frame_edges owner window t h
    exact ⟨_, SerialExecution.child_replay init.2 init.1 (children t h) w width⟩)
  exact ⟨forest, answer, snap, execution, owner, valueEq, valueEq ▸ reported, writes, valueEq ▸ checked, valueEq ▸ sound⟩

/-- Initialize root rows from the board, then execute one root attempt through the same stack. -/
def runFromBoard (c : SerialScout.Config) (inputs : Inputs mc fc) (opening count : Nat)
    (b : Position) (root : SerialRecursion.RootInput) (a z : BitVec 32) : Option (Nat × Int) := do
  let rows ← AspirationMir.NativeRootBirth.started mc fc b c.paired (inputs opening (context b a z)).memory
  let machine ← (Machine.running opening (context b a z)
    ((attempt c inputs opening b root rows 0 a z).map Answer.value) []).steps (body c inputs) (push mc fc) count
  match machine with
  | .finished last value => return (last, value)
  | .running .. => none

def FromBoardResult (c : SerialScout.Config) (inputs : Inputs mc fc)
    (opening : Nat) (b : Position) (root : SerialRecursion.RootInput) (a z : BitVec 32)
    (last : Nat) (value : Int) : Prop :=
    ∃ initial forest answer snap,
      AspirationMir.NativeRootBirth.started mc fc b c.paired (inputs opening (context b a z)).memory = some initial ∧
      Executes (body c inputs) (push mc fc) opening (context b a z)
        ((attempt c inputs opening b root initial 0 a z).map Answer.value) last value forest ∧
      Program.Runs (attempt c inputs opening b root initial 0 a z) (forest.map Tree.direct) answer ∧
      answer.value = value ∧ value = snap.state.best ∧
      RootSuffix.Writes initial 0 a.toInt z.toInt answer.rows snap ∧
      RootEvidence.Attempt mc fc b (initial.map Row.square) 0 a.toInt z.toInt value snap.scores ∧
      Aspiration.SoundResult (minimax mc fc b) a.toInt z.toInt value

theorem fromBoard {mc : X86Config} {fc : FlipConfig} (c : SerialScout.Config) (inputs : Inputs mc fc)
    (valid : ReadFacts inputs) (opening count : Nat) (b : Position) (root : SerialRecursion.RootInput)
    (a z : BitVec 32) (window : Window a.toInt z.toInt)
    (normal : runFromBoard c inputs opening count b root a z = some (last, value)) :
    FromBoardResult c inputs opening b root a z last value := by
  obtain ⟨initial, birth, proceed⟩ := Option.bind_eq_some_iff.mp normal
  obtain ⟨machine, stepped, stopped⟩ := Option.bind_eq_some_iff.mp proceed
  cases machine with
  | running => contradiction
  | finished endTime score =>
    obtain ⟨rfl,rfl⟩ := Prod.mk.inj (Option.some.inj stopped)
    obtain ⟨covers, nonempty⟩ := AspirationMir.NativeRootBirth.covers birth
    have pos := List.length_pos_iff.mpr nonempty
    obtain ⟨forest, answer, snap, execution, owner, valueEq, best, writes, evidence, sound⟩ :=
      returns c inputs valid opening b root initial 0 a z covers pos window stepped
    exact ⟨initial, forest, answer, snap, birth, execution, owner, valueEq, best, writes, evidence,
      Root.value_zero covers (by simpa using pos) ▸ sound.1⟩

def FromBoardContract : Prop :=
  ∀ (mc : X86Config) (fc : FlipConfig) (c : SerialScout.Config) (inputs : Inputs mc fc)
    (opening count : Nat) (b : Position) (root : SerialRecursion.RootInput)
    (a z : BitVec 32) (last : Nat) (value : Int),
    ReadFacts inputs → Window a.toInt z.toInt →
    runFromBoard c inputs opening count b root a z = some (last, value) →
    FromBoardResult c inputs opening b root a z last value

theorem fromBoardContract_proven : FromBoardContract := fun _ _ c inputs opening count b root a z _ _ valid =>
  fromBoard c inputs valid opening count b root a z

end NeuralReversi.PvsParallel.SerialRootExecution
