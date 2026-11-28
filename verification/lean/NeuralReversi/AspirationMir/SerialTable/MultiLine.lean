module
public import NeuralReversi.AspirationMir.SerialTable.MultiLevel
public import NeuralReversi.AspirationMir.SerialTable.MultiRows

@[expose] public section

namespace NeuralReversi.AspirationMir.SerialTable.MultiLine

open EndgameSearch.CacheHistory (Pool)
open PvsBoardTree (Position)
open PvsParallel
open PvsParallel.SerialTable (Memory)
open SelectiveLadder (Start)
open Repeated (ReadyMemory)
open RootMoveStorage (Row sortRows)
open PvsMoveLoop (scoreMax)

def previous (idx : Nat) (rows : List Row) : Option Int :=
  match idx with
  | 0 => none
  | i+1 => rows[i]?.map Row.score

def seed (idx : Nat) (base : Word) (rows : List Row) : Machine :=
  SerialEntry.windowMachine (RootAspiration.initialWindow idx base.toInt (previous idx rows))

def run (idx : Nat) (base : Word) (trials : Nat → SerialCache.Trial M) (levels : List Nat)
    (rows : List Row) (m : M) : Option (Start M) :=
  levels.foldlM (MultiLevel.advance idx trials) ⟨rows,seed idx base rows,m⟩

theorem stages (idx : Nat) (p : Pool) (c : SerialScout.Config) (b : Position) (ops : FullLadder.Inputs mc)
    (fc : Cutoffs.FlipConfig) (levels : List Nat) (s : Start (Memory l)) (addr : c.address = p.address)
    (ok : MultiLevel.Valid mc b idx (ReadyMemory p mc fc) s) :
    ∃ next, levels.foldlM (MultiLevel.advance idx (MultiLevel.trial idx p c b ops fc)) s = some next ∧
      MultiLevel.Valid mc b idx (ReadyMemory p mc fc) next ∧
      (next.rows.map Row.square).Perm (s.rows.map Row.square) ∧ next.rows.take idx = s.rows.take idx := by
  induction levels generalizing s with
  | nil => exact ⟨s,rfl,ok,.refl _,rfl⟩
  | cons level rest ih =>
    obtain ⟨last,step,_,valid,keys,front⟩ := MultiLevel.stage idx p c b ops fc level s addr ok
    obtain ⟨next,tail,good,names,pre⟩ := ih _ valid
    exact ⟨next,by simpa [List.foldlM_cons,step] using tail,good,names.trans keys,pre.trans front⟩

/-- The suffix sorted at exact acceptance supplies both the next seed and the frozen row. -/
theorem full (idx : Nat) (p : Pool) (c : SerialScout.Config) (b : Position) (ops : FullLadder.Inputs mc)
    (fc : Cutoffs.FlipConfig) (base : Word) (rows : List Row) (m : Memory l) (addr : c.address = p.address)
    (safe : ReadyMemory p mc fc m) (cover : Root.Covers mc b (rows.map Row.square)) (within : idx < rows.length)
    (range : -scoreMax ≤ base.toInt ∧ base.toInt ≤ scoreMax) (prev : ∀ v ∈ previous idx rows, -scoreMax ≤ v) :
    ∃ out, run idx base (MultiLevel.trial idx p c b ops fc) OuterMir.ladder rows m = some out ∧
      ReadyMemory p mc fc out.memory ∧ Root.Covers mc b (out.rows.map Row.square) ∧ out.rows.length = rows.length ∧
      (∀ v ∈ previous (idx+1) out.rows, -scoreMax ≤ v) ∧
      ∀ final, RootSuffix.Lines mc fc b (idx+1) out.rows final → RootSuffix.Lines mc fc b idx rows final := by
  let s : Start _ := ⟨rows,seed idx base rows,m⟩
  have win := SerialEntry.windowMachine_valid (RootAspiration.seedWindowsContract_proven.1 idx base.toInt _ range.1 range.2 prev)
  obtain ⟨active,early,ok,keys,front⟩ := stages idx p c b ops fc (OuterMir.ladder.take 4) s addr ⟨cover,within,safe,rfl,win⟩
  obtain ⟨last,step,accepted,post,names,_⟩ := MultiLevel.stage idx p c b ops fc 5 active addr ok
  let out := MultiLevel.successor idx last
  obtain ⟨snap,best,written,sound⟩ := accepted.certificate rfl
  have before : Root.Covers mc b (last.state.rows.map Row.square) := accepted.keys ▸ ok.cover
  have stateLen : last.state.rows.length = active.rows.length := by simpa using congrArg List.length accepted.keys
  have inside : last.state.regs.alpha.toInt < snap.state.best ∧ snap.state.best < last.state.regs.beta.toInt := best ▸ accepted.inside
  rw [best] at sound
  obtain ⟨row,chosen,_,bounded⟩ := MultiRows.accepted_previous before (stateLen ▸ ok.index) written inside sound
  have ordered : (out.rows.map Row.square).Perm (rows.map Row.square) := names.trans keys
  refine ⟨out,?_,post.safe,post.cover,by simpa using ordered.length_eq,?_,?_⟩
  · change ((OuterMir.ladder.take 4) ++ [5]).foldlM _ s = _
    simp [List.foldlM_append,early,step,out]
  · intro v mem
    change ((sortRows last.answer.rows idx)[idx]?.map Row.score) = some v at mem
    rw [chosen] at mem
    exact Option.some.inj mem ▸ bounded.1
  · intro final rest
    apply RootSuffix.Lines.acceptedWrites before ((List.Perm.of_eq accepted.keys).trans keys)
      (accepted.front.trans front) (stateLen ▸ ok.index) written inside sound rest

end NeuralReversi.AspirationMir.SerialTable.MultiLine
