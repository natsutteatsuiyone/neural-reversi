import NeuralReversi.AspirationMir.SerialTable.MultiLine

namespace NeuralReversi.AspirationMir.SerialTable.MultiBatch

open EndgameSearch.CacheHistory (Pool)
open PvsBoardTree (Position)
open PvsParallel
open PvsParallel.SerialTable (Memory)
open Repeated (ReadyMemory)
open RootMoveStorage (Row)
open PvsMoveLoop (scoreMax)

def lines (trials : Nat → Nat → SerialCache.Trial M) (levels : Nat → List Nat) (base : Word) :
    Nat → Nat → List Row → M → Option (List Row × M)
  | 0, _, rows, m => some (rows,m)
  | count+1, idx, rows, m => do
    let after ← MultiLine.run idx base (trials idx) (levels idx) rows m
    lines trials levels base count (idx+1) after.rows after.memory

abbrev Inputs (mc : Bitboard.Moves.X86Config) := Nat → FullLadder.Inputs mc

def trials {l : PvsParallel.SerialTable.Layout} (p : Pool) (c : SerialScout.Config) (b : Position) (ops : Inputs mc) (fc : Cutoffs.FlipConfig) :=
  fun idx => MultiLevel.trial (l := l) idx p c b (ops idx) fc

theorem exact_lines (p : Pool) (c : SerialScout.Config) (b : Position) (ops : Inputs mc) (fc : Cutoffs.FlipConfig)
    (levels : Nat → List Nat) (base : Word) (count idx : Nat) (rows : List Row) (m : Memory l)
    (addr : c.address = p.address) (safe : ReadyMemory p mc fc m) (cover : Root.Covers mc b (rows.map Row.square))
    (size : idx + count = rows.length) (range : -scoreMax ≤ base.toInt ∧ base.toInt ≤ scoreMax)
    (prev : ∀ v ∈ MultiLine.previous idx rows, -scoreMax ≤ v)
    (all : ∀ i, idx ≤ i → i < idx + count → levels i = OuterMir.ladder) :
    ∃ final next, lines (trials p c b ops fc) levels base count idx rows m = some (final,next) ∧
      ReadyMemory p mc fc next ∧ RootSuffix.Lines mc fc b idx rows final := by
  induction count generalizing idx rows m with
  | zero => exact ⟨rows,m,rfl,safe,(show idx = rows.length by omega) ▸ .done rows⟩
  | succ count ih =>
    obtain ⟨out,ran,ready,covers,len,seed,step⟩ :=
      MultiLine.full idx p c b (ops idx) fc base rows m addr safe cover (by omega) range prev
    obtain ⟨final,next,tail,good,exactRows⟩ := ih (idx+1) out.rows out.memory ready covers
      (by omega) seed (fun i start stop => all i (by omega) (by omega))
    refine ⟨final,next,?_,good,step _ exactRows⟩
    change ((MultiLine.run idx base _ (levels idx) rows m).bind _) = _
    rw [all idx (by omega) (by omega)]
    change ((MultiLine.run idx base (MultiLevel.trial idx p c b (ops idx) fc) OuterMir.ladder rows m).bind _) = _
    rw [ran]
    exact tail

end NeuralReversi.AspirationMir.SerialTable.MultiBatch
