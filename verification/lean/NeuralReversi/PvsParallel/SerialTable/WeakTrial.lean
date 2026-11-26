import NeuralReversi.PvsParallel.SerialTable.WeakTrial.Program
import NeuralReversi.PvsParallel.SerialTable.WeakStack

namespace NeuralReversi.PvsParallel.SerialTable.WeakTrial

open EndgameSearch.CacheHistory (Pool)
open PvsBoardTree (Position)
open SerialExecution (Inputs)
open SerialCalls.ScoreWords (Finite)
open TranspositionTable (Selectivity)
open AspirationMir.SerialStack (State)
open RootMoveStorage (Row)

theorem trial_returns (p : Pool) (c : SerialScout.Config) (address : c.address = p.address) (b : Position)
    (base : Inputs mc fc) (root : SerialRecursion.RootInput) (lvl : Selectivity) (weak : lvl.val < 5)
    (screen : WeakStack.Screen) (s : State) (prior : Memory l) (valid : prior.Valid p mc fc)
    (covers : PvsParallel.Root.Covers mc b (s.rows.map Row.square)) (within : 0 < s.rows.length)
    (window : Aspiration.Window s.regs.alpha.toInt s.regs.beta.toInt) :
    ∃ k ans next, trial p c b base root lvl screen s prior = some (k,ans,next) ∧ next.Valid p mc fc ∧
      Finite ans.value ∧ ans.rows.map Row.square = s.rows.map Row.square :=
  trial_of_children p c b base root lvl weak screen s prior valid covers within window
    (WeakStack.run_returns p c address base lvl weak screen)

end NeuralReversi.PvsParallel.SerialTable.WeakTrial
