module
public import NeuralReversi.AspirationMir.SerialEntry
public import NeuralReversi.AspirationMir.SerialTable.FromBoard
public import NeuralReversi.EndgameSearch.SerialWarmCache
public import NeuralReversi.PvsParallel.SerialTable.Prefix

@[expose] public section

namespace NeuralReversi.AspirationMir.SerialTable

open EndgameSearch.CacheHistory (Pool Visit)
open PvsParallel
open PvsParallel.SerialTable (Layout Table Memory EarlierStore)
open PvsBoardTree (Position)
open RootMoveStorage (Row)

/-- The earlier levels can return an approximate score. Only its acceptance inside the previous
    window, the retained root squares and the operands of their cache and TT effects are used. -/
def EarlierLevelContract : Prop :=
  ∀ (l : Layout) (p : Pool) mc fc (c : SerialScout.Config) (b : Position)
    (orders : SerialStack.State → MoveInputs mc) (t : Table l) (stores : List (EarlierStore l))
    (visits : List (Visit p)) (rows : List Row) (regs after : WindowRegs) (score : Word) sort nodes fallback,
    t.Good mc fc → c.address = p.address → Board.hasLegalMoves mc b.val = true →
    Root.Covers mc b (rows.map Row.square) →
    Aspiration.Window regs.alpha.toInt regs.beta.toInt → applyWiden regs score = some (after,false) →
    ∃ before : Memory l, before.tt = stores.foldl EarlierStore.write t ∧
      before.cache = EndgameSearch.CacheHistory.replay mc fc p visits ∧
      Published mc fc b (search p c b orders fc) (Memory.Good p mc fc) (recorded (search p c b orders fc))
        rows (SerialEntry.nextMachine score) before sort nodes fallback

theorem from_earlier_level : EarlierLevelContract := by
  intro l p mc fc c b orders t stores visits rows regs after score sort nodes fallback known address mobile cover window accepted
  have table := PvsParallel.SerialTable.earlier_good l mc fc t stores known
  let before : Memory l := ⟨_,table.supported,EndgameSearch.CacheHistory.replay mc fc p visits⟩
  have valid : before.Good p mc fc := ⟨table,EndgameSearch.SerialWarmCache.replay_reachable p mc fc visits⟩
  have pos : 0 < rows.length := by
    have legal : PvsBoardTree.legalSquares mc b.val ≠ [] := fun empty => by
      exact Bool.noConfusion (mobile.symm.trans (PvsBoardTree.stuck_of_nil empty))
    have len := cover.length_eq
    simp only [List.length_map] at len
    rw [len]
    exact List.length_pos_iff.mpr legal
  exact ⟨before,rfl,rfl,publication _ (works p c b orders fc address) valid cover pos _ rfl
    (SerialEntry.accepted_next_window window accepted) sort nodes fallback⟩

end NeuralReversi.AspirationMir.SerialTable
