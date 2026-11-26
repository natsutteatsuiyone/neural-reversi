import NeuralReversi.AspirationMir.SerialTable.FullLadder
import NeuralReversi.AspirationMir.SerialTable.PvInterior
import NeuralReversi.EndgameSearch.SerialWarmCache

namespace NeuralReversi.AspirationMir.SerialTable.MidgameHandoff

open PvsParallel
open EndgameSearch.CacheHistory (Pool Visit)
open PvsParallel.SerialTable (Table Memory EarlierStore FiniteHistory)
open PvsBoardTree (Position legalSquares)
open PvsMoveLoop (scoreMax)
open RootMoveStorage (Row)
open PvInterior (InsideRange)

/-- The PV returns used for a midgame estimate. A reduced scout, or its full-depth scout retry,
    contributes its last word at that move. Its numeric value is unrestricted. Every requested
    wide child has its own derivation at the reflected window. -/
inductive PvReturn (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) : Position → Int → Int → Int → Prop
  | evaluated (b a z raw) : PvReturn mc fc b a z (InitialEstimate.static raw b).val
  | terminal (b a z) : PvReturn mc fc b a z (PvsBoardTree.terminalScore b).val
  | wipeout (b a z) : PvReturn mc fc b a z scoreMax
  | stopped (b a z) : PvReturn mc fc b a z 0
  | passed (sub : PvReturn mc fc (PvsBoardTree.pass b) (-z) (-a) v) : PvReturn mc fc b a z (-v)
  | owner {b : Position} {calls : List (SerialCalls.MoveRequest (PvsBoardTree.GeneratedSquare mc b.val) × Int)}
      {out : SerialOwnerCalls.Result (PvsBoardTree.GeneratedSquare mc b.val)} (moves : List (PvsBoardTree.GeneratedSquare mc b.val)) (unique : moves.Nodup)
      (win : Aspiration.Window a z)
      (ran : SerialCalls.Program.Runs (SerialOwnerCalls.run Subtype.val z moves (RootReplay.start a)) calls out)
      (children : ∀ t ∈ calls, t.1.pv = true →
        PvReturn mc fc (PvsBoardTree.play mc fc b t.1.move).val t.1.alpha t.1.beta t.2) :
      PvReturn mc fc b a z out.2.state.best

theorem interior (pv : PvReturn mc fc b a z v) : InsideRange a z v := by
  induction pv with
  | evaluated => exact fun _ _ => (InitialEstimate.static _ _).property
  | terminal => exact fun _ _ => (PvsBoardTree.terminalScore _).property
  | wipeout => exact fun _ _ => ⟨by decide,Int.le_refl _⟩
  | stopped => exact fun _ _ => ⟨by decide,by decide⟩
  | passed _ ih =>
    intro low high
    have ⟨lo,hi⟩ := ih (by omega) (by omega)
    omega
  | owner moves unique win ran _ ih =>
    exact PvInterior.owner unique ran win (fun t mem isPv => ih t mem isPv)

/-- Includes every earlier TT store, even when the physical replacement rejects it. -/
private theorem earlier_finite (stores : List (EarlierStore l)) (t : Table l) (good : t.Good mc fc)
    (finite : FiniteHistory t.history) : FiniteHistory (stores.foldl EarlierStore.write t).history := by
  induction stores generalizing t with
  | nil => exact finite
  | cons s tail ih =>
    exact ih (s.write t) (t.store_other good _ _ _ _ _ _ _ _ _ s.other s.legal)
      (t.store_finite good.supported finite _ _ _ _ _ _ _ _ _)

/-- A timed midgame handoff passes the score of the completed root and its already sorted rows.
    Replay its cache visits and earlier legal-hint stores, then execute the entire endgame ladder.
    Only the preceding PV's interior score is needed, regardless of its depth or distance to disc minimax. -/
def Contract : Prop :=
  ∀ (l : PvsParallel.SerialTable.Layout) (p : Pool) mc fc (c : SerialScout.Config) (b : Position)
    (ops : FullLadder.Inputs mc) (t : Table l) (stores : List (EarlierStore l)) (visits : List (Visit p))
    (rows : List Row) (a z : Int) (v : Word) sort nodes fallback,
    c.address = p.address → t.Good mc fc → FiniteHistory t.history →
    Board.hasLegalMoves mc b.val = true → Root.Covers mc b (rows.map Row.square) →
    PvReturn mc fc b a z v.toInt → a < v.toInt → v.toInt < z →
    ∃ before : Memory l, before.tt = stores.foldl EarlierStore.write t ∧
      before.cache = EndgameSearch.CacheHistory.replay mc fc p visits ∧
      let start := SelectiveLadder.Start.mk rows (SerialEntry.firstMachine v) before
      let weak := FullLadder.trial (l := l) p c b ops fc
      let exactTrial := search (l := l) p c b (ops.order 5) fc
      ∃ final, (OuterMir.ladder.take 4).foldlM (SelectiveLadder.advance weak) start = some final ∧
        SelectiveLadder.Levels mc b (Memory.Valid p mc fc) weak (OuterMir.ladder.take 4) start final ∧
        Published mc fc b exactTrial (Memory.Good p mc fc) (recorded exactTrial)
          final.rows final.machine final.memory sort nodes fallback

theorem handoff : Contract := by
  intro l p mc fc c b ops t stores visits rows a z v sort nodes fallback addr good finite mobile cover pv lo hi
  have table := PvsParallel.SerialTable.earlier_good l mc fc t stores good
  let prior : Memory l := ⟨_,table.supported,EndgameSearch.CacheHistory.replay mc fc p visits⟩
  have valid : prior.Valid p mc fc :=
    ⟨⟨table,EndgameSearch.SerialWarmCache.replay_reachable ..⟩,earlier_finite stores t good finite⟩
  have nonempty : legalSquares mc b.val ≠ [] := fun eq =>
    Bool.noConfusion (mobile.symm.trans (PvsBoardTree.stuck_of_nil eq))
  have positive : 0 < rows.length := by
    have n := cover.length_eq
    simp only [List.length_map] at n
    exact n ▸ List.length_pos_iff.mpr nonempty
  have range := interior pv lo hi
  let start := SelectiveLadder.Start.mk rows (SerialEntry.firstMachine v) prior
  have ok : start.valid mc b (Memory.Valid p mc fc) :=
    ⟨cover,positive,valid,rfl,SerialEntry.firstMachine_window v range.1 range.2⟩
  obtain ⟨final,fold,steps,squares,more,current,pc,w⟩ := SelectiveLadder.ladder _ start ok (FullLadder.earlier_returns p c b ops fc addr)
  exact ⟨prior,rfl,rfl,final,fold,steps,publication _ (works p c b (ops.order 5) fc addr)
    current.1 squares more _ pc w sort nodes fallback⟩

end NeuralReversi.AspirationMir.SerialTable.MidgameHandoff
