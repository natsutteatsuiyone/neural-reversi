module
public import NeuralReversi.AspirationMir.SerialTable.MultiSession
public import NeuralReversi.AspirationMir.SerialTable.MultiPrefix
public import NeuralReversi.AspirationMir.SerialTable.MultiMidgamePrefix

@[expose] public section

namespace NeuralReversi.AspirationMir.SerialTable.MultiHistory

open EndgameSearch.CacheHistory (Pool)
open PvsParallel
open PvsParallel.SerialTable (Memory)
open Repeated (ReadyMemory)
open RootMoveStorage (Row)
open TranspositionTable (Selectivity)
open PvsMoveLoop (scoreMax)

private theorem solve_ready (q : Repeated.Call mc) (p : Pool) (c : SerialScout.Config) (fc : Cutoffs.FlipConfig)
    (ops : MultiBatch.Inputs mc) (clock : MultiPublic.Clock) (base : Word) (rows : List Row) (m : Memory l)
    (addr : c.address = p.address) (good : ReadyMemory p mc fc m) (cover : Root.Covers mc q.board (rows.map Row.square))
    (range : -scoreMax ≤ base.toInt ∧ base.toInt ≤ scoreMax) (enabled : 1 ≤ cap)
    (ran : MultiPublic.solve q p c fc ops cap minimum prior clock base rows m = some (result,after)) :
    ReadyMemory p mc fc after := by
  obtain ⟨⟨_,out⟩,batch,finished⟩ := Option.bind_eq_some_iff.mp ran
  obtain ⟨_,_,eq⟩ := Option.bind_eq_some_iff.mp finished
  cases eq
  obtain ⟨_,_,same,safe,_⟩ := MultiOrdinary.lines_returns p c q.board ops fc
    (MultiClock.levels cap minimum clock.prediction clock.elapsed) base rows.length 0 rows m addr good cover
    (by simp) range (by simp [MultiLine.previous]) (fun _ _ _ => MultiOrdinary.clock_nonempty enabled)
  rw [batch] at same
  cases same
  exact safe

/-- A previous request contributes the memory returned by its concrete loop. A timed
    request can also finish in the midgame without entering the endgame phase. -/
inductive Action (p : Pool) (c : SerialScout.Config) (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) :
    Memory l → Memory l → Prop where
  | direct (q : Repeated.Call mc) (ops : MultiBatch.Inputs mc) (clock : MultiPublic.Clock)
      (ran : MultiPublic.direct q p c fc ops clock before = some (result,after)) :
      Action p c mc fc before after
  | timed (q : Repeated.Call mc) (ops : MultiBatch.Inputs mc) (level : Selectivity) (mid : MultiMidgame.Inputs l)
      (time : MidgameIterate.Clock (Memory l)) (clock : MultiPublic.Clock)
      (mobile : Board.hasLegalMoves mc q.board.val = true)
      (ran : MultiPublic.timed q p c fc ops level mid time clock before = some (result,after)) :
      Action p c mc fc before after
  | directPrefix (q : Repeated.Call mc) (ops : MultiBatch.Inputs mc) (clock : MultiPublic.Clock) (point : MultiPrefix.Point)
      (born : NativeRootBirth.started mc fc q.board c.paired q.scratch = some rows)
      (ran : MultiPrefix.finish q p c fc ops (MultiClock.levels q.last.val 1 clock.prediction clock.elapsed)
        (InitialEstimate.estimate (Repeated.increment before).tt.image mc fc q.raw q.board) rows (Repeated.increment before) point = some after) :
      Action p c mc fc before after
  | timedPrefix (q : Repeated.Call mc) (ops : MultiBatch.Inputs mc) (level : Selectivity) (mid : MultiMidgame.Inputs l)
      (time : MidgameIterate.Clock (Memory l)) (clock : MultiPublic.Clock) (point : MultiPrefix.Point)
      (mobile : Board.hasLegalMoves mc q.board.val = true)
      (prepared : MultiMidgame.prepare q p c fc level mid (some time) before = some (.endgame h))
      (ran : MultiPrefix.finish q p c fc ops (MultiClock.levels 5 2 clock.prediction clock.elapsed)
        (BitVec.ofInt 32 h.base.score) h.state.rows h.state.memory point = some after) :
      Action p c mc fc before after
  | fixedMidgame (q : Repeated.Call mc) (level : Selectivity) (mid : MultiMidgame.Inputs l) (requested : Nat)
      (mobile : Board.hasLegalMoves mc q.board.val = true)
      (ran : MultiMidgame.prepareAt q p c fc level mid requested none before = some (.done s)) :
      Action p c mc fc before s.memory
  | midgamePrefix (q : Repeated.Call mc) (level : Selectivity) (mid : MultiMidgame.Inputs l) (requested : Nat)
      (point : MultiMidgamePrefix.Point) (mobile : Board.hasLegalMoves mc q.board.val = true)
      (ran : MultiMidgamePrefix.run q p c fc level mid requested point before = some after) :
      Action p c mc fc before after

theorem Action.ready (action : Action p c mc fc before after) (addr : c.address = p.address)
    (checked : Std.Tactic.BVDecide.LRAT.check certificate (LegalMoves.MoveBound.dimacsFormula 35) = true)
    (good : ReadyMemory p mc fc before) : ReadyMemory p mc fc after := by
  cases action with
  | direct q ops clock ran =>
    obtain ⟨rows,born,solved⟩ := Option.bind_eq_some_iff.mp ran
    have ready := Repeated.begin_valid good
    exact solve_ready q p c fc ops clock _ rows _ addr ready (NativeRootBirth.covers born).1
      (InitialEstimate.warm_bounded _ mc fc q.raw q.board (Repeated.increment before).supported ready.2) q.last.property.1 solved
  | timed q ops level mid time clock mobile ran =>
    obtain ⟨out,prepared,post⟩ := MultiMidgame.prepare_returns q p c fc level mid (some time) _ good checked mobile
    cases out with
    | done s =>
      simp only [MultiPublic.timed,prepared,bind,Option.bind,pure,Pure.pure,Option.some.injEq,Prod.mk.injEq] at ran
      exact ran.2 ▸ post.2.2.2
    | endgame h =>
      simp only [MultiPublic.timed,prepared,bind,Option.bind] at ran
      obtain ⟨⟨_,_⟩,solved,eq⟩ := Option.bind_eq_some_iff.mp ran
      cases eq
      exact solve_ready q p c fc ops clock _ _ _ addr post.valid.2.2.2 post.valid.1
        (MultiSession.base_range _ post.bounds) (by decide) solved
  | directPrefix q ops clock point born ran =>
    have ready := Repeated.begin_valid good
    obtain ⟨_,same,safe⟩ := MultiPrefix.finish_returns q p c fc ops
      (MultiClock.levels q.last.val 1 clock.prediction clock.elapsed) _ _ _ point addr ready (NativeRootBirth.covers born).1
      (InitialEstimate.warm_bounded _ mc fc q.raw q.board (Repeated.increment before).supported ready.2) (fun _ _ => MultiOrdinary.clock_nonempty q.last.property.1)
    rw [ran] at same
    cases same
    exact safe
  | timedPrefix q ops level mid time clock point mobile prepared ran =>
    obtain ⟨_,same,post⟩ := MultiMidgame.prepare_returns q p c fc level mid (some time) _ good checked mobile
    rw [prepared] at same
    cases same
    obtain ⟨_,same,safe⟩ := MultiPrefix.finish_returns q p c fc ops
      (MultiClock.levels 5 2 clock.prediction clock.elapsed) _ _ _ point addr post.valid.2.2.2 post.valid.1
      (MultiSession.base_range _ post.bounds) (fun _ _ => MultiOrdinary.clock_nonempty (by decide))
    rw [ran] at same
    cases same
    exact safe
  | fixedMidgame q level mid requested mobile ran =>
    obtain ⟨_,same,post⟩ := MultiMidgame.prepareAt_returns q p c fc level mid requested none _ good checked mobile
    rw [ran] at same
    cases same
    exact post.2.2.2
  | midgamePrefix q level mid requested point mobile ran =>
    obtain ⟨_,same,safe⟩ := MultiMidgamePrefix.returns q p c fc level mid requested point _ good checked mobile
    rw [ran] at same
    cases same
    exact safe

/-- The only additional writes come from Multi-PV requests. Single-PV requests and
    shared-table operations can occur between any two Multi-PV requests. -/
abbrev History (p : Pool) (c : SerialScout.Config) (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig)
    {l : PvsParallel.SerialTable.Layout} (m : Memory l) :=
  MidgameSession.Generated p c mc fc (fun _ => Action p c mc fc) m

def Contract : Prop := ∀ (l : PvsParallel.SerialTable.Layout) (p : Pool) (c : SerialScout.Config) mc fc
    (q : Repeated.Call mc) (ops : MultiBatch.Inputs mc) (clock : MultiPublic.Clock) (m : Memory l) certificate,
    c.address = p.address → History p c mc fc m → clock.ordered ((NativeRootBirth.initial mc q.board).length-1) →
    Std.Tactic.BVDecide.LRAT.check certificate (LegalMoves.MoveBound.dimacsFormula 35) = true →
    Board.hasLegalMoves mc q.board.val = true → MultiSession.Guarantee q p c fc ops clock m

theorem from_engine : Contract := by
  intro l p c mc fc q ops clock m cert addr history time checked mobile
  have ready := MidgameSession.Generated.ready history addr checked (fun _ _ step => step.ready addr checked)
  exact MultiSession.exact l p c mc fc q ops clock m cert addr ready time checked mobile

end NeuralReversi.AspirationMir.SerialTable.MultiHistory
