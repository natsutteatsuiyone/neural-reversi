module
public import NeuralReversi.AspirationMir.SerialTable.MultiPublic

@[expose] public section

namespace NeuralReversi.AspirationMir.SerialTable.MultiSession

open EndgameSearch.CacheHistory (Pool)
open PvsParallel
open PvsParallel.SerialTable (Memory Layout)
open PvsMoveLoop (scoreMax)
open PvsBoardTree (Position)
open RootMoveStorage (Row)
open Repeated (ReadyMemory)
open MultiPublic (Clock direct timed unlimited)
open MultiBatch (Inputs)
open TranspositionTable (Selectivity)

theorem base_range (value : Int) (range : -scoreMax ≤ value ∧ value ≤ scoreMax) :
    -scoreMax ≤ (BitVec.ofInt 32 value).toInt ∧ (BitVec.ofInt 32 value).toInt ≤ scoreMax := by
  have signed : (BitVec.ofInt 32 value).toInt = value :=
    BitVec.toInt_ofInt_eq_self (by decide) (by unfold scoreMax at range; omega) (by unfold scoreMax at range; omega)
  simpa [signed] using range

def Guarantee (q : Repeated.Call mc) (p : Pool) (c : SerialScout.Config) (fc : Cutoffs.FlipConfig)
    (ops : Inputs mc) (clock : Clock) (m : Memory l) : Prop :=
    (∀ report next, direct q p c fc ops clock m = some (.bestMove report,next) →
      report.endgame = true → report.stamp.selectivity = 5 →
      MultiRows.ReportExact mc fc q.board report ∧ ReadyMemory p mc fc next) ∧
    (q.last.val = 5 → ∃ report next, direct q p c fc ops unlimited m = some (.bestMove report,next) ∧
      MultiRows.ReportExact mc fc q.board report ∧ ReadyMemory p mc fc next) ∧
    ∀ (level : Selectivity) (mid : MultiMidgame.Inputs l),
    (∀ time report next, timed q p c fc ops level mid time clock m = some (some (.bestMove report),next) →
      report.endgame = true → report.stamp.selectivity = 5 →
      MultiRows.ReportExact mc fc q.board report ∧ ReadyMemory p mc fc next) ∧
    (∃ report next, timed q p c fc ops level mid MidgameIterate.continueToEnd unlimited m =
      some (some (.bestMove report),next) ∧ MultiRows.ReportExact mc fc q.board report ∧ ReadyMemory p mc fc next)

/-- The public fields select exact results. This includes the score of every legal move,
    the head selected on a tie, and the memory returned to the next request. -/
def Contract : Prop := ∀ (l : Layout) (p : Pool) (c : SerialScout.Config) mc fc (q : Repeated.Call mc)
    (ops : Inputs mc) (clock : Clock) (m : Memory l) certificate,
    c.address = p.address → ReadyMemory p mc fc m → clock.ordered ((NativeRootBirth.initial mc q.board).length-1) →
    Std.Tactic.BVDecide.LRAT.check certificate (LegalMoves.MoveBound.dimacsFormula 35) = true →
    Board.hasLegalMoves mc q.board.val = true →
    Guarantee q p c fc ops clock m

theorem exact : Contract := by
  intro l p c mc fc q ops clock m cert addr good times checked mobile
  have boot := started_of_certificate mc fc q.board c.paired q.scratch checked mobile
  obtain ⟨cover,pos⟩ := NativeRootBirth.covers boot
  let now := Repeated.increment m
  have ready : ReadyMemory p mc fc now := Repeated.begin_valid good
  have range := InitialEstimate.warm_bounded now.tt mc fc q.raw q.board now.supported ready.2
  constructor
  · intro r next ran tag full
    simp only [direct,boot,bind,Option.bind] at ran
    exact MultiPublic.from_flags q p c fc ops _ _ _ clock _ _ now addr ready cover pos range
      .direct q.last.property.1 times ran tag full
  constructor
  · intro cap
    simpa only [direct,boot,bind,Option.bind,cap] using
      MultiPublic.unlimited_returns q p c fc ops 1 none _ _ now addr ready cover pos range (by decide)
  intro level mid
  constructor
  · intro time r next ran tag full
    obtain ⟨out,prepared,post⟩ := MultiMidgame.prepare_returns q p c fc level mid (some time) m good checked mobile
    cases out with
    | done s => simp [timed,prepared] at ran
    | endgame h =>
      obtain ⟨state,_,_,_,flag,bounds⟩ := post
      simp only [timed,prepared,bind,Option.bind] at ran
      obtain ⟨⟨result,after⟩,solved,eq⟩ := Option.bind_eq_some_iff.mp ran
      cases eq
      have len : h.state.rows.length = (NativeRootBirth.initial mc q.board).length := by
        simpa only [List.length_map] using (state.1.trans cover.symm).length_eq
      exact MultiPublic.from_flags q p c fc ops 5 2 _ clock _ _ _ addr state.2.2.2 state.1
        (List.length_pos_iff.mp state.2.1) (base_range _ bounds) (.handoff h.saved flag) (by decide)
        (len ▸ times) solved tag full
  · obtain ⟨h,prepared,post⟩ := MultiMidgame.reaches_endgame q p c fc level mid m good checked mobile
    obtain ⟨report,next,solve,correct,valid⟩ := MultiPublic.unlimited_returns q p c fc ops 2 (some h.saved)
      (BitVec.ofInt 32 h.base.score) h.state.rows h.state.memory addr post.valid.2.2.2 post.valid.1
      (List.length_pos_iff.mp post.valid.2.1) (base_range _ post.bounds) (by decide)
    exact ⟨report,next,by simp [timed,prepared,solve],correct,valid⟩

end NeuralReversi.AspirationMir.SerialTable.MultiSession
