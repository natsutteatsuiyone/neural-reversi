module
public import NeuralReversi.Board.InitializationModel
public import NeuralReversi.AspirationMir.SerialTable.MidgameIterate
public import NeuralReversi.AspirationMir.SerialTable.MidgameRoot
public import NeuralReversi.PvsParallel.SerialTable.MidgameChild

@[expose] public section

namespace NeuralReversi.AspirationMir.SerialTable.TimedMidgame

open EndgameSearch.CacheHistory (Pool)
open PvsParallel
open PvsParallel.SerialTable (Memory Layout MidgameChild.Plan)
open PvsBoardTree (Position)
open Repeated (ReadyMemory)
open RootMoveStorage (Row)
open MidgameIterate (State Clock Outcome)
open MidgameAspiration (Attempt)
open TranspositionTable (Selectivity)

structure Frame (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) where
  scratch : MoveList.Initialization.Memory Move.Move
  order : SerialRecursion.RootInput
  children : (q : SerialCalls.Request) → PvsParallel.SerialTable.MidgameChild.Plan mc fc q

abbrev Inputs (l : Layout) (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) :=
  Nat → State (Memory l) → Attempt → Frame mc fc

def trial (p : Pool) (b : Position) (paired : Bool) (level : Selectivity) (ops : Inputs l mc fc) :
    MidgameIterate.Trial (Memory l) := fun d s w =>
  let f := ops d s w
  MidgameRoot.run mc fc b (BitVec.ofNat 32 d) paired f.scratch f.order level
    (PvsParallel.SerialTable.MidgameChild.run p f.children) s.rows (BitVec.ofInt 32 w.alpha) (BitVec.ofInt 32 w.beta) s.memory

theorem trial_works (p : Pool) (b : Position) (paired : Bool) (level : Selectivity) (ops : Inputs l mc fc) :
    MidgameIterate.Works mc b (ReadyMemory p mc fc) (trial p b paired level ops) := by
  intro d s w ok win
  obtain ⟨below,ordered,above⟩ := win
  have repr {x : Int} (low : -32767 ≤ x) (high : x ≤ 32767) :
      (BitVec.ofInt 32 x).toInt = x := BitVec.toInt_ofInt_eq_self (by decide) (by omega) (by omega)
  have la := repr below (by omega)
  have hz := repr (by omega) above
  let f := ops d s w
  have children : MidgameRoot.ChildReturns (l := l) p mc fc (PvsParallel.SerialTable.MidgameChild.run p f.children) :=
    fun q m window _ ready => PvsParallel.SerialTable.MidgameChild.returns p f.children q m window ready
  obtain ⟨ans,m,ran,ready,finite,inside,keys,hist,highest⟩ := MidgameRoot.returns p mc fc b
    (BitVec.ofNat 32 d) paired f.scratch f.order level _ s.rows (BitVec.ofInt 32 w.alpha) (BitVec.ofInt 32 w.beta)
    s.memory children ok.2.2.2 ok.1 ok.2.1 (by rw [la,hz]; exact ⟨below,ordered,above⟩)
  have count : ans.rows.length = s.rows.length := by simpa using congrArg List.length keys
  rw [la,hz] at inside
  rw [la] at highest
  exact ⟨ans,m,ran,⟨keys ▸ ok.1,count ▸ ok.2.1,hist ok.2.2.1,ready⟩,finite,inside,highest⟩

def prepare (q : Repeated.Call mc) (p : Pool) (c : SerialScout.Config) (fc : Cutoffs.FlipConfig)
    (level : Selectivity) (ops : Inputs l mc fc) (requested : Nat) (clock : Option (Clock (Memory l))) (m : Memory l) : Option (Outcome (Memory l)) := do
  let rows ← NativeRootBirth.started mc fc q.board c.paired q.scratch
  if q.board.val = Board.newBoard then return .done ⟨rows,Repeated.increment m⟩
  let target := min (max requested 1) (Board.emptyCount q.board.val).toNat
  MidgameIterate.run (trial p q.board c.paired level ops) q.sort level.val target clock ⟨rows,Repeated.increment m⟩

/-- The clock is polled after each completed endgame level with its zero-based step.
    Four decisions suffice since even a cutoff after the fifth level returns that level. -/
def completedLevel (clock : Option (Clock M)) : {n : Nat // 1 ≤ n ∧ n ≤ 5} :=
  let kept := ([0,1,2,3] : List Nat).takeWhile (fun step => !(clock.any (fun tm => tm.endgameStop step)))
  ⟨kept.length + 1,by
    have bound : kept.length ≤ 4 := (List.takeWhile_prefix (l := [0,1,2,3]) _).length_le
    omega⟩

/-- A stopped clock can report the completed midgame instead of entering the endgame.
    On the handoff path the score comes from the captured best row; the next
    search uses the live rows and memory, without incrementing the generation. -/
def run (q : Repeated.Call mc) (p : Pool) (c : SerialScout.Config) (fc : Cutoffs.FlipConfig)
    (level : Selectivity) (ops : Inputs l mc fc) (requested : Nat) (clock : Option (Clock (Memory l))) (m : Memory l) :
    Option (Option PublicationMir.SearchResult × Memory l) := do
  match ← prepare q p c fc level ops requested clock m with
  | .done s => return (none,s.memory)
  | .endgame out =>
    let start := SelectiveLadder.Start.mk out.state.rows (SerialEntry.firstMachine (BitVec.ofInt 32 out.base.score)) out.state.memory
    let last := completedLevel clock
    let (result,next) ← Repeated.finish p c fc {q with last} start
    if last.val < 2 then
      let saved ← PublicationMir.intoResult (some out.saved) q.nodes q.fallback
      return (some saved,next)
    return (some result,next)

theorem prepare_returns (q : Repeated.Call mc) (p : Pool) (c : SerialScout.Config) (fc : Cutoffs.FlipConfig)
    (level : Selectivity) (ops : Inputs l mc fc) (requested : Nat) (clock : Option (Clock (Memory l))) (m : Memory l)
    (ok : ReadyMemory p mc fc m)
    (checked : Std.Tactic.BVDecide.LRAT.check certificate (LegalMoves.MoveBound.dimacsFormula 35) = true)
    (mobile : Board.hasLegalMoves mc q.board.val = true) :
    ∃ out, prepare q p c fc level ops requested clock m = some out ∧
      MidgameIterate.Ensures mc q.board (ReadyMemory p mc fc) level.val
        (min (max requested 1) (Board.emptyCount q.board.val).toNat) out := by
  have born := started_of_certificate mc fc q.board c.paired q.scratch checked mobile
  obtain ⟨cover,pos⟩ := NativeRootBirth.covers born
  have histories : MidgameRows.Valid (NativeRootBirth.initial mc q.board) := by
    intro r member
    obtain ⟨sq,_,rfl⟩ := List.mem_map.mp member
    exact Or.inl rfl
  by_cases opening : q.board.val = Board.newBoard
  · exact ⟨.done ⟨NativeRootBirth.initial mc q.board,Repeated.increment m⟩,by simp [prepare,born,opening],cover,List.length_pos_iff.mpr pos,histories,Repeated.begin_valid ok⟩
  let target := min (max requested 1) (Board.emptyCount q.board.val).toNat
  obtain ⟨out,ran,post⟩ := MidgameIterate.returns (trial_works p q.board c.paired level ops)
    ⟨NativeRootBirth.initial mc q.board,Repeated.increment m⟩
    ⟨cover,List.length_pos_iff.mpr pos,histories,Repeated.begin_valid ok⟩ q.sort level.val target clock
  exact ⟨out,by simpa only [prepare,born,bind,Option.bind,ite_eq_right opening] using ran,post⟩

theorem returns (q : Repeated.Call mc) (p : Pool) (c : SerialScout.Config) (fc : Cutoffs.FlipConfig)
    (level : Selectivity) (ops : Inputs l mc fc) (requested : Nat) (clock : Option (Clock (Memory l))) (m : Memory l)
    (addr : c.address = p.address) (ok : ReadyMemory p mc fc m)
    (checked : Std.Tactic.BVDecide.LRAT.check certificate (LegalMoves.MoveBound.dimacsFormula 35) = true)
    (mobile : Board.hasLegalMoves mc q.board.val = true) :
    ∃ out report next, prepare q p c fc level ops requested clock m = some out ∧
      run q p c fc level ops requested clock m = some (report,next) ∧ ReadyMemory p mc fc next ∧
      match out with
      | .done s => report = none ∧ next = s.memory
      | .endgame handoff => PublicationMir.first handoff.saved = some handoff.base ∧
        ∃ r, report = some r ∧ PublicationMir.ExactScore.scoreIfExact r = ({q with last := completedLevel clock} : Repeated.Call mc).expected fc := by
  obtain ⟨out,call,post⟩ := prepare_returns q p c fc level ops requested clock m ok checked mobile
  cases out with
  | done s => exact ⟨_,none,s.memory,call,by simp [run,call],post.2.2.2,rfl,rfl⟩
  | endgame h =>
    obtain ⟨valid,captured,_,_,midgame,range⟩ := post
    let word := BitVec.ofInt 32 h.base.score
    have fits : word.toInt = h.base.score := BitVec.toInt_ofInt_eq_self (by decide)
      (by unfold PvsMoveLoop.scoreMax at range; omega) (by unfold PvsMoveLoop.scoreMax at range; omega)
    let last := completedLevel clock
    obtain ⟨r,next,finish,ready,score⟩ := Repeated.finish_returns p c fc {q with last}
      ⟨h.state.rows,SerialEntry.firstMachine word,h.state.memory⟩ addr
      ⟨valid.1,valid.2.1,valid.2.2.2,rfl,SerialEntry.firstMachine_window word (fits ▸ range.1) (fits ▸ range.2)⟩
    dsimp only [last,word] at finish
    by_cases early : last.val < 2
    · let saved := PublicationMir.build h.saved.rows h.base h.saved.depth h.saved.selectivity h.saved.endgame q.nodes
      have replay : PublicationMir.intoResult (some h.saved) q.nodes q.fallback = some saved := by
        simp [PublicationMir.intoResult,captured,saved]
      have weak : last.val ≠ 5 := by omega
      refine ⟨_,some saved,next,call,?_,ready,captured,saved,rfl,?_⟩
      · dsimp only [last] at early
        simp [run,call,finish,early,replay]
      · change PublicationMir.ExactScore.scoreIfExact saved = (if last.val = 5 then _ else none)
        simp [saved,PublicationMir.ExactScore.scoreIfExact,PublicationMir.build,midgame,weak]
    · dsimp only [last] at early
      exact ⟨_,some r,next,call,by simp [run,call,finish,early],
        ready,captured,r,rfl,score⟩

theorem reaches_endgame (q : Repeated.Call mc) (p : Pool) (c : SerialScout.Config) (fc : Cutoffs.FlipConfig)
    (level : Selectivity) (ops : Inputs l mc fc) (requested : Nat) (m : Memory l)
    (ok : ReadyMemory p mc fc m)
    (checked : Std.Tactic.BVDecide.LRAT.check certificate (LegalMoves.MoveBound.dimacsFormula 35) = true)
    (mobile : Board.hasLegalMoves mc q.board.val = true) (opening : q.board.val ≠ Board.newBoard) :
    ∃ out, prepare q p c fc level ops requested (some MidgameIterate.continueToEnd) m = some (.endgame out) := by
  have born := started_of_certificate mc fc q.board c.paired q.scratch checked mobile
  obtain ⟨cover,pos⟩ := NativeRootBirth.covers born
  have histories : MidgameRows.Valid (NativeRootBirth.initial mc q.board) := by
    intro r member
    obtain ⟨sq,_,rfl⟩ := List.mem_map.mp member
    exact Or.inl rfl
  obtain ⟨n,legal⟩ := (Board.hasLegalMoves_exact mc q.board.val).mp mobile
  let sq : PvsBoardTree.GeneratedSquare mc q.board.val := ⟨⟨n,legal.1⟩,(Board.getMoves_exact mc q.board.val _).mpr legal⟩
  have positive := (ExactDepth.move_depth mc fc q.board _ sq rfl).1
  let target := min (max requested 1) (Board.emptyCount q.board.val).toNat
  obtain ⟨out,ran,_⟩ := MidgameIterate.to_end (trial_works p q.board c.paired level ops)
    ⟨NativeRootBirth.initial mc q.board,Repeated.increment m⟩
    ⟨cover,List.length_pos_iff.mpr pos,histories,Repeated.begin_valid ok⟩ q.sort level.val target (by unfold target; omega)
  exact ⟨out,by simpa only [prepare,born,bind,Option.bind,ite_eq_right opening] using ran⟩

end NeuralReversi.AspirationMir.SerialTable.TimedMidgame
