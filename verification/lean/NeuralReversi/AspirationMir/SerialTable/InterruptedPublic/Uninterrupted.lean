module
public import NeuralReversi.AspirationMir.SerialTable.InterruptedPublic

@[expose] public section

namespace NeuralReversi.AspirationMir.SerialTable.InterruptedPublic.Uninterrupted

open PvsParallel
open PvsParallel.SerialTable (Layout Memory)
open EndgameSearch.CacheHistory (Pool)
open PvsBoardTree (Position)
open TranspositionTable (Selectivity)
open Repeated (ReadyMemory)
open SelectiveLadder (Start)
open PublicationMir (Iteration SliceSort)

/-- No stop is observed in a child, after a root call, or on either side of
    snapshot capture. The ordinary selectivity ladder therefore runs to its end. -/
def controls : Controls where
  toStops := InterruptedAttempt.noStops
  beforeCapture := fun _ _ _ => false
  afterCapture := fun _ _ _ => false

/-- Any finite sequence of levels followed by a reportable final level really
    captures that final level. This rules out satisfying the public-score
    guarantee merely by returning an unmarked fallback on every execution. -/
theorem loop_last (p : Pool) (c : SerialScout.Config) (b : Position) (ops : FullLadder.Inputs mc)
    (fc : Cutoffs.FlipConfig) (minimum : Nat) (sort : SliceSort) (earlier : List Selectivity)
    (level : Selectivity) (start : Start (Memory l)) (last : Option Iteration)
    (address : c.address = p.address) (initial : start.valid mc b (ReadyMemory p mc fc))
    (reportable : minimum ≤ level.val) :
    ∃ out it, loop p c b ops fc minimum controls sort (earlier ++ [level]) start last = some out ∧
      out.last = some it ∧ it.endgame = true ∧ it.selectivity = level.val := by
  induction earlier generalizing start last with
  | nil =>
    obtain ⟨visit,ran,accepted⟩ := InterruptedAttempt.uninterrupted_returns p c b ops fc level 12
      (SerialStack.first start.machine start.rows) start.memory address initial.2.2.1
      initial.1 initial.2.1 (first_ready _ initial.2.2.2.2) rfl
    let saved := PublicationMir.capture sort (SelectiveLadder.successor visit).rows
      (Board.emptyCount b.val).toNat level.val true
    exact ⟨⟨some saved,visit.after⟩,saved,
      by simp [loop,controls,ran,reportable,saved,SelectiveLadder.successor],rfl,rfl,rfl⟩
  | cons next earlier ih =>
    obtain ⟨visit,ran,accepted⟩ := InterruptedAttempt.uninterrupted_returns p c b ops fc next 12
      (SerialStack.first start.machine start.rows) start.memory address initial.2.2.1
      initial.1 initial.2.1 (first_ready _ initial.2.2.2.2) rfl
    let updated := if minimum ≤ next.val then
      some (PublicationMir.capture sort (SelectiveLadder.successor visit).rows
        (Board.emptyCount b.val).toNat next.val true) else last
    obtain ⟨out,it,rest,found,ending,final⟩ :=
      ih (SelectiveLadder.successor visit) updated accepted.toAccepted.successor_valid
    exact ⟨out,it,by simpa [loop,controls,ran,updated] using rest,found,ending,final⟩

private theorem saved_tagged (last : Option Iteration) (it : Iteration) (nodes : Nat)
    (fallback : Unit → PublicationMir.SearchResult) (result : PublicationMir.SearchResult)
    (found : last = some it) (ending : it.endgame = true) (full : it.selectivity = 5)
    (run : PublicationMir.intoResult last nodes fallback = some result) :
    ∃ value, PublicationMir.ExactScore.scoreIfExact result = some value := by
  rw [found,PublicationMir.intoResult] at run
  cases first : PublicationMir.first it with
  | none => simp [first] at run
  | some row =>
    simp only [first,bind,Option.bind,pure,Pure.pure,Option.some.injEq] at run
    subst result
    exact ⟨row.score,by simp [PublicationMir.build,PublicationMir.ExactScore.scoreIfExact,ending,full]⟩

/-- An uninterrupted request for exact selectivity returns an exact-tagged
    minimax result, rather than only a conditional guarantee about its tag. -/
theorem solve_exact (p : Pool) (c : SerialScout.Config) (fc : Cutoffs.FlipConfig) (q : Repeated.Call mc)
    (minimum : Nat) (evaluate : Board.Board → Int) (start : Start (Memory l))
    (last : Option Iteration) (address : c.address = p.address)
    (initial : start.valid mc q.board (ReadyMemory p mc fc)) (saved : SavedReady mc fc q.board last)
    (full : q.last.val = 5) (reportable : minimum ≤ 5) :
    ∃ result after, solve p c fc q minimum controls evaluate start last = some (result,after) ∧
      ReadyMemory p mc fc after ∧ PublicationMir.ExactScore.scoreIfExact result =
        some (256 * EndgameTreeCache.discValue mc fc q.board.val) := by
  obtain ⟨result,after,ran,safe,exactScore⟩ := solve_returns p c fc q minimum controls evaluate
    start last address initial saved
  let earlier : List Selectivity := [⟨1,by decide⟩,⟨2,by decide⟩,⟨3,by decide⟩,⟨4,by decide⟩]
  let final : Selectivity := ⟨5,by decide⟩
  have ladder : levels q.last.val = earlier ++ [final] := by rw [full]; rfl
  obtain ⟨out,it,computed,found,ending,level⟩ := loop_last p c q.board q.moves fc minimum q.sort
    earlier final start last address initial reportable
  rw [← ladder] at computed
  have body := ran
  simp only [solve,computed,bind,Option.bind] at body
  obtain ⟨r,returned,eq⟩ := Option.bind_eq_some_iff.mp body
  simp only [pure,Pure.pure,Option.some.injEq,Prod.mk.injEq] at eq
  rw [eq.1] at returned
  obtain ⟨value,marked⟩ := saved_tagged out.last it q.nodes _ result found ending level returned
  exact ⟨result,after,ran,safe,(exactScore value marked) ▸ marked⟩

theorem direct_exact (p : Pool) (c : SerialScout.Config) (fc : Cutoffs.FlipConfig) (q : Repeated.Call mc)
    (evaluate : Board.Board → Int) (before : Memory l) (address : c.address = p.address)
    (initial : ReadyMemory p mc fc before)
    (capacity : (PvsBoardTree.legalSquares mc q.board.val).length ≤ NativeRootBirth.capacity)
    (mobile : Board.hasLegalMoves mc q.board.val = true) (full : q.last.val = 5) :
    ∃ result after, direct p c fc q controls evaluate before = some (result,after) ∧
      ReadyMemory p mc fc after ∧ PublicationMir.ExactScore.scoreIfExact result =
        some (256 * EndgameTreeCache.discValue mc fc q.board.val) := by
  have nonempty : PvsBoardTree.legalSquares mc q.board.val ≠ [] :=
    fun empty => Bool.noConfusion ((PvsBoardTree.stuck_of_nil empty).symm.trans mobile)
  have born := (NativeRootBirth.started_iff mc fc q.board c.paired q.scratch).2 ⟨capacity,nonempty,rfl⟩
  obtain ⟨cover,positive⟩ := NativeRootBirth.covers born
  let prior := Repeated.increment before
  have ready : ReadyMemory p mc fc prior := Repeated.begin_valid initial
  have range := InitialEstimate.warm_bounded prior.tt mc fc q.raw q.board prior.supported ready.2
  let base := InitialEstimate.estimate prior.tt.image mc fc q.raw q.board
  obtain ⟨result,after,ran,safe,score⟩ := solve_exact p c fc q 1 evaluate
    ⟨NativeRootBirth.initial mc q.board,SerialEntry.firstMachine base,prior⟩ none address
    ⟨cover,List.length_pos_iff.mpr positive,ready,rfl,SerialEntry.firstMachine_window base range.1 range.2⟩
    saved_none full (by decide)
  exact ⟨result,after,by simpa [direct,born,prior,base] using ran,safe,score⟩

def Contract : Prop :=
  ∀ (l : Layout) (g : BitVec 8) (p : Pool) mc fc (c : SerialScout.Config) (q : Repeated.Call mc)
    (evaluate : Board.Board → Int),
    c.address = p.address → (Board.emptyCount q.board.val).toNat ≤ 35 →
    Board.hasLegalMoves mc q.board.val = true → q.last.val = 5 →
    ∃ result after, direct p c fc q controls evaluate (Memory.empty l g p) = some (result,after) ∧
      ReadyMemory p mc fc after ∧ PublicationMir.ExactScore.scoreIfExact result =
        some (256 * EndgameTreeCache.discValue mc fc q.board.val)

theorem cold_exact : Contract := by
  intro l g p mc fc c q evaluate address small mobile full
  apply direct_exact p c fc q evaluate (Memory.empty l g p) address
    ⟨Memory.cold_valid ..,PvsParallel.SerialTable.exactScores_empty⟩ _ mobile full
  have bound := legalMoveCount_le_empties mc q.board
  unfold NativeRootBirth.capacity
  omega

end NeuralReversi.AspirationMir.SerialTable.InterruptedPublic.Uninterrupted
