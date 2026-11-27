import NeuralReversi.AspirationMir.SerialTable.InterruptedAttempt
import NeuralReversi.PublicationMir.SerialFallback

namespace NeuralReversi.AspirationMir.SerialTable.InterruptedPublic

open PvsParallel
open PvsParallel.SerialTable (Layout Memory)
open EndgameSearch.CacheHistory (Pool)
open PvsBoardTree (Position minimax)
open RootMoveStorage (Row)
open TranspositionTable (Selectivity)
open Repeated (ReadyMemory)
open SelectiveLadder (Start)
open PublicationMir (Iteration SliceSort)
open PublicationMir.InterruptedExact (SavedSafe SnapshotSafe)

/-- The two root-side checks are distinct: an abort before capture keeps the
    old snapshot, whereas the clock/abort observed after capture keeps the new
    completed snapshot. Child and aspiration-root polls are inherited. -/
structure Controls extends InterruptedAttempt.Stops where
  beforeCapture : Nat → SerialStack.State → Nat → Bool
  afterCapture : Nat → SerialStack.State → Nat → Bool

structure Finished (l : Layout) where
  last : Option Iteration
  memory : Memory l

/-- A saved snapshot can be returned without another sort or a partial head
    operation. Exact-tagged snapshots additionally retain their true maximum. -/
def SavedReady (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : Position)
    (last : Option Iteration) : Prop :=
  (∀ it, last = some it → it.rows ≠ []) ∧
    SavedSafe (minimax mc fc b) (last.map PublicationMir.asSnapshot)

theorem saved_none : SavedReady mc fc b none := by
  constructor <;> simp [SavedSafe]

theorem rowScores_perm {truth : Int} {rows prior : List Row} (perm : rows.Perm prior)
    (scores : PublicationMir.ExactScore.RowScores truth prior) :
    PublicationMir.ExactScore.RowScores truth rows := by
  obtain ⟨upper,winner,member,score⟩ := scores
  exact ⟨fun row present => upper row (perm.mem_iff.mp present),
    winner,perm.mem_iff.mpr member,score⟩

theorem captured_ready (sort : SliceSort) (rows : List Row) (depth level : Nat)
    (positive : 0 < rows.length)
    (exactRows : level = 5 → PublicationMir.ExactScore.RowScores (minimax mc fc b) rows) :
    SavedReady mc fc b (some (PublicationMir.capture sort rows depth level true)) := by
  have perm := sort.permutation PublicationMir.key rows
  constructor
  · intro it found
    cases found
    intro empty
    have size := perm.length_eq
    change (PublicationMir.capture sort rows depth level true).rows.length = rows.length at size
    rw [empty] at size
    simp only [List.length_nil] at size
    omega
  · intro saved found
    simp only [Option.map_some,Option.some.injEq] at found
    subst saved
    intro _ full
    constructor
    · change PublicationMir.InterruptedExact.RowsExact (minimax mc fc b)
        ((PublicationMir.capture sort rows depth level true).rows.map PublicationMir.project)
      rw [PublicationMir.InterruptedExact.rowsExact_project]
      exact rowScores_perm perm (exactRows full)
    · apply List.pairwise_map.mpr
      exact sort.ordered PublicationMir.key rows

/-- A midgame handoff need not have an exact score, but it must contain the
    completed rows actually retained by the caller. -/
theorem saved_midgame (it : Iteration) (positive : it.rows ≠ []) (midgame : it.endgame = false) :
    SavedReady mc fc b (some it) := by
  constructor
  · intro saved found
    cases found
    exact positive
  · intro saved found
    simp only [Option.map_some,Option.some.injEq] at found
    subst saved
    intro ending
    change it.endgame = true at ending
    cases midgame.symm.trans ending

/-- Execute the single-PV selectivity loop. Only a completed, accepted attempt
    may replace `last`. Every exit uses the memory returned by that attempt,
    including an interrupted attempt with no publishable answer. -/
def loop (p : Pool) (c : SerialScout.Config) (b : Position) (ops : FullLadder.Inputs mc)
    (fc : Cutoffs.FlipConfig) (minimum : Nat) (controls : Controls) (sort : SliceSort) :
    List Selectivity → Start (Memory l) → Option Iteration → Option (Finished l)
  | [], start, last => some ⟨last,start.memory⟩
  | level :: levels, start, last => do
    let (answer,after) ← InterruptedAttempt.run p c b ops fc level controls.toStops 12
      (SerialStack.first start.machine start.rows) start.memory
    match answer with
    | none => return ⟨last,after⟩
    | some visit =>
      if controls.beforeCapture level.val visit.state visit.closing then return ⟨last,after⟩
      let next := SelectiveLadder.successor visit
      let saved := if minimum ≤ level.val then
        some (PublicationMir.capture sort next.rows (Board.emptyCount b.val).toNat level.val true)
        else last
      if controls.afterCapture level.val visit.state visit.closing then return ⟨saved,after⟩
      loop p c b ops fc minimum controls sort levels next saved

theorem loop_returns (p : Pool) (c : SerialScout.Config) (b : Position) (ops : FullLadder.Inputs mc)
    (fc : Cutoffs.FlipConfig) (minimum : Nat) (controls : Controls) (sort : SliceSort)
    (levels : List Selectivity) (start : Start (Memory l)) (last : Option Iteration)
    (address : c.address = p.address) (initial : start.valid mc b (ReadyMemory p mc fc))
    (saved : SavedReady mc fc b last) :
    ∃ out, loop p c b ops fc minimum controls sort levels start last = some out ∧
      ReadyMemory p mc fc out.memory ∧ SavedReady mc fc b out.last := by
  induction levels generalizing start last with
  | nil => exact ⟨⟨last,start.memory⟩,rfl,initial.2.2.1,saved⟩
  | cons level levels ih =>
    obtain ⟨cover,positive,good,_,window⟩ := initial
    obtain ⟨answer,after,ran,safe,accepted⟩ := InterruptedAttempt.returns p c b ops fc level
      controls.toStops 12 (SerialStack.first start.machine start.rows) start.memory address good
      cover positive (first_ready _ window) rfl
    cases answer with
    | none => exact ⟨⟨last,after⟩,by simp [loop,ran],safe,saved⟩
    | some visit =>
      obtain ⟨same,proof⟩ := accepted visit rfl
      by_cases stopped : controls.beforeCapture level.val visit.state visit.closing = true
      · exact ⟨⟨last,after⟩,by simp [loop,ran,stopped],safe,saved⟩
      · let next := SelectiveLadder.successor visit
        let updated := if minimum ≤ level.val then
          some (PublicationMir.capture sort next.rows (Board.emptyCount b.val).toNat level.val true)
          else last
        have valid : next.valid mc b (ReadyMemory p mc fc) := proof.toAccepted.successor_valid
        have retained : SavedReady mc fc b updated := by
          dsimp only [updated]
          split
          · exact captured_ready sort next.rows _ _ valid.2.1 (fun full =>
              rowScores_perm (RootMoveStorage.sortRowsContract_proven visit.answer.rows 0).1
                (proof.exactRows full))
          · exact saved
        by_cases finished : controls.afterCapture level.val visit.state visit.closing = true
        · exact ⟨⟨updated,after⟩,by simp [loop,ran,stopped,finished,next,updated],safe,retained⟩
        · obtain ⟨out,rest,good,snapshot⟩ := ih next updated valid retained
          exact ⟨out,by simpa [loop,ran,stopped,finished,next,updated] using rest,good,snapshot⟩

/-- The native five-level ladder, truncated at the requested selectivity. -/
def levels (last : Nat) : List Selectivity :=
  ([⟨1,by decide⟩,⟨2,by decide⟩,⟨3,by decide⟩,⟨4,by decide⟩,⟨5,by decide⟩] : List Selectivity).filter
    (fun level => level.val ≤ last)

/-- Read the already captured result, or the actual TT left by this execution.
    No externally supplied completion history, child score or fallback result
    is an input to this return path. -/
def solve (p : Pool) (c : SerialScout.Config) (fc : Cutoffs.FlipConfig) (q : Repeated.Call mc)
    (minimum : Nat) (controls : Controls) (evaluate : Board.Board → Int)
    (start : Start (Memory l)) (last : Option Iteration) :
    Option (PublicationMir.SearchResult × Memory l) := do
  let out ← loop p c q.board q.moves fc minimum controls q.sort (levels q.last.val) start last
  let result ← PublicationMir.intoResult out.last q.nodes
    (fun _ => PublicationMir.SerialFallback.fallback out.memory.tt mc fc q.board evaluate q.nodes)
  return (result,out.memory)

private theorem saved_returns (last : Option Iteration) (nodes : Nat)
    (fallback : Unit → PublicationMir.SearchResult) (nonempty : ∀ it, last = some it → it.rows ≠ []) :
    ∃ result, PublicationMir.intoResult last nodes fallback = some result := by
  cases last with
  | none => exact ⟨fallback (),rfl⟩
  | some it =>
    obtain ⟨head,tail,rows⟩ := List.exists_cons_of_ne_nil (nonempty it rfl)
    exact ⟨PublicationMir.build it.rows head it.depth it.selectivity it.endgame nodes,
      by simp [PublicationMir.intoResult,PublicationMir.first,rows]⟩

theorem solve_returns (p : Pool) (c : SerialScout.Config) (fc : Cutoffs.FlipConfig) (q : Repeated.Call mc)
    (minimum : Nat) (controls : Controls) (evaluate : Board.Board → Int)
    (start : Start (Memory l)) (last : Option Iteration) (address : c.address = p.address)
    (initial : start.valid mc q.board (ReadyMemory p mc fc)) (saved : SavedReady mc fc q.board last) :
    ∃ result after, solve p c fc q minimum controls evaluate start last = some (result,after) ∧
      ReadyMemory p mc fc after ∧ PublicationMir.ExactScore.Ensures mc fc q.board result := by
  obtain ⟨out,ran,safe,kept⟩ := loop_returns p c q.board q.moves fc minimum controls q.sort
    (levels q.last.val) start last address initial saved
  let fallback := fun _ : Unit => PublicationMir.SerialFallback.fallback out.memory.tt mc fc q.board evaluate q.nodes
  obtain ⟨result,returned⟩ := saved_returns out.last q.nodes fallback kept.1
  exact ⟨result,out.memory,by simp [solve,ran,returned,fallback],safe,
    PublicationMir.SerialFallback.intoResult_ensures out.memory.tt mc fc q.board evaluate q.sort
      out.last q.nodes result safe.1.1.1 kept.2 returned⟩

/-- A direct request constructs its own root vector, advances the TT generation
    and reads its initial aspiration estimate from the retained table. -/
def direct (p : Pool) (c : SerialScout.Config) (fc : Cutoffs.FlipConfig) (q : Repeated.Call mc)
    (controls : Controls) (evaluate : Board.Board → Int) (before : Memory l) := do
  let rows ← NativeRootBirth.started mc fc q.board c.paired q.scratch
  let prior := Repeated.increment before
  let base := InitialEstimate.estimate prior.tt.image mc fc q.raw q.board
  solve p c fc q 1 controls evaluate ⟨rows,SerialEntry.firstMachine base,prior⟩ none

/-- Capacity is checked on this board, rather than assumed through an
    unavailable global mobility certificate. In particular, at most 35 empty
    squares suffice for this premise. -/
theorem direct_returns (p : Pool) (c : SerialScout.Config) (fc : Cutoffs.FlipConfig) (q : Repeated.Call mc)
    (controls : Controls) (evaluate : Board.Board → Int) (before : Memory l)
    (address : c.address = p.address) (initial : ReadyMemory p mc fc before)
    (capacity : (PvsBoardTree.legalSquares mc q.board.val).length ≤ NativeRootBirth.capacity)
    (mobile : Board.hasLegalMoves mc q.board.val = true) :
    ∃ result after, direct p c fc q controls evaluate before = some (result,after) ∧
      ReadyMemory p mc fc after ∧ PublicationMir.ExactScore.Ensures mc fc q.board result := by
  have nonempty : PvsBoardTree.legalSquares mc q.board.val ≠ [] :=
    fun empty => Bool.noConfusion ((PvsBoardTree.stuck_of_nil empty).symm.trans mobile)
  have born := (NativeRootBirth.started_iff mc fc q.board c.paired q.scratch).2 ⟨capacity,nonempty,rfl⟩
  obtain ⟨cover,positive⟩ := NativeRootBirth.covers born
  let prior := Repeated.increment before
  have ready : ReadyMemory p mc fc prior := Repeated.begin_valid initial
  have range := InitialEstimate.warm_bounded prior.tt mc fc q.raw q.board prior.supported ready.2
  let base := InitialEstimate.estimate prior.tt.image mc fc q.raw q.board
  obtain ⟨result,after,ran,safe,score⟩ := solve_returns p c fc q 1 controls evaluate
    ⟨NativeRootBirth.initial mc q.board,SerialEntry.firstMachine base,prior⟩ none address
    ⟨cover,List.length_pos_iff.mpr positive,ready,rfl,SerialEntry.firstMachine_window base range.1 range.2⟩ saved_none
  exact ⟨result,after,by simpa [direct,born,prior,base] using ran,safe,score⟩

/-- Non-vacuous public-score guarantee for every stop schedule of the serial,
    single-PV driver, from a cold table and a mobile board with at most 35 empty
    squares. All child results, retries, saved rows and fallback probes are
    computed by the driver. -/
def Contract : Prop :=
  ∀ (l : Layout) (g : BitVec 8) (p : Pool) mc fc (c : SerialScout.Config) (q : Repeated.Call mc)
    (controls : Controls) (evaluate : Board.Board → Int),
    c.address = p.address → (Board.emptyCount q.board.val).toNat ≤ 35 →
    Board.hasLegalMoves mc q.board.val = true →
    ∃ result after, direct p c fc q controls evaluate (Memory.empty l g p) = some (result,after) ∧
      ReadyMemory p mc fc after ∧ PublicationMir.ExactScore.Ensures mc fc q.board result

theorem cold : Contract := by
  intro l g p mc fc c q controls evaluate address small mobile
  apply direct_returns p c fc q controls evaluate (Memory.empty l g p) address
    ⟨Memory.cold_valid ..,PvsParallel.SerialTable.exactScores_empty⟩ _ mobile
  have bound := legalMoveCount_le_empties mc q.board
  unfold NativeRootBirth.capacity
  omega

end NeuralReversi.AspirationMir.SerialTable.InterruptedPublic
