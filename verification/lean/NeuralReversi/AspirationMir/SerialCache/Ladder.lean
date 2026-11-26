import NeuralReversi.AspirationMir.SerialCache.Physical
import NeuralReversi.AspirationMir.SerialEntry
import NeuralReversi.EndgameSearch.SerialWarmCache

namespace NeuralReversi.AspirationMir.SerialCache.Ladder

open PvsParallel
open PvsBoardTree (Position)
open SerialExecution (Inputs)
open EndgameSearch.CacheHistory (Pool Local)
open EndgameSearch.SerialLeafCache (Reachable)
open RootMoveStorage (Row)

/-- Inputs left after specializing the MIR move evaluator to a mobile root
    with at most thirteen empties. Sorting equal keys can vary by target. -/
structure Ordering (mc : Bitboard.Moves.X86Config) where
  probe : Nat → SerialExecution.Frame → List TranspositionTable.ProbeRetry.RawAttempt
  tie : Nat → (frame : SerialExecution.Frame) → PvsBoardTree.GeneratedSquare mc frame.request.board.val → Fin 64
  rootProbe : Word → Word → List TranspositionTable.ProbeRetry.RawAttempt
  rootSort : RootFrameMir.LibraryOrder
  pvLine : Bitboard.Square → List Bitboard.Square

def Ordering.inputs (o : Ordering mc) : Inputs mc fc := fun i frame =>
  ⟨fun _ => none,[],[],o.probe i frame,[],fun _ => .fast,o.tie i frame⟩

def Ordering.root (o : Ordering mc) (fc : Cutoffs.FlipConfig) (b : Position) (a z : Word) : SerialRecursion.RootInput :=
  let hint := GeneralNonPvMir.probeHint b (o.rootProbe a z)
  let key := EndgameOrderMir.actualKey (EndgameOrderMir.priority mc fc b (fun _ => .fast)) hint
  ⟨fun sq => BitVec.ofInt 32 (key sq),o.rootSort,o.pvLine⟩

theorem frame_fast (s : SerialExecution.Frame) (aligned : s.empties.matches s.request.board)
    (full : ExactDepth.Full s.request.board s.request.depth) (small : s.request.depth.toNat ≤ 13) :
    ∃ i : Fin 64, i.val = s.empties.count.toNat ∧ EndgameOrderMir.useFast s.request.depth i = true := by
  have same : s.empties.count = s.request.depth := aligned.1.trans full.symm
  refine ⟨⟨s.request.depth.toNat,by omega⟩,by rw [same],?_⟩
  have e := EndgameOrderMir.fullDepth_fast (⟨s.request.depth.toNat,by omega⟩ : Fin 64)
  simpa [small] using e

structure State where
  rows : List Row
  machine : Machine
  cache : Local

def State.ready (s : State) (p : Pool) (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig)
    (b : Position) : Prop :=
  Root.Covers mc b (s.rows.map Row.square) ∧ 0 < s.rows.length ∧ Reachable p mc fc s.cache ∧
    s.machine.pc = 0 ∧ Aspiration.Window s.machine.regs.alpha.toInt s.machine.regs.beta.toInt

def State.contains (s : State) (v : Int) : Prop :=
  s.machine.regs.alpha.toInt < v ∧ v < s.machine.regs.beta.toInt

/-- The next selectivity receives the accepted window's score, sorted root
    rows, and both worker arrays left by the previous selectivity. -/
def advance (p : Pool) (c : SerialScout.Config) (b : Position) (inputs : Nat → Nat → Inputs mc fc)
    (roots : Nat → Roots) (s : State) (level : Nat) : Option State := do
  let (_,last,cache) ← Physical.run p c b (inputs level) (roots level) 12
    (SerialStack.first s.machine s.rows) s.cache
  return ⟨RootMoveStorage.sortRows last.answer.rows 0,SerialEntry.nextMachine last.call.score,cache⟩

private theorem levels_ready (p : Pool) (c : SerialScout.Config) (b : Position)
    (inputs : Nat → Nat → Inputs mc fc) (roots : Nat → Roots) (sort : PublicationMir.SliceSort)
    (fallback : Unit → PublicationMir.SearchResult) (levels : List Nat) (s : State)
    (address : c.address = p.address) (small : (Board.emptyCount b.val).toNat ≤ 13) (initial : s.ready p mc fc b) :
    ∃ out, levels.foldlM (advance p c b inputs roots) s = some out ∧ out.ready p mc fc b ∧
      (s.contains (PvsBoardTree.minimax mc fc b) ∨ levels ≠ [] → out.contains (PvsBoardTree.minimax mc fc b)) := by
  induction levels generalizing s with
  | nil => exact ⟨s,rfl,initial,by simp⟩
  | cons level rest ih =>
    obtain ⟨covers,pos,boot,pc,window⟩ := initial
    obtain ⟨prior,last,post,_,ran,ok,_,_,_,_,_,_,sameRows,positive,score⟩ :=
      Physical.publication mc fc p c b (inputs level) (roots level) s.rows s.machine s.cache
        sort 0 fallback address
        (fun i => shared_thirteen _ _ small) boot covers pos pc window
    have range : -PvsMoveLoop.scoreMax ≤ PvsBoardTree.minimax mc fc b ∧
        PvsBoardTree.minimax mc fc b ≤ PvsMoveLoop.scoreMax := PvsMoveLoop.minimax_range _
    rw [← score] at range
    have next := SerialEntry.nextMachine_window _ range.1 range.2
    obtain ⟨out,tail,valid,contains⟩ := ih ⟨RootMoveStorage.sortRows last.answer.rows 0,SerialEntry.nextMachine last.call.score,post⟩
      ⟨sameRows,positive,ok,rfl,next.1⟩
    exact ⟨out,by simpa [List.foldlM_cons,advance,ran] using tail,valid,
      fun _ => contains (.inl (score ▸ next.2))⟩

def Contract : Prop :=
  ∀ mc fc (p : Pool) (c : SerialScout.Config) (b : Position) (orders : Nat → Nat → Ordering mc)
    (mem : MoveList.Initialization.Memory Move.Move) (base : Word) (visits : List (EndgameSearch.CacheHistory.Visit p))
    (sort : PublicationMir.SliceSort) nodes fallback, c.address = p.address →
    (Board.emptyCount b.val).toNat ≤ 13 → Board.hasLegalMoves mc b.val = true →
    -PvsMoveLoop.scoreMax ≤ base.toInt → base.toInt ≤ PvsMoveLoop.scoreMax →
    let inputs : Nat → Nat → Inputs mc fc := fun level trial => (orders level trial).inputs
    let roots : Nat → Roots := fun level trial => (orders level trial).root fc b
    ∃ rows state last post result,
      NativeRootBirth.started mc fc b c.paired mem = some rows ∧
      (OuterMir.ladder.take 4).foldlM (advance p c b inputs roots)
        ⟨rows,SerialEntry.firstMachine base,EndgameSearch.CacheHistory.replay mc fc p visits⟩ = some state ∧
      Physical.run p c b (inputs 5) (roots 5) 12 (SerialStack.first state.machine state.rows) state.cache =
        some ([],last,post) ∧ Reachable p mc fc post ∧
      (∀ previous, OuterMir.finish 1 5 sort (RootMoveStorage.sortRows last.answer.rows 0)
        (Board.emptyCount b.val).toNat nodes previous fallback = some result) ∧
      PublicationMir.ExactScore.scoreIfExact result = some (256 * EndgameTreeCache.discValue mc fc b.val)

theorem publication : Contract := by
  intro mc fc p c b orders mem base visits sort nodes fallback address small mobile lo hi inputs roots
  have capacity := legalMoveCount_le_empties mc b
  have legal : PvsBoardTree.legalSquares mc b.val ≠ [] :=
    fun nil => Bool.noConfusion ((PvsBoardTree.stuck_of_nil nil).symm.trans mobile)
  have started := (NativeRootBirth.started_iff mc fc b c.paired mem).2
    ⟨by unfold NativeRootBirth.capacity; omega,legal,rfl⟩
  obtain ⟨covers,pos⟩ := NativeRootBirth.covers started
  obtain ⟨state,earlier,⟨rootCover,positive,boot,pc,window⟩,contains⟩ := levels_ready p c b inputs roots sort fallback
    (OuterMir.ladder.take 4) ⟨_,SerialEntry.firstMachine base,EndgameSearch.CacheHistory.replay mc fc p visits⟩ address small
    ⟨covers,List.length_pos_iff.mpr pos,EndgameSearch.SerialWarmCache.replay_reachable p mc fc visits,rfl,
      SerialEntry.firstMachine_window base lo hi⟩
  obtain ⟨prior,last,post,result,ran,ok,_,_,_,_,printed,score,_⟩ :=
    Physical.publication mc fc p c b (inputs 5) (roots 5) state.rows state.machine state.cache sort nodes fallback
      address (fun i => shared_thirteen _ _ small) boot rootCover positive pc window
  have center := contains (.inr (by decide))
  have first := Physical.first_accepts p c b (inputs 5) (roots 5) (SerialStack.first state.machine state.rows)
    state.cache address (shared_thirteen _ _ small) boot rootCover positive window center.1 center.2 ran
  exact ⟨_,state,last,post,result,started,earlier,first ▸ ran,ok,
    fun _ => by simpa [OuterMir.finish,PublicationMir.completed,PublicationMir.record,
      PublicationMir.capture,PublicationMir.snapshot] using printed,score⟩

end NeuralReversi.AspirationMir.SerialCache.Ladder
