module
public import NeuralReversi.EndgameSearch.SerialLeafRestore
public import NeuralReversi.PvsParallel.SerialCalls.Stack
public import NeuralReversi.PvsParallel.SerialCalls.ScoreWords
public import NeuralReversi.PvsParallel.SerialRecursion
public import NeuralReversi.PvsParallel.SerialNonPvCalls
public import NeuralReversi.PvsParallel.SerialPvCalls

@[expose] public section

namespace NeuralReversi.PvsParallel.SerialExecution

open SerialCalls (Request Program Machine Executes)
open PvsBoardTree (build minimax)
open EndgameSearch.Cancellation (EmptyState)
open Aspiration (Window)
open Bitboard.Moves (X86Config)
open Cutoffs (FlipConfig)

structure Frame where
  request : Request
  empties : EmptyState

def push (mc : X86Config) (fc : FlipConfig) (ctx : Frame) (r : Request) : Option Frame :=
  (SerialChildState.nextState mc fc ctx.request.board ctx.request.depth ctx.empties r.board r.depth).map
    fun empties => ⟨r, empties⟩

abbrev Inputs (mc : X86Config) (fc : FlipConfig) :=
  Nat → (ctx : Frame) → SerialRecursion.Operands mc fc ctx.request.board

abbrev raw := SerialCalls.ScoreWords.encode

def body (c : SerialScout.Config) (inputs : Inputs mc fc) (i : Nat) (ctx : Frame) : Program Request Int :=
  let r := ctx.request
  let op := inputs i ctx
  if r.pv then
    SerialPvCalls.frame mc fc r.board r.depth c.paired c.checked r.cut op.memory r.alpha r.beta
      (SerialRecursion.pvOrder r.board op)
  else if r.depth.toNat ≤ 12 then
    .ofOption ((EndgameSearch.SerialLeafRestore.leaf c mc fc r.board ctx.empties (raw r.alpha) (raw r.beta) op.cache).map (fun out => out.1.1))
  else
    SerialNonPvCalls.frame mc fc r.board r.depth (raw r.alpha) (raw r.beta) c.paired c.checked r.cut
      op.memory op.probe op.etc (EndgameOrderMir.framePriority mc fc r.board op.ordering op.tie)

/-- Only table-dependent exits use these facts about the native loads. An opening
    number distinguishes calls even if all six search arguments are identical. -/
structure ReadFacts (inputs : Inputs mc fc) : Prop where
  cache : ∀ i ctx, ctx.request.pv = false → 5 ≤ ctx.request.depth.toNat → ctx.request.depth.toNat ≤ 12 →
    EndgameTreeCache.soundHistory mc fc (inputs i ctx).cache
  probe : ∀ i ctx, ctx.request.pv = false → 12 < ctx.request.depth.toNat →
    TableReads.ProbeReads (inputs i ctx).table (inputs i ctx).probe
  table : ∀ i ctx, ctx.request.pv = false → 12 < ctx.request.depth.toNat →
    ExactTableHistory.Publications mc fc (inputs i ctx).table

def Invariant (mc : X86Config) (fc : FlipConfig) (ctx : Frame) (value : Int) : Prop :=
  let r := ctx.request
  ctx.empties.matches r.board → ExactDepth.Full r.board r.depth → Window r.alpha r.beta →
    (r.pv = false → r.beta = r.alpha + 1) →
    boundedReplay (Board.playRank mc r.board.val + 1) (build mc fc r.board) r.pv r.alpha r.beta value

theorem child_replay {ctx : Frame} {r : Request × Int} {mc fc}
    (full : ExactDepth.Full ctx.request.board ctx.request.depth) (aligned : ctx.empties.matches ctx.request.board)
    (issued : ∃ child, push mc fc ctx r.1 = some child ∧ Invariant mc fc child r.2)
    (window : Window r.1.alpha r.1.beta) (null : r.1.pv = false → r.1.beta = r.1.alpha + 1) :
    boundedReplay (Board.playRank mc ctx.request.board.val) (build mc fc r.1.board)
      r.1.pv r.1.alpha r.1.beta r.2 := by
  obtain ⟨child, entered, valid⟩ := issued
  obtain ⟨s, updated, rfl⟩ := Option.map_eq_some_iff.mp entered
  have edge := SerialChildState.edge_of_nextState updated
  obtain ⟨fitted, fittedEq, fits⟩ := SerialChildState.nextState_matches edge aligned
  have same := Option.some.inj (updated.symm.trans fittedEq)
  subst fitted
  exact boundedReplay_mono (Nat.succ_le_of_lt edge.earlier) _ _ _ _ _ (valid fits (edge.full full) window null)

/-- A returning child is the sole supplier of the word resuming its parent. -/
theorem localRule {mc fc} (c : SerialScout.Config) (inputs : Inputs mc fc) (valid : ReadFacts inputs)
    (i : Nat) (ctx : Frame) (children : List (Request × Int)) (word : Int)
    (ran : Program.Runs (body c inputs i ctx) children word)
    (earlier : ∀ r ∈ children, ∃ child, push mc fc ctx r.1 = some child ∧ Invariant mc fc child r.2) :
    Invariant mc fc ctx word := by
  intro aligned full window null
  let q := ctx.request
  let op := inputs i ctx
  have child (r : Request × Int) (member : r ∈ children) := child_replay full aligned (earlier r member)
  cases role : ctx.request.pv with
  | true =>
    simp only [body, role, ↓reduceIte] at ran
    exact SerialPvCalls.frame_replay full (EndgameOrderMir.ordinary_perm _ _ _ _ _) window ran child
  | false =>
    have narrow := null role
    obtain ⟨low, order, high⟩ := window
    have av : (raw q.alpha).raw.toInt = q.alpha :=
      BitVec.toInt_ofInt_eq_self (by decide) (by dsimp [q]; omega) (by dsimp [q]; omega)
    have zv : (raw q.beta).raw.toInt = q.beta :=
      BitVec.toInt_ofInt_eq_self (by decide) (by dsimp [q]; omega) (by dsimp [q]; omega)
    have w : Window (raw q.alpha).raw.toInt (raw q.beta).raw.toInt := by rw [av,zv]; exact ⟨low,order,high⟩
    have narrow' : (raw q.beta).raw.toInt = (raw q.alpha).raw.toInt + 1 := by rw [av,zv]; exact narrow
    by_cases small : ctx.request.depth.toNat ≤ 12
    · simp only [body, role, Bool.false_eq_true, ↓reduceIte, small] at ran
      have size : ctx.empties.count.toNat ≤ 12 := by rw [aligned.1, ← full]; exact small
      obtain ⟨v, after, computed, evidence⟩ := SerialScout.score_returns c mc fc q.board ctx.empties
        (raw q.alpha) (raw q.beta) op.cache aligned size (fun lo => valid.cache i ctx role (by rwa [full, ← aligned.1]) small) narrow'
      have found := (Program.runs_ofOption ran).1
      change (EndgameSearch.SerialLeafRestore.leaf c mc fc q.board ctx.empties (raw q.alpha) (raw q.beta) op.cache).map _ = some word at found
      rw [EndgameSearch.SerialLeafRestore.leaf_refines _ _ _ _ _ _ _ _ aligned size, computed] at found
      have same : v = word := Option.some.inj found
      rw [av,zv,same] at evidence
      exact boundedReplay_mono (Nat.le_add_left 1 _) _ _ _ _ _ evidence
    · simp only [body, role, Bool.false_eq_true, ↓reduceIte, small] at ran
      let priority := EndgameOrderMir.framePriority mc fc q.board op.ordering op.tie
      have priorityLow (m) (_ : m ∈ PvsBoardTree.legalSquares mc q.board.val) :=
        EndgameOrderMir.framePriority_low mc fc q.board op.ordering op.tie m
      have edges := SerialNonPvCalls.frame_edges mc fc q.board q.depth (raw q.alpha) (raw q.beta) c.paired
        c.checked q.cut op.memory op.probe op.etc priority priorityLow w narrow' ran
      have result := SerialNonPvCalls.frame_replay mc fc q.board q.depth (raw q.alpha) (raw q.beta) c.paired
        c.checked q.cut op.memory op.probe op.etc priority _ full w narrow'
        (valid.probe i ctx role (by omega)) (valid.table i ctx role (by omega)) priorityLow ran (by
          intro e seen
          obtain ⟨_, ew, en, _⟩ := edges e seen
          exact child e seen ew (fun _ => en))
      rwa [av,zv] at result

/-- A normal completion of the single worker's stack yields the chronological
    call tree, the search replay, and the minimax bound for the returned word. -/
theorem from_steps (c : SerialScout.Config) (inputs : Inputs mc fc) (valid : ReadFacts inputs)
    (ctx : Frame) (aligned : ctx.empties.matches ctx.request.board)
    (full : ExactDepth.Full ctx.request.board ctx.request.depth) (window : Window ctx.request.alpha ctx.request.beta)
    (null : ctx.request.pv = false → ctx.request.beta = ctx.request.alpha + 1)
    (ran : (Machine.running start ctx (body c inputs start ctx) []).steps (body c inputs) (push mc fc) count =
      some (.finished last word)) :
    ∃ forest, Executes (body c inputs) (push mc fc) start ctx (body c inputs start ctx) last word forest ∧
      SerialCalls.openingNumbers forest = List.range' (start+1) (last-start) ∧
      Replayed (build mc fc ctx.request.board) ctx.request.pv ctx.request.alpha ctx.request.beta word ∧
      Aspiration.SoundResult (minimax mc fc ctx.request.board) ctx.request.alpha ctx.request.beta word := by
  obtain ⟨forest, rec⟩ := SerialCalls.record_program_of_steps _ _ _ ran
  have ind := localRule c inputs valid
  have result := ind _ _ _ _ rec.directCalls (rec.children _ ind) aligned full window null
  exact ⟨forest, rec, rec.openings.1, ⟨_, result⟩, replayContract_proven _ _ _ _ _ _ window null result⟩

end NeuralReversi.PvsParallel.SerialExecution
