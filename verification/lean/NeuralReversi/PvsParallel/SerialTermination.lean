import NeuralReversi.PvsParallel.SerialRootExecution
import NeuralReversi.PvsParallel.SerialPvReady
import NeuralReversi.PvsParallel.SerialNonPvReady

namespace NeuralReversi.PvsParallel.SerialTermination

open SerialCalls (Program Request Machine Executes)
open SerialExecution (Frame Inputs ReadFacts body push)
open Aspiration (Window)
open SerialCalls.ScoreWords (encode)
open PvsBoardTree (Position minimax legalSquares)
open AspirationMir.SmallRoot (Answer)

/-- A native TT move used directly for a presearch belongs to its board. -/
def Hints (inputs : Inputs mc fc) : Prop :=
  ∀ i ctx, ctx.request.pv = false → 12 < ctx.request.depth.toNat → TableReads.LegalHints (inputs i ctx).table

def Entry (ctx : Frame) : Prop :=
  let r := ctx.request
  ctx.empties.matches r.board ∧ ExactDepth.Full r.board r.depth ∧ Window r.alpha r.beta ∧
    (r.pv = false → r.beta = r.alpha + 1)

def Allowed (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (ctx : Frame) (q : Request) : Prop :=
  ExactDepth.RecursiveCall mc fc ctx.request.board ctx.request.depth q.board q.depth ∧ Window q.alpha q.beta ∧
    (q.pv = false → q.beta = q.alpha + 1)

theorem frame_ready (c : SerialScout.Config) (inputs : Inputs mc fc) (valid : ReadFacts inputs)
    (hints : Hints inputs) (i : Nat) (ctx : Frame) (start : Entry ctx) :
    Program.Ready (Allowed mc fc ctx) (body c inputs i ctx) := by
  obtain ⟨aligned, full, window, null⟩ := start
  let q := ctx.request
  let op := inputs i ctx
  cases role : ctx.request.pv with
  | true =>
    simp only [body, role, ↓reduceIte]
    exact SerialPvReady.frame_ready full window c.paired c.checked q.cut op.memory (SerialRecursion.pvOrder q.board op)
  | false =>
    obtain ⟨low, order, high⟩ := window
    have av : (encode q.alpha).raw.toInt = q.alpha :=
      BitVec.toInt_ofInt_eq_self (by decide) (by dsimp [q]; omega) (by dsimp [q]; omega)
    have zv : (encode q.beta).raw.toInt = q.beta :=
      BitVec.toInt_ofInt_eq_self (by decide) (by dsimp [q]; omega) (by dsimp [q]; omega)
    have w : Window (encode q.alpha).raw.toInt (encode q.beta).raw.toInt := by rw [av,zv]; exact ⟨low,order,high⟩
    have width : (encode q.beta).raw.toInt = (encode q.alpha).raw.toInt + 1 := by rw [av,zv]; exact null role
    by_cases small : ctx.request.depth.toNat ≤ 12
    · simp only [body, role, Bool.false_eq_true, ↓reduceIte, small]
      have size : ctx.empties.count.toNat ≤ 12 := by rw [aligned.1, ← full]; exact small
      obtain ⟨v, after, computed, _⟩ := SerialScout.score_returns c mc fc q.board ctx.empties
        (encode q.alpha) (encode q.beta) op.cache aligned size (fun lo => valid.cache i ctx role (by rwa [full, ← aligned.1]) small) width
      change Program.Ready _ (.ofOption ((EndgameSearch.SerialLeafRestore.leaf c mc fc q.board ctx.empties
        (encode q.alpha) (encode q.beta) op.cache).map (fun out => out.1.1)))
      rw [EndgameSearch.SerialLeafRestore.leaf_refines _ _ _ _ _ _ _ _ aligned size, computed]
      exact .pure
    · simp only [body, role, Bool.false_eq_true, ↓reduceIte, small]
      exact SerialNonPvReady.frame_ready mc fc q.board q.depth _ _ c.paired c.checked q.cut op.memory op.probe
        op.etc _ full (by dsimp [q]; omega) w width (valid.probe i ctx role (by omega)) (hints i ctx role (by omega))

theorem child_entry {ctx : Frame} (entry : Entry ctx) (q : Request) (allowed : Allowed mc fc ctx q) :
    ∃ child, push mc fc ctx q = some child ∧ Entry child ∧
      Board.playRank mc child.request.board.val < Board.playRank mc ctx.request.board.val := by
  obtain ⟨edge, w, null⟩ := allowed
  obtain ⟨state, updated, fits⟩ := SerialChildState.nextState_matches edge entry.1
  exact ⟨⟨q,state⟩, by simp [push, updated], ⟨fits, edge.full entry.2.1, w, null⟩, edge.earlier⟩

/-- The number of steps and all returned words are outputs. A later activation
    can observe different operands even for a repeated board and window. -/
theorem terminates (c : SerialScout.Config) (inputs : Inputs mc fc) (valid : ReadFacts inputs)
    (hints : Hints inputs) (ctx : Frame) (entry : Entry ctx) (start : Nat) :
    ∃ last value forest, Executes (body c inputs) (push mc fc) start ctx (body c inputs start ctx) last value forest := by
  apply SerialCalls.executes_of_ready (frame_ready c inputs valid hints start ctx entry)
  intro q allowed i
  obtain ⟨child, pushed, fits, lower⟩ := child_entry entry q allowed
  obtain ⟨last, word, forest, ran⟩ := terminates c inputs valid hints child fits (i+1)
  exact ⟨child,last,word,forest,pushed,ran⟩
termination_by Board.playRank mc ctx.request.board.val

theorem root_terminates (c : SerialScout.Config) (inputs : Inputs mc fc) (valid : ReadFacts inputs)
    (hints : Hints inputs) (start : Nat) (b : Position) (root : SerialRecursion.RootInput)
    (initial : List RootMoveStorage.Row) (idx : Nat) (a z : BitVec 32)
    (covers : Root.Covers mc b (initial.map RootMoveStorage.Row.square)) (within : idx < initial.length)
    (window : Window a.toInt z.toInt) :
    ∃ count last value, (Machine.running start (SerialRootExecution.context b a z)
      ((SerialRootExecution.attempt c inputs start b root initial idx a z).map Answer.value) []).steps
        (body c inputs) (push mc fc) count = some (.finished last value) := by
  let ctx := SerialRootExecution.context b a z
  have facts : Entry ctx := by
    simp only [ctx, Entry, SerialRootExecution.context, ContextPathMir.start_eq]
    exact ⟨EndgameSearch.emptyState_initial b, rfl, window, by simp⟩
  have live := SerialRootCalls.frame_ready (fc := fc) (mem := (inputs start ctx).memory) (paired := c.paired)
    (key := root.key) (sort := root.sort) (pv := root.pvLine) covers within facts.2.1 window
  obtain ⟨last,word,forest,ran⟩ := SerialCalls.executes_of_ready (body := body c inputs) (push := push mc fc)
    (ctx := ctx) (live.map Answer.value) (fun req ok i => by
      obtain ⟨child, pushed, childFits, _⟩ := child_entry facts req ok
      obtain ⟨k,v,inside,exec⟩ := terminates c inputs valid hints child childFits (i+1)
      exact ⟨child,k,v,inside,pushed,exec⟩) start
  obtain ⟨n,eq⟩ := ran.to_steps
  exact ⟨n,last,word,eq⟩

def Contract : Prop :=
  ∀ certificate, Std.Tactic.BVDecide.LRAT.check certificate (LegalMoves.MoveBound.dimacsFormula 35) = true →
    ∀ mc fc (c : SerialScout.Config) (inputs : Inputs mc fc), ReadFacts inputs → Hints inputs →
    ∀ (start : Nat) (b : Position) (root : SerialRecursion.RootInput) (a z : BitVec 32),
      legalSquares mc b.val ≠ [] →
      Window a.toInt z.toInt →
      ∃ count last value, SerialRootExecution.runFromBoard c inputs start count b root a z = some (last,value) ∧
        SerialRootExecution.FromBoardResult c inputs start b root a z last value

theorem contract_proven : Contract := by
  intro certificate certified mc fc c inputs valid hints start b root a z mobile window
  let mem := (inputs start (SerialRootExecution.context b a z)).memory
  have boot := AspirationMir.NativeRootBirth.certified mc fc b c.paired mem certified
  have present : (AspirationMir.NativeRootBirth.initial mc b).isEmpty = false := by
    cases h : legalSquares mc b.val with
    | nil => exact (mobile h).elim
    | cons m rest => simp [AspirationMir.NativeRootBirth.initial, h]
  have birth : AspirationMir.NativeRootBirth.started mc fc b c.paired mem =
      some (AspirationMir.NativeRootBirth.initial mc b) := by
    unfold AspirationMir.NativeRootBirth.started
    rw [boot]
    change (if (AspirationMir.NativeRootBirth.initial mc b).isEmpty then none else some _) = _
    rw [present]
    rfl
  dsimp only [mem] at birth
  obtain ⟨covers, nonempty⟩ := AspirationMir.NativeRootBirth.covers birth
  obtain ⟨count,last,value,stepped⟩ := root_terminates c inputs valid hints start b root _ 0 a z covers
    (List.length_pos_iff.mpr nonempty) window
  have run : SerialRootExecution.runFromBoard c inputs start count b root a z = some (last,value) := by
    simp [SerialRootExecution.runFromBoard, birth, stepped]
  exact ⟨count,last,value,run,SerialRootExecution.fromBoard c inputs valid start count b root a z window run⟩

end NeuralReversi.PvsParallel.SerialTermination
