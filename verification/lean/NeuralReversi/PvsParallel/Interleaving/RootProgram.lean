import NeuralReversi.PvsParallel.Interleaving.Publication
import NeuralReversi.PvsParallel.Interleaving.ObservedJoin

namespace NeuralReversi.PvsParallel.Interleaving.RootProgram

open PvsBoardTree (GeneratedSquare Position)
open RootMoveStorage (Row)
open RootLadder (Attempt Reply Source)
open PvsMoveLoop (ChildRelation scoreMax)
open Bitboard (Square)
open RootOwnerMir (Result Replies fromMoves)
open RootTrial (labels ownerRows square)

structure Operands (mc : Bitboard.Moves.X86Config) (b : Position)
    (done_ : ChildRelation (GeneratedSquare mc b.val)) (beta : Int) where
  memory : MoveList.Initialization.Memory Move.Move
  paired : Bool
  key : Square → BitVec 32
  sort : RootFrameMir.LibraryOrder
  pv : Square → List Square
  child : Replies (GeneratedSquare mc b.val)
  returned : ∀ m cp a z v, child m cp a z = some v → done_ m cp a z v
  scheduling : Nat → Bool × Bool
  minimumSplitDepth : Nat
  width : Nat
  supported : 7 ≤ width
  split : (r : Result (GeneratedSquare mc b.val)) →
    Option (ObservedJoin.Trace (RootSchedule.task r (Board.emptyCount b.val).toNat beta false) (RootSchedule.indexed done_ r)
      (labels r).length (BitVec.ofNat width r.remaining.idx))

def run (fc : Root.FlipConfig) (idx : Nat) (s : AspirationMir.SerialStack.State)
    (op : Operands mc b done_ s.regs.beta.toInt) : Option AspirationMir.SmallRoot.Answer := do
  let roots := s.rows.map Row.square
  let g := MoveList.Generation.fill false op.paired fc b.val (Root.remainingMask mc b.val roots idx) op.memory
  match ← SerialRootCalls.entry g.len g.wipeout with
  | some sq => return ⟨scoreMax,FrameReturns.wipeRows s.rows idx sq op.pv⟩
  | none =>
    let d := (Board.emptyCount b.val).toNat
    let moves := AspirationMir.SmallRoot.candidates mc fc b roots idx op.key op.sort
    let r ← fromMoves Subtype.val op.child op.scheduling d op.minimumSplitDepth s.regs.alpha.toInt s.regs.beta.toInt moves
    let initial := ownerRows s.regs.alpha.toInt s.regs.beta.toInt s.rows idx r op.child op.pv
    if r.handoff then
      let joined ← op.split r
      let v ← joined.answer
      return ⟨v,FrameReturns.reportRows (square r) op.pv s.regs.beta.toInt initial r.snapshot (Schedule.reports joined.state)⟩
    return ⟨(rootEpilogue idx r.snapshot.state.best).2,initial⟩

theorem ran (op : Operands mc b done_ s.regs.beta.toInt) (h : run fc idx s op = some ans) (window : Aspiration.Window s.regs.alpha.toInt s.regs.beta.toInt) :
    Attempt mc fc b idx s done_ ans := by
  obtain ⟨opt,entry,body⟩ := Option.bind_eq_some_iff.mp h
  cases opt with
  | some sq =>
    cases body
    apply Attempt.wiped
    have pick {n : Option Nat} {c : Option (Option Square)} (found : SerialRootCalls.entry n c = some (some sq)) : c = some (some sq) := by
      cases n with
      | none => contradiction
      | some count =>
        cases c with
        | none => contradiction
        | some cached =>
          cases cached <;> by_cases yes : 0 < count <;> simp_all [SerialRootCalls.entry,rootEntry]
    exact pick entry
  | none =>
    obtain ⟨r,owner,answer⟩ := Option.bind_eq_some_iff.mp body
    have order := (RootFrameMir.ordered_candidates mc fc b (s.rows.map Row.square) idx op.key op.sort).1
    cases split_ : r.handoff with
    | false =>
      simp only [split_,Bool.false_eq_true,ite_false] at answer
      cases answer
      exact .searched (.owner r op.child _ op.pv order op.returned owner split_)
    | true =>
      simp only [split_,ite_true] at answer
      obtain ⟨receipt,_,returned⟩ := Option.bind_eq_some_iff.mp answer
      obtain ⟨v,joined,eq⟩ := Option.bind_eq_some_iff.mp returned
      cases eq
      obtain ⟨execution,_,positive⟩ := RootOwnerMir.refines Subtype.val op.child done_ op.returned _ _ _ _ _ _ _ _ _ _ (by simp) rfl owner
      have cursor := (RootTrial.layout_of_header owner).2
      have hit : 0 < r.visited.length := cursor ▸ positive split_
      obtain ⟨move,_,_⟩ := List.exists_cons_of_ne_nil (List.length_pos_iff.mp hit)
      have decreases := (Board.checkedPlacementProgress_proven true _ _ _ _ b.property
        (Board.precomputedMoveContract_proven mc fc true true b.val move.val b.property move.property).1).2
      have continues : r.outcome = .handoff r.snapshot.state := by simp [Result.outcome,split_]
      obtain ⟨_,_,path⟩ := execution.reproduced
      rw [continues] at path
      obtain ⟨win,cap,_⟩ := Prefix.continued_window true _ s.regs.alpha.toInt s.regs.beta.toInt r.visited _ _ _
        window (Int.le_max_left ..) path.1
      obtain ⟨j,word,posts⟩ := receipt.exit.observes s.regs.alpha.toInt win cap
      have same : Schedule.reports j.state = Schedule.reports receipt.state := congrArg (List.map Tickets.Post.report) posts
      rw [←same]
      exact .searched (.split r op.child _ op.pv order op.returned owner split_ (by omega) op.supported j (word.trans joined))

structure Input (M : Type) (mc : Bitboard.Moves.X86Config) (b : Position)
    (done_ : ChildRelation (GeneratedSquare mc b.val)) (beta : Int) where
  code : Operands mc b done_ beta
  closing : Nat
  after : M

abbrev Supplies (M : Type) (mc : Bitboard.Moves.X86Config) (b : Position) (done_ : RootLadder.Children mc b) :=
  (idx level : Nat) → (s : AspirationMir.SerialStack.State) → M → Option (Input M mc b (done_ level) s.regs.beta.toInt)

def source (fc : Root.FlipConfig) (ops : Supplies M mc b done_) : Source M mc fc b done_ := fun idx level s m => do
  let i ← ops idx level s m
  match h : run fc idx s i.code with
  | none => none
  | some ans => return ⟨i.closing,ans,i.after,ran i.code h⟩

end NeuralReversi.PvsParallel.Interleaving.RootProgram
