module
public import NeuralReversi.PvsParallel.RootOwnerMir
public import NeuralReversi.MoveList.Ordering

@[expose] public section

namespace NeuralReversi.PvsParallel.RootOwnerMir.Slice

open PvsPostpone (Queue step)
open RootReplay (Snapshot)
open MoveList.Ordering (postponeMove)
open Bitboard (Square)

/-- Move storage and owner fields on leaving the Rust loop, including the iterator
    start passed to a split. -/
structure Exit (Move : Type) where
  moves : List Move
  idx : Nat
  snapshot : Snapshot
  handoff : Bool

def header (replies : Replies Move) (moves : List Move) (idx raw depth minSplit : Nat)
    (free busy : Bool) (alpha beta : Int) : HeaderAction :=
  let head := moves[idx]?
  let scout := head.bind fun m => replies m false (-(alpha+1)) (-alpha)
  let wide := head.bind fun m => replies m true (-beta) (-alpha)
  headerAction idx moves.length raw depth minSplit free busy alpha (scout.getD 0) (wide.getD 0)

private theorem postponed (h : header r moves idx raw d limit free busy a z = .postpone next fresh) :
    idx + 1 < raw ∧ next = idx ∧ fresh = raw - 1 := by
  unfold header at h
  rw [headerAction_eq] at h
  split at h <;> try contradiction
  split at h <;> try contradiction
  split at h
  · rename_i rotate
    simp only [PvsPostpone.shouldPostpone,Bool.and_eq_true,decide_eq_true_eq] at rotate
    cases h
    exact ⟨rotate.2,rfl,rfl⟩
  · split at h <;> try contradiction
    split at h <;> contradiction

private theorem advanced (h : header r moves idx raw d limit free busy a z = .child next scout wide score) :
    idx < moves.length ∧ next = idx + 1 := by
  unfold header at h
  rw [headerAction_eq] at h
  split at h <;> try contradiction
  rename_i inside
  split at h <;> try contradiction
  split at h <;> try contradiction
  split at h
  · cases h; exact ⟨inside,rfl⟩
  · split at h <;> cases h <;> exact ⟨inside,rfl⟩

/-- Execute the extracted PV header and score updates with the actual indexed slice.
    On a busy sibling Rust rotates the suffix starting at idx and decrements fresh_end. -/
def run (square : Move → Square) (r : Replies Move) (schedule : Nat → Bool × Bool)
    (depth minSplit : Nat) (beta : Int) (i idx raw : Nat) (moves : List Move) (s : Snapshot) :
    Option (Exit Move) :=
  let a := s.state.alpha
  match act : header r moves idx raw depth minSplit (schedule i).1 (schedule i).2 a beta with
  | .finish => some ⟨moves,idx,s,false⟩
  | .split => some ⟨moves,idx,s,true⟩
  | .postpone next fresh =>
    run square r schedule depth minSplit beta (i+1) next fresh (postponeMove moves idx) s
  | .child next scout wide score => do
    let m ← moves[idx]?
    if (!scout || (r m false (-(a+1)) (-a)).isSome) && (!wide || (r m true (-beta) (-a)).isSome) then
      let d := RootUpdateMir.decision (decide (next = 1)) a beta s.state.best score
      let updated := RootUpdateMir.after s (square m) score d
      if d.cuts then some ⟨moves,next,updated,false⟩
      else run square r schedule depth minSplit beta (i+1) next raw moves updated
    else none
termination_by (moves.length - idx) + (raw - idx)
decreasing_by
  · obtain ⟨_,rfl,rfl⟩ := postponed act
    rw [MoveList.Ordering.postponeMove_length]
    omega
  · obtain ⟨_,rfl⟩ := advanced act
    omega

def layout (prior : List Move) (r : Result Move) : Exit Move :=
  ⟨prior ++ (r.visited ++ r.remaining.pending),r.remaining.idx,r.snapshot,r.handoff⟩

private theorem indexed {prior : List Move} {q : Queue Move} (index : prior.length = q.idx) :
    (prior ++ q.pending)[q.idx]? = q.pending.head? := by
  rw [← index,List.getElem?_append_right (Nat.le_refl _)]
  simp [List.head?_eq_getElem?]

private theorem sameHeader (r : Replies Move) (prior : List Move) (q : Queue Move)
    (index : prior.length = q.idx) (raw depth minSplit : Nat) (free busy : Bool) (a z : Int) :
    header r (prior ++ q.pending) q.idx raw depth minSplit free busy a z =
      issued r q raw depth minSplit free busy a z := by
  simp only [header,issued,indexed index,List.length_append,index]

private theorem rotated (prior : List Move) (q next : Queue Move) (index : prior.length = q.idx)
    (advance : step busy q = some (none,next)) :
    postponeMove (prior ++ q.pending) q.idx = prior ++ next.pending := by
  rcases q with ⟨idx,fresh,deferred⟩
  cases fresh with
  | nil => cases deferred <;> simp [step] at advance
  | cons m rest =>
    simp only [step] at advance
    split at advance
    · cases advance
      change prior.length = idx at index
      subst idx
      simp [postponeMove,Queue.pending,List.append_assoc]
    · simp at advance

/-- The array-index execution agrees with queue replay even when a child return is
    missing. The raw fresh_end may be behind idx after deferred moves are consumed. -/
theorem agrees (square : Move → Square) (r : Replies Move) (schedule : Nat → Bool × Bool)
    (depth minSplit : Nat) (beta : Int) (fuel i raw : Nat) (prior : List Move)
    (q : Queue Move) (s : Snapshot) (enough : q.budget < fuel)
    (index : prior.length = q.idx) (cursor : raw - q.idx = q.fresh.length) :
    run square r schedule depth minSplit beta i q.idx raw (prior ++ q.pending) s =
      (RootOwnerMir.run square r schedule depth minSplit beta fuel i raw q s).map (layout prior) := by
  induction fuel generalizing i raw prior q s with
  | zero => omega
  | succ fuel ih =>
    let free := (schedule i).1
    let busy := (schedule i).2
    let a := s.state.alpha
    let scout := (q.pending.head?.bind fun m => r m false (-(a+1)) (-a)).getD 0
    let wide := (q.pending.head?.bind fun m => r m true (-beta) (-a)).getD 0
    have same := sameHeader r prior q index raw depth minSplit free busy a beta
    rw [run,RootOwnerMir.run]
    split
    next physical =>
      have h := same.symm.trans physical
      dsimp only [free,busy,a] at h
      rw [h]
      rfl
    next physical =>
      have h := same.symm.trans physical
      dsimp only [free,busy,a] at h
      rw [h]
      rfl
    next n fresh physical =>
      have h := same.symm.trans physical
      have act := h
      change headerAction _ _ _ _ _ _ _ _ scout wide = _ at act
      rw [header_normalize q.idx _ raw,cursor] at act
      obtain ⟨next,move,idx,cut,less⟩ := header_postpone q depth minSplit free busy a scout wide act
      have unchanged : next.idx = q.idx := by simpa using PvsPostpone.step_idx move
      simp only [free,busy,a] at h
      dsimp only [busy] at move
      rw [h,move,rotated prior q next index move]
      rw [← idx]
      exact ih (i+1) fresh prior next s (by omega) (by omega) (by omega)
    next n needsScout needsWide value physical =>
      have h := same.symm.trans physical
      have act := h
      change headerAction _ _ _ _ _ _ _ _ scout wide = _ at act
      rw [header_normalize q.idx _ raw,cursor] at act
      obtain ⟨m,next,move,head,idx,less⟩ := header_take q depth minSplit free busy a scout wide act
      have addr : (prior ++ q.pending)[q.idx]? = some m := by rw [indexed index,head]; rfl
      have nextIdx : next.idx = q.idx + 1 := by simpa using PvsPostpone.step_idx move
      have nextMem : prior ++ q.pending = (prior ++ [m]) ++ next.pending := by simp [head]
      simp only [free,busy,a] at h
      dsimp only [busy] at move
      rw [h,move,addr]
      dsimp only [bind,Option.bind]
      split
      · split
        · simp [layout,head,idx]
        · rw [← idx,nextMem]
          rw [ih (i+1) raw (prior ++ [m]) next _ (by omega) (by simp [index,nextIdx])
            (by simpa using cursor_step q next raw cursor move)]
          rw [Option.map_map]
          apply congrArg (fun f => Option.map f _)
          funext tail
          simp [layout,Result.visit]
      · rfl

/-- A returning child may supply any score. Neither scheduling nor repeated busy
    claims can prevent the owner from reaching a cutoff, a handoff or the end. -/
theorem returns (square : Move → Square) (search : PvsMoveLoop.ChildSearch Move)
    (schedule : Nat → Bool × Bool) (depth limit : Nat) (beta : Int) (i idx raw : Nat)
    (moves : List Move) (s : Snapshot) :
    ∃ out, run square (fun m pv a z => some (search m pv a z)) schedule depth limit beta i idx raw moves s = some out := by
  rw [run]
  split
  next => exact ⟨_,rfl⟩
  next => exact ⟨_,rfl⟩
  next n fresh act =>
    obtain ⟨_,same,fewer⟩ := postponed act
    rw [same,fewer]
    exact returns square search schedule depth limit beta (i+1) idx (raw-1) (postponeMove moves idx) s
  next n scout wide score act =>
    obtain ⟨inside,rfl⟩ := advanced act
    simp only [List.getElem?_eq_getElem inside,Option.isSome_some,Bool.or_true,Bool.and_self,
      bind,Option.bind,ite_true]
    split
    next => exact ⟨_,rfl⟩
    next => exact returns square search schedule depth limit beta (i+1) (idx+1) raw moves _
termination_by (moves.length - idx) + (raw - idx)
decreasing_by
  · rw [MoveList.Ordering.postponeMove_length]; omega
  · omega

/-- Start from the exact operands of the Rust while loop. This equality holds for
    every pattern of helper availability, busy claims and child return words. -/
theorem fromMoves_eq (square : Move → Square) (r : Replies Move) (schedule : Nat → Bool × Bool)
    (depth minSplit : Nat) (alpha beta : Int) (moves : List Move) :
    run square r schedule depth minSplit beta 0 0 moves.length moves (RootReplay.start alpha) =
      (fromMoves square r schedule depth minSplit alpha beta moves).map (layout []) := by
  let q : Queue Move := ⟨0,moves,[]⟩
  simpa [q,fromMoves,Queue.pending] using
    agrees square r schedule depth minSplit beta (q.budget+1) 0 moves.length [] q
      (RootReplay.start alpha) (Nat.lt_succ_self _) rfl (by simp [q])

end NeuralReversi.PvsParallel.RootOwnerMir.Slice
