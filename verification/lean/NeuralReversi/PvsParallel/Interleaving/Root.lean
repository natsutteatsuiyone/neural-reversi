import NeuralReversi.PvsParallel.FrameRows
import NeuralReversi.PvsParallel.RootOwnerMir.Slice
import NeuralReversi.PvsParallel.RootReplay.Split

namespace NeuralReversi.PvsParallel.Interleaving.RootTrial

open Aspiration (SoundResult Window)
open PvsBoardTree (GeneratedSquare Position)
open Bitboard (Square)
open Threading
open SplitResult (Report)
open PvsMoveLoop (bestValue scoreInf scoreMax Bound)
open RootReplay (Snapshot start serial)
open RootOwnerMir (Result Replies fromMoves observed)

/-- The prefix contains moves searched by the owner; the suffix is in the physical
    order left by any postponed claims at the instant of the split. -/
def labels (r : Result (GeneratedSquare mc b)) : List Square :=
  (r.visited ++ r.remaining.pending).map Subtype.val

def square (r : Result (GeneratedSquare mc b)) (index : Nat) : Square := (labels r)[index]?.getD 0

def prefixValue (mc : Bitboard.Moves.X86Config) (fc : Root.FlipConfig) (b : Position)
    (r : Result (GeneratedSquare mc b.val)) : Int :=
  bestValue (Root.childValue mc fc b) r.visited (-scoreInf)

abbrev ChildSound (mc : Bitboard.Moves.X86Config) (fc : Root.FlipConfig) (b : Position)
    (replies : Replies (GeneratedSquare mc b.val)) : Prop :=
  ∀ m cp a z score, replies m cp a z = some score → Window a z → (cp = false → z = a+1) →
    SoundResult (-(Root.squareValue mc fc b m.val)) a z score

variable {mc : Bitboard.Moves.X86Config} {fc : Root.FlipConfig} {b : Position}
variable {moves : List (GeneratedSquare mc b.val)} {r : Result (GeneratedSquare mc b.val)}
variable {replies : Replies (GeneratedSquare mc b.val)} {schedule : Nat → Bool × Bool}
variable {roots : List Square} {idx depth limit : Nat} {alpha beta : Int}

private theorem refined (ran : fromMoves Subtype.val replies schedule depth limit alpha beta moves = some r) :
    Prefix.Serial.Run (fun m cp a z score => replies m cp a z = some score) (observed replies) true beta
      ⟨0,moves,[]⟩ (start alpha).state r.visited r.remaining r.outcome ∧
    r.snapshot = serial Subtype.val (observed replies) beta r.replayMoves (start alpha) ∧
    (r.handoff = true → 0 < r.remaining.idx) :=
  RootOwnerMir.refines Subtype.val replies (fun m cp a z score => replies m cp a z = some score)
    (fun _ _ _ _ _ h => h) schedule depth limit beta _ 0 moves.length ⟨0,moves,[]⟩ (start alpha) r (by simp) rfl ran

theorem layout_of_header (ran : fromMoves Subtype.val replies schedule depth limit alpha beta moves = some r) :
    (r.visited ++ r.remaining.pending).Perm moves ∧ r.remaining.idx = r.visited.length := by
  obtain ⟨perm,cursor,_⟩ := (refined ran).1.reproduced
  exact ⟨by simpa [PvsPostpone.Queue.pending] using perm,by simpa using cursor⟩

theorem child_cap (m : GeneratedSquare mc b.val) : Root.childValue mc fc b m ≤ scoreMax := by
  have h := (PvsMoveLoop.minimax_range (PvsBoardTree.build mc fc (PvsBoardTree.play mc fc b m).val)).1
  change -scoreMax ≤ PvsBoardTree.minimax mc fc (PvsBoardTree.play mc fc b m).val at h
  unfold Root.childValue
  omega

theorem ready (ran : fromMoves Subtype.val replies schedule depth limit alpha beta moves = some r)
    (split : r.handoff = true) (w : Window alpha beta) (children : ChildSound mc fc b replies) :
    0 < r.remaining.idx ∧ r.remaining.idx = r.visited.length ∧
    Window r.snapshot.state.alpha beta ∧ r.snapshot.state.alpha ≤ max alpha r.snapshot.state.best ∧
    Bound (prefixValue mc fc b r) alpha beta r.snapshot.state.best := by
  obtain ⟨trace,_,pos⟩ := refined ran
  obtain ⟨_,cursor,tr⟩ := trace.reproduced
  have same : r.outcome = .handoff r.snapshot.state := by simp [Result.outcome,split]
  rw [same] at tr
  obtain ⟨continued,calls⟩ := tr
  obtain ⟨window,cap,_,bound⟩ := Prefix.selectedContract_proven _ true (observed replies)
    (Root.childValue mc fc b) r.visited _ alpha beta r.snapshot.state w (by simp) continued
    (fun m _ => child_cap m) calls (fun m _ cp a z score win narrow child => by
      rw [← Root.square_value_eq_child m]; exact children m cp a z score child win narrow)
  exact ⟨pos split,by simpa using cursor,window,cap,bound⟩

/-- The scheduler supplies these facts at an accepted owner's return. They are indexed by
    the actual move array, cursor and split fields passed by the running root header. -/
structure Accepted (mc : Bitboard.Moves.X86Config) (fc : Root.FlipConfig) (b : Position)
    (r : Result (GeneratedSquare mc b.val)) (alpha beta : Int) (reports : List Report) (score : Int) : Prop where
  valid : ∀ x ∈ reports, x.index ∈ SplitResult.slots r.remaining.idx (labels r).length
  distinct : (reports.map Report.index).Nodup
  steps : SplitReturn.ReportSteps true beta r.snapshot.state (Root.squareValue mc fc b ∘ square r) reports
  bound : Bound (SplitResult.nodeValue (Root.squareValue mc fc b ∘ square r)
    r.remaining.idx (labels r).length (prefixValue mc fc b r)) alpha beta score
  returned : (SplitReturn.stateAfter true beta r.snapshot.state reports).best = score

private theorem node_value (order : (r.visited ++ r.remaining.pending).Perm (Root.choices mc b roots idx))
    (cursor : r.remaining.idx = r.visited.length) :
    SplitResult.nodeValue (Root.squareValue mc fc b ∘ square r) r.remaining.idx (labels r).length
      (prefixValue mc fc b r) = Root.value mc fc b roots idx := by
  let val := Root.squareValue mc fc b
  have convert (xs : List (GeneratedSquare mc b.val)) :
      bestValue val (xs.map Subtype.val) (-scoreInf) = bestValue (Root.childValue mc fc b) xs (-scoreInf) := by
    rw [PvsMoveLoop.bestValue_map]
    congr 1
    funext m
    exact Root.square_value_eq_child m
  have partition := SplitResult.partitionLabels (r.visited.map Subtype.val) (r.remaining.pending.map Subtype.val)
    (0 : Square) val (-scoreInf)
  rw [List.length_map,←List.map_append,convert,convert] at partition
  change SplitResult.nodeValue _ r.remaining.idx _ _ = _
  rw [cursor]
  exact partition.trans (PvsMoveLoop.bestValue_eq_of_same_moves _ _ (fun _ => order.mem_iff))

private theorem node_sound (covers : Root.Covers mc b roots) (inside : idx < roots.length) (w : Window alpha beta)
    (bound : Bound (Root.value mc fc b roots idx) alpha beta score) :
    SoundResult (Root.value mc fc b roots idx) alpha beta score := by
  obtain ⟨lo,hi⟩ := Root.value_range (fc := fc) covers inside
  have low := w.lower
  have ord := w.ordered
  have high := w.upper
  apply bound.toSoundResult w
  · by_cases good : -scoreInf < score
    · exact good
    have := bound.1 (by unfold scoreInf at good; omega)
    unfold scoreInf scoreMax at *
    omega
  · by_cases good : score < scoreInf
    · exact good
    have := bound.2 (by unfold scoreInf at good; omega)
    unfold scoreInf scoreMax at *
    omega

theorem handoff (ordered : moves.Perm (Root.choices mc b roots idx))
    (ran : fromMoves Subtype.val replies schedule depth limit alpha beta moves = some r)
    (split : r.handoff = true) (w : Window alpha beta) (covers : Root.Covers mc b roots)
    (inside : idx < roots.length) (children : ChildSound mc fc b replies)
    (tail : Accepted mc fc b r alpha beta reports score) :
    let final := RootReplay.Split.replay (square r) beta reports r.snapshot
    final.state.best = score ∧ RootOrder.SoundAttempt mc fc b roots idx alpha beta score final.scores := by
  obtain ⟨trace,snapshot,_⟩ := refined ran
  obtain ⟨permutation,count,tr⟩ := trace.reproduced
  have order : (r.visited ++ r.remaining.pending).Perm (Root.choices mc b roots idx) := by
    apply List.Perm.trans _ ordered
    simpa [PvsPostpone.Queue.pending] using permutation
  have cursor : r.remaining.idx = r.visited.length := by simpa using count
  have same : r.outcome = .handoff r.snapshot.state := by simp [Result.outcome,split]
  rw [same] at tr
  obtain ⟨_,calls⟩ := tr
  have snap : r.snapshot = serial Subtype.val (observed replies) beta r.visited (start alpha) := by
    simpa [Result.replayMoves,split] using snapshot
  let pre := r.visited.map Subtype.val
  let pending := r.remaining.pending.map Subtype.val
  have all : labels r = pre ++ pending := List.map_append
  have unique : (labels r).Nodup := (order.map Subtype.val).nodup_iff.mpr (Root.choices_squares_nodup ..)
  have names : ∀ m ∈ labels r, m ∈ roots.drop idx := fun m mem =>
    (Root.choices_squares_mem covers m).mp ((order.map Subtype.val).mem_iff.mp mem)
  have prev := RootReplay.serial_records Subtype.val (observed replies) (Root.squareValue mc fc b)
    r.visited (start alpha) (fun m cp a z s => replies m cp a z = some s) (original := alpha)
    (candidates := roots.drop idx) w .start (List.nodup_append.mp (all ▸ unique)).1
    (fun m mem => ⟨names m.val (all ▸ List.mem_append_left pending (List.mem_map_of_mem mem)),by simp [start]⟩)
    calls (fun m _ cp a z s win narrow child => children m cp a z s child win narrow)
  rw [←snap] at prev
  have prevSeen (sq : Square) (h : sq ∈ r.snapshot.seen) : sq ∈ pre := by
    rw [snap] at h
    exact ((RootReplay.serial_seen Subtype.val (observed replies) beta r.visited (start alpha)).1 sq h).resolve_left (by simp [start])
  have inBounds (x : Report) (h : x ∈ reports) := SplitResult.mem_slots.mp (tail.valid x h)
  have atIndex (i : Nat) (h : i < (labels r).length) : square r i = (labels r)[i] := by
    simp [square,List.getElem?_eq_getElem h]
  have fresh (x : Report) (h : x ∈ reports) : square r x.index ∈ roots.drop idx ∧ square r x.index ∉ r.snapshot.seen := by
    obtain ⟨lo,hi⟩ := inBounds x h
    rw [atIndex _ hi]
    refine ⟨names _ (List.getElem_mem _),?_⟩
    intro old
    obtain ⟨j,hj,eq⟩ := List.mem_iff_getElem.mp (prevSeen _ old)
    have used : j < (labels r).length := by rw [all,List.length_append]; omega
    have eqAt : (labels r)[j] = (labels r)[x.index] := by
      calc (labels r)[j] = pre[j] := by simp [all,List.getElem_append_left hj]
           _ = _ := eq
    have := unique.eq_of_getElem_eq used hi eqAt
    have preLen : pre.length = r.remaining.idx := by simp [pre,cursor]
    omega
  have keys := RootReplay.Split.mapped_distinct (square r) reports tail.distinct
    (fun x hx y hy eq => unique.eq_of_getElem_eq (inBounds x hx).2 (inBounds y hy).2 (by
      rwa [atIndex _ (inBounds x hx).2,atIndex _ (inBounds y hy).2] at eq))
  have records := RootReplay.Split.records_after (square r) (Root.squareValue mc fc b) r.snapshot reports
    tail.steps [] reports rfl keys fresh prev
  have best : (RootReplay.Split.replay (square r) beta reports r.snapshot).state.best = score := by
    rw [RootReplay.Split.replay_state]
    exact tail.returned
  have bound := tail.bound
  rw [node_value order cursor] at bound
  have sound := node_sound covers inside w bound
  exact ⟨best,sound,fun lo hi => RootReplay.selected _ covers inside w records (best ▸ sound) (best ▸ lo) (best ▸ hi)⟩

theorem returned (ordered : moves.Perm (Root.choices mc b roots idx))
    (ran : fromMoves Subtype.val replies schedule depth limit alpha beta moves = some r)
    (done : r.handoff = false) (w : Window alpha beta) (covers : Root.Covers mc b roots)
    (inside : idx < roots.length) (children : ChildSound mc fc b replies) :
    RootOrder.SoundAttempt mc fc b roots idx alpha beta r.snapshot.state.best r.snapshot.scores := by
  obtain ⟨trace,snapshot,_⟩ := refined ran
  obtain ⟨perm,_,tr⟩ := trace.reproduced
  have same : r.outcome = .returned r.snapshot.state.best := by simp [Result.outcome,done]
  rw [same] at tr
  obtain ⟨_,calls⟩ := tr
  have snap : r.snapshot = serial Subtype.val (observed replies) beta (r.visited ++ r.remaining.pending) (start alpha) := by
    simpa [Result.replayMoves,done] using snapshot
  rw [snap]
  exact (RootReplay.serial_of_children mc fc b roots idx _ (observed replies) _ alpha beta covers inside w
    ((by simpa [PvsPostpone.Queue.pending] using perm : (r.visited ++ r.remaining.pending).Perm moves).trans ordered) calls
    (fun m _ cp a z score win narrow child => children m cp a z score child win narrow)).2

open RootMoveStorage (Row)

def ownerRows (alpha beta : Int) (initial : List Row) (idx : Nat) (r : Result (GeneratedSquare mc b.val))
    (replies : Replies (GeneratedSquare mc b.val)) (pv : Square → List Square) : List Row :=
  RootSerialMir.rowsAfter Subtype.val (fun m => pv m.val) (observed replies) beta r.replayMoves
    (start alpha) (RootMoveStorage.resetFrom initial idx)

theorem owner_writes {initial : List Row} (covers : Root.Covers mc b (initial.map Row.square))
    (ordered : moves.Perm (Root.choices mc b (initial.map Row.square) idx))
    (ran : fromMoves Subtype.val replies schedule depth limit alpha beta moves = some r)
    (pv : Square → List Square) :
    RootSuffix.Writes initial idx alpha beta (ownerRows alpha beta initial idx r replies pv) r.snapshot := by
  obtain ⟨trace,snap,_⟩ := refined ran
  have perm : (r.visited ++ r.remaining.pending).Perm moves := by
    simpa [PvsPostpone.Queue.pending] using trace.reproduced.1
  rw [snap]
  apply RootSerialMir.rowsAfter_writes _ _ _ _ _ .reset (Root.roots_nodup covers)
  intro m mem
  have used : m ∈ r.visited ++ r.remaining.pending := by
    cases h : r.handoff <;> simp [Result.replayMoves,h] at mem <;> simp_all
  have present := (Root.choices_squares_mem (idx := idx) covers m.val).mp
    (List.mem_map_of_mem ((perm.trans ordered).mem_iff.mp used))
  simpa [List.map_drop] using present

theorem handoff_writes {initial : List Row} (covers : Root.Covers mc b (initial.map Row.square))
    (ordered : moves.Perm (Root.choices mc b (initial.map Row.square) idx))
    (ran : fromMoves Subtype.val replies schedule depth limit alpha beta moves = some r)
    (split : r.handoff = true) (reports : List Report)
    (valid : ∀ x ∈ reports, x.index ∈ SplitResult.slots r.remaining.idx (labels r).length)
    (pv : Square → List Square) :
    let physical := FrameReturns.reportRows (square r) pv beta (ownerRows alpha beta initial idx r replies pv) r.snapshot reports
    RootSuffix.Writes initial idx alpha beta physical (RootReplay.Split.replay (square r) beta reports r.snapshot) := by
  obtain ⟨trace,snap,pos⟩ := refined ran
  have perm : (r.visited ++ r.remaining.pending).Perm moves := by
    simpa [PvsPostpone.Queue.pending] using trace.reproduced.1
  have nonempty : r.visited ≠ [] := by
    have := pos split
    have cursor : r.remaining.idx = r.visited.length := by simpa using trace.reproduced.2.1
    exact List.length_pos_iff.mp (cursor ▸ this)
  have begun : r.snapshot.seen ≠ [] := by
    rw [snap]
    exact (RootReplay.serial_seen _ _ _ _ _).2 (Or.inl (by simpa [Result.replayMoves,split] using nonempty))
  apply FrameReturns.reportWrites (square r) pv beta reports r.snapshot (owner_writes covers ordered ran pv) begun
    (Root.roots_nodup covers)
  intro x h
  have inside := (SplitResult.mem_slots.mp (valid x h)).2
  have legal : square r x.index ∈ labels r := by simp only [square,List.getElem?_eq_getElem inside,Option.getD_some]; exact List.getElem_mem _
  have found := ((perm.trans ordered).map Subtype.val).mem_iff.mp legal
  simpa [List.map_drop] using (Root.choices_squares_mem (idx := idx) covers _).mp found

end NeuralReversi.PvsParallel.Interleaving.RootTrial
