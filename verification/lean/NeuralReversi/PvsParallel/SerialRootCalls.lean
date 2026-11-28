module
public import NeuralReversi.PvsParallel.SerialOwnerCalls
public import NeuralReversi.PvsParallel.SerialRoot
public import NeuralReversi.PvsParallel.EndgameOrderMir

@[expose] public section

namespace NeuralReversi.PvsParallel.SerialRootCalls

open Aspiration (Window)
open AspirationMir.SmallRoot (Answer candidates)
open Bitboard (Square)
open Bitboard.Moves (X86Config)
open Cutoffs (FlipConfig)
open PvsBoardTree (Position GeneratedSquare build play)
open RootMoveStorage (Row)
open SerialCalls (Program Request MoveRequest)

def entry (count : Option Nat) (wipe : Option (Option Square)) : Option (Option Square) := do
  let n ← count
  let cached ← wipe
  if h : 0 < n then
    match rootEntry n cached.isSome h, cached with
    | .wipeout, some sq => some (some sq)
    | .header _ _ _ _, none => some none
    | _, _ => none
  else none

private theorem entry_wipe (found : entry n wipe = some (some sq)) : wipe = some (some sq) := by
  cases n with
  | none => contradiction
  | some n =>
    cases wipe with
    | none => contradiction
    | some hit =>
      cases hit <;> by_cases h : 0 < n <;> simp [entry, rootEntry, h] at found ⊢
      exact found

def request (mc : X86Config) (fc : FlipConfig) (b : Position) (depth : ExactDepth.Depth)
    (q : MoveRequest (GeneratedSquare mc b.val)) : Request :=
  ⟨(play mc fc b q.move).val, depth - 1, q.pv, !q.pv, q.alpha, q.beta⟩

theorem request_scout (mc : X86Config) (fc : FlipConfig) (b : Position) (depth : ExactDepth.Depth)
    (m : GeneratedSquare mc b.val) (alpha : Int) :
    request mc fc b depth ⟨m, false, -(alpha+1), -alpha⟩ =
      SerialCalls.Sites.scoutRequest (play mc fc b m).val depth false alpha := rfl

theorem request_wide (mc : X86Config) (fc : FlipConfig) (b : Position) (depth : ExactDepth.Depth)
    (m : GeneratedSquare mc b.val) (alpha beta : Int) :
    request mc fc b depth ⟨m, true, -beta, -alpha⟩ =
      SerialCalls.Sites.wideRequest (play mc fc b m).val depth alpha beta := rfl

def ownerAnswer (initial : List Row) (idx : Nat) (pv : Square → List Square)
    (out : SerialOwnerCalls.Result (GeneratedSquare mc b)) : Answer :=
  ⟨(rootEpilogue idx out.2.state.best).2,
    (out.1.foldl (SerialOwnerCalls.writeVisit Subtype.val (fun m => pv m.val))
      (true, RootMoveStorage.resetFrom initial idx)).2⟩

def ordinary (mc : X86Config) (fc : FlipConfig) (b : Position) (depth : ExactDepth.Depth)
    (moves : List (GeneratedSquare mc b.val)) (initial : List Row) (idx : Nat)
    (pv : Square → List Square) (a z : BitVec 32) : Program Request Answer :=
  let start := RootReplay.start a.toInt
  ((SerialOwnerCalls.run Subtype.val z.toInt moves start).mapRequests (request mc fc b depth)).map
    (ownerAnswer initial idx pv)

def frame (mc : X86Config) (fc : FlipConfig) (b : Position) (depth : ExactDepth.Depth)
    (paired : Bool) (mem : MoveList.Initialization.Memory Move.Move) (initial : List Row) (idx : Nat)
    (key : Square → BitVec 32) (sort : RootFrameMir.LibraryOrder)
    (pv : Square → List Square) (a z : BitVec 32) : Program Request Answer :=
  let roots := initial.map Row.square
  let gen := MoveList.Generation.fill false paired fc b.val (Root.remainingMask mc b.val roots idx) mem
  match entry gen.len gen.wipeout with
  | none => .fail
  | some (some sq) =>
    let (writes, value) := rootWipeoutExit sq
    .pure ⟨value, writes.foldl (AspirationMir.applyRootWrite pv) (RootMoveStorage.resetFrom initial idx)⟩
  | some none => ordinary mc fc b depth (candidates mc fc b roots idx key sort) initial idx pv a z

private theorem eval_ordinary (mc : X86Config) (fc : FlipConfig) (b : Position) (depth : ExactDepth.Depth)
    (moves : List (GeneratedSquare mc b.val)) (initial : List Row) (idx : Nat)
    (pv : Square → List Square) (a z : BitVec 32) (respond : Request → Option Int) :
    let r := fun m p lo hi => respond (request mc fc b depth ⟨m,p,lo,hi⟩)
    let start := RootReplay.start a.toInt
    let rows := RootMoveStorage.resetFrom initial idx
    (ordinary mc fc b depth moves initial idx pv a z).eval respond = (do
      let out ← RootOwnerMir.Slice.run Subtype.val r RootOwnerMir.Serial.schedule depth.toNat
        rootMinimumSplitDepth z.toInt 0 0 moves.length moves start
      guard (!out.handoff)
      return ⟨(rootEpilogue idx out.snapshot.state.best).2,
        RootSerialMir.rowsAfter Subtype.val (fun m => pv m.val) (RootOwnerMir.observed r) z.toInt moves start rows⟩) := by
  intro r start rows
  simp only [ordinary, Program.eval_map, Program.eval_mapRequests]
  change ((SerialOwnerCalls.run Subtype.val z.toInt moves start).eval (SerialOwnerCalls.reply r)).map _ = _
  have agree := SerialOwnerCalls.eval_handoff Subtype.val r depth.toNat rootMinimumSplitDepth a.toInt z.toInt moves
  change ((SerialOwnerCalls.run Subtype.val z.toInt moves start).eval (SerialOwnerCalls.reply r)).map _ =
    (RootOwnerMir.Slice.run Subtype.val r RootOwnerMir.Serial.schedule depth.toNat rootMinimumSplitDepth z.toInt
      0 0 moves.length moves start).map _ at agree
  cases issued : (SerialOwnerCalls.run Subtype.val z.toInt moves start).eval (SerialOwnerCalls.reply r) <;>
    cases source : RootOwnerMir.Slice.run Subtype.val r RootOwnerMir.Serial.schedule depth.toNat rootMinimumSplitDepth
      z.toInt 0 0 moves.length moves start <;>
    simp only [issued, source, Option.map_none, Option.map_some, Option.some.injEq, Prod.mk.injEq,
      reduceCtorEq] at agree <;> try contradiction
  · rfl
  · obtain ⟨snap, joined⟩ := agree
    obtain ⟨calls, ran, supplied⟩ := Program.runs_of_eval _ _ issued
    have written := SerialOwnerCalls.runs_rows ran (fun m => pv m.val) rows (RootOwnerMir.observed r) (by
      intro t h
      have returned := supplied t h
      change r t.1.move t.1.pv t.1.alpha t.1.beta = some t.2 at returned
      simp [RootOwnerMir.observed, returned])
    change (List.foldl _ (true,rows) _).2 = _ at written
    simp [ownerAnswer, ← joined, guard, snap, ← written, rows]

/-- Equality with the previous root-frame interpretation includes the stored rows
    and rejection when a reached recursive reply is absent. -/
theorem eval_frame (mc : X86Config) (fc : FlipConfig) (b : Position) (depth : ExactDepth.Depth)
    (paired : Bool) (mem : MoveList.Initialization.Memory Move.Move) (initial : List Row) (idx : Nat)
    (key : Square → BitVec 32) (sort : RootFrameMir.LibraryOrder)
    (pv : Square → List Square) (a z : BitVec 32) (respond : Request → Option Int) :
    (frame mc fc b depth paired mem initial idx key sort pv a z).eval respond =
      SerialRoot.trial mc fc b depth paired mem initial idx key sort pv a z
        (fun m p lo hi => respond (request mc fc b depth ⟨m,p,lo,hi⟩)) := by
  dsimp only [frame, SerialRoot.trial]
  let gen := MoveList.Generation.fill false paired fc b.val (Root.remainingMask mc b.val (initial.map Row.square) idx) mem
  generalize cn : gen.len = count
  generalize cc : gen.wipeout = cache
  cases count with
  | none => rfl
  | some n =>
    cases cache with
    | none => rfl
    | some cached =>
      by_cases positive : 0 < n
      · cases cached with
        | some sq => simp [entry, positive, rootEntry, Program.eval]
        | none =>
          simpa only [entry, Option.bind_some, positive, ↓reduceDIte, Option.isSome_none, rootEntry, Bool.false_eq_true,
            ite_false, bind, Option.bind] using
            eval_ordinary mc fc b depth (candidates mc fc b (initial.map Row.square) idx key sort) initial idx pv a z respond
      · simp [entry, positive, Program.eval]

theorem frame_ready_at_depth {mc : X86Config} {fc : FlipConfig} {b : Position} {initial : List Row}
    {a z : BitVec 32} (covers : Root.Covers mc b (initial.map Row.square)) (within : idx < initial.length)
    (window : Window a.toInt z.toInt) :
    Program.Ready (fun r => ExactDepth.RecursiveCall mc fc b depth r.board r.depth ∧
      Window r.alpha r.beta ∧ (r.pv = false → r.beta = r.alpha + 1))
      (frame mc fc b depth paired mem initial idx key sort pv a z) := by
  let roots := initial.map Row.square
  let selected := Root.remainingMask mc b.val roots idx
  have subset : MoveList.Generation.IsSubset selected (Board.getMoves mc b.val) := by
    intro i bit
    simp only [selected, Root.remainingMask, BitVec.getLsbD_and, Bool.and_eq_true] at bit
    exact bit.1
  obtain ⟨_, _, count, nonempty, length⟩ := ConcurrentReplay.generatedRoot_positive mc fc b roots idx (Board.emptyCount b.val)
    paired mem covers (by simpa [roots] using within) rfl
  have cache := (MoveList.Generation.generationContract_proven mc fc b.val selected false paired mem b.property
    subset (by simp)).2.2.2.2
  generalize ((MoveList.Generation.candidatesForCache (MoveList.Generation.records fc b.val selected)).reverse.find?
    (MoveList.Wipeout.isMatch b.val.opponent)).map MoveList.Wipeout.Move.square = hit at cache
  dsimp only [frame]
  dsimp only [selected, roots] at cache length
  rw [cache, length]
  cases hit with
  | some sq =>
    simp only [entry, nonempty, ↓reduceDIte, rootEntry, Option.isSome_some, ite_true,
      bind, Option.bind]
    exact .pure
  | none =>
    simp only [entry, nonempty, ↓reduceDIte, rootEntry, Option.isSome_none, Bool.false_eq_true,
      ite_false, bind, Option.bind, ordinary]
    apply Program.Ready.map
    apply Program.Ready.mapRequests
    have prepared := SerialOwnerCalls.run_ready (s := RootReplay.start a.toInt) (square := Subtype.val)
      window (candidates mc fc b roots idx key sort)
    generalize SerialOwnerCalls.run Subtype.val z.toInt (candidates mc fc b roots idx key sort) (RootReplay.start a.toInt) =
      code at prepared ⊢
    induction prepared with
    | pure => exact .pure
    | @call q _ good _ ih => exact .call ⟨.ply b q.move depth, good⟩ ih

theorem frame_ready {mc : X86Config} {fc : FlipConfig} {b : Position} {initial : List Row}
    {a z : BitVec 32} (covers : Root.Covers mc b (initial.map Row.square)) (within : idx < initial.length)
    (_full : ExactDepth.Full b depth) (window : Window a.toInt z.toInt) :
    Program.Ready (fun r => ExactDepth.RecursiveCall mc fc b depth r.board r.depth ∧
      Window r.alpha r.beta ∧ (r.pv = false → r.beta = r.alpha + 1))
      (frame mc fc b depth paired mem initial idx key sort pv a z) :=
  frame_ready_at_depth covers within window

theorem frame_edges {mc : X86Config} {fc : FlipConfig} {b : Position}
    (ran : Program.Runs (frame mc fc b depth paired mem initial idx key sort pv a z) calls answer)
    (window : Window a.toInt z.toInt) (item) (member : item ∈ calls) :
    ∃ m : GeneratedSquare mc b.val,
      item.1.board = (play mc fc b m).val ∧ item.1.depth = depth - 1 ∧ item.1.cut = !item.1.pv ∧
      Window item.1.alpha item.1.beta ∧ (item.1.pv = false → item.1.beta = item.1.alpha + 1) := by
  dsimp only [frame] at ran
  split at ran
  · cases ran
  · cases ran; cases member
  · obtain ⟨result, issued, _⟩ := Program.runs_map ran
    obtain ⟨words, owner, rfl⟩ := Program.runs_mapRequests issued
    obtain ⟨t, found, rfl⟩ := List.mem_map.mp member
    exact ⟨t.1.move, rfl, rfl, rfl, SerialOwnerCalls.runs_windows owner window t found⟩

theorem frame_replay {mc : X86Config} {fc : FlipConfig} {b : Position}
    (ran : Program.Runs (frame mc fc b depth paired mem initial idx key sort pv a z) calls answer)
    (covers : Root.Covers mc b (initial.map Row.square)) (within : idx < initial.length)
    (window : Window a.toInt z.toInt)
    (child : ∀ t ∈ calls, Replayed (build mc fc t.1.board) t.1.pv t.1.alpha t.1.beta t.2) :
    ∃ snap : RootReplay.Snapshot, answer.value = snap.state.best ∧
      RootSuffix.Writes initial idx a.toInt z.toInt answer.rows snap ∧
      RootEvidence.Attempt mc fc b (initial.map Row.square) idx a.toInt z.toInt answer.value snap.scores ∧
      RootOrder.SoundAttempt mc fc b (initial.map Row.square) idx a.toInt z.toInt answer.value snap.scores := by
  let roots := initial.map Row.square
  have pos : idx < roots.length := by simpa [roots] using within
  let start := RootReplay.start a.toInt
  have cleared := RootSuffix.Writes.reset (initial := initial) (idx := idx) (alpha := a.toInt) (beta := z.toInt)
  dsimp only [frame] at ran
  split at ran
  · cases ran
  next sq found =>
    cases ran
    obtain ⟨m, member, square, wipe⟩ := ConcurrentReplay.generatedWipeout mc fc b roots idx paired mem sq covers
      (entry_wipe found)
    have legal := (Root.choices_squares_mem covers m.val).mp (List.mem_map_of_mem member)
    have shape := (cleared.scores (Root.roots_nodup covers)).1
    have inRows : sq ∈ ((RootMoveStorage.resetFrom initial idx).drop idx).map Row.square := by
      rw [List.map_drop, shape]
      simpa [roots, square] using legal
    have written := RootSuffix.Writes.record cleared (⟨sq, a.toInt, PvsMoveLoop.scoreMax⟩ : RootOrder.Report)
      true (pv sq) inRows rfl
    have attempt : RootEvidence.Attempt mc fc b roots idx a.toInt z.toInt PvsMoveLoop.scoreMax
        (start.record z.toInt ⟨sq, a.toInt, PvsMoveLoop.scoreMax⟩).scores := by
      rw [← square]
      exact .wipeout m member wipe
    exact ⟨_, rfl, written, attempt, attempt.sound covers pos window⟩
  next =>
    obtain ⟨result, issued, rfl⟩ := Program.runs_map ran
    obtain ⟨words, owner, rfl⟩ := Program.runs_mapRequests issued
    let moves := candidates mc fc b roots idx key sort
    have order : moves.Perm (Root.choices mc b roots idx) := (RootFrameMir.ordered_candidates mc fc b roots idx key sort).1
    have unique := order.nodup_iff.mpr ((EndgameOrderMir.generated_layout mc b).1.filter _)
    let moveKey := fun q : MoveRequest (GeneratedSquare mc b.val) => (q.move,q.pv)
    let search : PvsMoveLoop.ChildSearch (GeneratedSquare mc b.val) := fun m p _ _ => Program.returnAt moveKey words (m,p)
    have supplied (t) (h : t ∈ words) : search t.1.move t.1.pv t.1.alpha t.1.beta = t.2 :=
      Program.returnAt_mem moveKey words (SerialOwnerCalls.runs_distinct owner unique) t h
    have completed (t) (h : t ∈ words) :
        Replayed (build mc fc (play mc fc b t.1.move).val) t.1.pv t.1.alpha t.1.beta t.2 := by
      have found := List.mem_map_of_mem (f := fun r : MoveRequest (GeneratedSquare mc b.val) × Int =>
        (request mc fc b depth r.1, r.2)) h
      have returned := child _ found
      exact returned
    obtain ⟨snapshot, calls⟩ := SerialOwnerCalls.runs_serial owner search
      (fun m p lo hi score => Replayed (build mc fc (play mc fc b m).val) p lo hi score)
      fun t hu => ⟨supplied t hu, completed t hu⟩
    have attempt : RootEvidence.Attempt mc fc b roots idx a.toInt z.toInt result.2.state.best result.2.scores := by
      rw [snapshot]
      exact .sequential moves search order calls
    have stored := SerialOwnerCalls.runs_rows owner (fun m => pv m.val) (RootMoveStorage.resetFrom initial idx) search supplied
    have writes := RootSerialMir.rowsAfter_writes Subtype.val (fun m => pv m.val) search moves start cleared
      (Root.roots_nodup covers) (by
        intro m h
        rw [List.map_drop]
        exact (Root.choices_squares_mem covers m.val).mp (List.mem_map_of_mem (order.mem_iff.mp h)))
    refine ⟨result.2, rfl, ?_, attempt, attempt.sound covers pos window⟩
    rw [← snapshot] at writes
    simp only [ownerAnswer]
    exact stored.symm ▸ writes

end NeuralReversi.PvsParallel.SerialRootCalls
