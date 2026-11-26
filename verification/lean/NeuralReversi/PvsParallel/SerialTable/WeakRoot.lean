import NeuralReversi.PvsParallel.SerialTable.Root

namespace NeuralReversi.PvsParallel.SerialTable.WeakRoot

open Bitboard (Square)
open RootMoveStorage (Row)
open SerialCalls (Program)
open SerialCalls.ScoreWords (Finite)

private theorem write_keys (sq : Square) (score : Int) (visible : Bool) (pv : List Square) (rows : List Row) :
    (RootSuffix.writeFirst sq (fun r => RootSuffix.updateRow r score visible pv) rows).map Row.square =
      rows.map Row.square := by
  induction rows with
  | nil => rfl
  | cons r rs ih =>
    by_cases hit : r.square = sq
    · rw [RootSuffix.writeFirst,ite_eq_left hit]
      rfl
    · rw [RootSuffix.writeFirst,ite_eq_right hit,List.map_cons,List.map_cons,ih]

private theorem reset_keys (rows : List Row) (idx : Nat) :
    (RootMoveStorage.resetFrom rows idx).map Row.square = rows.map Row.square := by
  rw [RootMoveStorage.resetFrom,List.map_append,List.map_map]
  change (rows.take idx).map Row.square ++ (rows.drop idx).map Row.square = _
  rw [← List.map_append,List.take_append_drop]

private theorem visit_keys (square : M → Square) (pv : M → List Square) (visits : List (RootOwnerMir.Slice.Visit M))
    (s : Bool × List Row) : (visits.foldl (SerialOwnerCalls.writeVisit square pv) s).2.map Row.square = s.2.map Row.square := by
  induction visits generalizing s with
  | nil => rfl
  | cons v vs ih =>
    rw [List.foldl_cons,ih]
    exact write_keys ..

/-- Actual words returned to this root may be approximate and window-dependent. Both the wipeout
    return and the owner loop preserve every row square, in the same physical order. -/
theorem result {mc fc b d paired mem rows root a z} (issued : Program.Runs (Root.frame mc fc b d paired mem rows idx root a z) cs out)
    (covers : PvsParallel.Root.Covers mc b (rows.map Row.square)) (inside : idx < rows.length)
    (children : ∀ t ∈ cs, Finite t.2) :
    Finite out.answer.value ∧ out.answer.rows.map Row.square = rows.map Row.square := by
  dsimp only [Root.frame] at issued
  split at issued
  · cases issued
  next sq _ =>
    cases issued
    exact ⟨by change -32767 < (16384 : Int) ∧ (16384 : Int) < 32767; decide, (write_keys sq 16384 true (root.pvLine sq) _).trans (reset_keys rows idx)⟩
  next =>
    obtain ⟨owner,called,rfl⟩ := Program.runs_map issued
    obtain ⟨words,ran,rfl⟩ := Program.runs_mapRequests called
    let moves := AspirationMir.SmallRoot.candidates mc fc b (rows.map Row.square) idx root.key root.sort
    have selected : moves.Perm (PvsParallel.Root.choices mc b (rows.map Row.square) idx) :=
      (RootFrameMir.ordered_candidates mc fc b (rows.map Row.square) idx root.key root.sort).1
    have nonempty := (PvsParallel.Root.maskContract_proven mc b _ idx covers).2.2.mpr (by simpa using inside)
    have present : moves ≠ [] := by
      intro empty
      apply nonempty
      exact List.length_eq_zero_iff.mp (selected.length_eq.symm.trans (congrArg List.length empty))
    have finite (t) (ht : t ∈ words) : Finite t.2 :=
      children _ (List.mem_map_of_mem (f := fun x : SerialCalls.MoveRequest (PvsBoardTree.GeneratedSquare mc b.val) × Int =>
        (SerialRootCalls.request mc fc b d x.1,x.2)) ht)
    refine ⟨SerialOwnerCalls.run_finite present ran finite,?_⟩
    exact (visit_keys Subtype.val (fun m => root.pvLine m.val) owner.1 (true, _)).trans (reset_keys rows idx)

end NeuralReversi.PvsParallel.SerialTable.WeakRoot
