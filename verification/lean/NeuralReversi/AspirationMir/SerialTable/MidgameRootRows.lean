module
public import NeuralReversi.PvsParallel.SerialTable.WeakTrial.Bounds

@[expose] public section

namespace NeuralReversi.AspirationMir.SerialTable.MidgameRoot

open PvsParallel
open PvsBoardTree (Position GeneratedSquare)
open PvsMoveLoop (scoreInf scoreMax ChildSearch moveScore)
open RootMoveStorage (Row)
open SerialCalls (Program MoveRequest)
open Bitboard (Square)

/-- The first move is visible even on a fail-low. An accepted root must additionally
    expose the improvement over the opening alpha somewhere among its physical rows. -/
def RowMaximum (rows : List Row) (score : Int) : Prop :=
  (∃ r ∈ rows, r.score = score) ∧ ∀ r ∈ rows, r.score ≤ score

private structure Peak (a : Int) (s : RootReplay.Snapshot) : Prop where
  alpha : s.state.alpha ≤ max a s.state.best
  floor : -scoreInf ≤ s.state.best
  upper : ∀ sq, s.scores sq ≤ s.state.best
  winner : a < s.state.best → ∃ sq ∈ s.seen, s.scores sq = s.state.best

private theorem start_peak (lo : -scoreInf ≤ a) : Peak a (RootReplay.start a) :=
  ⟨Int.le_max_left ..,Int.le_refl _,fun _ => Int.le_refl _,fun h => False.elim (Int.not_lt.mpr lo h)⟩

private theorem step_peak (prior : Peak a s) (fresh : sq ∉ s.seen) (z v : Int) :
    Peak a (s.record z ⟨sq,s.state.alpha,v⟩) := by
  let best := max s.state.best v
  let visible := RootOrder.visibleScore s.seen.isEmpty s.state.alpha v
  have oldLe : s.state.best ≤ best := Int.le_max_left ..
  have valLe : v ≤ best := Int.le_max_right ..
  have floor := prior.floor
  have bound : visible ≤ best := by unfold visible RootOrder.visibleScore; split <;> omega
  refine ⟨?_,Int.le_trans floor oldLe,?_,?_⟩
  · change (if _ then v else s.state.alpha) ≤ max a best
    have := prior.alpha
    split <;> omega
  · intro q
    change RootOrder.writeScore s.scores sq visible q ≤ best
    by_cases same : q = sq
    · simpa [RootOrder.writeScore,same] using bound
    · simpa [RootOrder.writeScore,same] using Int.le_trans (prior.upper q) oldLe
  · change a < best → ∃ q ∈ sq :: s.seen, RootOrder.writeScore s.scores sq visible q = best
    intro greater
    by_cases improves : s.state.best < v
    · have winner : best = v := Int.max_eq_right (by omega)
      have raises : s.state.alpha < v := by have := prior.alpha; omega
      exact ⟨sq,List.mem_cons_self,by simp [RootOrder.writeScore,visible,RootOrder.visibleScore,raises,winner]⟩
    · have winner : best = s.state.best := Int.max_eq_left (by omega)
      obtain ⟨q,seen,won⟩ := prior.winner (by omega)
      have other : q ≠ sq := fun eq => fresh (eq ▸ seen)
      exact ⟨q,List.mem_cons_of_mem _ seen,by simpa [RootOrder.writeScore,other,winner] using won⟩

private theorem scan_peak (square : M → Square) (search : ChildSearch M) (z : Int) (moves : List M)
    (s : RootReplay.Snapshot) (prior : Peak a s) (unique : (moves.map square).Nodup)
    (apart : ∀ m ∈ moves, square m ∉ s.seen) : Peak a (RootReplay.serial square search z moves s) := by
  induction moves generalizing s with
  | nil => exact prior
  | cons m ms ih =>
    rw [RootReplay.serial]
    have record := step_peak prior (apart m List.mem_cons_self) z (moveScore true s.seen.isEmpty search m s.state.alpha z)
    split
    · exact record
    · apply ih _ record (List.nodup_cons.mp unique).2
      intro n hn seen
      rcases List.mem_cons.mp seen with same | old
      · change square n = square m at same
        exact (List.nodup_cons.mp unique).1 (same ▸ List.mem_map_of_mem (f := square) hn)
      · exact apart n (List.mem_cons_of_mem _ hn) old

private theorem maximum_of_stored {rows initial : List Row} (written : RootSuffix.Writes initial idx a z rows s)
    (unique : (initial.map Row.square).Nodup) (pk : Peak a s)
    (allSeen : ∀ sq ∈ s.seen, sq ∈ (initial.map Row.square).drop idx) (accepted : a < s.state.best) :
    RowMaximum (rows.drop idx) s.state.best := by
  obtain ⟨keys,_,scores⟩ := written.scores unique
  obtain ⟨sq,seen,won⟩ := pk.winner accepted
  have key := allSeen sq seen
  rw [← keys,← List.map_drop] at key
  obtain ⟨r,mem,eq⟩ := List.mem_map.mp key
  refine ⟨⟨r,mem,(scores r mem).trans (eq ▸ won)⟩,?_⟩
  intro row hr
  rw [scores row hr]
  exact pk.upper _

/-- With unrestricted midgame child replies, an improvement over the original alpha is
    the maximum visible row. This also accounts for unvisited rows on an early cutoff. -/
theorem root_suffix_rows {mc fc b d paired mem rows root a z} (issued : Program.Runs
    (PvsParallel.SerialTable.Root.frame mc fc b d paired mem rows idx root a z) cs out)
    (cover : Root.Covers mc b (rows.map Row.square)) (window : Aspiration.Window a.toInt z.toInt)
    : out.answer.rows.take idx = rows.take idx ∧
      (a.toInt < out.answer.value → RowMaximum (out.answer.rows.drop idx) out.answer.value) := by
  let roots := rows.map Row.square
  let start := RootReplay.start a.toInt
  have clear := RootSuffix.Writes.reset (initial := rows) (idx := idx) (alpha := a.toInt) (beta := z.toInt)
  have pk := start_peak window.lower
  dsimp only [PvsParallel.SerialTable.Root.frame] at issued
  split at issued
  · cases issued
  next sq hit =>
    cases issued
    have cache : (MoveList.Generation.fill false paired fc b.val (Root.remainingMask mc b.val roots idx) mem).wipeout = some (some sq) := by
      generalize (MoveList.Generation.fill false paired fc b.val (Root.remainingMask mc b.val (rows.map Row.square) idx) mem).len = count at hit
      change SerialRootCalls.entry count (MoveList.Generation.fill false paired fc b.val (Root.remainingMask mc b.val roots idx) mem).wipeout = _ at hit
      generalize (MoveList.Generation.fill false paired fc b.val (Root.remainingMask mc b.val roots idx) mem).wipeout = wipe at hit ⊢
      cases count with
      | none => contradiction
      | some n =>
        cases wipe with
        | none => contradiction
        | some v => cases v <;> by_cases pos : 0 < n <;> simp [SerialRootCalls.entry,rootEntry,pos] at hit ⊢; exact hit
    obtain ⟨m,member,eq,_⟩ := ConcurrentReplay.generatedWipeout mc fc b roots idx paired mem sq cover cache
    have legal : sq ∈ roots.drop idx := by simpa only [eq,roots] using (Root.choices_squares_mem cover m.val).mp (List.mem_map_of_mem member)
    have inRows : sq ∈ ((RootMoveStorage.resetFrom rows idx).drop idx).map Row.square := by
      simpa only [List.map_drop,(clear.scores (Root.roots_nodup cover)).1] using legal
    have written := RootSuffix.Writes.record clear (⟨sq,a.toInt,scoreMax⟩ : RootOrder.Report) true (root.pvLine sq) inRows rfl
    have peak := step_peak pk (sq := sq) (by simp [RootReplay.start]) z.toInt scoreMax
    refine ⟨(written.scores (Root.roots_nodup cover)).2.1,?_⟩
    intro accepted
    apply maximum_of_stored written (Root.roots_nodup cover) peak _ accepted
    intro q h
    change q ∈ [sq] at h
    exact (List.mem_singleton.mp h) ▸ legal
  next =>
    obtain ⟨owner,requests,rfl⟩ := Program.runs_map issued
    obtain ⟨words,ran,rfl⟩ := Program.runs_mapRequests requests
    let moves := SmallRoot.candidates mc fc b roots idx root.key root.sort
    have order : moves.Perm (Root.choices mc b roots idx) := (RootFrameMir.ordered_candidates ..).1
    have unique := order.nodup_iff.mpr ((EndgameOrderMir.generated_layout mc b).1.filter _)
    let key := fun q : MoveRequest (GeneratedSquare mc b.val) => (q.move,q.pv)
    let search : ChildSearch (GeneratedSquare mc b.val) := fun m p _ _ => Program.returnAt key words (m,p)
    have supplied t (h : t ∈ words) : search t.1.move t.1.pv t.1.alpha t.1.beta = t.2 :=
      Program.returnAt_mem key words (SerialOwnerCalls.runs_distinct ran unique) t h
    obtain ⟨snapshot,_⟩ := SerialOwnerCalls.runs_serial ran search (fun _ _ _ _ _ => True) (fun t h => ⟨supplied t h,trivial⟩)
    have stored := SerialOwnerCalls.runs_rows ran (fun m => root.pvLine m.val) (RootMoveStorage.resetFrom rows idx) search supplied
    have named m (h : m ∈ moves) : m.val ∈ roots.drop idx := by
      simpa only [roots] using (Root.choices_squares_mem cover m.val).mp (List.mem_map_of_mem (order.mem_iff.mp h))
    have written := RootSerialMir.rowsAfter_writes Subtype.val (fun m => root.pvLine m.val) search moves start clear
      (Root.roots_nodup cover) (by simpa only [roots,List.map_drop] using named)
    have peak := scan_peak Subtype.val search z.toInt moves start pk
      (unique.map Subtype.val (fun _ _ neq eq => neq (Subtype.ext eq))) (by simp [start,RootReplay.start])
    have allSeen (sq) (hs : sq ∈ (RootReplay.serial Subtype.val search z.toInt moves start).seen) : sq ∈ roots.drop idx := by
      rcases (RootReplay.serial_seen Subtype.val search z.toInt moves start).1 sq hs with absent | visited
      · cases absent
      · obtain ⟨m,h,rfl⟩ := List.mem_map.mp visited
        exact named m h
    constructor
    · change (owner.1.foldl (SerialOwnerCalls.writeVisit Subtype.val (fun m => root.pvLine m.val))
        ((RootReplay.start a.toInt).seen.isEmpty,RootMoveStorage.resetFrom rows idx)).2.take idx = rows.take idx
      rw [stored]
      exact (written.scores (Root.roots_nodup cover)).2.1
    intro accepted
    change a.toInt < owner.2.state.best at accepted
    change RowMaximum _ owner.2.state.best
    rw [snapshot] at accepted ⊢
    change RowMaximum ((owner.1.foldl (SerialOwnerCalls.writeVisit Subtype.val (fun m => root.pvLine m.val)) ((RootReplay.start a.toInt).seen.isEmpty,RootMoveStorage.resetFrom rows idx)).2.drop idx) _
    rw [stored]
    exact maximum_of_stored written (Root.roots_nodup cover) peak allSeen accepted

theorem root_maximum {mc fc b d paired mem rows root a z} (issued : Program.Runs
    (PvsParallel.SerialTable.Root.frame mc fc b d paired mem rows 0 root a z) cs out)
    (cover : Root.Covers mc b (rows.map Row.square)) (window : Aspiration.Window a.toInt z.toInt)
    (accepted : a.toInt < out.answer.value) : RowMaximum out.answer.rows out.answer.value := by
  simpa using (root_suffix_rows issued cover window).2 accepted

end NeuralReversi.AspirationMir.SerialTable.MidgameRoot
