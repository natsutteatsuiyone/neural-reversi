module
public import NeuralReversi.PvsParallel.Interleaving.RootSchedule
public import NeuralReversi.PvsParallel.SerialCalls.ScoreWords

@[expose] public section

namespace NeuralReversi.PvsParallel.Interleaving.RootSchedule

open PvsMoveLoop (ChildRelation ChildSearch ScanCalls MoveCalls moveScore scan scoreInf)
open PvsBoardTree (Position GeneratedSquare)
open SerialCalls.ScoreWords (Finite)
open SplitWorkerMir (issued retry)
open Threading
open RootOwnerMir (Result Replies fromMoves observed)
open RootMoveStorage (Row)

private theorem neg_finite (f : Finite n) : Finite (-n) := by
  obtain ⟨lo,hi⟩ := f
  exact ⟨by omega,by omega⟩

private theorem move_finite (call : MoveCalls done_ search pv first m a z)
    (child : ∀ p l h v, done_ m p l h v → Finite v) : Finite (moveScore pv first search m a z) := by
  unfold MoveCalls at call
  unfold moveScore
  split <;> rename_i eldest
  · exact neg_finite (child _ _ _ _ (by simpa only [eldest,ite_true] using call))
  · rw [ite_eq_right eldest] at call
    dsimp only
    split <;> rename_i retry
    · exact neg_finite (child _ _ _ _ (call.2 retry))
    · exact neg_finite (child _ _ _ _ call.1)

theorem scan_finite {moves : List Move} (runs : ScanCalls done_ search pv beta moves first alpha best)
    (child : ∀ m cp a z v, done_ m cp a z v → Finite v)
    (low : -32767 ≤ best) (high : best < 32767) (started : moves ≠ [] ∨ -32767 < best) :
    Finite (scan pv search beta moves first alpha best) := by
  induction moves generalizing first alpha best with
  | nil => exact ⟨started.resolve_left (by simp),high⟩
  | cons m rest ih =>
    have score := move_finite runs.1 (child m)
    have tail := runs.2
    rw [PvsMoveLoop.scan_step]
    cases next : PvsMoveLoop.nextScanState pv beta alpha best (moveScore pv first search m alpha beta) with
    | none => exact score
    | some pair =>
      obtain ⟨a,b⟩ := pair
      rw [next] at tail
      have changed : b = max best (moveScore pv first search m alpha beta) := by
        unfold PvsMoveLoop.nextScanState at next
        split at next <;> rename_i raised
        · rw [Int.max_eq_right (Int.le_of_lt raised)]
          split at next
          · split at next <;> cases next; rfl
          · cases next; rfl
        · cases next
          exact (Int.max_eq_left (Int.le_of_not_gt raised)).symm
      obtain ⟨lo,hi⟩ := score
      exact ih tail (by omega) (by omega) (.inr (by omega))

private theorem owner_finite {b : Position} {r : Result (GeneratedSquare mc b.val)}
    {replies : Replies (GeneratedSquare mc b.val)} {moves : List (GeneratedSquare mc b.val)}
    (ran : fromMoves Subtype.val replies schedule depth limit alpha beta moves = some r)
    (ordered : moves.Perm (Root.choices mc b roots idx)) (covers : Root.Covers mc b roots)
    (inside : idx < roots.length) (children : ∀ m cp a z v, replies m cp a z = some v → Finite v) :
    Finite r.snapshot.state.best := by
  obtain ⟨trace,snap,positive⟩ := RootOwnerMir.refines Subtype.val replies
    (fun m cp a z v => replies m cp a z = some v) (fun _ _ _ _ _ h => h) _ _ _ _ _ _ _ _ _ _
    (by simp) rfl ran
  obtain ⟨perm,count,path⟩ := trace.reproduced
  rw [snap,RootReplay.serial_best]
  cases handoff : r.handoff with
  | false =>
    have finished : r.outcome = .returned r.snapshot.state.best := by simp [Result.outcome,handoff]
    rw [finished] at path
    have mobile := (Root.maskContract_proven mc b roots idx covers).2.2.mpr inside
    have nonempty : r.visited ++ r.remaining.pending ≠ [] := by
      intro no
      have p : (r.visited ++ r.remaining.pending).Perm moves := by simpa [PvsPostpone.Queue.pending] using perm
      exact mobile ((no ▸ p.trans ordered).nil_eq.symm)
    simpa [Result.replayMoves,handoff,RootReplay.start] using scan_finite path.2 children (by change (-32767:Int) ≤ -32767; decide) (by change (-32767:Int) < 32767; decide) (.inl nonempty)
  | true =>
    have continuing : r.outcome = .handoff r.snapshot.state := by simp [Result.outcome,handoff]
    rw [continuing] at path
    have cursor : r.remaining.idx = r.visited.length := by simpa using count
    have visited : r.visited ≠ [] := List.length_pos_iff.mp (cursor ▸ positive handoff)
    simpa [Result.replayMoves,handoff,RootReplay.start] using scan_finite path.2 children (by change (-32767:Int) ≤ -32767; decide) (by change (-32767:Int) < 32767; decide) (.inl visited)

private theorem word_eq (f : Finite n) : (BitVec.ofInt 32 n).toInt = n :=
  BitVec.toInt_ofInt_eq_self (by decide) (by have := f.1; omega) (by have := f.2; omega)

private theorem Job_finite (j : Schedule.Job) (t : Schedule.Task) (ps : List Tickets.Post)
    (calls : j.Calls done_ t ps) (child : ∀ cp a z v, done_ j.index cp a z v → Finite v) :
    Finite (j.post t ps).report.score := by
  let i := j.input t ps
  let sc := j.reply j.index false (-(j.alpha t ps false+1)) (-(j.alpha t ps false))
  let wd := j.reply j.index true (-t.beta) (-(j.alpha t ps true))
  let again := SplitWindow.reSearch t.pv (j.alpha t ps false) sc
  have sf : Finite sc := child _ _ _ _ calls.1
  have f1 : Finite i.scout.toInt := word_eq sf ▸ sf
  have f2 : Finite i.wide.toInt := by
    change Finite (BitVec.ofInt 32 (if again then wd else 0)).toInt
    split
    · rename_i yes
      have wf := child _ _ _ _ (calls.2 yes)
      exact word_eq wf ▸ wf
    · exact ⟨by decide,by decide⟩
  change Finite (Worker.report i).score
  cases h : retry i with
  | false =>
    simp only [Worker.report,SplitWorkerMir.selected,issued,h,Bool.false_eq_true,ite_false]
    rw [Solve2.neg_safe _ (by have := f1.1; omega)]
    exact neg_finite f1
  | true =>
    simp only [Worker.report,SplitWorkerMir.selected,issued,h,ite_true]
    rw [Solve2.neg_safe _ (by have := f2.1; omega)]
    exact neg_finite f2

theorem reported_finite (run : Schedule.RunWith outside t done_ len start owner mask state)
    (children : ∀ i cp a z v, done_ i cp a z v → Finite v) :
    ∀ p ∈ state.reports, Finite p.report.score := by
  induction run with
  | initial => simp [Tickets.initial]
  | book _ _ _ step ih | finish _ _ _ step ih => simpa [step.reports_eq,Tickets.Action.posts] using ih
  | @work actions before a after prior step effect _ check ih =>
    intro p mem
    rw [step.reports_eq] at mem
    rcases List.mem_append.mp mem with old | posted
    · exact ih p old
    · obtain ⟨j,rfl,calls⟩ := check p posted
      exact Job_finite j t before.reports calls (children j.index)

theorem locked_finite (pv : Bool) (beta : Int) (s : SplitWindow.State)
    (posts : List Tickets.Post) (entry : Finite s.best) (reports : ∀ p ∈ posts, Finite p.report.score) :
    Finite (SplitReturn.stateAfter pv beta s (posts.map Tickets.Post.report)).best := by
  induction posts generalizing s with
  | nil => exact entry
  | cons p rest ih =>
    have ⟨a,b⟩ := reports p List.mem_cons_self
    have next : Finite (SplitWindow.publish pv beta s p.report.score).best := by
      change Finite (max s.best p.report.score)
      exact ⟨by have := entry.1; omega,by have := entry.2; omega⟩
    exact ih _ next (fun p h => reports p (List.mem_cons_of_mem _ h))

theorem Trial.finite {b : Position} {done_ : ChildRelation (GeneratedSquare mc b.val)}
    (trial : Trial mc b initial idx alpha beta done_ score rows) (covers : Root.Covers mc b (initial.map Row.square))
    (inside : idx < initial.length) (child : ∀ m cp a z v, done_ m cp a z v → Finite v) :
    Finite score ∧ rows.map Row.square = initial.map Row.square ∧ rows.take idx = initial.take idx := by
  cases trial with
  | owner r replies moves pv order source ran done =>
    have f := owner_finite ran order covers (by simpa using inside) (fun m cp a z v eq => child m cp a z v (source m cp a z v eq))
    have writes := (RootTrial.owner_writes covers order ran pv).scores (Root.roots_nodup covers)
    exact ⟨f,writes.1,writes.2.1⟩
  | @split schedule depth limit n cut score r replies moves pv order source ran handed positive width j returns =>
    have f := owner_finite ran order covers (by simpa using inside) (fun m cp a z v eq => child m cp a z v (source m cp a z v eq))
    have words := reported_finite j.path (fun i cp a z v reply => by
      obtain ⟨m,_,used⟩ := reply
      exact child m cp a z v used)
    have final := j.finished returns
    have finite := locked_finite true beta r.snapshot.state j.state.reports f words
    obtain ⟨fit,startEq⟩ := iterator_layout order ran width
    have valid (p) (member : p ∈ Schedule.reports j.state) : p.index ∈ SplitResult.slots r.remaining.idx (RootTrial.labels r).length := by
      obtain ⟨post,hit,rfl⟩ := List.mem_map.mp member
      exact startEq ▸ SplitResult.mem_slots.mpr (j.path.tickets.valid fit hit)
    have writes := (RootTrial.handoff_writes covers order ran handed _ valid pv).scores (Root.roots_nodup covers)
    exact ⟨final.2.2 ▸ finite,writes.1,writes.2.1⟩

end NeuralReversi.PvsParallel.Interleaving.RootSchedule
