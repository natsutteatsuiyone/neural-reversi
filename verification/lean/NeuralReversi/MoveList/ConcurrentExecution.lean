module
public import NeuralReversi.MoveList.ConcurrentTrace

@[expose] public section

namespace NeuralReversi.MoveList.ConcurrentTrace

open Concurrent (advance Claims)

inductive Local (w : Nat) where
  | idle
  | testing (opening : Nat) (prev : BitVec w)
  | held (write : Write w)
  | drained (opening : Nat) (seen : BitVec w)

structure State (w : Nat) where
  current : BitVec w
  writes : List (Write w)
  localAt : Nat → Local w

def replace (slots : Nat → Local w) (worker : Nat) (slot : Local w) (i : Nat) : Local w :=
  if i = worker then slot else slots i

/-- Interleavings of the load and weak CAS inside `try_update`. A pending CAS may fail
    spuriously. A successful RMW reads the immediately preceding modification. Loads and
    unsuccessful CASs may read any available modification, including older values. -/
inductive Execution (len : Nat) (start : BitVec w) : State w → Prop where
  | init : Execution len start ⟨start,[],fun _ => .idle⟩
  | load (run : Execution len start s) (free : s.localAt worker = .idle)
      (read : (words start s.writes)[(source : Nat)]? = some seen) (opening : Nat) :
      Execution len start { s with localAt := replace s.localAt worker (.testing opening seen) }
  | failed (run : Execution len start s) (active : s.localAt worker = .testing opening prev)
      (proposed : advance len prev = some wanted)
      (read : (words start s.writes)[(source : Nat)]? = some seen) :
      Execution len start { s with localAt := replace s.localAt worker (.testing opening seen) }
  | success (run : Execution len start s) (active : s.localAt worker = .testing opening s.current)
      (proposed : advance len s.current = some wanted) :
      Execution len start (let op : Write w := ⟨worker,opening,s.current,wanted⟩
        ⟨wanted,s.writes ++ [op],replace s.localAt worker (.held op)⟩)
  | decline (run : Execution len start s) (active : s.localAt worker = .testing opening prev)
      (ended : advance len prev = none) :
      Execution len start { s with localAt := replace s.localAt worker (.drained opening prev) }
  | returnedSome (run : Execution len start s) (held : s.localAt worker = .held op) :
      Execution len start { s with localAt := replace s.localAt worker .idle }
  | returnedNone (run : Execution len start s) (drained : s.localAt worker = .drained opening seen) :
      Execution len start { s with localAt := replace s.localAt worker .idle }

structure Drained (len : Nat) (start : BitVec w) (final : State w) (i : Nat) where
  snapshot : State w
  run : Execution len start snapshot
  prior : snapshot.writes <+: final.writes
  opening : Nat
  seen : BitVec w
  read : snapshot.localAt i = .drained opening seen

private theorem replay_append (xs ys : List (Write w)) (first : replay len start xs = some middle) :
    replay len start (xs ++ ys) = replay len middle ys := by
  induction xs generalizing start with
  | nil => cases first; rfl
  | cons x rest ih =>
    simp only [replay] at first
    split at first
    · rename_i consistent
      simpa [replay,consistent] using ih first
    · contradiction

theorem Execution.replayed (run : Execution len start s) : replay len start s.writes = some s.current := by
  induction run with
  | init => rfl
  | success _ _ proposed ih =>
    rw [replay_append _ _ ih]
    simp [replay,proposed]
  | _ => assumption

def Local.valid (len : Nat) (start : BitVec w) (ops : List (Write w)) (worker : Nat) : Local w → Prop
  | .idle => True
  | .testing _ prev => prev ∈ words start ops
  | .held op => op.worker = worker ∧ op ∈ ops
  | .drained _ prev => prev ∈ words start ops ∧ advance len prev = none

private theorem Local.valid_snoc (slot : Local w) (known : slot.valid len start ops worker) (op : Write w) :
    slot.valid len start (ops ++ [op]) worker := by
  have more {x} (mem : x ∈ words start ops) : x ∈ words start (ops ++ [op]) := by
    simpa [words] using (List.mem_append_left [op.stored] mem)
  cases slot with
  | idle => trivial
  | testing => exact more known
  | held => exact ⟨known.1,List.mem_append_left _ known.2⟩
  | drained => exact ⟨more known.1,known.2⟩

private theorem replace_valid {slots : Nat → Local w} (all : ∀ i, (slots i).valid len start ops i)
    (value : slot.valid len start ops worker) : ∀ i, (replace slots worker slot i).valid len start ops i := by
  intro i
  by_cases same : i = worker
  · simpa [replace,same] using value
  · simpa [replace,same] using all i

theorem Execution.local_valid (run : Execution len start s) : ∀ i, (s.localAt i).valid len start s.writes i := by
  induction run with
  | init => exact fun _ => trivial
  | load _ _ read _ ih | failed _ _ _ read ih => exact replace_valid ih (List.mem_of_getElem? read)
  | success _ _ _ ih =>
    apply replace_valid (fun i => Local.valid_snoc _ (ih i) _)
    exact ⟨rfl,by simp⟩
  | @decline state worker opening prev _ active stop ih =>
    have prior := ih worker
    rw [active] at prior
    exact replace_valid ih ⟨prior,stop⟩
  | returnedSome _ _ ih | returnedNone _ _ ih => exact replace_valid ih trivial

theorem Execution.claims (run : Execution len start s) :
    Claims len start (s.writes.map Write.old) s.current := replay_claims _ run.replayed

theorem Execution.end_of_list (run : Execution len start s) (done : s.localAt worker = .drained opening seen) :
    seen = s.current ∧ advance len s.current = none := by
  have h := run.local_valid worker
  rw [done] at h
  obtain ⟨_,tail⟩ := suffix_from_read _ run.replayed h.1
  cases tail with
  | done => exact ⟨rfl,h.2⟩
  | take write _ => rw [h.2] at write; contradiction

theorem Execution.move_at_return {s : State w} (moves : List α) (run : Execution moves.length start s)
    (ready : s.localAt worker = .held op) (fits : moves.length < 2 ^ w) :
    op.worker = worker ∧ ∃ move, moves[op.old.toNat]? = some move ∧
      reply moves op.old = some (move,op.stored) ∧ op.stored.toNat = op.old.toNat+1 := by
  have h := run.local_valid worker
  rw [ready] at h
  exact ⟨h.1,returned moves _ run.replayed h.2 fits⟩

theorem Execution.prior_end {first last : State w} (before : Execution len start first)
    (after : Execution len start last) (extendsWrites : first.writes <+: last.writes)
    (done : first.localAt worker = .drained opening seen) : advance len last.current = none := by
  obtain ⟨rest,eq⟩ := extendsWrites
  have localValue := before.local_valid worker
  rw [done] at localValue
  have member : seen ∈ words start last.writes := by
    rw [← eq]
    simpa [words] using (List.mem_append_left (rest.map Write.stored) localValue.1)
  obtain ⟨_,tail⟩ := suffix_from_read _ after.replayed member
  cases tail with
  | done => exact localValue.2
  | take next _ => rw [localValue.2] at next; contradiction

def forWorker (ops : List (Write w)) (worker : Nat) := ops.filter (fun op => op.worker == worker)

theorem partition (ops : List (Write w)) (workers : List Nat) (unique : workers.Nodup)
    (covers : ∀ op ∈ ops, op.worker ∈ workers) : ops.Perm (workers.flatMap (forWorker ops)) := by
  induction workers generalizing ops with
  | nil =>
    cases ops with
    | nil => exact .nil
    | cons head rest => simpa using covers head (by simp)
  | cons worker others ih =>
    obtain ⟨different,distinct⟩ := List.nodup_cons.mp unique
    let pending := ops.filter (fun op => !(op.worker == worker))
    have remains : ∀ op ∈ pending, op.worker ∈ others := by
      intro op mem
      obtain ⟨was,ne⟩ := List.mem_filter.mp mem
      exact (List.mem_cons.mp (covers _ was)).resolve_left (by simpa using ne)
    have tail := ih pending distinct remains
    have same : others.flatMap (forWorker pending) = others.flatMap (forWorker ops) := by
      apply congrArg List.flatten
      apply List.map_congr_left
      intro i member
      have ne : i ≠ worker := fun eq => different (eq ▸ member)
      unfold forWorker pending
      rw [List.filter_filter]
      apply List.filter_congr
      intro op _
      by_cases eq : op.worker = i <;> simp [eq,ne]
    rw [same] at tail
    exact (List.filter_append_perm (fun op : Write w => op.worker == worker) ops).symm.trans
      (List.Perm.append_left (forWorker ops worker) tail)

end NeuralReversi.MoveList.ConcurrentTrace
