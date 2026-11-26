import NeuralReversi.MoveList.Concurrent

namespace NeuralReversi.MoveList.ConcurrentTrace

open Concurrent (Claims advance)

/-- Successful compare-exchanges in modification order. The call identity survives the gap
    between the atomic instruction and delivery of the move to the worker. -/
structure Write (w : Nat) where
  worker : Nat
  opening : Nat
  old : BitVec w
  stored : BitVec w
  deriving DecidableEq

def replay (len : Nat) (current : BitVec w) : List (Write w) → Option (BitVec w)
  | [] => some current
  | op :: rest =>
    if current = op.old ∧ advance len op.old = some op.stored then replay len op.stored rest else none

theorem replay_claims (ops : List (Write w)) (ok : replay len start ops = some last) :
    Claims len start (ops.map Write.old) last := by
  induction ops generalizing start with
  | nil => cases ok; exact .done
  | cons op rest ih =>
    simp only [replay] at ok
    split at ok
    · rename_i same
      exact same.1 ▸ Claims.take same.2 (ih ok)
    · contradiction

theorem consecutive (ops : List (Write w)) (ok : replay len start ops = some last)
    (fits : len < 2 ^ w) :
    ops.map (fun op => op.old.toNat) = List.range' start.toNat ops.length ∧
      (ops.map (fun op => op.old.toNat)).Nodup := by
  have claims := replay_claims ops ok
  simpa [Function.comp_def] using And.intro (Concurrent.atomicClaimsContract_proven _ _ _ _ _ fits claims).1
    (claims.nodup fits)

def words (start : BitVec w) (ops : List (Write w)) := start :: ops.map Write.stored

theorem suffix_from_read (ops : List (Write w)) (ok : replay len start ops = some last)
    (observed : seen ∈ words start ops) : ∃ issued, Claims len seen issued last := by
  induction ops generalizing start with
  | nil =>
    simp [words] at observed
    cases ok
    exact observed ▸ ⟨[],.done⟩
  | cons op rest ih =>
    rcases List.mem_cons.mp observed with rfl | tail
    · exact ⟨_,replay_claims _ ok⟩
    · simp only [replay] at ok
      split at ok
      · exact ih ok tail
      · contradiction

/-- The first index is the load's reads-from source. Each subsequent index comes from a failed
    weak CAS, which may fail spuriously. None of those instructions modifies the counter. -/
def retries (len : Nat) (memory : List (BitVec w)) (prev : BitVec w) : List Nat → Option (BitVec w)
  | [] => some prev
  | index :: rest => do
    let _ ← advance len prev
    let observed ← memory[index]?
    retries len memory observed rest

def observed (len : Nat) (memory : List (BitVec w)) (load : Nat) (failures : List Nat) :
    Option (BitVec w) := do
  let first ← memory[load]?
  retries len memory first failures

private theorem retries_member (failures : List Nat) (first : prev ∈ memory)
    (ok : retries len memory prev failures = some seen) : seen ∈ memory := by
  induction failures generalizing prev with
  | nil => cases ok; exact first
  | cons index rest ih =>
    simp only [retries] at ok
    obtain ⟨_,_,after⟩ := Option.bind_eq_some_iff.mp ok
    obtain ⟨v,read,retried⟩ := Option.bind_eq_some_iff.mp after
    exact ih (List.mem_of_getElem? read) retried

theorem observed_member (ok : observed len memory load failures = some seen) : seen ∈ memory := by
  obtain ⟨first,read,retried⟩ := Option.bind_eq_some_iff.mp ok
  exact retries_member failures (List.mem_of_getElem? read) retried

/-- Source index of the last observation made by this invocation of try_update. -/
def lastSource (load : Nat) : List Nat → Nat
  | [] => load
  | i :: rest => lastSource i rest

private theorem retries_source (failures : List Nat) (first : memory[load]? = some prev)
    (ok : retries len memory prev failures = some seen) : memory[lastSource load failures]? = some seen := by
  induction failures generalizing load prev with
  | nil => cases ok; exact first
  | cons index rest ih =>
    obtain ⟨_,_,more⟩ := Option.bind_eq_some_iff.mp ok
    obtain ⟨v,read,retried⟩ := Option.bind_eq_some_iff.mp more
    exact ih read retried

theorem observed_source (ok : observed len memory load failures = some seen) :
    memory[lastSource load failures]? = some seen := by
  obtain ⟨first,read,retried⟩ := Option.bind_eq_some_iff.mp ok
  exact retries_source failures read retried

private theorem exhausted_rank (ops : List (Write w)) (ok : replay len start ops = some last)
    (read : (words start ops)[i]? = some seen) (stop : advance len seen = none) : i = ops.length := by
  induction ops generalizing start i with
  | nil => cases i <;> simp_all [words]
  | cons op rest ih =>
    simp only [replay] at ok
    split at ok
    · rename_i edge
      cases i with
      | zero =>
        have value : start = seen := Option.some.inj read
        have next := edge.2
        rw [←edge.1,value,stop] at next
        contradiction
      | succ k =>
        have inner : (words op.stored rest)[k]? = some seen := read
        exact congrArg Nat.succ (ih ok inner)
    · contradiction

/-- At exhaustion the source is the final modification, including when a failed
    weak CAS supplied the word and when the initial cursor was already exhausted. -/
theorem declined_source (ops : List (Write w)) (ok : replay len start ops = some last)
    (read : observed len (words start ops) load failures = some seen) (stop : advance len seen = none) :
    lastSource load failures = ops.length :=
  exhausted_rank ops ok (observed_source read) stop

theorem declined (ops : List (Write w)) (ok : replay len start ops = some last)
    (read : observed len (words start ops) load failures = some seen)
    (stop : advance len seen = none) : seen = last ∧ advance len last = none := by
  obtain ⟨_,tail⟩ := suffix_from_read ops ok (observed_member read)
  cases tail with
  | done => exact ⟨rfl,stop⟩
  | take wrote _ => rw [stop] at wrote; contradiction

theorem all_issued (ops : List (Write w)) (ok : replay len start ops = some last)
    (fits : len < 2 ^ w) (read : observed len (words start ops) load failures = some seen)
    (stop : advance len seen = none) :
    ops.map (fun op => op.old.toNat) = List.range' start.toNat (len - start.toNat) := by
  simpa [Function.comp_def] using (Concurrent.exhaustionContract_proven _ _ _ _ _ fits (replay_claims ops ok)
    (declined ops ok read stop).2).1

theorem issued (ops : List (Write w)) (ok : replay len start ops = some last)
    (member : op ∈ ops) : advance len op.old = some op.stored := by
  induction ops generalizing start with
  | nil => contradiction
  | cons head rest ih =>
    simp only [replay] at ok
    split at ok
    · rename_i same
      rcases List.mem_cons.mp member with rfl | tail
      · exact same.2
      · exact ih ok tail
    · contradiction

/-- The native return uses the old word twice: to index the immutable move list and to
    compute the one-based move count used by the split worker. -/
def reply (moves : List α) (old : BitVec w) : Option (α × BitVec w) :=
  moves[old.toNat]?.map (fun move => (move,old+1))

theorem returned (moves : List α) (ops : List (Write w))
    (ok : replay moves.length start ops = some last) (member : op ∈ ops)
    (fits : moves.length < 2 ^ w) :
    ∃ move, moves[op.old.toNat]? = some move ∧ reply moves op.old = some (move,op.stored) ∧
      op.stored.toNat = op.old.toNat+1 := by
  have write := issued ops ok member
  obtain ⟨inside,word⟩ := Concurrent.advance_some _ write
  exact ⟨moves[op.old.toNat],List.getElem?_eq_getElem inside,
    by simp [reply,inside,word],Concurrent.advance_nat _ fits write⟩

end NeuralReversi.MoveList.ConcurrentTrace
