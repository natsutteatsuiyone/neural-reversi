import NeuralReversi.TranspositionTable.ReaderMir.Lookup
import NeuralReversi.TranspositionTable.ReaderMir.Program
import NeuralReversi.TranspositionTable.WriterMir.SourceProof

namespace NeuralReversi.TranspositionTable.ReaderMir.SequenceCycle

open SeqLock (Word seqWord)

private def image (p o seq metadata : Word) : SeqLock.Snapshot := ⟨p,o,seq,metadata⟩

/-- One uncontended save. Both the requested CAS and the final payload come
    from the MIR programs for the exact save and the successful writer. -/
def store (endian : Bool) (p o : Word) (d : EntryData) (n : Nat) (s : SeqLock.Snapshot) :
    Option SeqLock.Snapshot := do
  let obs : Save.Observation := ⟨s.sequence,s.sequence,⟨s.player,s.opponent,s.metadata⟩,s.sequence⟩
  let (_,c) ← WriterMir.publishRound (fun i => 6*n+i) endian s.sequence (WriterMir.saveRound endian p o d obs)
  return image c.payload.player c.payload.opponent (SeqLock.publish c.expected) c.payload.metadata

def stores (endian : Bool) (p o : Word) (d : EntryData) (start : SeqLock.Snapshot) : Nat → Option SeqLock.Snapshot
  | 0 => some start
  | n+1 => (stores endian p o d start n).bind (store endian p o d (n+1))

private theorem store_at (exactTag : d.bound = Packing.boundByte .exact) (endian p o n s epoch)
    (stamp : s.sequence = seqWord (2*epoch)) :
    store endian p o d n s = some (image p o (seqWord (2*(epoch+1))) (Packing.pack endian d)) := by
  have even : s.sequence &&& 1#64 = 0#64 := stamp ▸ (SeqLock.seqWord_even_iff (2*epoch)).mpr (by omega)
  simp [store,WriterMir.saveRound,WriterMir.forward_exact exactTag,WriterMir.publishRound,
    WriterMir.exact_request,even,WriterMir.published]
  rw [stamp,SeqLock.publish_seqWord,Nat.mul_add]

/-- Induction represents the entire finite execution, including every intervening
    successful claim and release. No list of length 2^63 needs to be allocated. -/
theorem stores_at (exactTag : d.bound = Packing.boundByte .exact) (endian p o s epoch)
    (stamp : s.sequence = seqWord (2*epoch)) (n : Nat) :
    stores endian p o d s (n+1) = some (image p o (seqWord (2*(epoch+n+1))) (Packing.pack endian d)) := by
  induction n with
  | zero => exact store_at exactTag endian p o _ s epoch stamp
  | succ n ih =>
    rw [stores,ih,Option.bind_some]
    exact store_at exactTag endian p o _ _ (epoch+n+1) rfl

private def register (s : SeqLock.Snapshot) : SeqLock.Location → Word
  | .sequence => s.sequence
  | .payload .player => s.player
  | .payload .opponent => s.opponent
  | .payload .metadata => s.metadata

/-- A sequentially consistent schedule: every load sees the current field.
    The entire writer interval occurs between two reader instructions. -/
private def samples (actions : List AtomicAction) (cut : Nat) (first last : SeqLock.Snapshot) (pc : Nat) : Word :=
  match actions[pc]? with
  | some (.load loc _) => register (if pc < cut then first else last) loc
  | _ => 0

private def lookupOps : List AtomicAction :=
  [.load .sequence .acquire,.load (.payload .player) .relaxed,.load (.payload .opponent) .relaxed,
   .load (.payload .metadata) .relaxed,.fence .acquire,.load .sequence .relaxed]

private def snapshotOps : List AtomicAction :=
  [.load .sequence .acquire,.load (.payload .metadata) .relaxed,.load (.payload .player) .relaxed,
   .load (.payload .opponent) .relaxed,.fence .acquire,.load .sequence .relaxed]

/-- Two independent readers of the same writer history. Lookup first captures
    the old board. The occupied snapshot used by probe first captures its metadata. -/
def readers (endian : Bool) (first last : SeqLock.Snapshot) :=
  let lookup := lookupRun endian first.player first.opponent (samples lookupOps 3 first last)
  let snap := run endian (samples snapshotOps 2 first last)
  (lookup, snap)

def scheduleFrom (endian : Bool) (p o q v : Word) (old new : EntryData) (start : SeqLock.Snapshot) (count : Nat) := do
  let first ← store endian p o old 0 start
  let last ← stores endian q v new first count
  return readers endian first last

def schedule (endian : Bool) (p o q v : Word) (old new : EntryData) (count : Nat) :=
  scheduleFrom endian p o q v old new (image 0 0 0 0) count

/-- The previous occupants and the number of writes before the pending read are arbitrary. -/
private theorem aliased_from (endian : Bool) (p o q v : Word) (old new : EntryData)
    (fromOld : old.bound = Packing.boundByte .exact) (fromNew : new.bound = Packing.boundByte .exact)
    (start : SeqLock.Snapshot) (epoch : Nat) (stamp : start.sequence = seqWord (2*epoch))
    (count n : Nat) (positive : count = n+1) (again : seqWord (2*(epoch+1+n+1)) = seqWord (2*(epoch+1))) :
    scheduleFrom endian p o q v old new start count =
    some ((lookupOps,some new),⟨snapshotOps,some (image q v (seqWord (2*(epoch+1))) (Packing.pack endian old))⟩) := by
  unfold scheduleFrom
  rw [store_at fromOld endian p o 0 start epoch stamp]
  simp only [bind,Option.bind_some]
  rw [positive,stores_at fromNew endian q v _ (epoch+1) rfl n,again]
  simp [readers,lookupRun,run,samples,lookupOps,snapshotOps,register,image,Packing.unpack_pack,
    sameBoardBits_exact,fromOld,fromNew,Packing.boundByte]
  exact (SeqLock.seqWord_even_iff _).mpr (by omega)

theorem full_cycle_from (endian : Bool) (p o q v : Word) (old new : EntryData)
    (fromOld : old.bound = Packing.boundByte .exact) (fromNew : new.bound = Packing.boundByte .exact)
    (start : SeqLock.Snapshot) (epoch : Nat) (stamp : start.sequence = seqWord (2*epoch)) :
    scheduleFrom endian p o q v old new start (2^63) =
    some ((lookupOps,some new),⟨snapshotOps,some (image q v (seqWord (2*(epoch+1))) (Packing.pack endian old))⟩) := by
  apply aliased_from endian p o q v old new fromOld fromNew start epoch stamp _ (2^63-1) (by decide)
  rw [Nat.add_assoc (epoch+1),Nat.mul_add]
  change BitVec.ofNat 64 (2*(epoch+1) + 18446744073709551616) = _
  rw [BitVec.ofNat_add]
  exact BitVec.add_zero _

theorem full_cycle (endian : Bool) (p o q v : Word) (old new : EntryData)
    (fromOld : old.bound = Packing.boundByte .exact) (fromNew : new.bound = Packing.boundByte .exact) :
    schedule endian p o q v old new (2^63) =
    some ((lookupOps,some new),⟨snapshotOps,some (image q v 2 (Packing.pack endian old))⟩) :=
  full_cycle_from endian p o q v old new fromOld fromNew _ 0 rfl

end NeuralReversi.TranspositionTable.ReaderMir.SequenceCycle
