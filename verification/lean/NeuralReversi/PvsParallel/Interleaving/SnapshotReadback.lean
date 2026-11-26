import NeuralReversi.PvsParallel.Interleaving.ReadCandidates

namespace NeuralReversi.PvsParallel.Interleaving.SnapshotReadback

open TranspositionTable
open SeqLock (Location Word)
open FrameTableReads (Address)
open NativeTableWrites.Journal (Calls Record Even)

private def ranks (c : Calls) (endian : Bool) (rs : List Record) (a : Address)
    (initial : Location → Word) (loc : Location) (seen : Word) :=
  ReadCandidates.sources c endian rs a loc (initial loc) seen (fun _ => false)

/-- Epochs whose three payload words occur at one common modification rank. -/
def payloadEpochs (c : Calls) (endian : Bool) (rs : List Record) (a : Address)
    (initial : Location → Word) (snapshot : SeqLock.Snapshot) : List Nat :=
  let words := ranks c endian rs a initial
  let players := words (.payload .player) snapshot.player
  let opponents := words (.payload .opponent) snapshot.opponent
  (words (.payload .metadata) snapshot.metadata).filter fun k => players.contains k && opponents.contains k

/-- An occupied MIR return contains the common word of the two validated sequence
    loads. Both must read sequence rank 2*k when the three fields read rank k. -/
def epochs (c : Calls) (endian : Bool) (rs : List Record) (a : Address)
    (initial : Location → Word) (snapshot : SeqLock.Snapshot) : List Nat :=
  let sequence := ranks c endian rs a initial .sequence snapshot.sequence
  (payloadEpochs c endian rs a initial snapshot).filter fun k => sequence.contains (2*k)

variable {c : Calls} {endian : Bool} {rs : List Record} {a : Address}
  {result : SeqLock.Snapshot} {ev source : Nat → Nat}

private theorem hasRank (x : NativeTableWrites.Journal.Execution c endian rs)
    {loc : Location} {word : Word} {r i : Nat}
    (read : (x.memory a).rf ((x.memory a).write loc i) r ∧ (x.memory a).readValue r = word) :
    i ∈ ranks c endian rs a (fun l => (x.memory a).writeValue l 0) loc word := by
  rw [← read.2]
  exact ReadCandidates.complete x (fun _ => false) (by simp) read.1

theorem complete (x : NativeTableWrites.Journal.Execution c endian rs) (init : Even (x.memory a))
    (steps : ReaderMir.Realizes (x.memory a) (ReaderMir.run endian ((x.memory a).readValue ∘ ev)).actions ev source)
    (hit : (ReaderMir.run endian ((x.memory a).readValue ∘ ev)).returned = some result)
    (occupied : Snapshot.empty endian result.metadata = false)
    (noWrap : source 5 < source 0 + SeqLock.modulus) :
    source 0 / 2 ∈ epochs c endian rs a (fun l => (x.memory a).writeValue l 0) result := by
  obtain ⟨same,m,p,o,even⟩ := ReaderMir.occupied_indices _ (x.slot a init) endian ev source result steps hit occupied noWrap
  obtain ⟨first,metadata,player,opponent,last⟩ := ReaderMir.occupied_loads _ endian ev source result steps hit occupied
  have metadataRank := hasRank x metadata
  have playerRank := hasRank x player
  have opponentRank := hasRank x opponent
  have sequenceRank := hasRank x last
  rw [m] at metadataRank
  rw [p] at playerRank
  rw [o] at opponentRank
  rw [same] at sequenceRank
  exact List.mem_filter.mpr ⟨List.mem_filter.mpr ⟨metadataRank,by simpa using And.intro playerRank opponentRank⟩,
    by simpa only [List.contains_iff_mem,← even] using sequenceRank⟩

/-- When the five-word intersection has one epoch, it identifies every
    reads-from rank even if individual payload words occur in other writes. -/
theorem identified (x : NativeTableWrites.Journal.Execution c endian rs) (init : Even (x.memory a))
    (steps : ReaderMir.Realizes (x.memory a) (ReaderMir.run endian ((x.memory a).readValue ∘ ev)).actions ev source)
    (hit : (ReaderMir.run endian ((x.memory a).readValue ∘ ev)).returned = some result)
    (occupied : Snapshot.empty endian result.metadata = false)
    (noWrap : source 5 < source 0 + SeqLock.modulus)
    (one : epochs c endian rs a (fun l => (x.memory a).writeValue l 0) result = [k]) :
    source 0 = 2*k ∧ source 1 = k ∧ source 2 = k ∧ source 3 = k ∧ source 5 = 2*k := by
  have atEpoch : source 0 / 2 = k := by simpa only [one,List.mem_singleton] using complete x init steps hit occupied noWrap
  obtain ⟨last,metadata,player,opponent,first⟩ := ReaderMir.occupied_indices _ (x.slot a init) endian ev source result steps hit occupied noWrap
  exact ⟨by omega,metadata.trans atEpoch,player.trans atEpoch,opponent.trans atEpoch,by omega⟩

def lookupSnapshot (endian : Bool) (b : PvsBoardTree.Position) (sequence : Word) (e : EntryData) : SeqLock.Snapshot :=
  ⟨b.val.player,b.val.opponent,sequence,Packing.pack endian e⟩

private theorem lookupMember (x : NativeTableWrites.Journal.Execution c endian rs)
    (b : PvsBoardTree.Position) (h : Table.LookupSources (x.memory a) endian b ev source e) :
    source 0 / 2 ∈ epochs c endian rs a (fun l => (x.memory a).writeValue l 0)
      (lookupSnapshot endian b ((x.memory a).readValue (ev 0)) e) := by
  obtain ⟨last,p,o,m,first⟩ := h.indices
  have playerRank := hasRank x h.player
  have opponentRank := hasRank x h.opponent
  have metaRank := hasRank x h.metadata
  have sequenceRank := hasRank x h.last
  rw [p] at playerRank
  rw [o] at opponentRank
  rw [m] at metaRank
  rw [last] at sequenceRank
  exact List.mem_filter.mpr ⟨List.mem_filter.mpr ⟨metaRank,by simpa [lookupSnapshot] using And.intro playerRank opponentRank⟩,
    by simpa only [List.contains_iff_mem,lookupSnapshot,← first] using sequenceRank⟩

/-- The occupied fast lookup returns only metadata. Its returned EntryData,
    requested board and observed opening sequence reconstruct the sampled words. -/
theorem lookup_complete (x : NativeTableWrites.Journal.Execution c endian rs) (init : Even (x.memory a))
    (b : PvsBoardTree.Position)
    (steps : ReaderMir.Realizes (x.memory a)
      (ReaderMir.lookupRun endian b.val.player b.val.opponent ((x.memory a).readValue ∘ ev)).1 ev source)
    (hit : (ReaderMir.lookupRun endian b.val.player b.val.opponent ((x.memory a).readValue ∘ ev)).2 = some e)
    (noWrap : source 5 < source 0 + SeqLock.modulus) :
    source 0 / 2 ∈ epochs c endian rs a (fun l => (x.memory a).writeValue l 0)
      (lookupSnapshot endian b ((x.memory a).readValue (ev 0)) e) :=
  lookupMember x b (Table.seen_lookup_sources x a init b ev source steps hit noWrap).2

/-- Source indices are in fast-path instruction order: sequence, player, opponent,
    metadata, then sequence again after the acquire fence at PC 4. -/
theorem lookup_identified (x : NativeTableWrites.Journal.Execution c endian rs) (init : Even (x.memory a))
    (b : PvsBoardTree.Position)
    (steps : ReaderMir.Realizes (x.memory a)
      (ReaderMir.lookupRun endian b.val.player b.val.opponent ((x.memory a).readValue ∘ ev)).1 ev source)
    (hit : (ReaderMir.lookupRun endian b.val.player b.val.opponent ((x.memory a).readValue ∘ ev)).2 = some e)
    (noWrap : source 5 < source 0 + SeqLock.modulus)
    (one : epochs c endian rs a (fun l => (x.memory a).writeValue l 0)
      (lookupSnapshot endian b ((x.memory a).readValue (ev 0)) e) = [k]) :
    source 0 = 2*k ∧ source 1 = k ∧ source 2 = k ∧ source 3 = k ∧ source 5 = 2*k := by
  have h := (Table.seen_lookup_sources x a init b ev source steps hit noWrap).2
  have atEpoch : source 0 / 2 = k := by simpa only [one,List.mem_singleton] using lookupMember x b h
  obtain ⟨last,p,o,m,first⟩ := h.indices
  exact ⟨by omega,p.trans atEpoch,o.trans atEpoch,m.trans atEpoch,by omega⟩

namespace Regression

private def addr : Address := ⟨7,1⟩
private def oldBoard : PvsBoardTree.Position := ⟨⟨1,2⟩,rfl⟩
private def newBoard : PvsBoardTree.Position := ⟨⟨4,8⟩,rfl⟩
private def entryWord : Word := 16777216

private def calls (different : Bool) : Calls := fun k =>
  ⟨addr,if k = 0 then ⟨0,rfl⟩ else ⟨2,rfl⟩,
    .store none (if different && k == 1 then newBoard else oldBoard) (Packing.unpack false entryWord),
    true,fun pc => 1 + 7*k + pc⟩

private def writer (w k : Nat) : Joint.Tape Unit :=
  .engine (.beginWrite w k ()) :: List.replicate 6 (.engine (.atomic w k))

private def tape (snap : SeqLock.Snapshot) : Joint.Tape Unit :=
  writer 0 0 ++ writer 1 1 ++ [
    .issue 2 (.load addr .sequence .acquire) 0 snap.sequence,
    .issue 2 (.load addr (.payload .metadata) .relaxed) 0 snap.metadata,
    .issue 2 (.load addr (.payload .player) .relaxed) 0 snap.player,
    .issue 2 (.load addr (.payload .opponent) .relaxed) 0 snap.opponent,
    .issue 2 (.fence .acquire) 0 0,
    .issue 2 (.load addr .sequence .relaxed) 0 snap.sequence]

private def records (different : Bool) (snap : SeqLock.Snapshot) :=
  ((CallJournal.run ⟨false,false⟩ ⟨false,false,false⟩ (calls different) false
    ((tape snap).map Joint.Event.engineEvent)).map CallJournal.State.records).getD []

private def read (different : Bool) (snap : SeqLock.Snapshot) :=
  epochs (calls different) false (records different snap) addr (fun _ => 0) snap

private def snapshot (seq : Word) : SeqLock.Snapshot := ⟨1,2,seq,entryWord⟩

-- Each publisher is on a different worker from the reader. No producer HB
-- edge is supplied to the candidate calculation.
example : ((List.range 14).filter (ReadCandidates.sameWorkerBefore (tape (snapshot 4)) 14)) = [] := rfl
example : (ReaderMir.run false (fun j => Joint.word (tape (snapshot 4)) (14+j))).returned =
    some (snapshot 4) := rfl
example : payloadEpochs (calls false) false (records false (snapshot 4)) addr (fun _ => 0) (snapshot 4) =
    [1,2] := rfl
example : read false (snapshot 2) = [1] := rfl
example : read false (snapshot 4) = [2] := rfl
-- Both the old player and new opponent occur in the slot, but their ranks differ.
private def mixed : SeqLock.Snapshot := ⟨1,8,4,entryWord⟩
example : (ReaderMir.run false (fun j => Joint.word (tape mixed) (14+j))).returned = some mixed := rfl
example : read true mixed = [] := rfl

private def fastWords (s : SeqLock.Snapshot) : Nat → Word
  | 0 | 5 => s.sequence
  | 1 => s.player
  | 2 => s.opponent
  | 3 => s.metadata
  | _ => 0

-- Equal player, opponent and metadata words in two successive publications;
-- lookupRun reads them in key-first order and the observed sequence selects the second.
example : (ReaderMir.lookupRun false 1 2 (fastWords (snapshot 4))).2 = some (Packing.unpack false entryWord) ∧
    read false (lookupSnapshot false oldBoard 4 (Packing.unpack false entryWord)) = [2] := by decide

example : (ReaderMir.lookupRun false 1 8 (fastWords mixed)).2 = some (Packing.unpack false entryWord) ∧
    read true (lookupSnapshot false ⟨⟨1,8⟩,rfl⟩ 4 (Packing.unpack false entryWord)) = [] := by decide

end Regression

end NeuralReversi.PvsParallel.Interleaving.SnapshotReadback
