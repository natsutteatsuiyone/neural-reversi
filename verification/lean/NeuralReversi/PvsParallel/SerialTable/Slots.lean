import NeuralReversi.PvsParallel.SerialTable.Image
import NeuralReversi.TranspositionTable.ReaderMir.Program

namespace NeuralReversi.PvsParallel.SerialTable

open TranspositionTable
open PvsBoardTree (Position)

def snapshotWords (endian : Bool) (s : Clear.SlotState) : Nat → SeqLock.Word
  | 0 | 5 => s.seq
  | 1 => s.data
  | 2 => s.player
  | 3 => if (Packing.unpack endian s.data).bound = 0 then s.seq else s.opponent
  | _ => 0

def lookupWords (s : Clear.SlotState) : Nat → SeqLock.Word
  | 0 | 5 => s.seq
  | 1 => s.player
  | 2 => s.opponent
  | 3 => s.data
  | _ => 0

theorem snapshot_mir (endian : Bool) (s : Clear.SlotState) :
    (ReaderMir.run endian (snapshotWords endian s)).returned = Snapshot.read endian s.seq s.seq (payload s) := by
  by_cases even : s.seq &&& 1#64 = 0#64 <;> by_cases empty : (Packing.unpack endian s.data).bound = 0#8 <;>
    simp [ReaderMir.run,snapshotWords,Snapshot.read,Snapshot.empty,SeqLock.snapshotDecision,payload,even,empty]

theorem lookup_mir (endian : Bool) (s : Clear.SlotState) (p o : SeqLock.Word) :
    (ReaderMir.lookupRun endian p o (lookupWords s)).2 =
      LookupCut.readOccupied endian s.seq s.seq (payload s) p o := by
  by_cases even : s.seq &&& 1#64 = 0#64 <;> by_cases same : s.player = p ∧ s.opponent = o <;>
    by_cases empty : (Packing.unpack endian s.data).bound = 0#8 <;>
    simp [ReaderMir.lookupRun,lookupWords,LookupCut.readOccupied,SeqLock.lookupDecision,payload,
      even,sameBoardBits_exact,same,empty]

private def slotAt (im : Image l) (i : Nat) := im.slots[i]?.getD Clear.emptySlot

private theorem slot_at (im : Image l) (i : l.Index) : slotAt im i.val = im.get i := by
  have h : i.val < im.slots.size := im.size.symm ▸ i.isLt
  simp [slotAt,Image.get,h]

/-- Both the retained slot and the ETC reads use the absolute cluster selected by the native board hash. -/
theorem Image.lookup_cluster (im : Image l) (b : Position) :
    ReaderMir.lookupCluster l.clusters (Board.Hash.hash l.bigEndian b.val) b.val.player b.val.opponent
      (fun i p o => (ReaderMir.lookupRun l.bigEndian p o (lookupWords (slotAt im i))).2) = im.lookup b := by
  unfold ReaderMir.lookupCluster
  have atSlot (i : Fin 2) : (ReaderMir.lookupRun l.bigEndian b.val.player b.val.opponent (lookupWords
      (slotAt im (Clear.getClusterIdx (Board.Hash.hash l.bigEndian b.val) l.clusters + i.val)))).2 =
      im.lookupSlot b i := by
    rw [show Clear.getClusterIdx (Board.Hash.hash l.bigEndian b.val) l.clusters + i.val = (l.index b i).val from rfl,slot_at,lookup_mir]
    rfl
  have front := atSlot 0
  have back := atSlot 1
  simp only [Fin.val_zero,Nat.add_zero,Fin.val_one] at front back
  simp only [front,back]
  cases read : im.lookupSlot b 0 <;> simp [Image.lookup,read]

private def observed (im : Image l) (b : Position) (i : Fin 2) :=
  ProbeRetry.classify (Cutoffs.key b) (sample l.bigEndian (im.cell b i))

private theorem stable {l : Layout} (s : Clear.SlotState) (even : s.seq &&& 1 = 0) (key : ProbeRetry.Key) :
    ProbeRetry.classify key (sample l.bigEndian s) ≠ .busy := by
  change s.seq &&& 1#64 = 0#64 at even
  by_cases empty : (Packing.unpack l.bigEndian s.data).bound = 0#8 <;>
    simp [sample,Snapshot.read,Snapshot.empty,SeqLock.snapshotDecision,payload,ProbeRetry.classify,even,empty]
  split <;> simp

private theorem actualPass (gen : BitVec 8) (a b : ProbeRetry.Observation) (xa : a ≠ .busy) (xb : b ≠ .busy) :
    let second := if ReaderMir.readsSecond a then b else .busy
    let generation := if ReaderMir.readsGeneration a second then gen else 0
    ∃ v, (ReaderMir.probePass gen a b).exit = .done v ∧
      ReaderMir.probePass generation a second = ReaderMir.probePass gen a b := by
  cases a <;> cases b <;> first | contradiction | exact ⟨_,rfl,rfl⟩

private theorem round_observed (im : Image l) (b : Position) :
    let a := observed im b 0
    let back := if ReaderMir.readsSecond a then observed im b 1 else .busy
    ProbeRetry.observe (Cutoffs.key b) (im.round b) =
      ⟨a,back,if ReaderMir.readsGeneration a back then im.generation else 0⟩ := by
  simp only [observed,Image.round,ProbeRetry.observe]
  by_cases need : ReaderMir.readsSecond (ProbeRetry.classify (Cutoffs.key b) (sample l.bigEndian (im.cell b 0))) = true
  · simp [need]
  · simp [need]
    exact ⟨rfl,rfl⟩

/-- A serial stable cluster answers in its first round. Its result chooses the exact retained offset
    from the extracted probe program, including both occupied mismatches and the generation tie. -/
theorem Image.stable_probe (im : Image l) (supported : im.Supported h) (b : Position) :
    im.attempts b = [im.round b] ∧
      (ReaderMir.probePass im.generation
        (ProbeRetry.classify (Cutoffs.key b) (sample l.bigEndian (im.cell b 0)))
        (ProbeRetry.classify (Cutoffs.key b) (sample l.bigEndian (im.cell b 1)))).exit = .done (im.probe b) := by
  let a := observed im b 0
  let z := observed im b 1
  obtain ⟨v,result,total⟩ := actualPass im.generation a z
    (stable (im.cell b 0) (supported.even _) _) (stable (im.cell b 1) (supported.even _) _)
  let round := ProbeRetry.observe (Cutoffs.key b) (im.round b)
  have decided : (ReaderMir.probePass round.generation round.first round.second).exit = .done v := by
    dsimp only [round]
    rw [round_observed]
    exact (congrArg ReaderMir.Pass.exit total).trans result
  have once : im.attempts b = [im.round b] := by
    unfold Image.attempts
    dsimp only
    rw [decided]
  refine ⟨once,?_⟩
  have answer : im.probe b = v := by
    unfold Image.probe
    rw [once]
    change ProbeRetry.scan 0 [round] = v
    rw [ReaderMir.pass_scan,decided]
  rw [answer]
  exact result

end NeuralReversi.PvsParallel.SerialTable
