module
public import NeuralReversi.TranspositionTable.WriterMir.SourceSyntax

@[expose] public section

namespace NeuralReversi.TranspositionTable.WriterMir

open SeqLock (Word)

def saveStep (more found cas : Bool) (p o : Word) (entry : EntryData) (saved : SeqLock.Snapshot) (old : EntryData) : SourceStep :=
  let step0 : SourceStep := ⟨[.exact p o entry], false⟩
  let step1 : SourceStep := ⟨[.snapshot, .claim ⟨saved.sequence, p, o, entry⟩], false⟩
  let step2 : SourceStep := ⟨[.snapshot, .claim ⟨saved.sequence, p, o, entry⟩, .spin], true⟩
  let step3 : SourceStep := if cas then step1 else step2
  let step4 : SourceStep := ⟨[.snapshot, .claim ⟨saved.sequence, p, o, (Packing.withBestMoveOf entry old)⟩], false⟩
  let step5 : SourceStep := ⟨[.snapshot, .claim ⟨saved.sequence, p, o, (Packing.withBestMoveOf entry old)⟩, .spin], true⟩
  let step6 : SourceStep := if cas then step4 else step5
  let step7 : SourceStep := if entry.bestMove ≠ Packing.noMove then step3 else step6
  let step8 : SourceStep := ⟨[.snapshot], false⟩
  let step9 : SourceStep := if entry.generation ≠ old.generation then step7 else step8
  let step10 : SourceStep := if old.depth.toNat ≤ (entry.depth.setWidth 32 + 4).toNat then step7 else step9
  let step11 : SourceStep := if selectivityRank old.selectivity < entry.selectivity.toNat then step10 else step9
  let step12 : SourceStep := if old.depth.toNat ≤ (entry.depth.setWidth 32 + 2).toNat then step7 else step11
  let step13 : SourceStep := if sameBoardBits p o saved.player saved.opponent then step12 else step3
  let step14 : SourceStep := if old.bound ≠ 0 then step13 else step3
  let step15 : SourceStep := ⟨[.snapshot, .spin], true⟩
  let step16 : SourceStep := if found then step14 else step15
  let step17 : SourceStep := ⟨[], false⟩
  let step18 : SourceStep := if more then step16 else step17
  let step19 : SourceStep := if entry.bound = Packing.boundByte .exact then step0 else step18
  step19

def exactStep (more cas : Bool) (seq p o : Word) (entry : EntryData) : SourceStep :=
  let step0 : SourceStep := ⟨[.load .sequence .acquire, .spin], true⟩
  let step1 : SourceStep := ⟨[.load .sequence .acquire, .claim ⟨seq, p, o, entry⟩], false⟩
  let step2 : SourceStep := ⟨[.load .sequence .acquire, .claim ⟨seq, p, o, entry⟩, .spin], true⟩
  let step3 : SourceStep := if cas then step1 else step2
  let step4 : SourceStep := if seq &&& 1 ≠ 0 then step0 else step3
  let step5 : SourceStep := ⟨[], false⟩
  let step6 : SourceStep := if more then step4 else step5
  step6

def clearStep (endian more cas : Bool) (seq metadata last : Word) : SourceStep :=
  let step0 : SourceStep := ⟨[.load .sequence .acquire, .spin], true⟩
  let step1 : SourceStep := ⟨[.load .sequence .acquire, .load (.payload .metadata) .relaxed, .fence .acquire, .load .sequence .relaxed], false⟩
  let step2 : SourceStep := ⟨[.load .sequence .acquire, .load (.payload .metadata) .relaxed, .fence .acquire, .load .sequence .relaxed, .spin], true⟩
  let step3 : SourceStep := if last = seq then step1 else step2
  let step4 : SourceStep := ⟨[.load .sequence .acquire, .load (.payload .metadata) .relaxed, .claim ⟨seq, 0, 0, (Packing.unpack endian 0)⟩], false⟩
  let step5 : SourceStep := ⟨[.load .sequence .acquire, .load (.payload .metadata) .relaxed, .claim ⟨seq, 0, 0, (Packing.unpack endian 0)⟩, .spin], true⟩
  let step6 : SourceStep := if cas then step4 else step5
  let step7 : SourceStep := if metadata = 0 then step3 else step6
  let step8 : SourceStep := if seq &&& 1 ≠ 0 then step0 else step7
  let step9 : SourceStep := ⟨[], false⟩
  let step10 : SourceStep := if more then step8 else step9
  step10

def penaltyStep (found : Bool) (p o : Word) (entry : EntryData) (saved : SeqLock.Snapshot) (old : EntryData) : SourceStep :=
  let step0 : SourceStep := ⟨[], false⟩
  let step1 : SourceStep := ⟨[.snapshot], false⟩
  let step2 : SourceStep := ⟨[.snapshot, .claim ⟨saved.sequence, p, o, (Packing.penalized entry)⟩], false⟩
  let step3 : SourceStep := if sameBoardBits p o saved.player saved.opponent then step2 else step1
  let step4 : SourceStep := if old ≠ entry then step1 else step3
  let step5 : SourceStep := if found then step4 else step1
  let step6 : SourceStep := if entry.depth = 0 then step0 else step5
  step6

end NeuralReversi.TranspositionTable.WriterMir
