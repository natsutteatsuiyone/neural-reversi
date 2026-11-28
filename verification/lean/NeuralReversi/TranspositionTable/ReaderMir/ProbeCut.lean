module
public import NeuralReversi.PvsParallel.TraceCheck
public import NeuralReversi.TranspositionTable.ReaderMir.Publication

@[expose] public section

namespace NeuralReversi.PvsParallel.TraceCheck.Sample

open TranspositionTable

def ofSnapshot (worker scope slot : Nat) (result : SeqLock.Snapshot) : Sample :=
  ⟨worker, scope, slot, result.sequence.toNat, result.sequence.toNat,
    result.player.toNat, result.opponent.toNat, result.metadata.toNat⟩

theorem mir_probe_cut_sound {frame : FrameReturn} {fc : Flip.Dispatch.X86Config}
    (m : SeqLock.Memory Nat) (slot : WriterTrace.Slot m) (ev source : Nat → Nat)
    (worker scope slotIndex : Nat) (result : SeqLock.Snapshot)
    (steps : ReaderMir.Realizes m (ReaderMir.run false (m.readValue ∘ ev)).actions ev source)
    (hit : (ReaderMir.run false (m.readValue ∘ ev)).returned = some result)
    (occupied : Snapshot.empty false result.metadata = false)
    (noWrap : source 5 < source 0 + SeqLock.modulus)
    (initial : Snapshot.empty false (slot.payloadAt 0).metadata = false →
      ∃ publication : Sample, publication.payload = slot.payloadAt 0 ∧ publication.BoardBound fc)
    (certified : ∀ call, slot.history[source 0 - 1]? = some (.publish call) → m.hb call.finish (ev 0) →
      ∃ publication : Sample, publication.payload = call.payload ∧ publication.BoardBound fc)
    (matched : (ofSnapshot worker scope slotIndex result).cutMatches frame = true) (sane : frame.sane = true) :
    frame.BoardSound fc := by
  let words : SeqLock.Payload := ⟨result.player, result.opponent, result.metadata⟩
  have sourceBound : ∃ publication : Sample, words = publication.payload ∧ publication.BoardBound fc := by
    rcases ReaderMir.origin_of_occupied_return m slot false ev source result steps hit occupied noWrap with
      ⟨_, inherited⟩ | ⟨call, present, acquired, written⟩
    · have same := congrArg SeqLock.Payload.metadata inherited
      change result.metadata = _ at same
      obtain ⟨publication, stored, sound⟩ := initial (same ▸ occupied)
      exact ⟨publication, inherited.trans stored.symm, sound⟩
    · obtain ⟨publication, stored, sound⟩ := certified call present acquired
      exact ⟨publication, written.trans stored.symm, sound⟩
  obtain ⟨publication, stored, sound⟩ := sourceBound
  have equal : (ofSnapshot worker scope slotIndex result).payload = publication.payload := by
    simpa [ofSnapshot, payload, words] using stored
  apply cut_sound matched sane
  unfold BoardBound board at sound ⊢
  simpa only [equal] using sound

end NeuralReversi.PvsParallel.TraceCheck.Sample
