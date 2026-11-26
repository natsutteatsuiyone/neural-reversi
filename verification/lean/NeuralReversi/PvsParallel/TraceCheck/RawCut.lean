import NeuralReversi.PvsParallel.TraceCheck

namespace NeuralReversi.PvsParallel.TraceCheck

open TranspositionTable

theorem Sample.raw_probe_cut_sound {sample publication : Sample} {frame : FrameReturn}
    {fc : Flip.Dispatch.X86Config} (raw : Cutoffs.RawSample) {result : SeqLock.Snapshot}
    (accepted : Snapshot.read raw.bigEndian raw.first raw.last raw.payload = some result)
    (observed : sample.payload = raw.payload)
    (saved : publication.payload = raw.slot.payloadAt (raw.firstIndex / 2))
    (matched : sample.cutMatches frame = true) (sane : frame.sane = true)
    (sound : publication.BoardBound fc) : frame.BoardSound fc := by
  have stable := (SeqLock.snapshot_success_iff _ _ _ _ _).mp accepted
  let writers := raw.slot.writers
  let reader := raw.reader.asStable writers stable.1
  have index := raw.reader.evenIndex writers stable.1
  have noWrap : raw.finalIndex < 2 * (raw.firstIndex / 2) + SeqLock.modulus := by
    rw [← index]
    exact raw.noWrap
  have sequence := stable.2.1
  change raw.memory.readValue reader.first = raw.memory.readValue reader.last at sequence
  rw [SeqLock.reader_first_value writers reader, SeqLock.reader_last_value writers reader] at sequence
  have player := SeqLock.validated_field_value writers reader
    (raw.player.asStable writers stable.1) noWrap sequence
  have opponent := SeqLock.validated_field_value writers reader
    (raw.opponent.asStable writers stable.1) noWrap sequence
  have metadata := SeqLock.validated_field_value writers reader
    (raw.metadata.asStable writers stable.1) noWrap sequence
  change raw.memory.readValue raw.player.event = _ at player
  change raw.memory.readValue raw.opponent.event = _ at opponent
  change raw.memory.readValue raw.metadata.event = _ at metadata
  have payload : raw.payload = raw.slot.payloadAt (raw.firstIndex / 2) := by
    change (⟨raw.memory.readValue raw.player.event, raw.memory.readValue raw.opponent.event,
      raw.memory.readValue raw.metadata.event⟩ : SeqLock.Payload) = _
    rw [player, opponent, metadata]
    rfl
  have equal := observed.trans (payload.trans saved.symm)
  apply Sample.cut_sound matched sane
  unfold BoardBound board at sound ⊢
  simpa only [equal] using sound

theorem Sample.raw_probe_cut_from_publisher {sample : Sample} {frame : FrameReturn}
    {fc : Flip.Dispatch.X86Config} (raw : Cutoffs.RawSample) {result : SeqLock.Snapshot}
    (accepted : Snapshot.read raw.bigEndian raw.first raw.last raw.payload = some result)
    (observed : sample.payload = raw.payload) (positive : 0 < raw.firstIndex)
    (certified : ∀ call, raw.slot.history[raw.firstIndex - 1]? = some (.publish call) →
      ∃ publication : Sample, publication.payload = call.payload ∧ publication.BoardBound fc)
    (matched : sample.cutMatches frame = true) (sane : frame.sane = true) : frame.BoardSound fc := by
  have even := ((SeqLock.snapshot_success_iff _ _ _ _ _).mp accepted).1
  obtain ⟨call, actual, payload, _⟩ := raw.publishingCall even positive
  obtain ⟨publication, saved, sound⟩ := certified call actual
  exact raw_probe_cut_sound raw accepted observed (saved.trans payload.symm) matched sane sound

end NeuralReversi.PvsParallel.TraceCheck
