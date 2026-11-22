import NeuralReversi.TranspositionTable.Packing

namespace NeuralReversi.TranspositionTable.LookupCut

theorem different_published_key_misses {Event : Type} {memory : SeqLock.Memory Event}
    {epoch finalIndex : Nat} (writers : SeqLock.Writers memory)
    (reader : SeqLock.Reader memory epoch finalIndex)
    {pe oe de : Nat} (pr : SeqLock.FieldRead reader .player pe)
    (op : SeqLock.FieldRead reader .opponent oe) (dr : SeqLock.FieldRead reader .metadata de)
    (qp qo : SeqLock.Word) (noWrap : finalIndex < 2 * epoch + SeqLock.modulus)
    (different : memory.writeValue (.payload .player) epoch ≠ qp ∨
      memory.writeValue (.payload .opponent) epoch ≠ qo) :
    SeqLock.lookupDecision (memory.readValue reader.first) (memory.readValue reader.last)
      ⟨memory.readValue pr.event, memory.readValue op.event, memory.readValue dr.event⟩ qp qo =
        none := by
  cases decision : SeqLock.lookupDecision (memory.readValue reader.first)
      (memory.readValue reader.last)
      ⟨memory.readValue pr.event, memory.readValue op.event, memory.readValue dr.event⟩ qp qo with
  | none => rfl
  | some result =>
    obtain ⟨hp, ho, _⟩ := SeqLock.lookup_consistent writers reader pr op dr qp qo result
      noWrap decision
    rcases different with different | different
    · exact False.elim (different hp)
    · exact False.elim (different ho)

theorem lookup_query_exact {Event : Type} {memory : SeqLock.Memory Event}
    {epoch finalIndex : Nat} (writers : SeqLock.Writers memory)
    (reader : SeqLock.Reader memory epoch finalIndex)
    {pe oe de : Nat} (pr : SeqLock.FieldRead reader .player pe)
    (op : SeqLock.FieldRead reader .opponent oe) (dr : SeqLock.FieldRead reader .metadata de)
    (qp qo result : SeqLock.Word) (bigEndian checked : Bool) (entry : EntryData)
    (encoded : memory.writeValue (.payload .metadata) epoch = Packing.pack bigEndian entry)
    (noWrap : finalIndex < 2 * epoch + SeqLock.modulus)
    (accepted : SeqLock.lookupDecision (memory.readValue reader.first)
      (memory.readValue reader.last)
      ⟨memory.readValue pr.event, memory.readValue op.event, memory.readValue dr.event⟩ qp qo =
        some result)
    (beta : Score) (depth : Depth) (selectivity : Selectivity) (requireEndgame : Bool) :
    memory.writeValue (.payload .player) epoch = qp ∧
    memory.writeValue (.payload .opponent) epoch = qo ∧
    Packing.unpack bigEndian result = entry ∧
    canCut checked (Packing.unpack bigEndian result) beta depth selectivity requireEndgame =
      (if Eligible entry depth selectivity requireEndgame then
        if checked && decide (entry.score.toInt = -32768) then .error scoreError
        else .ok (decide (CutWitness entry beta))
      else .ok false) := by
  obtain ⟨hp, ho, raw⟩ := SeqLock.lookup_consistent writers reader pr op dr qp qo result
    noWrap accepted
  have decoded := Packing.decode_published bigEndian result entry (raw.trans encoded)
  refine ⟨hp, ho, decoded, ?_⟩
  rw [decoded]
  exact canCut_contract checked entry beta depth selectivity requireEndgame

theorem lookup_cut_sound {Event : Type} {memory : SeqLock.Memory Event}
    {epoch finalIndex : Nat} (writers : SeqLock.Writers memory)
    (reader : SeqLock.Reader memory epoch finalIndex)
    {pe oe de : Nat} (pr : SeqLock.FieldRead reader .player pe)
    (op : SeqLock.FieldRead reader .opponent oe) (dr : SeqLock.FieldRead reader .metadata de)
    (qp qo result : SeqLock.Word) (bigEndian checked : Bool) (entry : EntryData)
    (produced : Packing.Produced entry)
    (encoded : memory.writeValue (.payload .metadata) epoch = Packing.pack bigEndian entry)
    (noWrap : finalIndex < 2 * epoch + SeqLock.modulus)
    (accepted : SeqLock.lookupDecision (memory.readValue reader.first)
      (memory.readValue reader.last)
      ⟨memory.readValue pr.event, memory.readValue op.event, memory.readValue dr.event⟩ qp qo =
        some result)
    (beta : Score) (depth : Depth) (selectivity : Selectivity) (requireEndgame : Bool)
    (queryValue : SeqLock.Word → SeqLock.Word → Int)
    (sound : BoundSound entry
      (queryValue (memory.writeValue (.payload .player) epoch)
        (memory.writeValue (.payload .opponent) epoch)))
    (cut : canCut checked (Packing.unpack bigEndian result) beta depth selectivity
      requireEndgame = .ok true) :
    memory.writeValue (.payload .player) epoch = qp ∧
    memory.writeValue (.payload .opponent) epoch = qo ∧
    Packing.unpack bigEndian result = entry ∧
    Packing.EnumSafe (Packing.unpack bigEndian result) ∧
    (beta.raw.toInt ≤ (Packing.unpack bigEndian result).score.toInt ↔
      beta.raw.toInt ≤ queryValue qp qo) ∧
    ((Packing.unpack bigEndian result).score.toInt < beta.raw.toInt ↔
      queryValue qp qo < beta.raw.toInt) := by
  obtain ⟨hp, ho, decoded, safe⟩ := Packing.lookup_decodes_produced writers reader pr op dr
    qp qo result bigEndian entry produced encoded noWrap accepted
  have soundForQuery : BoundSound entry (queryValue qp qo) := by
    simpa only [hp, ho] using sound
  have entryCut : canCut checked entry beta depth selectivity requireEndgame = .ok true := by
    simpa only [decoded] using cut
  have sides := canCut_sound checked entry beta depth selectivity requireEndgame
    (queryValue qp qo) soundForQuery entryCut
  refine ⟨hp, ho, decoded, safe, ?_⟩
  simpa only [decoded] using sides

private def fixture (score : Int) (bound : Nat) : EntryData :=
  ⟨BitVec.ofInt 16 score, Packing.noMove, BitVec.ofNat 8 bound, 12, 5, 0, 1⟩

example : BoundSound (fixture 0 1) 256 ∧
    canCut true (fixture 0 1) ⟨0⟩ 12 ⟨5, by decide⟩ true = .ok true ∧
    (fixture 0 1).score.toInt ≠ 256 := by
  constructor
  · simp [BoundSound, fixture]
  · constructor
    · rfl
    · decide

example : BoundSound (fixture 0 2) (-256) ∧
    canCut true (fixture 0 2) ⟨1⟩ 12 ⟨5, by decide⟩ true = .ok true ∧
    (fixture 0 2).score.toInt ≠ -256 := by
  constructor
  · simp [BoundSound, fixture]
  · constructor
    · rfl
    · decide

example : SeqLock.lookupDecision 0 0 ⟨7, 8, Packing.pack false (fixture 0 3)⟩ 7 9 = none := by
  decide

example : canCut false (fixture 0 1) ⟨0⟩ 12 ⟨5, by decide⟩ true = .ok true ∧
    ¬ BoundSound (fixture 0 1) (-1) := by
  constructor
  · rfl
  · simp [BoundSound, fixture]

end NeuralReversi.TranspositionTable.LookupCut
