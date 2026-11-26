import NeuralReversi.TranspositionTable.Packing

namespace NeuralReversi.TranspositionTable.LookupCut

def readOccupied (bigEndian : Bool) (first last : SeqLock.Word) (sample : SeqLock.Payload)
    (qp qo : SeqLock.Word) : Option EntryData :=
  (SeqLock.lookupDecision first last sample qp qo).bind fun raw =>
    let entry := Packing.unpack bigEndian raw
    if entry.bound = 0#8 then none else some entry

def ReadContract : Prop :=
  ∀ bigEndian first last (sample : SeqLock.Payload) qp qo entry,
    readOccupied bigEndian first last sample qp qo = some entry ↔
      first &&& 1 = 0 ∧ sample.player = qp ∧ sample.opponent = qo ∧ first = last ∧
      Packing.unpack bigEndian sample.metadata = entry ∧ entry.bound ≠ 0#8

theorem readContract_proven : ReadContract := by
  intro bigEndian first last sample qp qo entry
  unfold readOccupied
  rw [Option.bind_eq_some_iff]
  constructor
  · rintro ⟨raw, published, occupied⟩
    obtain ⟨even, player, opponent, stable, rfl⟩ :=
      (SeqLock.lookup_success_iff first last sample qp qo raw).mp published
    dsimp only at occupied
    split at occupied
    · contradiction
    · rename_i nonzero
      have decoded := Option.some.inj occupied
      exact ⟨even, player, opponent, stable, decoded, decoded ▸ nonzero⟩
  · rintro ⟨even, player, opponent, stable, decoded, nonzero⟩
    exact ⟨sample.metadata, (SeqLock.lookup_success_iff _ _ _ _ _ _).mpr
      ⟨even, player, opponent, stable, rfl⟩, by simp [decoded, nonzero]⟩

def ClusterContract : Prop :=
  ∀ (left right : Option EntryData) entry,
    left.or right = some entry ↔ left = some entry ∨ (left = none ∧ right = some entry)

theorem clusterContract_proven : ClusterContract := by
  intro left right entry
  cases left <;> simp

def PublishedReadContract : Prop :=
  ∀ {Event : Type} {memory : SeqLock.Memory Event}
    {epoch finalIndex : Nat} (_writers : SeqLock.Writers memory)
    (reader : SeqLock.Reader memory epoch finalIndex)
    {pe oe de : Nat} (pr : SeqLock.FieldRead reader .player pe)
    (op : SeqLock.FieldRead reader .opponent oe) (dr : SeqLock.FieldRead reader .metadata de)
    (qp qo : SeqLock.Word) (bigEndian : Bool) (entry : EntryData),
    finalIndex < 2 * epoch + SeqLock.modulus →
    readOccupied bigEndian (memory.readValue reader.first) (memory.readValue reader.last)
      ⟨memory.readValue pr.event, memory.readValue op.event, memory.readValue dr.event⟩ qp qo = some entry →
    memory.writeValue (.payload .player) epoch = qp ∧
    memory.writeValue (.payload .opponent) epoch = qo ∧
    Packing.unpack bigEndian (memory.writeValue (.payload .metadata) epoch) = entry ∧ entry.bound ≠ 0#8

theorem publishedReadContract_proven : PublishedReadContract := by
  intro Event memory epoch finalIndex writers reader pe oe de pr op dr qp qo bigEndian entry noWrap accepted
  obtain ⟨even, player, opponent, stable, decoded, nonzero⟩ :=
    (readContract_proven _ _ _ _ _ _ _).mp accepted
  have raw := (SeqLock.lookup_success_iff _ _ _ _ _ _).mpr ⟨even, player, opponent, stable, rfl⟩
  obtain ⟨hp, ho, metadata⟩ := SeqLock.lookup_consistent writers reader pr op dr qp qo _ noWrap raw
  exact ⟨hp, ho, metadata ▸ decoded, nonzero⟩

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

example : readOccupied false 0 0 ⟨0, 0, 0⟩ 0 0 = none := by decide
example : readOccupied true 0 0 ⟨0, 0, Packing.pack true (fixture 0 3)⟩ 0 0 =
    some (fixture 0 3) := by decide
example : readOccupied false 0 0 ⟨7, 8, Packing.pack false (fixture 0 0)⟩ 7 8 = none := by decide

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
