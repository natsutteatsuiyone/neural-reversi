module
public import NeuralReversi.PvsParallel.NativeTableWrites.Schedule
public import NeuralReversi.PvsParallel.SerialTable.Store

@[expose] public section

namespace NeuralReversi.PvsParallel.Interleaving.Table

open TranspositionTable
open NativeTableWrites.Journal
open PvsBoardTree (Position minimax)
open Cutoffs (key ExactEndgame)
open SeqLock (Payload)
open FrameTableReads (Address)

def EntrySafe (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : Position) (e : EntryData) : Prop :=
  (ExactEndgame e → BoundSound e (minimax mc fc b) ∧ -32767 < e.score.toInt ∧ e.score.toInt < 32767) ∧ TableReads.LegalHint b e

theorem EntrySafe.cut (safe : EntrySafe mc fc b e) (checked : Bool) (beta : Score) (depth : Depth)
    (narrow : beta.raw.toInt = alpha+1)
    (cut : canCut checked e beta depth Cutoffs.exactSelectivity true = .ok true) :
    Aspiration.SoundResult (minimax mc fc b) alpha beta.raw.toInt e.score.toInt := by
  obtain ⟨eligible,_,side⟩ := (canCut_true_iff ..).mp cut
  obtain ⟨bound,lo,hi⟩ := safe.1 (Cutoffs.exactEndgame_of_eligible eligible)
  refine ⟨lo,hi,?_,?_,by omega⟩
  · intro low
    rcases side with ⟨higher,_⟩ | ⟨_,upper⟩
    · omega
    · exact bound.2 upper
  · intro high
    rcases side with ⟨_,lower⟩ | ⟨below,_⟩
    · exact bound.1 lower
    · omega

def Safe (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (endian : Bool) (p : Payload) : Prop :=
  ∀ b, key b = (p.player,p.opponent) → (Packing.unpack endian p.metadata).bound ≠ 0 →
    EntrySafe mc fc b (Packing.unpack endian p.metadata)

private theorem key_inj {b z : Position} (eq : key b = key z) : b = z := by
  apply Subtype.ext
  cases b with | mk b hb =>
    cases z with | mk z hz =>
      cases b; cases z; simpa [key] using eq

private theorem admitted_finite (past : ExactTableHistory.Publications mc fc history) :
    ∀ b e, (b,e) ∈ history → ExactEndgame e → -32767 < e.score.toInt ∧ e.score.toInt < 32767 := by
  induction past with
  | initial => simp
  | append previous write ih =>
    intro b e mem full
    rcases List.mem_append.mp mem with old | new
    · exact ih _ _ old full
    · obtain ⟨rfl,rfl⟩ := Prod.mk.inj (List.mem_singleton.mp new)
      cases write with
      | other _ outside => exact (outside full).elim
      | penalized d old _ => exact ih _ d old full
      | searched pv lo hi score mv d gen tag priorKey prior output window result saved =>
        have same := (Cutoffs.storeContract_proven _ _ _ _ pv lo hi score mv d gen tag priorKey prior _ previous.sound window result saved).1
        rw [same]
        exact ⟨result.lower,result.upper⟩

theorem warm (t : SerialTable.Table l) (good : t.Good mc fc) (i : l.Index) :
    Safe mc fc l.bigEndian (SerialTable.payload (t.image.get i)) := by
  intro b eq occupied
  obtain ⟨z,hit,p,o⟩ := good.supported.resident i occupied
  have same : b = z := key_inj (eq.trans (Prod.ext p o))
  subst z
  exact ⟨fun full => ⟨good.publications.sound _ _ hit full,admitted_finite good.publications _ _ hit full⟩,good.hints _ _ hit⟩

/-- The first load can synchronize with an earlier publication on this same
    slot. Later replacements or a suspended writer need not be visible. -/
inductive Seen (x : NativeTableWrites.Journal.Execution c endian rs) (a : Address) (init : Even (x.memory a)) (consumer : Nat) : Payload → Prop where
  | initial : Seen x a init consumer ((x.slot a init).payloadAt 0)
  | stored (index k : Nat) (atIndex : (mods c rs a .sequence)[index]? = some ⟨k,.finish⟩)
      (prior : (x.memory a).hb ((c k).event 5) consumer) : Seen x a init consumer ((c k).call endian).payload

theorem Seen.later {later : Nat} (h : Seen x a init earlier p) (ordered : (x.memory a).hb earlier later) : Seen x a init later p := by
  cases h with
  | initial => exact .initial
  | stored index k hit hb => exact .stored index k hit ((x.memory a).hb_trans hb ordered)

theorem seen_snapshot (x : NativeTableWrites.Journal.Execution c endian rs) (a : Address) (init : Even (x.memory a)) (ev source : Nat → Nat)
    (steps : ReaderMir.Realizes (x.memory a) (ReaderMir.run endian ((x.memory a).readValue ∘ ev)).actions ev source)
    (returned : (ReaderMir.run endian ((x.memory a).readValue ∘ ev)).returned = some s)
    (occupied : Snapshot.empty endian s.metadata = false) (noWrap : source 5 < source 0 + SeqLock.modulus) :
    Seen x a init (ev 0) ⟨s.player,s.opponent,s.metadata⟩ := by
  rcases x.origin a init ev source steps returned occupied noWrap with ⟨_,same⟩ | ⟨k,_,_,_,atIndex,hb,same⟩
  · rw [same]; exact .initial
  · rw [same]; exact .stored _ k atIndex hb

/-- The fast reader loads player, opponent, then metadata. The requested key and returned entry
    determine their words. The two sequence loads identify their common epoch. -/
structure LookupSources (m : SeqLock.Memory Nat) (endian : Bool) (b : Position)
    (ev source : Nat → Nat) (e : EntryData) : Prop where
  indices : source 5 = source 0 ∧ source 1 = source 0 / 2 ∧ source 2 = source 0 / 2 ∧
    source 3 = source 0 / 2 ∧ source 0 = 2 * (source 0 / 2)
  first : m.rf (m.write .sequence (source 0)) (ev 0)
  last : m.rf (m.write .sequence (source 5)) (ev 5) ∧ m.readValue (ev 5) = m.readValue (ev 0)
  player : m.rf (m.write (.payload .player) (source 1)) (ev 1) ∧ m.readValue (ev 1) = b.val.player
  opponent : m.rf (m.write (.payload .opponent) (source 2)) (ev 2) ∧ m.readValue (ev 2) = b.val.opponent
  metadata : m.rf (m.write (.payload .metadata) (source 3)) (ev 3) ∧ m.readValue (ev 3) = Packing.pack endian e

/-- Fast lookup loads the keys before the metadata. All three field reads and
    both sequence loads below are the actual events of the successful path. -/
theorem seen_lookup_sources (x : NativeTableWrites.Journal.Execution c endian rs) (a : Address)
    (init : Even (x.memory a)) (b : Position) (ev source : Nat → Nat)
    (steps : ReaderMir.Realizes (x.memory a)
      (ReaderMir.lookupRun endian b.val.player b.val.opponent ((x.memory a).readValue ∘ ev)).1 ev source)
    (returned : (ReaderMir.lookupRun endian b.val.player b.val.opponent ((x.memory a).readValue ∘ ev)).2 = some e)
    (noWrap : source 5 < source 0 + SeqLock.modulus) :
    (Seen x a init (ev 0) (SlotJournal.packed endian b e) ∧ e.bound ≠ 0) ∧
    LookupSources (x.memory a) endian b ev source e := by
  let m := x.memory a
  let w := m.readValue ∘ ev
  change (ReaderMir.lookupRun endian b.val.player b.val.opponent w).2 = some e at returned
  unfold ReaderMir.lookupRun at returned
  split at returned <;> try contradiction
  rename_i even
  split at returned <;> try contradiction
  rename_i matchKey
  split at returned <;> try contradiction
  rename_i same
  dsimp only at returned
  split at returned <;> try contradiction
  rename_i occupied
  cases returned
  change ReaderMir.Realizes m (ReaderMir.lookupRun endian b.val.player b.val.opponent w).1 ev source at steps
  simp only [ReaderMir.lookupRun,ite_eq_right even,ite_eq_left matchKey,ite_eq_left same] at steps
  have hb (i j : Nat) (lt : i < j) (size : j < 6) : m.sb (ev i) (ev j) := by
    induction j with
    | zero => omega
    | succ j ih =>
      by_cases last : i = j
      · subst i; exact steps.adjacent j size
      · exact m.sb_trans (ih (by omega) (by omega)) (steps.adjacent j size)
  let rd : SeqLock.RawReader m (source 0) (source 5) :=
    ⟨ev 0,ev 4,ev 5,steps.acquire 0 _ rfl,steps.fence 4 rfl,hb 0 4 (by decide) (by decide),
      hb 4 5 (by decide) (by decide),steps.reads 0 _ _ rfl,steps.reads 5 _ _ rfl⟩
  let raw : Cutoffs.RawSample := ⟨m,source 0,source 5,x.slot a init,rd,source 1,source 2,source 3,
    ⟨ev 1,hb 0 1 (by decide) (by decide),hb 1 4 (by decide) (by decide),steps.reads 1 _ _ rfl⟩,
    ⟨ev 2,hb 0 2 (by decide) (by decide),hb 2 4 (by decide) (by decide),steps.reads 2 _ _ rfl⟩,
    ⟨ev 3,hb 0 3 (by decide) (by decide),hb 3 4 (by decide) (by decide),steps.reads 3 _ _ rfl⟩,noWrap,endian⟩
  have stable : raw.first &&& 1 = 0 := Classical.not_not.mp even
  let ws := raw.slot.writers
  have keyBits := (sameBoardBits_exact ..).mp matchKey
  have short : raw.finalIndex < 2 * (raw.firstIndex/2) + SeqLock.modulus := by
    rw [←raw.reader.evenIndex ws stable]; exact noWrap
  have first := raw.reader.evenIndex ws stable
  have last := raw.reader.validatedIndex ws noWrap same
  have field {f : SeqLock.Field} {k : Nat} (read : SeqLock.FieldRead raw.reader f k) : k = source 0 / 2 := by
    apply SeqLock.validated_field_same_epoch ws (raw.reader.asStable ws stable) (read.asStable ws stable) short
    change SeqLock.seqWord (2 * ws.baseEpoch + 2 * (source 0 / 2)) = SeqLock.seqWord (2 * ws.baseEpoch + source 5)
    rw [← first,last]
  refine ⟨⟨?_,occupied⟩,⟨⟨last.symm,field raw.player,field raw.opponent,field raw.metadata,first⟩,
    rd.first_rf,⟨rd.last_rf,same.symm⟩,⟨raw.player.reads_from,keyBits.1⟩,⟨raw.opponent.reads_from,keyBits.2⟩,
    ⟨raw.metadata.reads_from,(Packing.pack_unpack endian (w 3)).symm⟩⟩⟩
  have entry := SeqLock.lookup_consistent ws (raw.reader.asStable ws stable)
    (raw.player.asStable ws stable) (raw.opponent.asStable ws stable) (raw.metadata.asStable ws stable)
    b.val.player b.val.opponent (w 3) short
    ((SeqLock.lookup_success_iff ..).mpr ⟨stable,keyBits.1,keyBits.2,same,rfl⟩)
  have sameWords : SlotJournal.packed endian b (Packing.unpack endian (w 3)) = raw.slot.payloadAt (raw.firstIndex/2) := by
    simp only [SlotJournal.packed,Packing.pack_unpack,WriterTrace.Slot.payloadAt]
    rw [←entry.1,←entry.2.1,entry.2.2]
  rw [sameWords]
  by_cases zero : raw.firstIndex = 0
  · rw [zero]; exact .initial
  · obtain ⟨call,atIndex,payload,sync⟩ := raw.publishingCall stable (by omega)
    change ((mods c rs a .sequence).map (fun s => s.action c endian))[raw.firstIndex-1]? = some (.publish call) at atIndex
    obtain ⟨⟨k,pc⟩,site,eq⟩ := (show ∃ s, (mods c rs a .sequence)[raw.firstIndex-1]? = some s ∧
      s.action c endian = .publish call by simpa only [List.getElem?_map,Option.map_eq_some_iff] using atIndex)
    have loc := (mem_mods.mp (List.mem_of_getElem? site)).2.2.2
    have finish : pc = .finish := by
      cases pc <;> first | rfl | contradiction | simp [Pc.location] at loc
    subst pc
    cases eq
    rw [payload]
    exact .stored _ k site sync

theorem seen_lookup (x : NativeTableWrites.Journal.Execution c endian rs) (a : Address)
    (init : Even (x.memory a)) (b : Position) (ev source : Nat → Nat)
    (steps : ReaderMir.Realizes (x.memory a)
      (ReaderMir.lookupRun endian b.val.player b.val.opponent ((x.memory a).readValue ∘ ev)).1 ev source)
    (returned : (ReaderMir.lookupRun endian b.val.player b.val.opponent ((x.memory a).readValue ∘ ev)).2 = some e)
    (noWrap : source 5 < source 0 + SeqLock.modulus) :
    Seen x a init (ev 0) (SlotJournal.packed endian b e) ∧ e.bound ≠ 0 :=
  (seen_lookup_sources x a init b ev source steps returned noWrap).1

/-- Exact writers classify the frame's returned score at its entry window. A
    previous midgame or selective store needs only its generated legal move. -/
inductive Input (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : Position) : EntryData → Prop where
  | searched (pv : Bool) (lo hi score : Score) (depth : Depth) (move : Option Bitboard.Square) (gen : BitVec 8) (ttPv : Bool)
      (window : Aspiration.Window lo.raw.toInt hi.raw.toInt)
      (result : Aspiration.SoundResult (minimax mc fc b) lo.raw.toInt hi.raw.toInt score.raw.toInt)
      (legal : TableReads.LegalChoice b move) :
      Input mc fc b (Packing.newData score move (classify pv score lo hi) depth Cutoffs.exactSelectivity gen true ttPv)
  | weaker (e : EntryData) (other : ¬ExactEndgame e) (hint : TableReads.LegalHint b e) : Input mc fc b e

theorem Input.safe (h : Input mc fc b e) : EntrySafe mc fc b e := by
  cases h with
  | weaker entry other hint => exact ⟨fun yes => (other yes).elim,hint⟩
  | searched pv lo hi score depth move gen ttPv window result legal =>
    refine ⟨fun _ => ?_,TableReads.newHint legal⟩
    have packed := StoredBound.contract_proven pv score lo hi _ window result move depth Cutoffs.exactSelectivity gen true ttPv false
    simp only [Packing.unpack_pack] at packed
    exact ⟨packed.2.1,packed.1 ▸ result.lower,packed.1 ▸ result.upper⟩

/-- Observations are required only where an occupied old value affects the
    data written by save or penalize. A successful exact overwrite can skip the snapshot. -/
inductive Operands (endian : Bool) (supplied : Position → EntryData → Prop)
    (seen : Payload → Prop) : Origin → Prop where
  | clear : Operands endian supplied seen .clear
  | save (frame : Option Nat) (b : Position) (incoming : EntryData) (input : supplied b incoming)
      (obs : Save.Observation) (seq : SeqLock.Word) (entry : EntryData)
      (wrote : Save.attempt endian (key b) incoming obs = .written seq entry)
      (inherited : ∀ s, incoming.bound ≠ Packing.boundByte .exact → incoming.bestMove = Packing.noMove →
        Snapshot.read endian obs.first obs.last obs.sample = some s → (s.player,s.opponent) = key b →
        (Packing.unpack endian s.metadata).bound ≠ 0 → seen ⟨s.player,s.opponent,s.metadata⟩) :
      Operands endian supplied seen (.store frame b entry)
  | penalty (frame : Option Nat) (b : Position) (e : EntryData) (s : Option SeqLock.Snapshot) (cur : SeqLock.Word)
      (entry : EntryData) (wrote : Penalty.tryPenalize endian b.val.player b.val.opponent e s cur = some entry)
      (sampled : ∀ snap, s = some snap → (Packing.unpack endian snap.metadata).bound ≠ 0 →
        seen ⟨snap.player,snap.opponent,snap.metadata⟩) :
      Operands endian supplied seen (.store frame b entry)

abbrev Produced (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (endian : Bool) :=
  Operands endian (Input mc fc)

theorem Operands.mapInputs (h : Operands endian supplied seen o) (lift : ∀ b e, supplied b e → supplied' b e) :
    Operands endian supplied' seen o := by
  cases h with
  | clear => exact .clear
  | save f b incoming input obs seq d written previous => exact .save f b incoming (lift _ _ input) obs seq d written previous
  | penalty f b d s cur e w r => exact .penalty f b d s cur e w r

private theorem after_save (input : Input mc fc b incoming) (wrote : Save.attempt endian (key b) incoming obs = .written seq entry)
    (inheritSafe : ∀ s, incoming.bound ≠ Packing.boundByte .exact → incoming.bestMove = Packing.noMove →
      Snapshot.read endian obs.first obs.last obs.sample = some s → (s.player,s.opponent) = key b →
      (Packing.unpack endian s.metadata).bound ≠ 0 → TableReads.LegalHint b (Packing.unpack endian s.metadata)) :
    EntrySafe mc fc b entry := by
  by_cases exactTag : incoming.bound = Packing.boundByte .exact
  · rw [Save.exactAttemptContract_proven _ _ _ _ exactTag] at wrote
    split at wrote <;> cases wrote
    exact input.safe
  · obtain ⟨s,read,_,_,proposal⟩ := ((Save.otherAttemptContract_proven endian _ _ obs exactTag).2 _ _).mp wrote
    rw [(Save.contract_proven ..).1] at proposal
    split at proposal
    · contradiction
    · cases Option.some.inj proposal
      unfold Save.payload
      split
      next uses => exact ⟨input.safe.1,inheritSafe s exactTag uses.2.2.2 read uses.2.2.1.symm uses.2.1⟩
      next => exact input.safe

private theorem packed_safe (b : Position) (d : EntryData) (ok : d.bound ≠ 0 → EntrySafe mc fc b d) :
    Safe mc fc endian (SlotJournal.packed endian b d) := by
  intro z eq occupied
  have same : z = b := key_inj eq
  subst z
  simpa only [SlotJournal.packed,Packing.unpack_pack] using
    ok (by simpa only [SlotJournal.packed,Packing.unpack_pack] using occupied)

theorem Produced.safe {w : Attempt} (prod : Produced mc fc endian seen w.origin)
    (reads : ∀ p, seen p → Safe mc fc endian p) : Safe mc fc endian (w.call endian).payload := by
  cases hw : w.origin with
  | clear =>
    intro b eq occupied
    simp only [Attempt.call,Attempt.args,hw,Packing.unpack_pack] at occupied
    cases endian <;> contradiction
  | store frame b d =>
    rw [hw] at prod
    change Safe mc fc endian (w.call endian).payload
    simp only [Attempt.call,Attempt.args,hw]
    apply packed_safe
    intro occupied
    cases prod with
    | save _ _ incoming input obs seq entry wrote inherited =>
      apply after_save input wrote
      intro s normal absent read keys used
      exact (reads _ (inherited s normal absent read keys used) _ keys.symm used).2
    | penalty _ _ e s cur d wrote sampled =>
      obtain ⟨_,snap,atSnap,p,o,data,_,rfl⟩ := (Penalty.writeContract_proven ..).mp wrote
      have old := reads _ (sampled snap atSnap (by rw [data]; exact occupied)) b (Prod.ext p.symm o.symm)
        (by rw [data]; exact occupied)
      rwa [data] at old

/-- Induction follows the modification order of this slot's actual finish instructions.
    An inherited move may be several publications old. Its source still precedes the
    writer that inherited it, including if the sequence word eventually wraps. -/
theorem publication_invariant (x : NativeTableWrites.Journal.Execution c endian rs) (a : Address) (init : Even (x.memory a))
    (holds : Payload → Prop) (seed : holds ((x.slot a init).payloadAt 0))
    (release : ∀ k, Record.writer ⟨k,.finish⟩ ∈ rs → (c k).address = a → (c k).success = true →
      (∀ p, Seen x a init ((c k).event 5) p → holds p) → holds ((c k).call endian).payload) :
    ∀ (n k : Nat), (mods c rs a .sequence)[n]? = some ⟨k,.finish⟩ → holds ((c k).call endian).payload := by
  intro n
  induction n using Nat.strongRecOn with
  | ind n ih =>
    intro k hit
    have ⟨inLog,whereAt,done,_⟩ := mem_mods.mp (List.mem_of_getElem? hit)
    apply release k inLog whereAt done
    intro p fromSlot
    cases fromSlot with
    | initial => exact seed
    | stored j old oldHit before =>
      have current := (x.modified a .sequence n ⟨k,.finish⟩ hit).1
      have prior := (x.modified a .sequence j ⟨old,.finish⟩ oldHit).1
      have smaller := (x.memory a).write_write (loc := .sequence) (i := j+1) (j := n+1)
        (by rw [prior,current]; exact before)
      exact ih j (by omega) old oldHit

theorem publications (x : NativeTableWrites.Journal.Execution c endian rs) (a : Address) (init : Even (x.memory a))
    (seed : Safe mc fc endian ((x.slot a init).payloadAt 0))
    (operands : ∀ k, Record.writer ⟨k,.finish⟩ ∈ rs → (c k).address = a → (c k).success = true →
      Produced mc fc endian (Seen x a init ((c k).event 5)) (c k).origin) :
    ∀ (n k : Nat), (mods c rs a .sequence)[n]? = some ⟨k,.finish⟩ → Safe mc fc endian ((c k).call endian).payload := by
  apply publication_invariant x a init _ seed
  intro k recorded address success earlier
  exact (operands k recorded address success).safe earlier

theorem reads (x : NativeTableWrites.Journal.Execution c endian rs) (a : Address) (init : Even (x.memory a))
    (seed : Safe mc fc endian ((x.slot a init).payloadAt 0))
    (operands : ∀ k, Record.writer ⟨k,.finish⟩ ∈ rs → (c k).address = a → (c k).success = true →
      Produced mc fc endian (Seen x a init ((c k).event 5)) (c k).origin)
    (observed : Seen x a init consumer words) : Safe mc fc endian words := by
  cases observed with
  | initial => exact seed
  | stored n k hit _ => exact publications x a init seed operands n k hit

end NeuralReversi.PvsParallel.Interleaving.Table
