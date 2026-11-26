import NeuralReversi.EndgameCutoffs
import NeuralReversi.PvsEnhancedCutoff
import NeuralReversi.TranspositionTable.ProbeRetry
import NeuralReversi.TranspositionTable.Save
import NeuralReversi.TranspositionTable.StoredBound
import NeuralReversi.TranspositionTable.Penalty
import NeuralReversi.TranspositionTable.WriterTrace

namespace NeuralReversi.PvsParallel.Cutoffs

open Aspiration (SoundResult Window)
open Bitboard.Moves (X86Config)
open PvsBoardTree (Position GeneratedSquare legalSquares play minimax)
open TranspositionTable

abbrev FlipConfig := Flip.Dispatch.X86Config
abbrev History := List (Position × EntryData)
def exactSelectivity : Selectivity := ⟨5, by decide⟩

def key (b : Position) : ProbeRetry.Key := (b.val.player, b.val.opponent)

def ExactEndgame (entry : EntryData) : Prop :=
  entry.flags.getLsbD 0 = true ∧ 5 ≤ entry.selectivity.toNat

theorem exactEndgame_of_eligible {entry : EntryData} {depth : Depth}
    (eligible : Eligible entry depth exactSelectivity true) : ExactEndgame entry := by
  refine ⟨eligible.1.resolve_left (by decide), ?_⟩
  exact Nat.le_trans eligible.2.2 (Nat.min_le_left _ _)

def SoundHistory (mc : X86Config) (fc : FlipConfig) (history : History) : Prop :=
  ∀ b entry, (b, entry) ∈ history → ExactEndgame entry → BoundSound entry (minimax mc fc b)

theorem SoundHistory.append_other {mc : X86Config} {fc : FlipConfig} {history : History}
    (sound : SoundHistory mc fc history) (b : Position) (entry : EntryData)
    (other : ¬ExactEndgame entry) : SoundHistory mc fc (history ++ [(b, entry)]) := by
  intro board data present eligible
  rcases List.mem_append.mp present with earlier | new
  · exact sound board data earlier eligible
  · obtain ⟨rfl, rfl⟩ := Prod.mk.inj (List.mem_singleton.mp new)
    exact (other eligible).elim

theorem SoundHistory.append_penalty {mc : X86Config} {fc : FlipConfig} {history : History}
    (sound : SoundHistory mc fc history) (b : Position) (entry : EntryData)
    (previous : (b, entry) ∈ history) :
    SoundHistory mc fc (history ++ [(b, Packing.penalized entry)]) := by
  intro board data present eligible
  rcases List.mem_append.mp present with earlier | new
  · exact sound board data earlier eligible
  · obtain ⟨rfl, rfl⟩ := Prod.mk.inj (List.mem_singleton.mp new)
    exact (Penalty.penalized_boundSound entry _).mpr (sound _ entry previous eligible)

def StoreContract : Prop :=
  ∀ (mc : X86Config) (fc : FlipConfig) (history : History) (b : Position)
    (pv : Bool) (alpha beta score : Score) (move : Option (Fin 64)) (depth : Depth)
    (generation : BitVec 8) (ttPv : Bool) (previousKey : Save.Key) (previous written : EntryData),
    SoundHistory mc fc history → Window alpha.raw.toInt beta.raw.toInt →
    SoundResult (minimax mc fc b) alpha.raw.toInt beta.raw.toInt score.raw.toInt →
    let incoming := Packing.newData score move (classify pv score alpha beta) depth
      exactSelectivity generation true ttPv
    Save.proposal (key b) previousKey incoming previous = some written →
    written.score.toInt = score.raw.toInt ∧ SoundHistory mc fc (history ++ [(b, written)])

theorem storeContract_proven : StoreContract := by
  intro mc fc history b pv alpha beta score move depth generation ttPv previousKey previous written
    historySound window result incoming saved
  have packed := StoredBound.contract_proven pv score alpha beta (minimax mc fc b) window result
    move depth exactSelectivity generation true ttPv false
  simp only [Packing.unpack_pack] at packed
  have data := (Save.contract_proven (key b) previousKey incoming previous).1
  rw [data] at saved
  split at saved
  · contradiction
  · have same := Option.some.inj saved
    subst written
    have fields : (Save.payload (key b) previousKey incoming previous).score = incoming.score ∧
        (Save.payload (key b) previousKey incoming previous).bound = incoming.bound := by
      unfold Save.payload
      split <;> exact ⟨rfl, rfl⟩
    refine ⟨fields.1 ▸ packed.1, ?_⟩
    intro board entry member eligible
    rcases List.mem_append.mp member with earlier | new
    · exact historySound board entry earlier eligible
    · obtain ⟨rfl, rfl⟩ := Prod.mk.inj (List.mem_singleton.mp new)
      simpa only [BoundSound, fields.1, fields.2] using packed.2.1

theorem cutBound (entry : EntryData) (alpha beta : Score) (depth : Depth) (checked : Bool)
    (truth : Int) (narrow : beta.raw.toInt = alpha.raw.toInt + 1) (sound : BoundSound entry truth)
    (cut : canCut checked entry beta depth exactSelectivity true = .ok true) :
    PvsMoveLoop.Bound truth alpha.raw.toInt beta.raw.toInt entry.score.toInt := by
  rcases ((TranspositionTable.canCut_true_iff checked entry beta depth exactSelectivity true).mp cut).2.2 with
    ⟨high, lower⟩ | ⟨low, upper⟩
  · exact ⟨fun impossible => False.elim (by omega), fun _ => sound.1 lower⟩
  · exact ⟨fun _ => sound.2 upper, fun impossible => False.elim (by omega)⟩

def ProbeMatches (history : History) (b : Position) (attempts : List ProbeRetry.RawAttempt)
    (slot : Fin 2) (entry : EntryData) : Prop :=
  ∀ a ∈ attempts.take 3, a.readSlot slot = some (key b, entry) →
    entry.bound ≠ 0 → (b, entry) ∈ history

def SlotPublications (history : History) {memory : SeqLock.Memory Nat}
    (slot : WriterTrace.Slot memory) (bigEndian : Bool) : Prop :=
  ∀ epoch, 2 * epoch ≤ slot.history.length →
    let entry := Packing.unpack bigEndian (memory.writeValue (.payload .metadata) epoch)
    entry.bound ≠ 0 →
    ∃ board : Position, (board, entry) ∈ history ∧
      memory.writeValue (.payload .player) epoch = board.val.player ∧
      memory.writeValue (.payload .opponent) epoch = board.val.opponent

structure RawSample where
  memory : SeqLock.Memory Nat
  firstIndex : Nat
  finalIndex : Nat
  slot : WriterTrace.Slot memory
  reader : SeqLock.RawReader memory firstIndex finalIndex
  pe : Nat
  oe : Nat
  de : Nat
  player : SeqLock.FieldRead reader .player pe
  opponent : SeqLock.FieldRead reader .opponent oe
  metadata : SeqLock.FieldRead reader .metadata de
  noWrap : finalIndex < firstIndex + SeqLock.modulus
  bigEndian : Bool

def RawSample.publication (sample : RawSample) (history : History) : Prop :=
  let payload := sample.slot.payloadAt (sample.firstIndex / 2)
  let entry := Packing.unpack sample.bigEndian payload.metadata
  entry.bound ≠ 0 →
  ∃ board : Position, (board, entry) ∈ history ∧
    payload.player = board.val.player ∧ payload.opponent = board.val.opponent

structure ReadSample (history : History) extends RawSample where
  published : toRawSample.publication history

def ReadSample.weaken (sample : ReadSample history) (included : ∀ pair ∈ history, pair ∈ later) :
    ReadSample later :=
  ⟨sample.toRawSample, fun occupied => by
    obtain ⟨board, present, player, opponent⟩ := sample.published occupied
    exact ⟨board, included _ present, player, opponent⟩⟩

def RawSample.first (sample : RawSample) : SeqLock.Word :=
  sample.memory.readValue sample.reader.first

def RawSample.last (sample : RawSample) : SeqLock.Word :=
  sample.memory.readValue sample.reader.last

def RawSample.payload (sample : RawSample) : SeqLock.Payload :=
  ⟨sample.memory.readValue sample.player.event, sample.memory.readValue sample.opponent.event,
    sample.memory.readValue sample.metadata.event⟩

theorem RawSample.publishingCall (raw : RawSample) (even : raw.first &&& 1 = 0)
    (positive : 0 < raw.firstIndex) :
    ∃ call, raw.slot.history[raw.firstIndex - 1]? = some (.publish call) ∧
      raw.slot.payloadAt (raw.firstIndex / 2) = call.payload ∧
      raw.memory.hb call.finish raw.reader.first := by
  let writers := raw.slot.writers
  let reader := raw.reader.asStable writers even
  have index := raw.reader.evenIndex writers even
  have known := raw.slot.source _ _ raw.reader.first_rf
  obtain ⟨call, event, words⟩ := raw.slot.publishedPayload (raw.firstIndex / 2) (by omega) (by omega)
  rw [← index] at event
  have physical := (raw.slot.modified _ _ event).1
  rw [show raw.firstIndex - 1 + 1 = raw.firstIndex by omega] at physical
  have prior := reader.publicationAcquired writers (by omega)
  change raw.memory.hb (raw.memory.write .sequence (2 * (raw.firstIndex / 2))) raw.reader.first at prior
  rw [← index, physical] at prior
  exact ⟨call, event, words, prior⟩

structure Lookup (history : History) (b : Position) (entry : EntryData)
    extends ReadSample history where
  raw : SeqLock.Word
  accepted : SeqLock.lookupDecision toReadSample.first toReadSample.last toReadSample.payload
    b.val.player b.val.opponent = some raw
  decoded : Packing.unpack bigEndian raw = entry
  occupied : entry.bound ≠ 0

def Lookup.weaken (lookup : Lookup history b entry) (included : ∀ pair ∈ history, pair ∈ later) :
    Lookup later b entry :=
  ⟨lookup.toReadSample.weaken included, lookup.raw, lookup.accepted, lookup.decoded, lookup.occupied⟩

private theorem Lookup.even (lookup : Lookup history b entry) :
    lookup.memory.readValue lookup.reader.first &&& 1 = 0 :=
  ((SeqLock.lookup_success_iff _ _ _ _ _ _).mp lookup.accepted).1

theorem Lookup.source (lookup : Lookup history b entry) : (b, entry) ∈ history := by
  let writers := lookup.slot.writers
  let reader := lookup.reader.asStable writers lookup.even
  have noWrap : lookup.finalIndex < 2 * (lookup.firstIndex / 2) + SeqLock.modulus := by
    rw [← lookup.reader.evenIndex writers lookup.even]
    exact lookup.noWrap
  obtain ⟨player, opponent, data⟩ := SeqLock.lookup_consistent writers reader
    (lookup.player.asStable writers lookup.even) (lookup.opponent.asStable writers lookup.even)
    (lookup.metadata.asStable writers lookup.even) b.val.player b.val.opponent lookup.raw noWrap lookup.accepted
  have payload := (congrArg (Packing.unpack lookup.bigEndian) data).symm.trans lookup.decoded
  obtain ⟨stored, valid, storedPlayer, storedOpponent⟩ :=
    lookup.published (by
      change (Packing.unpack lookup.bigEndian
        (lookup.memory.writeValue (.payload .metadata) (lookup.firstIndex / 2))).bound ≠ 0
      rw [payload]
      exact lookup.occupied)
  have equal : stored = b := by
    apply Subtype.ext
    have left := storedPlayer.symm.trans player
    have right := storedOpponent.symm.trans opponent
    calc
      stored.val = ⟨stored.val.player, stored.val.opponent⟩ := rfl
      _ = ⟨b.val.player, b.val.opponent⟩ := by rw [left, right]
      _ = b.val := rfl
  change (stored, Packing.unpack lookup.bigEndian
    (lookup.memory.writeValue (.payload .metadata) (lookup.firstIndex / 2))) ∈ history at valid
  simpa only [equal, payload] using valid

def ReadProvenanceContract : Prop :=
  ∀ history b entry (lookup : Lookup history b entry),
    (b, entry) ∈ history ∧ lookup.firstIndex = lookup.finalIndex ∧
    (0 < lookup.firstIndex →
      lookup.memory.hb (lookup.memory.write .sequence lookup.firstIndex) lookup.reader.first)

theorem readProvenanceContract_proven : ReadProvenanceContract := by
  intro history b entry lookup
  let writers := lookup.slot.writers
  have valid := (SeqLock.lookup_success_iff _ _ _ _ _ _).mp lookup.accepted
  refine ⟨lookup.source, lookup.reader.validatedIndex writers lookup.noWrap valid.2.2.2.1, ?_⟩
  intro positive
  have index := lookup.reader.evenIndex writers lookup.even
  have acquired := (lookup.reader.asStable writers lookup.even).publicationAcquired writers (by omega)
  change lookup.memory.hb (lookup.memory.write .sequence (2 * (lookup.firstIndex / 2)))
    lookup.reader.first at acquired
  rw [← index] at acquired
  exact acquired

def PublishedProbe (history : History) (b : Position) (attempts : List ProbeRetry.RawAttempt) : Prop :=
  ∀ a ∈ attempts.take 3, ∀ slot entry,
    a.readSlot slot = some (key b, entry) → entry.bound ≠ 0 → Nonempty (Lookup history b entry)

theorem PublishedProbe.matches
    (reads : PublishedProbe history b attempts) (slot : Fin 2) (entry : EntryData) :
    ProbeMatches history b attempts slot entry := by
  intro attempted present read occupied
  obtain ⟨lookup⟩ := reads attempted present slot entry read occupied
  exact lookup.source

def ProbedChildren (mc : X86Config) (fc : FlipConfig) (context : History) (b : Position)
    (q : PvsEnhancedCutoff.Query) (probes : List PvsEnhancedCutoff.Probe) : Prop :=
  ∀ probe ∈ probes, ∀ entry, probe.2 = some entry → Eligible entry (q.depth - 1) q.selectivity q.endgame →
    ∃ move : GeneratedSquare mc b.val, move.val = probe.1 ∧ move ∈ legalSquares mc b.val ∧
      Nonempty (Lookup context (play mc fc b move).val entry)

theorem etcBoundFrom (mc : X86Config) (fc : FlipConfig) (b : Position) (q : PvsEnhancedCutoff.Query)
    (probes : List PvsEnhancedCutoff.Probe) (hit : PvsEnhancedCutoff.Hit)
    (probed : ∀ probe ∈ probes, ∀ entry, probe.2 = some entry →
      Eligible entry (q.depth-1) q.selectivity q.endgame →
      ∃ move ∈ legalSquares mc b.val, BoundSound entry (minimax mc fc (play mc fc b move).val))
    (cut : PvsEnhancedCutoff.scan q probes = some hit) :
    PvsMoveLoop.Bound (minimax mc fc b) q.alpha.raw.toInt (q.alpha.raw.toInt + 1) hit.2 := by
  obtain ⟨before, probe, after, order, ⟨entry, probedEntry, eligible, tag, high, same⟩, _⟩ :=
    ((PvsEnhancedCutoff.contract_proven q probes).2.1 hit).mp cut
  obtain ⟨move,legal,bound⟩ := probed probe (by simp [order]) entry probedEntry eligible
  have upper := bound.2 (by
    rcases tag with exactTag | upperTag
    · rw [exactTag]; decide
    · rw [upperTag]; decide)
  have child : -(minimax mc fc (play mc fc b move).val) ≤ minimax mc fc b := by
    cases moves : legalSquares mc b.val with
    | nil => simp [moves] at legal
    | cons first rest =>
      rw [PvsBoardTree.minimax_moves mc fc b first rest moves]
      rw [moves] at legal
      exact PvsMoveLoop.bestValue_ge_move _ _ _ legal
  have returned : hit.2 = -entry.score.toInt := (congrArg Prod.snd same).symm
  exact ⟨fun impossible => False.elim (by omega), fun _ => by omega⟩

private theorem stabilityBound (mc : X86Config) (fc : FlipConfig) (b : Position)
    (alpha : ScoreRefresh.Score) (result : Stability.Cutoff.DiscScore)
    (raw : ScoreRefresh.Score) (checked : Bool)
    (hit : EndgameStability.tryCutoff b.val (Board.emptyCount b.val).toNat
      ⟨(ScoreRefresh.toDiscDiff alpha).raw.toInt⟩ = some result)
    (converted : ScoreRefresh.fromDiscDiff checked ⟨BitVec.ofInt 32 result.value⟩ = .ok raw) :
    PvsMoveLoop.Bound (minimax mc fc b) alpha.raw.toInt (alpha.raw.toInt + 1) raw.raw.toInt := by
  obtain ⟨lo, hi, upper, below⟩ :=
    (EndgameStability.cutoffContract_proven mc fc b _ _).2 result hit
  have disc : (BitVec.ofInt 32 result.value).toInt = result.value :=
    BitVec.toInt_ofInt_eq_self (by decide) (by omega) (by omega)
  obtain ⟨score, cast, _, scaled, _⟩ := ScoreRefresh.bounded_roundtrip
    ⟨BitVec.ofInt 32 result.value⟩ (by change -127 ≤ (BitVec.ofInt 32 result.value).toInt; omega)
    (by change (BitVec.ofInt 32 result.value).toInt ≤ 127; omega)
  have canonical : ScoreRefresh.fromDiscDiff checked ⟨BitVec.ofInt 32 result.value⟩ = .ok score := by
    have range : ¬ (result.value < -127 ∨ result.value > 127) := by omega
    simpa [ScoreRefresh.fromDiscDiff, disc, range] using cast
  have equal := Except.ok.inj (converted.symm.trans canonical)
  subst raw
  change score.raw.toInt = (BitVec.ofInt 32 result.value).toInt * 256 at scaled
  rw [disc] at scaled
  have floor := ScoreRefresh.disc_floor alpha
  change result.value ≤ (ScoreRefresh.toDiscDiff alpha).raw.toInt at below
  exact ⟨fun _ => by omega, fun impossible => False.elim (by omega)⟩

inductive Execution (mc : X86Config) (fc : FlipConfig) (b : Position) :
    Bool → Int → Int → Int → Prop
  | wipeout (move : GeneratedSquare mc b.val) : move ∈ legalSquares mc b.val →
      EndgameCutoffs.wipesOut fc b.val move.val = true →
      Execution mc fc b pv alpha beta PvsMoveLoop.scoreMax
  | stability (alpha : ScoreRefresh.Score) (result : Stability.Cutoff.DiscScore)
      (raw : ScoreRefresh.Score) (checked : Bool) :
      EndgameStability.tryCutoff b.val (Board.emptyCount b.val).toNat
        ⟨(ScoreRefresh.toDiscDiff alpha).raw.toInt⟩ = some result →
      ScoreRefresh.fromDiscDiff checked ⟨BitVec.ofInt 32 result.value⟩ = .ok raw →
      Execution mc fc b false alpha.raw.toInt (alpha.raw.toInt + 1) raw.raw.toInt
  | probe (context : History) (attempts : List ProbeRetry.RawAttempt) (slot : Fin 2)
      (entry : EntryData) (alpha beta : Score) (depth : Depth) (checked : Bool) :
      SoundHistory mc fc context → ProbeMatches context b attempts slot entry →
      ProbeRetry.probe (attempts.map (ProbeRetry.observe (key b))) = .hit slot entry →
      canCut checked entry beta depth exactSelectivity true = .ok true →
      Execution mc fc b false alpha.raw.toInt beta.raw.toInt entry.score.toInt
  | enhanced (context : History) (probes : List PvsEnhancedCutoff.Probe) (q : PvsEnhancedCutoff.Query)
      (hit : PvsEnhancedCutoff.Hit) :
      q.selectivity = exactSelectivity → q.endgame = true → SoundHistory mc fc context →
      ProbedChildren mc fc context b q probes → PvsEnhancedCutoff.scan q probes = some hit →
      Execution mc fc b false q.alpha.raw.toInt (q.alpha.raw.toInt + 1) hit.2

def Contract : Prop :=
  ∀ (mc : X86Config) (fc : FlipConfig) (b : Position) (pv : Bool) (alpha beta score : Int),
    (pv = false → beta = alpha + 1) → Execution mc fc b pv alpha beta score →
    PvsMoveLoop.Bound (minimax mc fc b) alpha beta score

theorem contract_proven : Contract := by
  intro mc fc b pv alpha beta score null source
  cases source with
  | wipeout move member hit =>
    rw [EndgameCutoffs.wipeout_value mc fc b move member hit]
    exact ⟨fun _ => Int.le_refl _, fun _ => Int.le_refl _⟩
  | stability alpha result raw checked hit converted =>
    exact stabilityBound mc fc b alpha result raw checked hit converted
  | probe context attempts slot entry alpha beta depth checked historySound matched hit cut =>
    obtain ⟨attempt, member, published, occupied⟩ :=
      ProbeRetry.contract_proven.2 (key b) attempts slot entry hit
    exact cutBound entry alpha beta depth checked _ (null rfl)
      (historySound b entry (matched attempt member published occupied)
        (exactEndgame_of_eligible (canCut_true_iff _ _ _ _ _ _ |>.mp cut).1)) cut
  | enhanced context probes query hit exact endgame historySound matched cut =>
    apply etcBoundFrom mc fc b query probes hit _ cut
    intro probe member e found eligible
    obtain ⟨m,_,legal,⟨read⟩⟩ := matched probe member e found eligible
    exact ⟨m,legal,historySound _ _ read.source (exactEndgame_of_eligible (by simpa only [exact,endgame] using eligible))⟩

def PublishedCutContract : Prop :=
  ∀ mc fc b history attempts slot entry (alpha beta : Score) depth checked,
    SoundHistory mc fc history → PublishedProbe history b attempts →
    beta.raw.toInt = alpha.raw.toInt + 1 →
    ProbeRetry.probe (attempts.map (ProbeRetry.observe (key b))) = .hit slot entry →
    canCut checked entry beta depth exactSelectivity true = .ok true →
    PvsMoveLoop.Bound (minimax mc fc b) alpha.raw.toInt beta.raw.toInt entry.score.toInt

theorem publishedCutContract_proven : PublishedCutContract := by
  intro mc fc b history attempts slot entry alpha beta depth checked sound reads narrow hit cut
  exact contract_proven mc fc b false _ _ _ (fun _ => narrow)
    (.probe history attempts slot entry alpha beta depth checked sound (reads.matches slot entry) hit cut)

private def scalarMoves : X86Config := ⟨false, false⟩
private def scalarFlips : FlipConfig := ⟨false, false, false⟩
private def winningBoard : Position := ⟨⟨0x10101#64, 2#64⟩, by rfl⟩
private def winningMove : GeneratedSquare scalarMoves winningBoard.val := ⟨2, by decide⟩
private def winningEntry : EntryData :=
  Packing.newData ⟨16384⟩ (some 2) .exact 60 exactSelectivity 0 true true

example : Execution scalarMoves scalarFlips winningBoard false 0 1 16384 := by
  have value := EndgameCutoffs.wipeout_value scalarMoves scalarFlips winningBoard winningMove
    (by decide) (by decide)
  have searched : SoundResult (minimax scalarMoves scalarFlips winningBoard) (-32767) 32767 16384 := by
    rw [value]
    exact ⟨by decide, by decide, fun _ => Int.le_refl _, fun _ => Int.le_refl _, fun _ _ => rfl⟩
  have stored := storeContract_proven scalarMoves scalarFlips [] winningBoard true
    ⟨-32767⟩ ⟨32767⟩ ⟨16384⟩ (some 2) 60 0 true (0, 0) Packing.zeroData winningEntry
    (by simp [SoundHistory]) ⟨by decide, by decide, by decide⟩ searched rfl
  let attempts : List ProbeRetry.RawAttempt :=
    [⟨fun slot =>
      if slot = 0 then some ((winningBoard.val.player, 4#64), winningEntry)
      else some (key winningBoard, winningEntry), 0⟩]
  exact .probe [(winningBoard, winningEntry)] attempts 1 winningEntry ⟨0⟩ ⟨1⟩ 60 true
    stored.2 (by simp [ProbeMatches]) rfl rfl

end NeuralReversi.PvsParallel.Cutoffs
