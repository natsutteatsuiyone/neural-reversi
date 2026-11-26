import NeuralReversi.PvsParallel.ExactTableHistory
import NeuralReversi.TranspositionTable.LookupCut

namespace NeuralReversi.PvsParallel.TableReads

open Bitboard.Moves (X86Config)
open PvsBoardTree (Position GeneratedSquare legalSquares play minimax)
open TranspositionTable
open Cutoffs (History RawSample Lookup FlipConfig key exactSelectivity PublishedProbe)
open ExactTableHistory (Admission Publications)
open Aspiration (Window)

inductive SnapshotReadFrom (source : RawSample → Prop) : Option (ProbeRetry.Key × EntryData) → Prop
  | unavailable : SnapshotReadFrom source none
  | vacant (entry : EntryData) : entry.bound = 0 → SnapshotReadFrom source (some ((0, 0), entry))
  | sampled (sample : RawSample) (snapshot : SeqLock.Snapshot) :
      source sample →
      Snapshot.read sample.bigEndian sample.first sample.last sample.payload = some snapshot →
      SnapshotReadFrom source
        (some ((snapshot.player, snapshot.opponent), Packing.unpack sample.bigEndian snapshot.metadata))

abbrev SnapshotRead (history : History) := SnapshotReadFrom (·.publication history)

theorem SnapshotReadFrom.mapEvidence (s : SnapshotReadFrom source observed)
    (lift : ∀ raw, source raw → target raw) : SnapshotReadFrom target observed := by
  cases s with
  | unavailable => exact .unavailable
  | vacant entry empty => exact .vacant entry empty
  | sampled sample snapshot known accepted => exact .sampled sample snapshot (lift sample known) accepted

theorem SnapshotReadFrom.lookup (s : SnapshotRead history observed) (hit : observed = some (key b, entry))
    (occupied : entry.bound ≠ 0) : Nonempty (Lookup history b entry) := by
  cases s with
  | unavailable => contradiction
  | vacant data empty =>
    have same : data = entry := congrArg Prod.snd (Option.some.inj hit)
    exact (occupied (same ▸ empty)).elim
  | sampled sample snapshot known accepted =>
    have keyEqual : (snapshot.player, snapshot.opponent) = key b :=
      congrArg Prod.fst (Option.some.inj hit)
    have dataEqual : Packing.unpack sample.bigEndian snapshot.metadata = entry :=
      congrArg Prod.snd (Option.some.inj hit)
    have metadata := (Snapshot.readContract_proven _ _ _ _ _).mp accepted
    have occupiedSample : Snapshot.empty sample.bigEndian sample.payload.metadata = false := by
      simpa [Snapshot.empty, ← metadata.2.2.1, dataEqual] using occupied
    have keys := metadata.2.2.2.2
    simp only [occupiedSample, Bool.false_eq_true, ↓reduceIte] at keys
    have p : sample.payload.player = b.val.player :=
      keys.1.symm.trans (congrArg Prod.fst keyEqual)
    have o : sample.payload.opponent = b.val.opponent :=
      keys.2.symm.trans (congrArg Prod.snd keyEqual)
    exact ⟨⟨⟨sample, known⟩, snapshot.metadata, (SeqLock.lookup_success_iff _ _ _ _ _ _).mpr
      ⟨metadata.1, p, o, metadata.2.1, metadata.2.2.1.symm⟩, dataEqual, occupied⟩⟩

def ProbeReadsFrom (source : RawSample → Prop) (attempts : List ProbeRetry.RawAttempt) : Prop :=
  ∀ a ∈ attempts.take 3, ∀ i, SnapshotReadFrom source (a.readSlot i)

abbrev ProbeReads (history : History) := ProbeReadsFrom (·.publication history)

theorem ProbeReadsFrom.mapEvidence {attempts : List ProbeRetry.RawAttempt} (reads : ProbeReadsFrom source attempts)
    (lift : ∀ raw, source raw → target raw) : ProbeReadsFrom target attempts :=
  fun a member slot => (reads a member slot).mapEvidence lift

theorem ProbeReads.published {attempts : List ProbeRetry.RawAttempt}
    (reads : ProbeReads history attempts) : PublishedProbe history b attempts := by
  intro a handled slot entry hit occupied
  exact (reads a handled slot).lookup hit occupied

def LegalHint (b : Position) (entry : EntryData) : Prop :=
  entry.bestMove ≠ Packing.noMove → Save.legal (key b) entry.bestMove

def LegalHints (history : History) : Prop :=
  ∀ b entry, (b, entry) ∈ history → LegalHint b entry

def LegalChoice (b : Position) (move : Option Bitboard.Square) : Prop :=
  ∀ sq, move = some sq → Bitboard.Moves.LegalMove b.val.player b.val.opponent sq.val

theorem LegalChoice.encoded (legal : LegalChoice b move)
    (encoding : entry.bestMove = Packing.encodeBestMove move) : LegalHint b entry := by
  unfold LegalHint
  rw [encoding]
  cases move with
  | none => simp [Packing.encodeBestMove]
  | some sq =>
    intro _
    have inside := sq.isLt
    simpa [Packing.encodeBestMove, Save.legal, key, BitVec.toNat_ofNat,
      Nat.mod_eq_of_lt (show sq.val < 256 by omega)] using legal sq rfl

theorem newHint (legal : LegalChoice b move) :
    LegalHint b (Packing.newData score move bound depth selectivity generation endgame pv) :=
  legal.encoded rfl

def PreservedHintFrom (source : RawSample → Prop) (board : Position)
    (bigEndian : Bool) (incoming : EntryData) (obs : Save.Observation) : Prop :=
  ∀ snapshot, incoming.bound ≠ Packing.boundByte .exact → incoming.bestMove = Packing.noMove →
    Snapshot.read bigEndian obs.first obs.last obs.sample = some snapshot →
    (snapshot.player, snapshot.opponent) = key board →
    (Packing.unpack bigEndian snapshot.metadata).bound ≠ 0 →
    ∃ raw, source raw ∧ raw.bigEndian = bigEndian ∧ raw.first = obs.first ∧
      raw.last = obs.last ∧ raw.payload = obs.sample

theorem PreservedHintFrom.mapEvidence (prior : PreservedHintFrom source board endian incoming obs)
    (lift : ∀ raw, source raw → target raw) : PreservedHintFrom target board endian incoming obs :=
  fun snapshot bound noMove read same occupied => by
    obtain ⟨raw, seen, byteOrder, first, last, words⟩ := prior snapshot bound noMove read same occupied
    exact ⟨raw, lift raw seen, byteOrder, first, last, words⟩

theorem savedHint (hints : LegalHints history) (fresh : LegalHint board incoming)
    (prior : PreservedHintFrom (·.publication history) board endian incoming obs)
    (saved : Save.attempt endian (key board) incoming obs = .written seq entry) :
    LegalHint board entry := by
  by_cases exactBound : incoming.bound = Packing.boundByte .exact
  · rw [Save.exactAttemptContract_proven _ _ _ _ exactBound] at saved
    split at saved
    · cases saved
      exact fresh
    · contradiction
  · obtain ⟨snapshot, read, _, _, proposal⟩ :=
      ((Save.otherAttemptContract_proven _ _ _ _ exactBound).2 _ _).mp saved
    let old := Packing.unpack endian snapshot.metadata
    have chosen := (Save.contract_proven (key board) (snapshot.player, snapshot.opponent) incoming old).1
    rw [chosen] at proposal
    split at proposal
    · contradiction
    · cases Option.some.inj proposal
      unfold LegalHint Save.payload
      split
      · rename_i copied
        obtain ⟨_, occupied, same, absent⟩ := copied
        obtain ⟨raw, seen, byteOrder, first, last, words⟩ :=
          prior snapshot exactBound absent read same.symm occupied
        have accepted : Snapshot.read raw.bigEndian raw.first raw.last raw.payload = some snapshot := by
          simpa only [byteOrder, first, last, words] using read
        have source : SnapshotRead history (some (key board, old)) := by
          simpa only [byteOrder, same, old] using
            (SnapshotReadFrom.sampled raw snapshot seen accepted : SnapshotRead history _)
        obtain ⟨origin⟩ := source.lookup rfl occupied
        exact hints board old origin.source
      · exact fresh

theorem ProbeReads.legalHint (reads : ProbeReads history attempts) (hints : LegalHints history)
    (hit : ProbeRetry.probe (attempts.map (ProbeRetry.observe (key board))) = .hit slot entry) :
    LegalHint board entry := by
  obtain ⟨attempt, tried, read, occupied⟩ := ProbeRetry.contract_proven.2 _ _ _ _ hit
  exact hints board entry (reads.published.matches slot entry attempt tried read occupied)

theorem LegalHint.generated (legal : LegalHint b entry)
    (decoded : Packing.decodeBestMove entry.bestMove = some sq) :
    ∃ move ∈ legalSquares mc b.val, move.val = sq := by
  have present : entry.bestMove ≠ Packing.noMove := by
    intro absent
    rw [absent, Packing.decodeBestMove_noMove] at decoded
    contradiction
  unfold Packing.decodeBestMove at decoded
  split at decoded
  · cases Option.some.inj decoded
    exact (PvsBoardTree.legalSquares_complete mc _ _).mpr (legal present)
  · contradiction

theorem parentExecution (trace : Publications mc fc history) (reads : ProbeReads history attempts)
    (alpha beta : Score)
    (hit : ProbeRetry.probe (attempts.map (ProbeRetry.observe (key b))) = .hit slot entry)
    (cut : canCut checked entry beta depth exactSelectivity true = .ok true) :
    Cutoffs.Execution mc fc b false alpha.raw.toInt beta.raw.toInt entry.score.toInt :=
  .probe history attempts slot entry alpha beta depth checked trace.sound
    (reads.published.matches slot entry) hit cut

def ParentContract : Prop :=
  ∀ mc fc b history attempts slot entry (alpha beta : Score) depth checked,
    Publications mc fc history → ProbeReads history attempts →
    beta.raw.toInt = alpha.raw.toInt + 1 →
    ProbeRetry.probe (attempts.map (ProbeRetry.observe (key b))) = .hit slot entry →
    canCut checked entry beta depth exactSelectivity true = .ok true →
    PvsMoveLoop.Bound (minimax mc fc b) alpha.raw.toInt beta.raw.toInt entry.score.toInt

theorem parentContract_proven : ParentContract := by
  intro mc fc b history attempts slot entry alpha beta depth checked historyTrace reads null hit cut
  exact Cutoffs.contract_proven mc fc b false _ _ _ (fun _ => null)
    (parentExecution historyTrace reads alpha beta hit cut)

inductive SlotLookupFrom (source : RawSample → Prop) (b : Position) : Option EntryData → Prop
  | miss : SlotLookupFrom source b none
  | hit (sample : RawSample) (entry : EntryData) :
      source sample →
      LookupCut.readOccupied sample.bigEndian sample.first sample.last sample.payload
        b.val.player b.val.opponent = some entry →
      SlotLookupFrom source b (some entry)

abbrev SlotLookup (history : History) := SlotLookupFrom (·.publication history)

theorem SlotLookupFrom.mapEvidence (read : SlotLookupFrom source b result)
    (lift : ∀ raw, source raw → target raw) : SlotLookupFrom target b result := by
  cases read with
  | miss => exact .miss
  | hit sample entry known accepted => exact .hit sample entry (lift sample known) accepted

theorem SlotLookupFrom.published (read : SlotLookup history b (some entry)) :
    Nonempty (Lookup history b entry) := by
  cases read with
  | hit sample _ known accepted =>
    obtain ⟨even, p, o, equal, decoded, occupied⟩ :=
      (LookupCut.readContract_proven _ _ _ _ _ _ _).mp accepted
    exact ⟨⟨⟨sample, known⟩, sample.payload.metadata,
      (SeqLock.lookup_success_iff _ _ _ _ _ _).mpr ⟨even, p, o, equal, rfl⟩, decoded, occupied⟩⟩

structure ChildLookupFrom (mc : X86Config) (fc : FlipConfig) (b : Position) (source : RawSample → Prop) where
  move : GeneratedSquare mc b.val
  front : Option EntryData
  back : Option EntryData
  readFront : SlotLookupFrom source (play mc fc b move).val front
  readBack : SlotLookupFrom source (play mc fc b move).val back

abbrev ChildLookup (mc : X86Config) (fc : FlipConfig) (b : Position) (history : History) :=
  ChildLookupFrom mc fc b (·.publication history)

def ChildLookupFrom.observation (read : ChildLookupFrom mc fc b source) : PvsEnhancedCutoff.Probe :=
  (read.move.val, read.front.or read.back)

theorem enhancedHint (query : PvsEnhancedCutoff.Query) (reads : List (ChildLookupFrom mc fc b source))
    (found : PvsEnhancedCutoff.scan query (reads.map ChildLookupFrom.observation) = some hit) :
    LegalChoice b (some hit.1) := by
  intro sq selected
  obtain ⟨before, probe, after, order, ⟨entry, _, _, _, _, same⟩, _⟩ :=
    ((PvsEnhancedCutoff.contract_proven query (reads.map ChildLookupFrom.observation)).2.1 hit).mp found
  have member : probe ∈ reads.map ChildLookupFrom.observation := by rw [order]; simp
  obtain ⟨read, _, rfl⟩ := List.mem_map.mp member
  have square := congrArg Prod.fst same
  rw [← Option.some.inj selected, ← square]
  exact (Board.getMoves_exact mc b.val _).mp read.move.property

def ChildLookupFrom.mapEvidence (read : ChildLookupFrom mc fc b source)
    (lift : ∀ raw, source raw → target raw) : ChildLookupFrom mc fc b target :=
  ⟨read.move, read.front, read.back, read.readFront.mapEvidence lift, read.readBack.mapEvidence lift⟩

theorem ChildLookupFrom.observations_mapEvidence {target : RawSample → Prop}
    (reads : List (ChildLookupFrom mc fc b source))
    (lift : ∀ raw, source raw → target raw) :
    ((reads.map fun read => read.mapEvidence lift).map ChildLookupFrom.observation) =
      reads.map ChildLookupFrom.observation := by
  rw [List.map_map]
  rfl

theorem ChildLookupFrom.source (read : ChildLookup mc fc b history) (entry : EntryData)
    (found : read.observation.2 = some entry) :
    Nonempty (Lookup history (play mc fc b read.move).val entry) := by
  rcases (LookupCut.clusterContract_proven read.front read.back entry).mp found with
    front | ⟨_, back⟩
  · have actual := read.readFront
    rw [front] at actual
    exact actual.published
  · have actual := read.readBack
    rw [back] at actual
    exact actual.published

private theorem move_member {mc : X86Config} {b : Position}
    (move : GeneratedSquare mc b.val) : move ∈ legalSquares mc b.val := by
  obtain ⟨generated, present, square⟩ :=
    (PvsBoardTree.legalSquares_complete mc b.val move.val).mpr
      ((Board.getMoves_exact mc b.val move.val).mp move.property)
  have same : generated = move := Subtype.ext square
  exact same ▸ present

theorem enhancedExecution (query : PvsEnhancedCutoff.Query) (reads : List (ChildLookup mc fc b history))
    (hit : PvsEnhancedCutoff.Hit) (trace : Publications mc fc history)
    (exact : query.selectivity = exactSelectivity) (endgame : query.endgame = true)
    (found : PvsEnhancedCutoff.scan query (reads.map ChildLookupFrom.observation) = some hit) :
    Cutoffs.Execution mc fc b false query.alpha.raw.toInt (query.alpha.raw.toInt + 1) hit.2 := by
  refine .enhanced history (reads.map ChildLookupFrom.observation) query hit exact endgame
    trace.sound ?_ found
  intro observed member entry present _
  obtain ⟨read, _, rfl⟩ := List.mem_map.mp member
  exact ⟨read.move, rfl, move_member read.move, read.source entry present⟩

def EnhancedContract : Prop :=
  ∀ mc fc b history (query : PvsEnhancedCutoff.Query) (reads : List (ChildLookup mc fc b history)) hit,
    Publications mc fc history → query.selectivity = exactSelectivity → query.endgame = true →
    PvsEnhancedCutoff.scan query (reads.map ChildLookupFrom.observation) = some hit →
    PvsMoveLoop.Bound (minimax mc fc b) query.alpha.raw.toInt (query.alpha.raw.toInt + 1) hit.2

theorem enhancedContract_proven : EnhancedContract := by
  intro mc fc b history query reads hit trace exact endgame found
  exact Cutoffs.contract_proven mc fc b false _ _ _ (fun _ => rfl)
    (enhancedExecution query reads hit trace exact endgame found)

def EnhancedStoreContract : Prop :=
  ∀ mc fc b history (query : PvsEnhancedCutoff.Query) (reads : List (ChildLookup mc fc b history))
      hit (beta score : Score) generation ttPv bigEndian observed seq written,
    Publications mc fc history → query.selectivity = exactSelectivity → query.endgame = true →
    PvsEnhancedCutoff.scan query (reads.map ChildLookupFrom.observation) = some hit →
    Window query.alpha.raw.toInt beta.raw.toInt → beta.raw.toInt = query.alpha.raw.toInt + 1 →
    score.raw.toInt = hit.2 →
    Save.attempt bigEndian (key b)
      (Packing.newData score (some hit.1) .lower query.depth exactSelectivity generation true ttPv)
      observed = .written seq written →
    Admission mc fc history b written

theorem enhancedStoreContract_proven : EnhancedStoreContract := by
  intro mc fc b history query reads hit beta score generation ttPv bigEndian observed seq written
    trace selectivity endgame found window narrow scoreEq saved
  obtain ⟨_, _, _, _, witness, _⟩ :=
    ((PvsEnhancedCutoff.contract_proven query (reads.map ChildLookupFrom.observation)).2.1 hit).mp found
  obtain ⟨entry, _, _, _, above, equal⟩ := witness
  have negated : -entry.score.toInt = hit.2 := congrArg Prod.snd equal
  have high : beta.raw.toInt ≤ score.raw.toInt := by omega
  have kind : classify false score query.alpha beta = .lower := by simp [classify, high]
  have bounds := enhancedContract_proven mc fc b history query reads hit trace selectivity endgame found
  rw [← narrow, ← scoreEq] at bounds
  have sound := PvsParallel.bound_sound (PvsBoardTree.build mc fc b) window bounds
  exact .saved history b false query.alpha beta score (some hit.1) query.depth generation ttPv
    window sound (by rwa [kind])

end NeuralReversi.PvsParallel.TableReads
