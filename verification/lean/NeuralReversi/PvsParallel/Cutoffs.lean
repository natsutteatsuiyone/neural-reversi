import NeuralReversi.EndgameCutoffs
import NeuralReversi.PvsEnhancedCutoff
import NeuralReversi.TranspositionTable.ProbeRetry
import NeuralReversi.TranspositionTable.Save
import NeuralReversi.TranspositionTable.StoredBound

namespace NeuralReversi.PvsParallel.Cutoffs

open Aspiration (SoundResult Window)
open Bitboard.Moves (X86Config)
open PvsBoardTree (Position GeneratedSquare legalSquares play minimax)
open TranspositionTable

abbrev FlipConfig := Flip.Dispatch.X86Config
abbrev History := List (Position × EntryData)
def exactSelectivity : Selectivity := ⟨5, by decide⟩

def key (b : Position) : ProbeRetry.Key := (b.val.player, b.val.opponent)

def SoundHistory (mc : X86Config) (fc : FlipConfig) (history : History) : Prop :=
  ∀ b entry, (b, entry) ∈ history → BoundSound entry (minimax mc fc b)

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
    intro board entry member
    rcases List.mem_append.mp member with earlier | new
    · exact historySound board entry earlier
    · obtain ⟨rfl, rfl⟩ := Prod.mk.inj (List.mem_singleton.mp new)
      simpa only [BoundSound, fields.1, fields.2] using packed.2.1

private theorem cutBound (entry : EntryData) (alpha beta : Score) (depth : Depth) (checked : Bool)
    (truth : Int) (narrow : beta.raw.toInt = alpha.raw.toInt + 1) (sound : BoundSound entry truth)
    (cut : canCut checked entry beta depth exactSelectivity true = .ok true) :
    PvsMoveLoop.Bound truth alpha.raw.toInt beta.raw.toInt entry.score.toInt := by
  rcases ((TranspositionTable.canCut_true_iff checked entry beta depth exactSelectivity true).mp cut).2.2 with
    ⟨high, lower⟩ | ⟨low, upper⟩
  · exact ⟨fun impossible => False.elim (by omega), fun _ => sound.1 lower⟩
  · exact ⟨fun _ => sound.2 upper, fun impossible => False.elim (by omega)⟩

def ProbeMatches (history : History) (b : Position) (attempts : List ProbeRetry.RawAttempt)
    (slot : Fin 2) (entry : EntryData) : Prop :=
  ∀ a ∈ attempts.take 3, a.readSlot slot = some (key b, entry) → (b, entry) ∈ history

structure Lookup (history : History) (b : Position) (entry : EntryData) where
  memory : SeqLock.Memory Nat
  epoch : Nat
  finalIndex : Nat
  writers : SeqLock.Writers memory
  reader : SeqLock.Reader memory epoch finalIndex
  pe : Nat
  oe : Nat
  de : Nat
  player : SeqLock.FieldRead reader .player pe
  opponent : SeqLock.FieldRead reader .opponent oe
  metadata : SeqLock.FieldRead reader .metadata de
  noWrap : finalIndex < 2 * epoch + SeqLock.modulus
  raw : SeqLock.Word
  bigEndian : Bool
  accepted : SeqLock.lookupDecision (memory.readValue reader.first) (memory.readValue reader.last)
    ⟨memory.readValue player.event, memory.readValue opponent.event, memory.readValue metadata.event⟩
    b.val.player b.val.opponent = some raw
  decoded : Packing.unpack bigEndian raw = entry
  stored : Position
  storedPlayer : memory.writeValue (.payload .player) epoch = stored.val.player
  storedOpponent : memory.writeValue (.payload .opponent) epoch = stored.val.opponent
  publication : (stored, Packing.unpack bigEndian (memory.writeValue (.payload .metadata) epoch)) ∈ history

theorem Lookup.source (lookup : Lookup history b entry) : (b, entry) ∈ history := by
  obtain ⟨player, opponent, data⟩ := SeqLock.lookup_consistent lookup.writers lookup.reader
    lookup.player lookup.opponent lookup.metadata b.val.player b.val.opponent lookup.raw
    lookup.noWrap lookup.accepted
  have equal : lookup.stored = b := by
    apply Subtype.ext
    have left := lookup.storedPlayer.symm.trans player
    have right := lookup.storedOpponent.symm.trans opponent
    calc
      lookup.stored.val = ⟨lookup.stored.val.player, lookup.stored.val.opponent⟩ := rfl
      _ = ⟨b.val.player, b.val.opponent⟩ := by rw [left, right]
      _ = b.val := rfl
  have payload := (congrArg (Packing.unpack lookup.bigEndian) data).symm.trans lookup.decoded
  simpa only [equal, payload] using lookup.publication

def ProbedChildren (mc : X86Config) (fc : FlipConfig) (context : History) (b : Position)
    (q : PvsEnhancedCutoff.Query) (probes : List PvsEnhancedCutoff.Probe) : Prop :=
  ∀ probe ∈ probes, ∀ entry, probe.2 = some entry → Eligible entry (q.depth - 1) q.selectivity q.endgame →
    ∃ move : GeneratedSquare mc b.val, move.val = probe.1 ∧ move ∈ legalSquares mc b.val ∧
      Nonempty (Lookup context (play mc fc b move).val entry)

private theorem etcBound (mc : X86Config) (fc : FlipConfig) (b : Position) (q : PvsEnhancedCutoff.Query)
    (context : History) (probes : List PvsEnhancedCutoff.Probe) (hit : PvsEnhancedCutoff.Hit)
    (historySound : SoundHistory mc fc context) (probed : ProbedChildren mc fc context b q probes)
    (cut : PvsEnhancedCutoff.scan q probes = some hit) :
    PvsMoveLoop.Bound (minimax mc fc b) q.alpha.raw.toInt (q.alpha.raw.toInt + 1) hit.2 := by
  obtain ⟨before, probe, after, order, ⟨entry, probedEntry, eligible, tag, high, same⟩, _⟩ :=
    ((PvsEnhancedCutoff.contract_proven q probes).2.1 hit).mp cut
  obtain ⟨move, _, legal, ⟨lookup⟩⟩ :=
    probed probe (by simp [order]) entry probedEntry eligible
  have upper := (historySound _ _ lookup.source).2 (by
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
    obtain ⟨attempt, member, published, _⟩ :=
      ProbeRetry.contract_proven.2 (key b) attempts slot entry hit
    exact cutBound entry alpha beta depth checked _ (null rfl)
      (historySound b entry (matched attempt member published)) cut
  | enhanced context probes query hit _ _ historySound matched cut =>
    exact etcBound mc fc b query context probes hit historySound matched cut

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
