module
public import NeuralReversi.PvsParallel.RootEvidence
public import NeuralReversi.PvsParallel.SlotJournal

@[expose] public section

namespace NeuralReversi.PvsParallel.CausalReplay

open Aspiration (SoundResult Window)
open Bitboard.Moves (X86Config)
open PvsBoardTree (Position GeneratedSquare build legalSquares)
open PvsMoveLoop (Tree scoreInf scoreMax)
open TranspositionTable
open Cutoffs (History FlipConfig RawSample exactSelectivity key)
open ExactTableHistory (Admission Publications)
open SlotJournal (Journal)
open TableReads (ChildLookupFrom ProbeReadsFrom SlotLookupFrom LegalChoice LegalHint LegalHints PreservedHintFrom)

structure ScoreResult where
  tree : Tree
  pv : Bool
  alpha : Int
  beta : Int
  score : Int

def Recorded (results : List ScoreResult) : Search := fun tree pv alpha beta score =>
  (⟨tree, pv, alpha, beta, score⟩ : ScoreResult) ∈ results

inductive PrefixExitFrom (mc : X86Config) (fc : FlipConfig) (source : RawSample → Prop) : Exit
  | wipeout (move : GeneratedSquare mc board.val) : move ∈ legalSquares mc board.val →
      EndgameCutoffs.wipesOut fc board.val move.val = true →
      PrefixExitFrom mc fc source mc fc board pv alpha beta scoreMax
  | stability (alpha : ScoreRefresh.Score) (result : Stability.Cutoff.DiscScore)
      (raw : ScoreRefresh.Score) (checked : Bool) :
      EndgameStability.tryCutoff board.val (Board.emptyCount board.val).toNat
        ⟨(ScoreRefresh.toDiscDiff alpha).raw.toInt⟩ = some result →
      ScoreRefresh.fromDiscDiff checked ⟨BitVec.ofInt 32 result.value⟩ = .ok raw →
      PrefixExitFrom mc fc source mc fc board false alpha.raw.toInt (alpha.raw.toInt + 1) raw.raw.toInt
  | probe (attempts : List ProbeRetry.RawAttempt) (slot : Fin 2) (entry : EntryData)
      (alpha beta : Score) (depth : Depth) (checked : Bool) :
      ProbeReadsFrom source attempts →
      ProbeRetry.probe (attempts.map (ProbeRetry.observe (key board))) = .hit slot entry →
      canCut checked entry beta depth exactSelectivity true = .ok true →
      PrefixExitFrom mc fc source mc fc board false alpha.raw.toInt beta.raw.toInt entry.score.toInt
  | enhanced (query : PvsEnhancedCutoff.Query) (reads : List (ChildLookupFrom mc fc board source))
      (hit : PvsEnhancedCutoff.Hit) :
      query.selectivity = exactSelectivity → query.endgame = true →
      PvsEnhancedCutoff.scan query (reads.map ChildLookupFrom.observation) = some hit →
      PrefixExitFrom mc fc source mc fc board false query.alpha.raw.toInt (query.alpha.raw.toInt + 1) hit.2

abbrev PrefixExit (mc : X86Config) (fc : FlipConfig) (history : History) :=
  PrefixExitFrom mc fc (·.publication history)

theorem PrefixExitFrom.mapEvidence (exit : PrefixExitFrom mc fc source emc efc board pv alpha beta score)
    (lift : ∀ raw, source raw → target raw) :
    PrefixExitFrom mc fc target emc efc board pv alpha beta score := by
  cases exit with
  | wipeout move legal hit => exact .wipeout move legal hit
  | stability alpha result raw checked hit converted => exact .stability alpha result raw checked hit converted
  | probe attempts slot entry alpha beta depth checked read found cut =>
    exact .probe attempts slot entry alpha beta depth checked (read.mapEvidence lift) found cut
  | enhanced query reads hit exact endgame found =>
    exact .enhanced query (reads.map fun read => read.mapEvidence lift) hit exact endgame
      (by rwa [ChildLookupFrom.observations_mapEvidence])

theorem PrefixExitFrom.checked (exit : PrefixExit mc fc history emc efc board pv alpha beta score)
    (publications : Publications mc fc history) : Cutoffs.Execution emc efc board pv alpha beta score := by
  cases exit with
  | wipeout move legal hit => exact .wipeout move legal hit
  | stability alpha result raw checked hit converted => exact .stability alpha result raw checked hit converted
  | probe _ _ _ alpha beta _ _ observed found cut =>
    exact TableReads.parentExecution publications observed alpha beta found cut
  | enhanced query reads hit exact endgame found =>
    exact TableReads.enhancedExecution query reads hit publications exact endgame found

structure SearchSave where
  alpha : Score
  beta : Score
  score : Score
  move : Option (Fin 64)
  depth : Depth
  generation : BitVec 8
  ttPv : Bool
  observed : Save.Observation
  sequence : SeqLock.Word

def SearchSave.incoming (save : SearchSave) (bound : Bound) : EntryData :=
  Packing.newData save.score save.move bound save.depth exactSelectivity save.generation true save.ttPv

inductive SearchHint (mc : X86Config) (board : Position) :
    Bool → Int → Int → Int → Option Bitboard.Square → Prop
  | absent : SearchHint mc board pv alpha beta score none
  | chosen (move : GeneratedSquare mc board.val) : SearchHint mc board pv alpha beta score (some move.val)
  | serial (observe : PvsMoveLoop.ChildSearch (GeneratedSquare mc board.val))
      (q remaining : PvsPostpone.Queue (GeneratedSquare mc board.val))
      (visited : List (GeneratedSquare mc board.val)) :
      q.idx = 0 →
      Prefix.Serial.Run (fun _ _ _ _ _ => True) observe pv beta q
        ⟨alpha, -scoreInf⟩ visited remaining (.returned score) →
      SearchHint mc board pv alpha beta score
        ((Prefix.Serial.bestMove pv observe beta visited true ⟨alpha, -scoreInf⟩ none).map Subtype.val)
  | split (observe : PvsMoveLoop.ChildSearch (GeneratedSquare mc board.val))
      (q remaining : PvsPostpone.Queue (GeneratedSquare mc board.val))
      (visited : List (GeneratedSquare mc board.val))
      (split : SplitExecution (visited ++ remaining.pending).length pv alpha beta score)
      (picked : Option (GeneratedSquare mc board.val))
      (read : SplitExecution.MoveRead split (fun m => Packing.encodeBestMove (m.map Subtype.val))
        (fun i => (visited ++ remaining.pending)[i]?)
        (Prefix.Serial.bestMove pv observe beta visited true ⟨alpha, -scoreInf⟩ none) picked) :
      q.idx = 0 → split.start.toNat = remaining.idx →
      Prefix.Serial.Run (fun _ _ _ _ _ => True) observe pv beta q
        ⟨alpha, -scoreInf⟩ visited remaining (.handoff split.entry) →
      SearchHint mc board pv alpha beta score
        (Packing.decodeBestMove ((read.writes.get read.load.source).value.setWidth 8))

private theorem generatedChoice (move : Option (GeneratedSquare mc board.val)) :
    LegalChoice board (move.map Subtype.val) := by
  intro sq same
  cases move with
  | none => contradiction
  | some mv =>
    have value : mv.val = sq := Option.some.inj same
    rw [← value]
    exact (Board.getMoves_exact mc board.val _).mp mv.property

theorem SearchHint.legal (choice : SearchHint mc board pv alpha beta score move) : LegalChoice board move := by
  cases choice with
  | absent => intro sq impossible; contradiction
  | chosen m => exact generatedChoice (some m)
  | serial => exact generatedChoice _
  | split _ _ _ _ _ picked read _ _ _ =>
    rw [read.value, Packing.decodeBestMove_encodeBestMove]
    exact generatedChoice picked

def SearchHintContract : Prop :=
  ∀ mc board pv alpha beta score move,
    SearchHint mc board pv alpha beta score move → LegalChoice board move

theorem searchHintContract_proven : SearchHintContract :=
  fun _ _ _ _ _ _ _ choice => choice.legal

def shallowBestMove (beta : Int) : List (Move × Int) → Int → Option Move → Option Move
  | [], _, previous => previous
  | (move, score) :: rest, alpha, previous =>
    if beta ≤ score then some move
    else if alpha < score then shallowBestMove beta rest score (some move)
    else shallowBestMove beta rest alpha previous

inductive OtherHint (mc : X86Config) (fc : FlipConfig) (board : Position) :
    Option Bitboard.Square → Prop
  | fromBoard (move : Option (GeneratedSquare mc board.val)) : OtherHint mc fc board (move.map Subtype.val)
  | pvs : SearchHint mc board pv alpha beta score move → OtherHint mc fc board move
  | shallow (searched : List (GeneratedSquare mc board.val × Int)) (alpha beta : Int) :
      OtherHint mc fc board ((shallowBestMove beta searched alpha none).map Subtype.val)
  | enhanced (query : PvsEnhancedCutoff.Query) (reads : List (ChildLookupFrom mc fc board (fun _ => True))) :
      PvsEnhancedCutoff.scan query (reads.map ChildLookupFrom.observation) = some hit →
      OtherHint mc fc board (some hit.1)

theorem OtherHint.legal (choice : OtherHint mc fc board move) : LegalChoice board move := by
  cases choice with
  | fromBoard => exact generatedChoice _
  | pvs matched => exact matched.legal
  | shallow => exact generatedChoice _
  | enhanced query reads found => exact TableReads.enhancedHint query reads found

def OtherHintContract : Prop :=
  ∀ mc fc board move entry, OtherHint mc fc board move →
    entry.bestMove = Packing.encodeBestMove move → LegalHint board entry

theorem otherHintContract_proven : OtherHintContract :=
  fun _ _ _ _ _ choice encoded => choice.legal.encoded encoded

inductive PrefixWriteFrom (mc : X86Config) (fc : FlipConfig) (bigEndian : Bool)
    (computed : Search) (source : RawSample → Prop) : Position → EntryData → Prop
  | exact (save : SearchSave) (pv : Bool) :
      Window save.alpha.raw.toInt save.beta.raw.toInt →
      (pv = false → save.beta.raw.toInt = save.alpha.raw.toInt + 1) →
      computed (build mc fc board) pv save.alpha.raw.toInt save.beta.raw.toInt save.score.raw.toInt →
      SearchHint mc board pv save.alpha.raw.toInt save.beta.raw.toInt save.score.raw.toInt save.move →
      PreservedHintFrom source board bigEndian
        (save.incoming (classify pv save.score save.alpha save.beta)) save.observed →
      Save.attempt bigEndian (key board) (save.incoming (classify pv save.score save.alpha save.beta))
        save.observed = .written save.sequence entry →
      PrefixWriteFrom mc fc bigEndian computed source board entry
  | localBoundary (save : SearchSave) :
      Window save.alpha.raw.toInt save.beta.raw.toInt → save.move = none →
      SoundResult (PvsBoardTree.minimax mc fc board) save.alpha.raw.toInt save.beta.raw.toInt save.score.raw.toInt →
      PreservedHintFrom source board bigEndian (save.incoming (classify false save.score save.alpha save.beta)) save.observed →
      Save.attempt bigEndian (key board) (save.incoming (classify false save.score save.alpha save.beta))
        save.observed = .written save.sequence entry → PrefixWriteFrom mc fc bigEndian computed source board entry
  | root (save : SearchSave) (roots : List Bitboard.Square) (idx : Nat) :
      Window save.alpha.raw.toInt save.beta.raw.toInt → Root.Covers mc board roots → idx < roots.length →
      Root.storeAllowed true idx = true →
      Root.Execution computed mc fc board roots idx save.alpha.raw.toInt save.beta.raw.toInt
        save.score.raw.toInt →
      SearchHint mc board true save.alpha.raw.toInt save.beta.raw.toInt save.score.raw.toInt save.move →
      PreservedHintFrom source board bigEndian
        (save.incoming (classify true save.score save.alpha save.beta)) save.observed →
      Save.attempt bigEndian (key board) (save.incoming (classify true save.score save.alpha save.beta))
        save.observed = .written save.sequence entry →
      PrefixWriteFrom mc fc bigEndian computed source board entry
  | other (incoming : EntryData) (observed : Save.Observation) (sequence : SeqLock.Word)
      (move : Option Bitboard.Square) :
      ¬Cutoffs.ExactEndgame incoming →
      OtherHint mc fc board move → incoming.bestMove = Packing.encodeBestMove move →
      PreservedHintFrom source board bigEndian incoming observed →
      Save.attempt bigEndian (key board) incoming observed = .written sequence entry →
      PrefixWriteFrom mc fc bigEndian computed source board entry
  | penalty (expected : EntryData) (snapshot : Option SeqLock.Snapshot) (current : SeqLock.Word) :
      SlotLookupFrom source board (some expected) →
      Penalty.tryPenalize bigEndian board.val.player board.val.opponent expected snapshot current = some entry →
      PrefixWriteFrom mc fc bigEndian computed source board entry
  | enhanced (query : PvsEnhancedCutoff.Query) (reads : List (ChildLookupFrom mc fc board source))
      (hit : PvsEnhancedCutoff.Hit) (save : SearchSave) :
      query.selectivity = exactSelectivity → query.endgame = true →
      PvsEnhancedCutoff.scan query (reads.map ChildLookupFrom.observation) = some hit →
      Window query.alpha.raw.toInt save.beta.raw.toInt →
      save.beta.raw.toInt = query.alpha.raw.toInt + 1 → save.score.raw.toInt = hit.2 →
      Save.attempt bigEndian (key board)
        (Packing.newData save.score (some hit.1) .lower query.depth exactSelectivity
          save.generation true save.ttPv) save.observed = .written save.sequence entry →
      PrefixWriteFrom mc fc bigEndian computed source board entry

abbrev PrefixWrite (mc : X86Config) (fc : FlipConfig) (bigEndian : Bool) (computed : Search) (history : History) :=
  PrefixWriteFrom mc fc bigEndian computed (·.publication history)

theorem PrefixWriteFrom.mapEvidence (write : PrefixWriteFrom mc fc bigEndian computed source board entry)
    (earlierResults : ∀ tree pv a b s, computed tree pv a b s → accepted tree pv a b s)
    (lift : ∀ raw, source raw → target raw) :
    PrefixWriteFrom mc fc bigEndian accepted target board entry := by
  cases write with
  | exact save pv window narrow result legal inherited stored =>
    exact .exact save pv window narrow (earlierResults _ _ _ _ _ result) legal (inherited.mapEvidence lift) stored
  | localBoundary save window absent score inherited stored =>
    exact .localBoundary save window absent score (inherited.mapEvidence lift) stored
  | root save roots idx window covers inside guard returned legal inherited stored =>
    refine .root save roots idx window covers inside guard ?_ legal (inherited.mapEvidence lift) stored
    generalize save.score.raw.toInt = score at returned ⊢
    cases returned with
    | searched tree completed => exact .searched tree (earlierResults _ _ _ _ _ completed)
    | wipeout move legal wipes => exact .wipeout move legal wipes
  | other incoming observed seq move outside legal encoded inherited saved =>
    exact .other incoming observed seq move outside legal encoded (inherited.mapEvidence lift) saved
  | penalty expected snapshot current source saved =>
    exact .penalty expected snapshot current (source.mapEvidence lift) saved
  | enhanced query reads hit save exact endgame cut window narrow scoreEq saved =>
    exact .enhanced query (reads.map fun read => read.mapEvidence lift) hit save exact endgame
      (by rwa [ChildLookupFrom.observations_mapEvidence]) window narrow scoreEq saved

def Certified (returns : List ScoreResult) : Prop :=
  ∀ tree pv alpha beta score, Recorded returns tree pv alpha beta score →
    boundedReplay returns.length tree pv alpha beta score

private theorem Certified.replayed (verified : Certified returns)
    (present : Recorded returns tree pv alpha beta score) : Replayed tree pv alpha beta score :=
  ⟨returns.length, verified _ _ _ _ _ present⟩

private theorem Certified.rootReplay (verified : Certified returns)
    (returned : Root.Execution (Recorded returns) mc fc board roots idx alpha beta score) :
    Root.Execution Replayed mc fc board roots idx alpha beta score := by
  cases returned with
  | searched found node => exact .searched found (verified.replayed node)
  | wipeout move legal wipes => exact .wipeout move legal wipes

private theorem PrefixWriteFrom.admitted (write : PrefixWrite mc fc endian (Recorded returns) history board entry)
    (verified : Certified returns) (publications : Publications mc fc history) :
    Admission mc fc history board entry := by
  cases write with
  | exact save pv window narrow returned _ _ stored =>
    exact Admission.completed history board pv save.alpha save.beta save.score save.move save.depth
      save.generation save.ttPv Replayed replay_follows window narrow (verified.replayed returned) stored
  | localBoundary save window _ score _ stored =>
    exact Admission.saved history board false save.alpha save.beta save.score save.move save.depth
      save.generation save.ttPv window score stored
  | root save roots idx window covers inside guard returned _ _ stored =>
    exact Admission.rootCompleted history board roots idx save.alpha save.beta save.score save.move save.depth
      save.generation save.ttPv Replayed replay_follows covers inside guard window
      (verified.rootReplay returned) stored
  | other incoming observed seq _ outside _ _ _ saved => exact Admission.otherSaved history board incoming outside saved
  | penalty expected snapshot current source saved =>
    obtain ⟨read⟩ := source.published
    exact SlotJournal.penaltyAdmission read snapshot current saved
  | enhanced query reads hit save exact endgame cut window narrow scoreEq saved =>
    exact TableReads.enhancedStoreContract_proven mc fc board history query reads hit save.beta save.score
      save.generation save.ttPv endian save.observed save.sequence entry publications exact endgame cut
      window narrow scoreEq saved

theorem PrefixWriteFrom.legalHint (write : PrefixWrite mc fc endian computed history board entry)
    (hints : LegalHints history) : LegalHint board entry := by
  cases write with
  | exact _ _ _ _ _ legal inherited saved =>
    exact TableReads.savedHint hints (TableReads.newHint legal.legal) inherited saved
  | localBoundary save _ absent _ inherited saved =>
    apply TableReads.savedHint hints (TableReads.newHint (move := save.move) ?_) inherited saved
    intro sq hit
    rw [absent] at hit
    contradiction
  | root _ _ _ _ _ _ _ _ legal inherited saved =>
    exact TableReads.savedHint hints (TableReads.newHint legal.legal) inherited saved
  | other _ _ _ _ _ legal encoded inherited saved =>
    exact TableReads.savedHint hints (legal.legal.encoded encoded) inherited saved
  | penalty expected snapshot current source saved =>
    obtain ⟨origin⟩ := source.published
    obtain ⟨_, _, _, _, _, _, _, rfl⟩ :=
      (Penalty.writeContract_proven endian board.val.player board.val.opponent expected snapshot current entry).mp saved
    exact hints board expected origin.source
  | enhanced query reads hit save _ _ cut _ _ _ saved =>
    have legal := TableReads.newHint (TableReads.enhancedHint query reads cut)
      (score := save.score) (bound := .lower) (depth := query.depth)
      (selectivity := exactSelectivity) (generation := save.generation) (endgame := true) (pv := save.ttPv)
    refine TableReads.savedHint hints legal ?_ saved
    intro snapshot _ absent
    have decoded := Packing.decodeBestMove_encodeBestMove (some hit.1)
    change Packing.encodeBestMove (some hit.1) = Packing.noMove at absent
    rw [absent, Packing.decodeBestMove_noMove] at decoded
    contradiction

inductive Run (mc : X86Config) (fc : FlipConfig) (bigEndian : Bool) :
    List ScoreResult → List (WriterTrace.Call Nat) → History → Prop
  | start : Run mc fc bigEndian [] [] []
  | scored (result : ScoreResult) : Run mc fc bigEndian earlier calls history →
      ExecutionWith (PrefixExit mc fc history) (Recorded earlier)
        result.tree result.pv result.alpha result.beta result.score →
      Run mc fc bigEndian (result :: earlier) calls history
  | clear (call : WriterTrace.Call Nat) : Run mc fc bigEndian returns calls history →
      call.payload = ⟨0, 0, 0⟩ → Run mc fc bigEndian returns (calls ++ [call]) history
  | stored (call : WriterTrace.Call Nat) (board : Position) (entry : EntryData) :
      Run mc fc bigEndian returns calls history →
      PrefixWrite mc fc bigEndian (Recorded returns) history board entry →
      call.payload = SlotJournal.packed bigEndian board entry →
      Run mc fc bigEndian returns (calls ++ [call]) (history ++ [(board, entry)])

theorem Run.legalHints (run : Run mc fc endian results calls history) : LegalHints history := by
  induction run with
  | start => intro b entry missing; simp at missing
  | scored _ _ _ ih | clear _ _ _ ih => exact ih
  | stored _ _ _ _ write _ ih =>
    intro board entry member
    rcases List.mem_append.mp member with old | recent
    · exact ih board entry old
    · cases List.mem_singleton.mp recent
      exact write.legalHint ih

def Contract : Prop :=
  ∀ mc fc bigEndian returns calls history, Run mc fc bigEndian returns calls history →
    Certified returns ∧ Journal mc fc bigEndian calls history

theorem contract_proven : Contract := by
  intro mc fc bigEndian returns calls history run
  induction run with
  | start => exact ⟨fun _ _ _ _ _ missing => by simp [Recorded] at missing, .start .initial⟩
  | @scored earlier calls history result _ execution ih =>
    have checked : Execution (boundedReplay earlier.length)
        result.tree result.pv result.alpha result.beta result.score :=
      execution.weaken ih.1 (fun _ _ _ _ _ _ _ exit => PrefixExitFrom.checked exit ih.2.publications)
    refine ⟨?_, ih.2⟩
    intro tree pv alpha beta score present
    rcases List.mem_cons.mp present with current | older
    · cases current
      exact checked
    · exact boundedReplay_mono (Nat.le_succ _) _ _ _ _ _ (ih.1 _ _ _ _ _ older)
  | clear call _ zero ih => exact ⟨ih.1, .clear call ih.2 zero⟩
  | stored call board entry _ written packed ih =>
    exact ⟨ih.1, .store call board entry ih.2 (written.admitted ih.1 ih.2.publications) packed⟩

def FullWindowContract : Prop :=
  ∀ mc fc bigEndian returns calls history, Run mc fc bigEndian returns calls history →
    ∀ board score, Recorded returns (build mc fc board) true (-scoreInf) scoreInf score →
      score = PvsBoardTree.minimax mc fc board

theorem fullWindowContract_proven : FullWindowContract := by
  intro mc fc bigEndian returns calls history run board score returned
  exact fullWindow_exact Replayed replay_follows mc fc board score
    ((contract_proven mc fc bigEndian returns calls history run).1.replayed returned)

def RootChecked (mc : X86Config) (fc : FlipConfig) (bigEndian : Bool) (board : Position)
    (roots : List Bitboard.Square) (idx : Nat) (search : RootAspiration.Search) : Prop :=
  ∀ i alpha beta, Window alpha beta → (search i alpha beta).aborted = false →
    ∃ returns calls history, Run mc fc bigEndian returns calls history ∧
      RootEvidence.AttemptWith (Recorded returns) mc fc board roots idx alpha beta
        (search i alpha beta).score (search i alpha beta).payload

theorem RootChecked.checked (root : RootChecked mc fc bigEndian board roots idx search) :
    RootEvidence.Checked mc fc board roots idx search := by
  intro i alpha beta window running
  obtain ⟨returns, calls, history, causal, attempt⟩ := root i alpha beta window running
  have verified := (contract_proven mc fc bigEndian returns calls history causal).1
  exact attempt.weaken (fun _ _ _ _ _ prior => verified.replayed prior)

def SingleResultContract : Prop :=
  ∀ mc fc bigEndian board roots alpha beta search (pv : Bitboard.Square → List Bitboard.Square)
      (depth nodes : Nat),
    Root.Covers mc board roots → 0 < roots.length → Window alpha beta →
    RootChecked mc fc bigEndian board roots 0 search →
    (∀ i a z, Window a z → (search i a z).aborted = false) →
    ∃ accepted result,
      Aspiration.runWith search 11 0 alpha beta 256 = some accepted ∧
      let rows := roots.map (fun sq => (⟨sq, accepted.trial.payload sq, pv sq⟩ : SearchPublication.Row))
      SearchFallback.fromSnapshot (SearchPublication.capture rows ⟨depth, 5⟩ true) nodes = some result ∧
        result.best.score = PvsBoardTree.minimax mc fc board

theorem singleResultContract_proven : SingleResultContract := by
  intro mc fc bigEndian board roots alpha beta search pv depth nodes covers nonempty window observed running
  exact RootEvidence.singleContract_proven mc fc board roots alpha beta search pv depth nodes
    covers nonempty window observed.checked running

def AllMovesResultContract : Prop :=
  ∀ mc fc bigEndian board roots (observe : RootMultiPV.Observations) (seed : RootMultiPV.Seed)
      (pv : Bitboard.Square → List Bitboard.Square) (depth nodes : Nat),
    Root.Covers mc board roots → 0 < roots.length →
    (∀ order idx, Root.Covers mc board order → idx < order.length →
      Window (seed order idx).1 (seed order idx).2 ∧
      RootChecked mc fc bigEndian board order idx (observe order idx) ∧
      ∀ i a z, Window a z → (observe order idx i a z).aborted = false) →
    ∃ batch result,
      RootMultiPV.run observe seed roots.length 0 roots = some batch ∧
      let rows := batch.picks.map (fun pick => (⟨pick.square, pick.score, pv pick.square⟩ : SearchPublication.Row))
      SearchFallback.fromSnapshot (SearchPublication.capture rows ⟨depth, 5⟩ true) nodes = some result ∧
        result.best.score = PvsBoardTree.minimax mc fc board ∧ result.endgame = true

theorem allMovesResultContract_proven : AllMovesResultContract := by
  intro mc fc bigEndian board roots observe seed pv depth nodes covers nonempty matched
  refine RootEvidence.allMovesContract_proven mc fc board roots observe seed pv depth nodes covers nonempty ?_
  intro order idx coverage inBounds
  obtain ⟨window, observed, running⟩ := matched order idx coverage inBounds
  exact ⟨window, observed.checked, running⟩

end NeuralReversi.PvsParallel.CausalReplay
