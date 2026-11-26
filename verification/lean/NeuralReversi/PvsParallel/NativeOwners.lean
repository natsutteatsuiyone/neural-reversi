import NeuralReversi.PvsParallel.NativeLeafBranches
import NeuralReversi.PvsParallel.NativeSplitFrame
import NeuralReversi.PvsParallel.FrameRows
import NeuralReversi.PvsParallel.NativeFrames.OwnerSites.Admission
import NeuralReversi.PvsParallel.GeneralOwnerMir.Calls
import NeuralReversi.PvsParallel.FrameTableReads.Execution

namespace NeuralReversi.PvsParallel.NativeOwners

open TranspositionTable
open NativeFrames (Row Capture logOf whole)
open FrameReturns (Log Samples ScoreStep)
open FrameTableReads (Table)
open PvsBoardTree (GeneratedSquare)
open TranspositionTable.ProbeRetry (RawAttempt)

structure Ordering (mc : Bitboard.Moves.X86Config) (r : Row) where
  paired : Bool
  checked : Bool
  memory : MoveList.Initialization.Memory Move.Move
  mode : Bitboard.Square → EndgameOrderMir.Heuristic
  tie : GeneratedSquare mc r.frame.position.val → Fin 64

structure Pv (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (r : Row) (cap : Capture pool)
    (identify : Threading.AtomicWord.Event → Nat) (before : Nat → Nat → Prop)
    (history : Threading.CutoffVisibility.History) extends Ordering mc r where
  probe : Option (FrameTableReads.Operations.Execution.ProbeCapture (Cutoffs.key r.frame.position))
  probeSite : let generated := PvEntryMir.generated mc fc r.frame.position paired memory
    probe.isSome = decide (PvEntryMir.dispatch r.depth.toNat (generated.len.getD 0 != 0)
      (Board.hasLegalMoves mc (PvsBoardTree.pass r.frame.position).val) generated.wipeout.join.isSome = .continueSearch)
  schedule : Nat → Bool × Bool
  split : Option (NativeSplit.Packet mc fc r.frame true r.beta.raw.toInt cap identify before history)

def Pv.hint (p : Pv mc fc r cap identify before history) : Option Bitboard.Square :=
  p.probe.bind fun captured => ReaderMir.bestMove captured.result

def Pv.moves {r : Row} (p : Pv mc fc r cap identify before history) : List (GeneratedSquare mc r.frame.position.val) :=
  EndgameOrderMir.pvOrder mc fc r.frame.position p.mode p.tie p.hint

def Pv.returned (p : Pv mc fc r cap identify before history)
    (seq : ∀ a b, Threading.AtomicWord.SequencedBefore a b → before (identify a) (identify b)) : Option Int :=
  let log := logOf mc fc cap.rows cap.table identify before seq
  let f := r.frame
  PvSerialMir.scheduled mc fc f.position r.depth p.paired p.checked p.memory r.alpha.raw.toInt r.beta.raw.toInt
    (fun b d => log.reply f.id f.use b d true false) p.moves (log.pvReplies f r.depth) p.schedule
    (NativeSplit.pvReply p.split)

/-- Replay the internal PV owner from the child return words, up to its normal
    return or its handoff of the actual move array to the split. -/
theorem Pv.returned_of_calls {cap : Capture pool} (p : Pv mc fc r cap identify before history) (seq)
    (entered : let generated := PvEntryMir.generated mc fc r.frame.position p.paired p.memory
      PvEntryMir.dispatch r.depth.toNat (generated.len.getD 0 != 0)
        (Board.hasLegalMoves mc (PvsBoardTree.pass r.frame.position).val) generated.wipeout.join.isSome = .continueSearch)
    (path : RootOwnerMir.Slice.Path Subtype.val source p.schedule r.depth.toNat rootMinimumSplitDepth r.frame.beta
      0 [] p.moves p.moves.length (RootReplay.start r.frame.alpha) visits physical)
    (captured : NativeFrames.decode pool inputs stream = some cap)
    (issued : NativeFrames.OwnerSites.Execution pool inputs stream r.frame (NativeFrames.OwnerSites.moveSite false)
      (FrameReturns.OwnerCalls.emitted fc r.frame r.depth visits)) :
    p.returned seq = if physical.handoff then NativeSplit.pvReply p.split physical else some physical.snapshot.state.best := by
  have unique : p.moves.Nodup := (EndgameOrderMir.pvOrder_perm _ _ _ _ _ _).nodup_iff.mpr
    (EndgameOrderMir.generated_layout mc r.frame.position).1
  have returns := issued.observed mc fc captured identify before seq
  have replay := path.replays _ (FrameReturns.OwnerCalls.supplies _ _ _ _ (path.distinct unique).1 returns)
  simp only [List.nil_append,List.length_nil] at replay
  simp only [Pv.returned,PvSerialMir.scheduled,entered,← r.lo,← r.hi,replay,bind,Option.bind]

theorem Pv.returned_of_pass {cap : Capture pool} (p : Pv mc fc r cap identify before history) (seq)
    (entered : let generated := PvEntryMir.generated mc fc r.frame.position p.paired p.memory
      PvEntryMir.dispatch r.depth.toNat (generated.len.getD 0 != 0)
        (Board.hasLegalMoves mc (PvsBoardTree.pass r.frame.position).val) generated.wipeout.join.isSome = .pass)
    (parsed : NativeFrames.decode pool inputs stream = some cap)
    (issued : NativeFrames.OwnerSites.Execution pool inputs stream r.frame (fun _ _ => .passChild)
      [FrameReturns.OwnerCalls.passCall r.frame r.depth true word]) : p.returned seq = some (-word) := by
  have reply := FrameReturns.OwnerCalls.pass_reply _ _ _ _ _ (issued.observed mc fc parsed identify before seq)
  simp only [Bool.not_true,Bool.false_and] at reply
  simp only [Pv.returned,PvSerialMir.scheduled,PvEntryMir.run,entered,← r.lo,← r.hi,reply,bind,Option.bind]
  rfl

structure SlotQuery (α : Type) where
  allocation : Nat
  clusters : TranspositionTable.SeqLock.Word
  key : TranspositionTable.SeqLock.Word
  records : List α

abbrev Lookup {log : Log mc fc} (t : Table log) (r : Row) (endian : Bool) :=
  (move : GeneratedSquare mc r.frame.position.val) × SlotQuery
    (FrameTableReads.Packet t r.frame (FrameTableReads.lookupCode endian (PvsBoardTree.play mc fc r.frame.position move).val))

def Lookup.read {log : Log mc fc} {t : Table log} (p : Lookup t r endian) :=
  FrameTableReads.decodedChild t p.2.allocation p.2.clusters p.2.key r.frame endian r.frame.position p.1 p.2.records

def attempts {log : Log mc fc} {t : Table log} {r : Row} (p : Option (NativeLeafBranches.Probe t r endian)) :
    Option (List RawAttempt) :=
  match p with
  | none => some []
  | some data =>
    FrameTableReads.decodeProbe t data.allocation data.clusters data.key r.frame endian (Cutoffs.key r.frame.position) data.rounds

theorem observed {log : Log mc fc} {t : Table log} {r : Row} (p : Option (NativeLeafBranches.Probe t r endian))
    (ok : attempts p = some result) : TableReads.ProbeReadsFrom (Samples log endian r.frame) result := by
  cases p with
  | none => cases ok; intro attempt member; contradiction
  | some packet => exact FrameTableReads.decoded_probe_reads t ok

structure Scout (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (r : Row) (cap : Capture pool)
    (identify before seq) (t : Table (logOf mc fc cap.rows cap.table identify before seq)) (endian : Bool)
    (history : Threading.CutoffVisibility.History) extends Ordering mc r where
  probe : Option (NativeLeafBranches.Probe t r endian)
  lookups : List (Lookup t r endian)
  schedule : GeneralOwnerMir.Schedule (GeneratedSquare mc r.frame.position.val)
  split : Option (NativeSplit.Packet mc fc r.frame false r.beta.raw.toInt cap identify before history)

def Scout.returned {cap : Capture pool} {t : Table (logOf mc fc cap.rows cap.table identify before seq)}
    (p : Scout mc fc r cap identify before seq t endian history) : Option Int := do
  let probed ← attempts p.probe
  let etc ← p.lookups.mapM Lookup.read
  let log := logOf mc fc cap.rows cap.table identify before seq
  let f := r.frame
  GeneralOwnerMir.scheduled mc fc f.position r.depth r.alpha r.beta p.paired p.checked f.cut p.memory probed etc
    (EndgameOrderMir.framePriority mc fc f.position p.mode p.tie) (log.nonPv f.id f.use) p.schedule
    (NativeSplit.scoutReply p.split)

def scoutVisit (alpha : Int) (c : Move × Int) : RootOwnerMir.Slice.Visit Move :=
  ⟨c.1,alpha,some c.2,none⟩

theorem scout_admissible (r : Row) (calls : List (GeneratedSquare mc r.frame.position.val × Int)) :
    ∀ c ∈ FrameReturns.OwnerCalls.emitted fc r.frame r.depth (calls.map (scoutVisit r.alpha.raw.toInt)),
      NativeFrames.OwnerSites.Admissible c := by
  apply NativeFrames.OwnerSites.emitted_admissible
  rintro v mem
  obtain ⟨c,_,rfl⟩ := List.mem_map.mp mem
  exact r.lo ▸ r.window

def Scout.body {cap : Capture pool} {t : Table (logOf mc fc cap.rows cap.table identify before seq)}
    (p : Scout mc fc r cap identify before seq t endian history) (probed : List RawAttempt)
    (first : GeneratedSquare mc r.frame.position.val) (pre : Int) (source : Nat → GeneratedSquare mc r.frame.position.val → Int) :=
  let generated := PvsBoardTree.legalSquares mc r.frame.position.val
  let hint := GeneralNonPvMir.checkedHint mc r.frame.position (GeneralNonPvMir.probeHint r.frame.position probed)
  let ordered := GeneralNonPvMir.order (EndgameOrderMir.framePriority mc fc r.frame.position p.mode p.tie) hint generated
  GeneralOwnerMir.Calls.body p.schedule source r.depth.toNat r.frame.cut ordered first generated.length hint
    r.alpha.raw.toInt r.beta.raw.toInt pre (NativeSplit.scoutReply p.split)

/-- Construct the physical NonPV calls from the presearch word and the return
    register of each subsequent scout, including the zero-visit split. -/
def Scout.issued {cap : Capture pool} {t : Table (logOf mc fc cap.rows cap.table identify before seq)}
    (p : Scout mc fc r cap identify before seq t endian history) (probed : List RawAttempt)
    (first : GeneratedSquare mc r.frame.position.val) (pre : ScoreRefresh.Score)
    (src : Nat → GeneratedSquare mc r.frame.position.val → ScoreRefresh.Score)
    (ctx : NativeFrames.Invocations.Scope pool stream r.frame) (sites : Nat → NativeFrames.OwnerSites.Site)
    (after : List NativeFrames.OwnerSites.SideCall) (others : Nat → Nat → Option NativeFrames.Args)
    (body : let calls := (p.body probed first pre.raw.toInt (fun i m => (src i m).raw.toInt)).calls
      let op := NativeFrames.OwnerSites.moveSite (decide ((PvsBoardTree.legalSquares mc r.frame.position.val).length = 1) ||
        (GeneralNonPvMir.probeHint r.frame.position probed).isSome)
      ctx.body = NativeFrames.OwnerSites.issue op ctx.entry.received sites after 0
        (FrameReturns.OwnerCalls.emitted fc r.frame r.depth (calls.map (scoutVisit r.alpha.raw.toInt)))) :=
  NativeFrames.OwnerSites.Execution.from_instructions ctx _ sites after _ others body
    (NativeFrames.OwnerSites.emitted_fits r _ (by
      rintro v mem
      obtain ⟨c,seen,rfl⟩ := List.mem_map.mp mem
      refine ⟨r.lo ▸ r.window,?_⟩
      have words := GeneralOwnerMir.Calls.body_words p.schedule (fun i m => (src i m).raw.toInt) r.depth.toNat
        r.frame.cut _ first _ _ r.alpha.raw.toInt r.beta.raw.toInt pre.raw.toInt (NativeSplit.scoutReply p.split)
        (fun x => ∃ w : BitVec 32, w.toInt = x) (fun _ _ => ⟨_,rfl⟩) ⟨_,rfl⟩ c seen
      intro pv x got
      cases pv with
      | false => exact Option.some.inj got ▸ words
      | true => contradiction))

/-- The forced/TT child and all subsequent scouts use one direct-return stream.
    The MIR determines which calls occur, including a cutoff before sorting and
    a split before the owner searches any move. -/
theorem Scout.returned_of_calls {cap : Capture pool} {t : Table (logOf mc fc cap.rows cap.table identify before seq)}
    (p : Scout mc fc r cap identify before seq t endian history) (pre : Int) (source)
    {etc : List (TableReads.ChildLookupFrom mc fc r.frame.position
      (Samples (logOf mc fc cap.rows cap.table identify before seq) endian r.frame))}
    (probe : attempts p.probe = some probed) (reads : p.lookups.mapM Lookup.read = some etc)
    (entered : GeneralNonPvMir.entryAction mc fc r.frame.position r.depth r.alpha r.beta
      p.paired p.checked p.memory probed etc = .rest)
    (moves : PvsBoardTree.legalSquares mc r.frame.position.val = first :: rest)
    (legalHint : (GeneralNonPvMir.checkedHint mc r.frame.position (GeneralNonPvMir.probeHint r.frame.position probed)).isSome =
      (GeneralNonPvMir.probeHint r.frame.position probed).isSome)
    (captured : NativeFrames.decode pool inputs stream = some cap)
    (issued : NativeFrames.OwnerSites.Execution pool inputs stream r.frame
      (NativeFrames.OwnerSites.moveSite (decide ((PvsBoardTree.legalSquares mc r.frame.position.val).length = 1) ||
        (GeneralNonPvMir.probeHint r.frame.position probed).isSome))
      (FrameReturns.OwnerCalls.emitted fc r.frame r.depth
        ((p.body probed first pre source).calls.map (scoutVisit r.alpha.raw.toInt)))) :
    p.returned = (p.body probed first pre source).value := by
  let f := r.frame
  let log := logOf mc fc cap.rows cap.table identify before seq
  let generated := PvsBoardTree.legalSquares mc f.position.val
  let hint := GeneralNonPvMir.checkedHint mc f.position (GeneralNonPvMir.probeHint f.position probed)
  let key := EndgameOrderMir.framePriority mc fc f.position p.mode p.tie
  let ordered := GeneralNonPvMir.order key hint generated
  let a := r.alpha.raw.toInt
  let z := r.beta.raw.toInt
  let child := log.nonPv f.id f.use
  let reply := GeneralOwnerMir.scoutReply mc fc f.position r.depth f.cut child a
  let search := GeneralNonPvMir.observe mc fc f.position r.depth f.cut child
  let b := p.body probed first pre source
  have legal (m : GeneratedSquare mc f.position.val) : m ∈ generated := by
    obtain ⟨v,mem,eq⟩ := (PvsBoardTree.legalSquares_complete mc f.position.val m.val).mpr
      ((Board.getMoves_exact mc f.position.val _).mp m.property)
    exact (Subtype.ext eq : v = m) ▸ mem
  have low (m : GeneratedSquare mc f.position.val) := EndgameOrderMir.framePriority_low mc fc f.position p.mode p.tie m
  have perm := GeneralNonPvMir.orderPerm key generated hint (fun m _ => legal m) (fun m _ _ _ _ => low m)
  have head : generated.length = 1 ∨ hint.isSome → ordered.head? = some (hint.getD first) := by
    intro used
    dsimp only [ordered]
    cases h : hint with
    | some m => exact (GeneralNonPvMir.orderHint key generated m (legal m) (fun m _ _ => low m)).2
    | none =>
      have one : generated.length = 1 := used.resolve_right (by simp [h])
      change generated = first :: rest at moves
      cases rest with
      | nil => simp [moves,GeneralNonPvMir.order]
      | cons => simp [moves] at one
  have unique : (b.calls.map Prod.fst).Nodup := GeneralOwnerMir.Calls.body_distinct p.schedule source r.depth.toNat
    f.cut ordered first generated.length hint a z pre _ (perm.nodup_iff.mpr (EndgameOrderMir.generated_layout mc f.position).1) head
  have sameVisits : ((b.calls.map (scoutVisit a)).map RootOwnerMir.Slice.Visit.move).Nodup := by
    simpa only [List.map_map,Function.comp_def,scoutVisit] using unique
  have live := issued.observed mc fc captured identify before seq
  have supplied := FrameReturns.OwnerCalls.supplies log f r.depth _ sameVisits live
  have words (c) (mem : c ∈ b.calls) : reply c.1 = some c.2 :=
    supplied (scoutVisit a c) (List.mem_map.mpr ⟨c,mem,rfl⟩) false c.2 rfl
  have body := GeneralOwnerMir.Calls.body_replays p.schedule source r.depth.toNat f.cut ordered first generated.length
    hint a z pre (NativeSplit.scoutReply p.split) search reply (fun _ => rfl) words
  simp only [Scout.returned,probe,reads,bind,Option.bind,GeneralOwnerMir.scheduled,GeneralNonPvMir.frame,entered]
  rw [moves]
  simp only [guard,legalHint,↓reduceIte,pure]
  change (do
    if (first :: rest).length = 1 || (GeneralNonPvMir.probeHint f.position probed).isSome then let _ ← reply (hint.getD first)
    GeneralNonPvMir.resumeOwner (GeneralOwnerMir.recordedOwner p.schedule r.depth.toNat f.cut search reply
      (NativeSplit.scoutReply p.split) a (GeneralNonPvMir.order key hint (first::rest)))
      (GeneralNonPvMir.presearch (first::rest).length hint first a z (fun m => search m false))) = b.value
  change generated = first :: rest at moves
  rw [← moves,← legalHint]
  exact body

theorem Scout.returned_of_pass {cap : Capture pool} {t : Table (logOf mc fc cap.rows cap.table identify before seq)}
    (p : Scout mc fc r cap identify before seq t endian history)
    {etc : List (TableReads.ChildLookupFrom mc fc r.frame.position
      (Samples (logOf mc fc cap.rows cap.table identify before seq) endian r.frame))}
    (probe : attempts p.probe = some probed) (reads : p.lookups.mapM Lookup.read = some etc)
    (entered : GeneralNonPvMir.entryAction mc fc r.frame.position r.depth r.alpha r.beta
      p.paired p.checked p.memory probed etc = .pass)
    (parsed : NativeFrames.decode pool inputs stream = some cap)
    (issued : NativeFrames.OwnerSites.Execution pool inputs stream r.frame (fun _ _ => .passChild)
      [FrameReturns.OwnerCalls.passCall r.frame r.depth false word]) : p.returned = some (-word) := by
  have reply := FrameReturns.OwnerCalls.pass_reply _ _ _ _ _ (issued.observed mc fc parsed identify before seq)
  simp only [Bool.not_false,Bool.true_and] at reply
  simp only [Scout.returned,probe,reads,bind,Option.bind,GeneralOwnerMir.scheduled,GeneralNonPvMir.frame,entered,
    GeneralNonPvMir.runPrefix]
  unfold GeneralNonPvMir.entryAction at entered
  rw [entered]
  simp only [FrameReturns.Log.nonPv,← r.lo,← r.hi,reply]
  rfl

theorem Scout.valid {cap : Capture pool} {t : Table (logOf mc fc cap.rows cap.table identify before seq)}
    (p : Scout mc fc r cap identify before seq t endian history) (tag : r.frame.pv = false)
    (ok : p.returned = some r.frame.score) :
    ScoreStep (logOf mc fc cap.rows cap.table identify before seq) endian r.frame r.depth r.alpha r.beta := by
  obtain ⟨ps,pr,tail⟩ := Option.bind_eq_some_iff.mp ok
  obtain ⟨etc,_,done⟩ := Option.bind_eq_some_iff.mp tail
  exact NativeSplit.nonPv p.split seq p.paired p.checked p.memory ps etc p.mode p.tie p.schedule tag
    (observed p.probe pr) done

abbrev RootInput (log : Log mc fc) (r : Row) :=
  ∃ initial pv physical, Root.Covers mc r.frame.position (initial.map RootMoveStorage.Row.square) ∧
    0 < initial.length ∧ FrameReturns.RootRows log r.frame r.depth initial 0 r.frame.alpha r.frame.beta pv r.frame.score physical

/-- Root, PV and NonPV frames have different entry points. In particular no
    root call with a restricted suffix can be used as a full-board bound. -/
structure Operands {cap : Capture pool} (t : Table (logOf mc fc cap.rows cap.table identify before seq)) (endian : Bool)
    (history : Threading.CutoffVisibility.History) where
  pv : (r : Row) → Option (Pv mc fc r cap identify before history)
  scout : (r : Row) → Option (Scout mc fc r cap identify before seq t endian history)

def Operands.check {cap : Capture pool} {t : Table (logOf mc fc cap.rows cap.table identify before seq)}
    (p : Operands t endian history) (r : Row) : Bool :=
  if r.rootLine.isSome then r.frame.pv && !r.frame.cut
  else if r.frame.pv then decide ((p.pv r).bind (Pv.returned · seq) = some r.frame.score)
  else decide ((p.scout r).bind Scout.returned = some r.frame.score)

theorem Operands.valid {cap : Capture pool} {t : Table (logOf mc fc cap.rows cap.table identify before seq)}
    (p : Operands t endian history) (r : Row) (first : whole r = true)
    (root : r.rootLine = some 0 → RootInput (logOf mc fc cap.rows cap.table identify before seq) r)
    (ok : p.check r = true) : ScoreStep (logOf mc fc cap.rows cap.table identify before seq) endian r.frame r.depth r.alpha r.beta := by
  cases line : r.rootLine with
  | some idx =>
    obtain ⟨pv,cut⟩ := (by simpa [Operands.check,line] using ok : r.frame.pv = true ∧ r.frame.cut = false)
    have zero : idx = 0 := by simpa [whole,line] using first
    obtain ⟨initial,pvs,physical,covers,nonempty,rows⟩ := root (zero ▸ line)
    obtain ⟨s,source,_⟩ := rows.checked covers
    exact .firstRoot _ s.scores pv cut covers (by simpa using nonempty) (r.lo ▸ r.hi ▸ source)
  | none =>
    cases tag : r.frame.pv with
    | true =>
      obtain ⟨op,_,ran⟩ := Option.bind_eq_some_iff.mp (by simpa [Operands.check,line,tag] using ok :
        (p.pv r).bind (Pv.returned · seq) = some r.frame.score)
      exact NativeSplit.pv op.split seq op.paired op.checked op.memory op.mode op.tie op.hint op.schedule tag ran
    | false =>
      obtain ⟨op,_,ran⟩ := Option.bind_eq_some_iff.mp (by simpa [Operands.check,line,tag] using ok :
        (p.scout r).bind Scout.returned = some r.frame.score)
      exact op.valid tag ran

def larger (r : Row) : Bool := whole r && (r.frame.pv || decide (12 < r.depth.toNat))

theorem branches {out : NativeFrameCache.Checked pool}
    (cache : NativeFrameCache.decode pool mc fc inputs values stream = some out)
    (t : Table (logOf mc fc out.physical.rows out.physical.table identify before seq))
    (small : NativeLeafBranches.Registers) (packets : (r : Row) → Option (NativeLeafBranches.Probe t r endian))
    (leafOK : NativeLeafBranches.checkRows t out stream small packets = true) (op : Operands t endian history)
    (root : ∀ r ∈ out.physical.rows, r.rootLine = some 0 →
      RootInput (logOf mc fc out.physical.rows out.physical.table identify before seq) r)
    (ownerOK : (out.physical.rows.filter larger).all op.check = true) :
    ∀ r ∈ out.physical.rows, whole r = true → NativeFrames.Branch out.physical
      (logOf mc fc out.physical.rows out.physical.table identify before seq) endian r := by
  apply (NativeLeafBranches.branches t out cache small packets leafOK ?_).2
  intro r member whole big
  apply op.valid r whole (root r member)
  apply List.all_eq_true.mp ownerOK
  apply List.mem_filter.mpr
  exact ⟨member,by cases pv : r.frame.pv <;> simp_all [larger]⟩

end NeuralReversi.PvsParallel.NativeOwners
