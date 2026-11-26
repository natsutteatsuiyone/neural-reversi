import NeuralReversi.PvsParallel.NativeFrameCache
import NeuralReversi.PvsParallel.FrameTableReads

namespace NeuralReversi.PvsParallel.NativeLeafBranches

open EndgameSearch.CacheHistory (Pool Input Arm Visit)
open EndgameTreeCache (Store Address)
open ScoreRefresh (Score DiscScore fromDiscDiff toDiscDiff)
open NativeFrames (Capture Row Record)
open EndgameSearch
open FrameTableReads (Table)
open TranspositionTable.ProbeRetry (RawAttempt)

inductive Operand where
  | entry (raw : BitVec 16)
  | disc (raw : DiscScore)
  deriving DecidableEq

def Operand.score (checked : Bool) : Operand → Option Score
  | .entry raw => some ⟨raw.signExtend 32⟩
  | .disc raw => (fromDiscDiff checked raw).toOption

/-- The register returned to the PVS caller and the caller's permission to consume it
    differ when a split has been cut. Rust still returns a numeric register in that case. -/
structure Returned where
  operand : Operand
  useScore : Bool

def Returned.score (checked : Bool) (r : Returned) : Option Int := do
  let v ← r.operand.score checked
  if r.useScore then some v.raw.toInt else none

def result (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (address : Address)
    (i : Input) (history : List Store) : Arm → Option Returned
  | .failed => none
  | .table entry => some ⟨.entry entry.score,true⟩
  | .fixed f => (EndgameSearch.Direct.Mir.discResult f i.target i.bmi2 i.board.val i.empties.links i.empties.parity
      i.checked (toDiscDiff i.alpha).raw).map (fun raw => ⟨.disc raw,true⟩)
  | .shallow =>
    let r := ShallowMir.run i.target i.bmi2 mc fc address i.board i.empties (toDiscDiff i.alpha).raw.toInt history
    some ⟨.disc ⟨BitVec.ofInt 32 r.score⟩,true⟩
  | .deep active =>
    let r := DeepRecursionMir.run i.target i.bmi2 i.paired mc fc address i.board (toDiscDiff i.alpha).raw.toInt
      (DeepLeafMir.pollInput (i.state history) active)
    some ⟨.disc ⟨BitVec.ofInt 32 (r.score.getD DeepRecursionMir.stopped)⟩,!(r.state.polls.cache.latched || i.aborted)⟩

theorem accepted (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (addr : Address)
    (i : Input) (past : List Store) :
    (result mc fc addr i past i.arm).bind (Returned.score i.checked) = (i.run mc fc addr past).score := by
  unfold Input.run
  cases i.arm with
  | failed => rfl
  | table e => exact congrArg some (BitVec.toInt_signExtend_of_le (by decide : 16 ≤ 32))
  | fixed f =>
    simp only [result,Input.eval]
    cases h : EndgameSearch.Direct.Mir.discResult f i.target i.bmi2 i.board.val i.empties.links i.empties.parity i.checked
        (toDiscDiff i.alpha).raw with
    | none => rfl
    | some raw =>
      change ((fromDiscDiff i.checked raw).toOption.bind (fun s => some s.raw.toInt)) =
        (fromDiscDiff i.checked raw).toOption.map (fun s => s.raw.toInt)
      cases h' : (fromDiscDiff i.checked raw).toOption <;> rfl
  | shallow =>
    let raw : DiscScore := ⟨BitVec.ofInt 32 (ShallowMir.run i.target i.bmi2 mc fc addr i.board i.empties
      (toDiscDiff i.alpha).raw.toInt past).score⟩
    change ((fromDiscDiff i.checked raw).toOption.bind (fun s => some s.raw.toInt)) =
      (fromDiscDiff i.checked raw).toOption.map (fun s => s.raw.toInt)
    cases (fromDiscDiff i.checked raw).toOption <;> rfl
  | deep active =>
    let child := DeepRecursionMir.run i.target i.bmi2 i.paired mc fc addr i.board (toDiscDiff i.alpha).raw.toInt
      (DeepLeafMir.pollInput (i.state past) active)
    change (do
      let v ← (fromDiscDiff i.checked ⟨BitVec.ofInt 32 (child.score.getD 0)⟩).toOption
      if !(child.state.polls.cache.latched || i.aborted) then some v.raw.toInt else none) =
        (do let v ← (fromDiscDiff i.checked ⟨BitVec.ofInt 32 (child.score.getD 0)⟩).toOption
            let saved ← EndgameSearch.Cancellation.publish child i.aborted v
            return saved.raw.toInt)
    cases h : (fromDiscDiff i.checked ⟨BitVec.ofInt 32 (child.score.getD 0)⟩).toOption with
    | none => rfl
    | some v =>
      change (if !(child.state.polls.cache.latched || i.aborted) then some v.raw.toInt else none) =
        (EndgameSearch.Cancellation.publish child i.aborted v).bind (fun s => some s.raw.toInt)
      unfold EndgameSearch.Cancellation.publish
      cases (child.state.polls.cache.latched || i.aborted) <;> rfl

structure ChildReturn where
  opening : Nat
  closed : Nat
  site : Nat
  callee : CacheOwner.Node

private structure OpenCall where
  opening : Nat
  site : Nat
  callee : CacheOwner.Node

private def walk : List OpenCall × List ChildReturn → CacheHistory.NativeTrace.Raw →
    Option (List OpenCall × List ChildReturn)
  | (stack,rows), .enter step site callee .. => some (⟨step,site,callee⟩ :: stack,rows)
  | (top :: rest,rows), .leave step opening callee _ =>
    if opening != top.opening || callee != top.callee then none else
    some (rest,if rest.length = 1 then ⟨opening,step,top.site,callee⟩ :: rows else rows)
  | ([],_), .leave .. => none

def children (stream : List Record) (worker first last : Nat) : Option (List ChildReturn) := do
  let raw := CacheHistory.NativeTrace.workerEvents worker (NativeFrameCache.project stream)
  let scope := raw.filter fun event => decide (first ≤ event.step ∧ event.step ≤ last)
  match scope.head?,scope.getLast? with
  | some (.enter start site .boundary ..), some (.leave finish opening .boundary _) =>
    if start ≠ first || finish ≠ last || opening ≠ first || site ≠ DeepLeafMir.boundaryPvsSite then none else
    let (remaining,found) ← scope.foldlM walk ([],[])
    if remaining.isEmpty then some found.reverse else none
  | _,_ => none

structure BoundaryRegisters where
  value : Score
  beforeScale : Option DiscScore

structure Registers where
  boundary : Nat → Nat → Option BoundaryRegisters
  callee : Nat → Nat → Option DiscScore

/-- A terminal or solve1–4 result remains in the boundary's disc register. The two
    recursive paths additionally read the register of the matched direct child return. -/
def operand (regs : Registers) (worker : Nat) (args : BoundaryRegisters) (sites : List ChildReturn) : Arm → Option Operand
  | .failed => none
  | .table e => if sites.isEmpty && args.beforeScale.isNone then some (.entry e.score) else none
  | .fixed _ => do
    let [access] := sites | none
    if access.site != 0 || access.callee != .access then none else args.beforeScale.map Operand.disc
  | .shallow | .deep _ => do
    let [access,child] := sites | none
    if access.site != 0 || access.callee != .access then none else
    let disc ← regs.callee worker child.closed
    if args.beforeScale = some disc then some (.disc disc) else none

def siteKinds (sites : List ChildReturn) := sites.map (fun x => (x.site,x.callee))

structure BoundaryResult where
  raw : Score
  useScore : Bool
  deriving DecidableEq

def replay (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (addr : Address)
    (i : Input) (history : List Store) (actual : Operand) (returned : Score) : Option BoundaryResult := do
  let produced ← result mc fc addr i history i.arm
  if produced.operand = actual ∧ actual.score i.checked = some returned then some ⟨returned,produced.useScore⟩ else none

theorem replay_score (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (addr : Address) (i : Input)
    (past : List Store) (hit : replay mc fc addr i past actual returned = some answer) :
    (i.run mc fc addr past).score = if answer.useScore then some answer.raw.raw.toInt else none := by
  have source := accepted mc fc addr i past
  unfold replay at hit
  obtain ⟨r,produced,ok⟩ := Option.bind_eq_some_iff.mp hit
  split at ok <;> try contradiction
  rename_i same
  cases ok
  rw [← source,produced]
  simp [Returned.score,same.1,same.2]

def returned {p : Pool} (c : Capture p) (stream : List Record) (regs : Registers)
    (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (v : Visit p) : Option BoundaryResult := do
  let actual ← regs.boundary v.worker v.last
  let sites ← children stream v.worker v.first v.last
  if CacheHistory.NativeTrace.expectedSites v.input.arm != some (siteKinds sites) then none else
  let raw ← operand regs v.worker actual sites v.input.arm
  replay mc fc p.address v.input (c.timeline.readAt mc fc v.worker v.first).history raw actual.value

theorem returned_score {p : Pool} {c : Capture p} {stream regs mc fc} {v : Visit p}
    (hit : returned c stream regs mc fc v = some answer) :
    (v.input.run mc fc p.address (c.timeline.readAt mc fc v.worker v.first).history).score =
      if answer.useScore then some answer.raw.raw.toInt else none := by
  unfold returned at hit
  obtain ⟨_,_,rest⟩ := Option.bind_eq_some_iff.mp hit
  obtain ⟨_,_,more⟩ := Option.bind_eq_some_iff.mp rest
  split at more <;> try contradiction
  obtain ⟨_,_,ran⟩ := Option.bind_eq_some_iff.mp more
  exact replay_score mc fc p.address v.input _ ran

inductive TailEvent where
  | boundary (step opening scoreAt : Nat)
  | scored (step : Nat)
  | pvs (step opening : Nat)

/-- Project each tail event at its original place in the worker stream. The physical
    result word is retained by the register capture even when no scored row is emitted. -/
def BoundaryResult.event (r : BoundaryResult) : TailEvent → Option NativeFrames.Event
  | .boundary t opened scoredAt =>
    some (.leave t opened .boundary (if r.useScore then some scoredAt else none) none)
  | .scored t => if r.useScore then some (.scored t r.raw) else none
  | .pvs t opened => some (if r.useScore then .leave t opened .pvs none (some r.raw) else .discarded t opened r.raw)

theorem discarded {p : Pool} {c : Capture p} {stream regs mc fc} {v : Visit p}
    (hit : returned c stream regs mc fc v = some answer) (stopped : answer.useScore = false) :
    (v.input.run mc fc p.address (c.timeline.readAt mc fc v.worker v.first).history).score = none ∧
    answer.event (.scored scoreAt) = none ∧
    answer.event (.boundary v.last v.first scoreAt) = some (.leave v.last v.first .boundary none none) ∧
    answer.event (.pvs finished opened) = some (.discarded finished opened answer.raw) := by
  exact ⟨by simpa [stopped] using returned_score hit,by simp [BoundaryResult.event,stopped],
    by simp [BoundaryResult.event,stopped],by simp [BoundaryResult.event,stopped]⟩

private instance : DecidableEq RawAttempt := fun x y =>
  decidable_of_iff (x.generation = y.generation ∧ ∀ i, x.readSlot i = y.readSlot i) (by
    cases x; cases y; simp [funext_iff,and_comm])

structure Probe {log : FrameReturns.Log mc fc} (t : Table log) (r : Row) (endian : Bool) where
  allocation : Nat
  clusters : TranspositionTable.SeqLock.Word
  key : TranspositionTable.SeqLock.Word
  rounds : List (BitVec 8 × List (FrameTableReads.Packet t r.frame (FrameTableReads.snapCode endian)))

def probe {log : FrameReturns.Log mc fc} {t : Table log} {r : Row} (i : Input) (packet : Option (Probe t r endian)) : Bool :=
  if i.depth.toNat = 12 ∧ i.parallel then
    match packet with
    | none => false
    | some p => decide (FrameTableReads.decodeProbe t p.allocation p.clusters p.key r.frame endian (Cutoffs.key i.board) p.rounds =
        some i.attempts)
  else decide (packet.isNone = true ∧ i.attempts = [])

theorem probe_reads {log : FrameReturns.Log mc fc} {t : Table log} {r : Row} (i : Input)
    {packet : Option (Probe t r endian)} (ok : probe i packet = true) :
    TableReads.ProbeReadsFrom (FrameReturns.Samples log endian r.frame) i.attempts := by
  unfold probe at ok
  split at ok
  · cases packet with
    | none => contradiction
    | some p => exact FrameTableReads.decoded_probe_reads t (of_decide_eq_true ok)
  · obtain ⟨_,noProbe⟩ := of_decide_eq_true ok
    simp [noProbe,TableReads.ProbeReadsFrom]

def scored (stream : List Record) (r : Row) : Bool := stream.any fun
  | .localCall w (.scored atStep word) => decide (w = r.frame.worker ∧ atStep = r.frame.scoreStep ∧ word = r.rawScore)
  | _ => false

def check {pool : Pool} {log : FrameReturns.Log mc fc} (t : Table log) (out : NativeFrameCache.Checked pool)
    (stream : List Record) (regs : Registers) (r : Row) (packet : Option (Probe t r endian)) : Bool :=
  match NativeFrames.cacheSite out.physical r with
  | none => false
  | some site => match returned out.physical stream regs mc fc site.visit with
    | none => false
    | some answer => answer.useScore && decide (answer.raw = r.rawScore) && scored stream r && probe site.visit.input packet

theorem branch {pool : Pool} {log : FrameReturns.Log mc fc} (t : Table log) (out : NativeFrameCache.Checked pool)
    (stream : List Record) (regs : Registers) (r : Row) (packet : Option (Probe t r endian))
    (small : r.frame.pv = false ∧ r.depth.toNat ≤ 12) (ok : check t out stream regs r packet = true) :
    NativeFrames.Branch out.physical log endian r := by
  unfold NativeFrames.Branch
  rw [ite_eq_left small]
  unfold check at ok
  cases found : NativeFrames.cacheSite out.physical r with
  | none => simp [found] at ok
  | some site =>
    cases ran : returned out.physical stream regs mc fc site.visit with
    | none => simp [found,ran] at ok
    | some answer =>
      obtain ⟨⟨⟨used,same⟩,_⟩,read⟩ := (by simpa only [found,ran,Bool.and_eq_true,decide_eq_true_eq] using ok :
        ((answer.useScore = true ∧ answer.raw = r.rawScore) ∧ _) ∧ probe site.visit.input packet = true)
      refine ⟨fun _ => probe_reads site.visit.input read,?_⟩
      have hs := returned_score ran
      simpa [site.worker,used,same,←r.value] using hs

def checkRows {pool : Pool} {log : FrameReturns.Log mc fc} (t : Table log) (out : NativeFrameCache.Checked pool)
    (stream : List Record) (regs : Registers) (packets : (r : Row) → Option (Probe t r endian)) : Bool :=
  out.physical.rows.all fun r => if NativeFrames.whole r ∧ r.frame.pv = false ∧ r.depth.toNat ≤ 12 then
    check t out stream regs r (packets r) else true

theorem branches {pool : Pool} {log : FrameReturns.Log mc fc} (t : Table log) (out : NativeFrameCache.Checked pool)
    (captured : NativeFrameCache.decode pool mc fc inputs values stream = some out)
    (regs : Registers) (packets : (r : Row) → Option (Probe t r endian))
    (checked : checkRows t out stream regs packets = true)
    (larger : ∀ r ∈ out.physical.rows, NativeFrames.whole r = true → ¬(r.frame.pv = false ∧ r.depth.toNat ≤ 12) →
      FrameReturns.ScoreStep log endian r.frame r.depth r.alpha r.beta) :
    NativeFrames.decode pool inputs stream = some out.physical ∧
      ∀ r ∈ out.physical.rows, NativeFrames.whole r = true → NativeFrames.Branch out.physical log endian r := by
  refine ⟨NativeFrameCache.decode_physical captured,?_⟩
  intro r member whole
  by_cases small : r.frame.pv = false ∧ r.depth.toNat ≤ 12
  · have each := List.all_eq_true.mp checked r member
    apply branch t out stream regs r (packets r) small
    simpa [whole,small.1,small.2] using each
  · simpa [NativeFrames.Branch,small] using larger r member whole small

end NeuralReversi.PvsParallel.NativeLeafBranches
