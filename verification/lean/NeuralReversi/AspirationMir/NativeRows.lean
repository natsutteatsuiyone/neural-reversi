module
public import NeuralReversi.AspirationMir.NativeTrials
public import NeuralReversi.PvsParallel.RootStorageMir.Rows

@[expose] public section

namespace NeuralReversi.AspirationMir.NativeRows

open PvsParallel NativeFrames
open RootStorageMir (Branch Packet Slots Stored)
open EndgameSearch.CacheOwner (Node)

inductive Mutation where
  | reset (index : Nat) (original : List RootMoveStorage.Row)
  | sort (index : Nat)
  | update (square : Bitboard.Square) (value : BitVec 32) (visible : Bool) (slots : Slots)

inductive Sample where
  | engine (record : Record)
  | locked (worker tick receiver : Nat) (action : Mutation)

def engine (samples : List Sample) : List Record :=
  samples.filterMap fun | .engine r => some r | _ => none

inductive Phase where | before | active | scored | done deriving DecidableEq

structure Scan where
  stack : Nat → List (Nat × Node) := fun _ => []
  tick : Nat → Option Nat := fun _ => none
  phase : Phase := .before
  index : Option (Nat × Nat × List RootMoveStorage.Row) := none
  reversed : List Packet := []

def put (f : Nat → A) (n : Nat) (a : A) (i : Nat) := if i = n then a else f i

def Scan.atTick (s : Scan) (worker t : Nat) : Option Scan :=
  if (s.tick worker).all (· < t) then some {s with tick := put s.tick worker (some t)} else none

structure CallSite where
  worker : Nat
  opening : Nat
  step : Nat
  returned : Nat
  score : ScoreRefresh.Score

def Scan.advance (call : CallSite) (returnAt : Option Nat) (receiver : Nat) (s : Scan) : Sample → Option Scan
  | .engine (.table _) => some s
  | .engine (.localCall w (.entry t _ node ..)) => do
    let s ← s.atTick w t
    let next := {s with stack := put s.stack w ((t,node) :: s.stack w)}
    if w = call.worker ∧ t = call.opening then
      if s.phase == .before && node == .pvs then some {next with phase := .active} else none
    else some next
  | .engine (.localCall w (.scored t value)) => do
    let s ← s.atTick w t
    if w = call.worker ∧ t = call.step then
      if s.phase == .active && (s.stack w).head? == some (call.opening,.pvs) && value.raw == call.score.raw then
        some {s with phase := .scored} else none
    else some s
  | .engine (.localCall w (.leave t opening node _ normal)) => do
    let s ← s.atTick w t
    let top :: rest := s.stack w | none
    if top != (opening,node) then none else
    let next := {s with stack := put s.stack w rest}
    if w = call.worker ∧ opening = call.opening then
      if s.phase == .scored && node == .pvs && normal == some call.score && returnAt == some t then
        some {next with phase := .done} else none
    else some next
  | .engine (.localCall w (.discarded t opening _)) => do
    let s ← s.atTick w t
    let top :: rest := s.stack w | none
    if top != (opening,.pvs) || (w == call.worker && opening == call.opening) then none
    else some {s with stack := put s.stack w rest}
  | .locked w t object mutation => do
    let s ← s.atTick w t
    if object != receiver || s.phase == .before || s.phase == .done then some s else
    if s.phase != .active then none else
    let top ← (s.stack w).head?
    let owner := w == call.worker && top == (call.opening,.pvs)
    match mutation with
    | .sort _ => none
    | .reset idx original =>
      if owner && s.index.isNone then some {s with index := some (idx,t,original)} else none
    | .update sq val shown slots =>
      if s.index.isSome && (owner || top.2 == .splitSearch) then
        some {s with reversed := ⟨w,t,sq,val,shown,slots⟩ :: s.reversed} else none

/-- One root-vector lifetime in the common native trace. The selected owner must perform
    the reset. Delegated writes must be in a split-worker body on the same vector. -/
structure Locks where
  index : Nat
  resetAt : Nat
  changes : List Packet
  original : List RootMoveStorage.Row

def Locks.steps (w : Nat) (locks : Locks) : List (Nat × Nat) :=
  (w,locks.resetAt) :: locks.changes.map fun p => (p.worker,p.tick)

def scanLocks (call : CallSite) (returnAt : Option Nat) (receiver : Nat) (samples : List Sample) : Option Locks := do
  let final ← samples.foldlM (Scan.advance call returnAt receiver) {}
  if final.phase != (if returnAt.isSome then .done else .scored) then none else
    let (idx,t,original) ← final.index
    return ⟨idx,t,final.reversed.reverse,original⟩

def collectAt (call : CallSite) (receiver : Nat) (samples : List Sample) : Option Locks :=
  scanLocks call (some call.returned) receiver samples

def collect (row : Row) (receiver : Nat) (samples : List Sample) : Option Locks := do
  scanLocks ⟨row.frame.worker,row.opening,row.frame.scoreStep,row.frame.returnStep.getD 0,row.rawScore⟩
    row.frame.returnStep receiver samples

def fields (r : RootMoveStorage.Row) := (r.square,r.score,r.average,r.meanSquared,r.pv)

def read {log : FrameReturns.Log mc fc} (row : Row) (branch : Branch log row.frame row.depth initial idx)
    (receiver : Nat) (samples : List Sample) : Option Stored := do
  let locks ← collect row receiver samples
  if locks.original.map fields = initial.map fields then RootStorageMir.replay branch locks.index locks.changes else none

theorem physical {log : FrameReturns.Log mc fc} (row : Row) (branch : Branch log row.frame row.depth initial idx)
    (ok : read row branch receiver samples = some out) :
    FrameReturns.RootRows log row.frame row.depth initial idx row.frame.alpha row.frame.beta
      (RootStorageMir.line out.packets) row.frame.score out.rows := by
  obtain ⟨_,_,replayed⟩ := Option.bind_eq_some_iff.mp ok
  split at replayed <;> try contradiction
  exact RootStorageMir.rows_of_replay branch replayed

theorem publicScore {out : NativeFrameCache.Checked pool}
    (cache : NativeFrameCache.decode pool mc fc inputs values (engine samples) = some out)
    (identify before sequenced) (irrefl : ∀ i, ¬before i i)
    (trans : ∀ {i j k}, before i j → before j k → before i k)
    (scores : let c := out.physical
      let log := NativeFrames.logOf mc fc c.rows c.table identify before sequenced
      ∀ row ∈ c.rows, NativeFrames.whole row = true → NativeFrames.Branch c log endian row)
    (stores : let c := out.physical
      let log := NativeFrames.logOf mc fc c.rows c.table identify before sequenced
      ∀ e ∈ c.table, FrameReturns.WriteStep log endian e)
    (b : PvsBoardTree.Position) (w solve opening : Nat) {trial : NativeTrials.Line out.physical b w 0}
    (search : NativeTrials.decode out.physical inputs (engine samples) b w 0 solve opening = some trial)
    (initial : List RootMoveStorage.Row)
    (covers : Root.Covers mc b (initial.map RootMoveStorage.Row.square)) (positive : 0 < initial.length)
    (branch : RootStorageMir.Branch (NativeFrames.logOf mc fc out.physical.rows out.physical.table identify before sequenced)
      trial.result.row.frame trial.result.row.depth initial 0)
    (stored : Stored) (fetched : read trial.result.row branch receiver samples = some stored)
    (sort : PublicationMir.SliceSort) (minimum nodes : Nat) (reportable : minimum ≤ 5)
    (old : Option PublicationMir.Iteration) (fallback : Unit → PublicationMir.SearchResult) :
    ∃ (t : NativeTrials.Trial) (result : PublicationMir.SearchResult),
      (NativeTrials.collect w solve opening (engine samples)).bind (fun ts => ts[trial.result.last.index]?) = some t ∧
      t.opening = trial.result.row.opening ∧
      OuterMir.finish minimum 5 sort stored.rows trial.result.row.depth.toNat nodes old fallback = some result ∧
      PublicationMir.score result = some t.score.toInt ∧ t.score.toInt = PvsBoardTree.minimax mc fc b :=
  NativeTrials.publicScore cache identify before sequenced irrefl trans scores stores b w solve opening search
    initial stored.rows (RootStorageMir.line stored.packets) covers positive (physical trial.result.row branch fetched)
    sort minimum nodes reportable old fallback

abbrev RootBranches (log : FrameReturns.Log mc fc) :=
  (row : Row) → (initial : List RootMoveStorage.Row) → Option (Branch log row.frame row.depth initial 0)

/-- Check every physical exact root frame for line zero, including a failed window and
    a scored frame awaiting its normal return. The original vector is read under its reset lock. -/
def rootCheck {log : FrameReturns.Log mc fc} (supply : RootBranches log) (receiver : Row → Nat)
    (samples : List Sample) (row : Row) : Bool := Id.run do
  if row.rootLine != some 0 then return true
  let some locks := collect row (receiver row) samples | return false
  let some branch := supply row locks.original | return false
  return 0 < locks.original.length && decide (Root.Covers mc row.frame.position (locks.original.map RootMoveStorage.Row.square)) &&
    (RootStorageMir.replay branch locks.index locks.changes).isSome

theorem rootInputs {log : FrameReturns.Log mc fc} {rows : List Row} (supply : RootBranches log) (receiver : Row → Nat)
    (ok : rows.all (rootCheck supply receiver samples) = true) (row : Row) (mem : row ∈ rows) (zero : row.rootLine = some 0) :
    ∃ initial pv physical, Root.Covers mc row.frame.position (initial.map RootMoveStorage.Row.square) ∧ 0 < initial.length ∧
      FrameReturns.RootRows log row.frame row.depth initial 0 row.frame.alpha row.frame.beta pv row.frame.score physical := by
  have check := List.all_eq_true.mp ok row mem
  cases got : collect row (receiver row) samples with
  | none => simp [rootCheck,zero,got] at check
  | some locks => cases supplied : supply row locks.original with
    | none => simp [rootCheck,zero,got,supplied] at check
    | some branch =>
      change rootCheck supply receiver samples row = true at check
      simp [rootCheck,zero,got,supplied] at check
      obtain ⟨⟨len,cover⟩,present⟩ := check
      obtain ⟨_,replayed⟩ := Option.isSome_iff_exists.mp present
      exact ⟨_,_,_,cover,len,RootStorageMir.rows_of_replay branch replayed⟩


end NeuralReversi.AspirationMir.NativeRows
