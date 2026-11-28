module
public import NeuralReversi.EndgameSearch.CacheHistory.NativeTrace
public import NeuralReversi.PvsParallel.FrameWrites
public import NeuralReversi.PvsParallel.FrameCache

@[expose] public section

namespace NeuralReversi.PvsParallel.NativeFrames

open EndgameSearch.CacheHistory (Pool)
open EndgameSearch.CacheHistory.NativeTrace (Worker Cursor Raw)
open FrameReturns (Frame Log ScoreStep WriteStep Decoded)

structure Args where
  board : Board.Board
  depth : TranspositionTable.Depth
  alpha : ScoreRefresh.Score
  beta : ScoreRefresh.Score
  endgame : Bool
  selectivity : TranspositionTable.Selectivity
  pv : Bool
  cut : Bool
  rootLine : Option Nat

def Args.exact (args : Args) : Bool := args.endgame && decide (args.selectivity = Cutoffs.exactSelectivity)

structure Row where
  opening : Nat
  rootLine : Option Nat
  frame : Frame
  depth : TranspositionTable.Depth
  alpha : ScoreRefresh.Score
  beta : ScoreRefresh.Score
  rawScore : ScoreRefresh.Score
  full : ExactDepth.Full frame.position depth
  atDepth : frame.depth = depth.toNat
  lo : frame.alpha = alpha.raw.toInt
  hi : frame.beta = beta.raw.toInt
  value : frame.score = rawScore.raw.toInt
  window : Aspiration.Window frame.alpha frame.beta
  narrow : frame.pv = false → frame.beta = frame.alpha+1

def frameKey (size worker opening : Nat) : Nat := opening * size + worker + 1

theorem frameKey_unique {size i j s t : Nat} (left : i < size) (right : j < size)
    (same : frameKey size i s = frameKey size j t) : i = j ∧ s = t := by
  have eq : s * size + i = t * size + j := by simpa [frameKey] using same
  have workers := congrArg (· % size) eq
  simp [Nat.add_mod, Nat.mod_eq_of_lt left, Nat.mod_eq_of_lt right] at workers
  subst j
  have mul := Nat.add_right_cancel eq
  exact ⟨rfl, Nat.eq_of_mul_eq_mul_right (by omega : 0 < size) mul⟩

def Args.row (args : Args) (size worker opening parent step : Nat) (value : ScoreRefresh.Score) : Option Row := do
  if wf : args.board.player &&& args.board.opponent = 0 then
    let b : PvsBoardTree.Position := ⟨args.board,wf⟩
    let a := args.alpha.raw.toInt
    let z := args.beta.raw.toInt
    if h : args.depth = Board.emptyCount b.val ∧ -32767 ≤ a ∧ a < z ∧ z ≤ 32767 ∧
        (args.pv = false → z = a+1) then
      let f : Frame := {
        parent, worker, depth := args.depth.toNat, alpha := a, beta := z, pv := args.pv,
        player := args.board.player.toNat, opponent := args.board.opponent.toNat, score := value.raw.toInt,
        id := frameKey size worker opening, cut := args.cut, scoreStep := step, returnStep := none,
        scoreBeforeReturn := by simp, boardValid := by simpa [Board.wellFormed,TraceCheck.FrameReturn.board] using wf }
      return ⟨opening,args.rootLine,f,args.depth,args.alpha,args.beta,value,
        by simpa [f,ExactDepth.Full,Frame.position,TraceCheck.FrameReturn.board,b] using h.1,
        rfl,rfl,rfl,rfl,⟨h.2.1,h.2.2.1,h.2.2.2.1⟩,h.2.2.2.2⟩
    else none
  else none

def Row.finish (r : Row) (time : Nat) (returned : ScoreRefresh.Score) : Option Row := do
  if later : r.frame.scoreStep < time then
    if r.frame.returnStep.isNone && decide (r.frame.score = returned.raw.toInt) then
      have ordered : ∀ n, some time = some n → r.frame.scoreStep < n :=
        fun n eq => (Option.some.inj eq : time = n) ▸ later
      let f : Frame := {r.frame with returnStep := some time, scoreBeforeReturn := ordered}
      have pos : f.position = r.frame.position := rfl
      return ⟨r.opening,r.rootLine,f,r.depth,r.alpha,r.beta,r.rawScore,pos.symm ▸ r.full,
        r.atDepth,r.lo,r.hi,r.value,r.window,r.narrow⟩
    else none
  else none

inductive Event where
  | entry (step site : Nat) (callee : EndgameSearch.CacheOwner.Node) (received : EndgameSearch.CacheOwner.Resource)
      (args : Option EndgameSearch.CacheHistory.NativeTrace.Operands)
  | scored (step : Nat) (score : ScoreRefresh.Score)
  | leave (step opening : Nat) (callee : EndgameSearch.CacheOwner.Node) (boundaryScore : Option Nat)
      (normal : Option ScoreRefresh.Score)
  | discarded (step opening : Nat) (raw : ScoreRefresh.Score)

def Event.step : Event → Nat
  | .entry t .. | .scored t .. | .leave t .. | .discarded t .. => t

def Event.edge? : Event → Option Raw
  | .entry t site node r args => some (.enter t site node r args)
  | .scored .. => none
  | .leave t first node stamp _ => some (.leave t first node stamp)
  | .discarded t first _ => some (.leave t first .pvs none)

structure State (w : Worker p) where
  cursor : Cursor w
  rows : List Row

def finishRow (opening time : Nat) (score : ScoreRefresh.Score) : List Row → Option (List Row)
  | [] => none
  | r :: rs => if r.opening = opening then (r.finish time score).map (· :: rs)
    else (finishRow opening time score rs).map (r :: ·)

def State.advance {p : Pool} {w : Worker p} (s : State w) (inputs : Nat → Option Args) : Event → Option (State w)
  | .entry t site node rec args => do
    if (inputs t).isSome != (node == .pvs) then none else
      let next ← s.cursor.advance (.enter t site node rec args)
      if node = .boundary then
        let op ← args
        let size := (EndgameSearch.CacheOwner.poolPlan p.size).length
        if s.cursor.parent.node != .pvs || op.frame != frameKey size w.ordinal s.cursor.parent.first then none
        else return ⟨next,s.rows⟩
      else return ⟨next,s.rows⟩
  | .scored t score => do
    let top :: rest := s.cursor.stack | none
    if s.cursor.clock > t || top.node != .pvs then none else
      let args ← inputs top.first
      let next := {s.cursor with clock := t+1}
      if !args.exact then return ⟨next,s.rows⟩
      if s.rows.any (fun r => r.opening == top.first) then none else
        let size := (EndgameSearch.CacheOwner.poolPlan p.size).length
        let parent := match rest with
          | caller :: _ => if caller.node == .pvs then frameKey size w.ordinal caller.first else 0
          | [] => 0
        let row ← args.row size w.ordinal top.first parent t score
        return ⟨next,row :: s.rows⟩
  | .leave t opening node stamp normal => do
    let next ← s.cursor.advance (.leave t opening node stamp)
    if node = .pvs then
      let args ← inputs opening
      match normal with
      | some score => if args.exact then return ⟨next,← finishRow opening t score s.rows⟩ else return ⟨next,s.rows⟩
      | none => return ⟨next,s.rows⟩
    else if normal.isSome then none else return ⟨next,s.rows⟩
  | .discarded t opening _ => do
    let _ ← inputs opening
    if s.rows.any (fun r => r.opening = opening) then none else
    return ⟨← s.cursor.advance (.leave t opening .pvs none),s.rows⟩

def run (w : Worker p) (inputs : Nat → Option Args) (stream : List Event) : Option (State w) :=
  stream.foldlM (fun s e => s.advance inputs e) ⟨Cursor.initial w,[]⟩

/-- Every score can be used by its own writer, even if the execution prefix ends before it
    returns. Restricted root-line scores only contribute to that line's published rows. -/
def whole (r : Row) : Bool := r.rootLine.getD 0 == 0

def logOf (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (rows : List Row)
    (writes : List ConcurrentReplay.Event) (identify : Threading.AtomicWord.Event → Nat)
    (before : Nat → Nat → Prop)
    (sequenced : ∀ a b, Threading.AtomicWord.SequencedBefore a b → before (identify a) (identify b)) : Log mc fc :=
  ⟨(rows.filter whole).map Row.frame,writes,identify,before,sequenced⟩

theorem decoded {mc fc} (rows : List Row) (writes : List ConcurrentReplay.Event) (identify before sequenced)
    (irrefl : ∀ i, ¬before i i) (trans : ∀ {i j k}, before i j → before j k → before i k)
    (scores : let log := logOf mc fc rows writes identify before sequenced
      ∀ row ∈ rows, whole row = true → ScoreStep log endian row.frame row.depth row.alpha row.beta)
    (stores : let log := logOf mc fc rows writes identify before sequenced
      ∀ e ∈ writes, WriteStep log endian e) : Decoded (logOf mc fc rows writes identify before sequenced) endian := by
  apply FrameReturns.decodedOfSteps _ irrefl trans _ stores
  intro f member
  obtain ⟨r,kept,rfl⟩ := List.mem_map.mp member
  obtain ⟨present,eligible⟩ := List.mem_filter.mp kept
  exact ⟨r.depth,r.alpha,r.beta,r.full,r.atDepth,r.lo,r.hi,r.window,r.narrow,scores r present eligible⟩

inductive Record where
  | localCall (worker : Nat) (event : Event)
  | table (event : ConcurrentReplay.Event)

structure Capture (p : Pool) where
  rows : List Row
  table : List ConcurrentReplay.Event
  timeline : EndgameSearch.CacheHistory.Timeline p
  inputs : Nat → Nat → Option Args
  stream : List Record

def workerRun (p : Pool) (inputs : Nat → Nat → Option Args) (stream : List Record) (worker : Nat) :
    Option ((w : Worker p) × State w) := do
  let w ← EndgameSearch.CacheHistory.NativeTrace.workerAt p worker
  let localCalls := stream.filterMap fun
    | .localCall n e => if worker = n then some e else none
    | .table _ => none
  return ⟨w,← run w (inputs worker) localCalls⟩

def tableEvents (stream : List Record) := stream.filterMap fun
  | .localCall .. => none
  | .table event => some event

def decode (p : Pool) (inputs : Nat → Nat → Option Args) (stream : List Record) : Option (Capture p) := do
  let size := (EndgameSearch.CacheOwner.poolPlan p.size).length
  if !(stream.all fun | .localCall n _ => decide (n < size) | .table _ => true) then none else
    let workers ← (List.range size).mapM (workerRun p inputs stream)
    let visits := workers.flatMap fun w => w.2.cursor.completed
    if ord : visits.Pairwise EndgameSearch.CacheHistory.Earlier then
      return ⟨workers.flatMap (fun w => w.2.rows),tableEvents stream,⟨visits,ord⟩,inputs,stream⟩
    else none

theorem decode_keeps (ok : decode p inputs stream = some c) :
    c.table = tableEvents stream ∧ ∃ ws, (List.range (EndgameSearch.CacheOwner.poolPlan p.size).length).mapM
        (workerRun p inputs stream) = some ws ∧ c.rows = ws.flatMap (fun w => w.2.rows) ∧
        c.timeline.visits = ws.flatMap (fun w => w.2.cursor.completed) := by
  unfold decode at ok
  dsimp only at ok
  split at ok <;> try contradiction
  cases workers : (List.range (EndgameSearch.CacheOwner.poolPlan p.size).length).mapM
      (workerRun p inputs stream) with
  | none => simp [workers] at ok
  | some ws =>
    simp [workers] at ok
    obtain ⟨_,rfl⟩ := ok
    exact ⟨rfl,ws,rfl,rfl,rfl⟩

theorem source_of_decode (ok : decode p inputs stream = some c) : c.inputs = inputs ∧ c.stream = stream := by
  obtain ⟨_,ws,workers,_⟩ := decode_keeps ok
  unfold decode at ok
  dsimp only at ok
  split at ok <;> try contradiction
  simp [workers] at ok
  obtain ⟨_,rfl⟩ := ok
  exact ⟨rfl,rfl⟩

def cacheSite {p : Pool} (c : Capture p) (r : Row) : Option (FrameReturns.CacheSite c.timeline r.frame r.depth r.alpha r.beta) := do
  let sameVisit := fun (v : EndgameSearch.CacheHistory.Visit p) =>
    decide (v.worker = r.frame.worker ∧ v.frame = r.frame.id ∧ v.scoredAt = some r.frame.scoreStep)
  match h : c.timeline.visits.find? sameVisit with
  | none => none
  | some v =>
    if ok : v.input.board = r.frame.position ∧ v.input.depth = r.depth ∧ v.input.alpha = r.alpha ∧
        v.input.beta = r.beta ∧ v.input.selectivity = Cutoffs.exactSelectivity ∧ r.frame.pv = false ∧
        v.input.empties.count.toNat ≤ 12 then
      have same : v.worker = r.frame.worker ∧ v.frame = r.frame.id ∧ v.scoredAt = some r.frame.scoreStep := by
        simpa [sameVisit] using List.find?_some h
      return ⟨v,List.mem_of_find?_eq_some h,same.1,same.2.1,same.2.2,
        ok.1,ok.2.1,ok.2.2.1,ok.2.2.2.1,ok.2.2.2.2.1,ok.2.2.2.2.2.1,ok.2.2.2.2.2.2⟩
    else none

def Branch {p : Pool} (c : Capture p) (log : Log mc fc) (endian : Bool) (r : Row) : Prop :=
  if r.frame.pv = false ∧ r.depth.toNat ≤ 12 then
    match cacheSite c r with
    | none => False
    | some site =>
      (6 < site.visit.input.empties.count.toNat →
        TableReads.ProbeReadsFrom (FrameReturns.Samples log endian r.frame) site.visit.input.attempts) ∧
      (site.visit.input.run mc fc p.address
        (c.timeline.readAt mc fc r.frame.worker site.visit.first).history).score = some r.frame.score
  else ScoreStep log endian r.frame r.depth r.alpha r.beta

theorem Branch.checked {p : Pool} {c : Capture p} {log : Log mc fc} (source : Branch c log endian r) :
    ScoreStep log endian r.frame r.depth r.alpha r.beta := by
  unfold Branch at source
  split at source
  · split at source
    · contradiction
    · rename_i site _
      exact site.scored r.full source.1 source.2
  · exact source

theorem decode_observations {p : Pool} {c : Capture p} (processed : decode p inputs stream = some c)
    (mc fc identify before sequenced) (irrefl : ∀ i, ¬before i i)
    (trans : ∀ {i j k}, before i j → before j k → before i k)
    (scores : let log := logOf mc fc c.rows (tableEvents stream) identify before sequenced
      ∀ r ∈ c.rows, whole r = true → Branch c log endian r)
    (stores : let log := logOf mc fc c.rows (tableEvents stream) identify before sequenced
      ∀ e ∈ c.table, WriteStep log endian e) :
    Decoded (logOf mc fc c.rows (tableEvents stream) identify before sequenced) endian := by
  apply decoded _ _ _ _ _ irrefl trans (fun r mem all => (scores r mem all).checked)
  simpa only [(decode_keeps processed).1] using stores

end NeuralReversi.PvsParallel.NativeFrames
