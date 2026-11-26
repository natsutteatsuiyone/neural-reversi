import NeuralReversi.PvsParallel.Interleaving.JournalEntry
import NeuralReversi.PvsParallel.Interleaving.JointReads
import NeuralReversi.PvsParallel.NativeFrames.EntryStream
import NeuralReversi.PvsParallel.NativeFrames.CacheCalls

namespace NeuralReversi.PvsParallel.Interleaving.JournalBodies

open NativeFrames.EntryStream (Sample)
open EndgameSearch.CacheHistory (Input Pool Timeline Visit)
open PvsBoardTree (Position)
open CausalTable (Frame Values Lookups)
open TranspositionTable hiding Score
open CallJournal (Choice)
open ScoreRefresh (Score)

private theorem except_word {out : Except String A} (got : out.toOption = some v) : out = .ok v := by
  cases out <;> cases got
  rfl

private def localInput (i : Input) : Input := {i with attempts := [], selectivity := Cutoffs.exactSelectivity}

private theorem local_lookup (i : Input) : (localInput i).lookup = some none := by
  unfold localInput Input.lookup
  split <;> rfl

/-- Once the actual strategy lookup falls through, no later solver instruction
    reads the parent probe or its selectivity. -/
private theorem local_score (i : Input) (continues : i.lookup = some none) :
    ((localInput i).run mc fc addr past).score = (i.run mc fc addr past).score := by
  unfold Input.run Input.arm
  rw [local_lookup,continues]
  change ((localInput i).eval mc fc addr past (match NonPvMir.dispatch i.empties.count.toNat with
    | some (.fixed a) => .fixed a | some .shallow => .shallow (decide (i.empties.count.toNat = 6))
    | none => .deep i.active)).score = _
  cases which : NonPvMir.dispatch i.empties.count.toNat with
  | none => rfl
  | some a => cases a <;> rfl

private theorem direct_dispatch (n : Nat) (small : n ≤ 4) :
    NonPvMir.dispatch n = (EndgameSearch.Direct.Mir.dispatch n).map NonPvMir.Arm.fixed := by
  have := (show n = 0 ∨ n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 by omega)
  rcases this with h|h|h|h|h <;> subst n <;> rfl

private theorem direct_score (i : Input) (local_ : i.lookup = some none) (small : i.empties.count.toNat ≤ 4) :
    (i.run mc fc addr past).score = (EndgameSearch.Direct.Mir.run i.target i.bmi2 i.board.val
      i.empties i.alpha i.checked).map (·.raw.toInt) := by
  unfold Input.run Input.arm
  rw [local_,direct_dispatch _ small]
  dsimp only
  unfold EndgameSearch.Direct.Mir.run
  cases arm : EndgameSearch.Direct.Mir.dispatch i.empties.count.toNat with
  | some a => rfl
  | none =>
    have counts : i.empties.count.toNat = 0 ∨ i.empties.count.toNat = 1 ∨ i.empties.count.toNat = 2 ∨
        i.empties.count.toNat = 3 ∨ i.empties.count.toNat = 4 := by omega
    rcases counts with e|e|e|e|e <;> rw [e] at arm <;> contradiction

/-- Bodies independent of a prior or a table reader can be used at any
    selectivity in each of the three actual score histories. -/
structure Local (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : Position) (pv : Bool)
    (alpha beta score : Int) : Prop where
  exact : ∀ prior reads, Nested.Body mc fc prior reads b pv alpha beta score
  weak : ∀ prior reads, WeakFrame.Frame mc fc prior reads b pv alpha beta score

private theorem local_bodies (p : Pool) (trace : Timeline p) (worker opening : Nat) (i : Input)
    (same : i.depth = i.empties.count) (within : i.depth.toNat ≤ 12)
    (continued : i.lookup = some none) (pool : i.parallel = DeepLeafMir.parallelPool p.size)
    (win : Aspiration.Window i.alpha.raw.toInt i.beta.raw.toInt) (narrow : i.beta.raw.toInt = i.alpha.raw.toInt+1)
    (ran : (i.run mc fc p.address (trace.readAt mc fc worker opening).history).score = some v) :
    Local mc fc i.board false i.alpha.raw.toInt i.beta.raw.toInt v := by
  let past := (trace.readAt mc fc worker opening).history
  have normalized : ((localInput i).run mc fc p.address past).score = some v := (local_score i continued).trans ran
  constructor
  · intro prior reads
    by_cases small : i.empties.count.toNat ≤ 4
    · obtain ⟨raw,result,score⟩ := Option.map_eq_some_iff.mp ((direct_score i continued small).symm.trans ran)
      rw [←score]
      exact .direct i.alpha i.beta raw i.aligned small result
    by_cases short : i.empties.count.toNat ≤ 6
    · have arm : NonPvMir.dispatch i.empties.count.toNat = some .shallow := by
        rcases (show i.empties.count.toNat = 5 ∨ i.empties.count.toNat = 6 by omega) with k|k <;> rw [k] <;> rfl
      unfold Input.run Input.arm at ran
      rw [continued] at ran
      dsimp only at ran
      rw [arm] at ran
      change ((ScoreRefresh.fromDiscDiff i.checked ⟨BitVec.ofInt 32 (EndgameSearch.ShallowMir.run
        i.target i.bmi2 mc fc p.address i.board i.empties (ScoreRefresh.toDiscDiff i.alpha).raw.toInt past).score⟩
        ).toOption.map (·.raw.toInt)) = some v at ran
      rw [EndgameSearch.ShallowMir.agrees _ _ _ _ _ _ _ _ _ i.aligned
        (EndgameSearch.ShallowMir.scoreThreshold _) (trace.reads mc fc worker opening).1] at ran
      obtain ⟨raw,converted,score⟩ := Option.map_eq_some_iff.mp ran
      rw [←score]
      exact .shallow p trace worker opening i.alpha i.beta raw (i.aligned.1 ▸ (by omega))
        (i.aligned.1 ▸ short) (except_word converted)
    · have raw := EndgameSearch.CacheHistory.deep_score (mc := mc) (fc := fc) (addr := p.address)
        (past := past) (localInput i) same (by change 6 < i.empties.count.toNat; omega) (same ▸ within) rfl
      rw [raw] at normalized
      apply Nested.Body.boundary (size := p.size) p trace worker opening i.alpha i.beta i.depth
        (i.state past) [] rfl i.aligned
      · intro slot e impossible
        contradiction
      · simpa only [localInput,Input.state,pool] using normalized
  · intro prior reads
    apply WeakFrame.Frame.leaf p trace worker opening (localInput i) rfl same within win narrow _ normalized
    intro slot e impossible
    contradiction

structure Bodies (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (level : Selectivity)
    (reads : Lookups) (f : Frame) : Prop where
  exact : level = Cutoffs.exactSelectivity → ∀ prior, Nested.Body mc fc prior reads f.board f.pv f.alpha f.beta f.score
  weak : ∀ prior, WeakFrame.Frame mc fc prior reads f.board f.pv f.alpha f.beta f.score

/-- A malformed score in a recognized native branch rejects this execution.
    Only an instruction outside the supported branch carries a remaining obligation. -/
inductive Check (P : Prop) where
  | outside
  | rejected
  | proved (result : P)

def Check.requires (status : Check P) (rest : Prop) : Prop :=
  match status with | .outside => rest | .rejected => False | .proved _ => True

theorem Check.derive (status : Check P) (produce : P → Q) (checked : status.requires Q) : Q := by
  cases status with
  | outside => exact checked
  | rejected => exact checked.elim
  | proved result => exact produce result

variable {x : NativeTableWrites.Journal.Execution c endian rs} {es : Joint.Tape Op}
variable {m : Joint.Atomics x es}

abbrev Probes (m : Joint.Atomics x es) := (event : Nat) → (b : Position) →
  Option (JointReads.Call m event (.probe (Cutoffs.key b)))

private def evaluated (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig)
    (init : ∀ a, NativeTableWrites.Journal.Even (x.memory a)) (probes : Probes m)
    (p : Pool) (trace : Timeline p) (visit : Visit p) (level : Selectivity) (f : Frame) :
    Check (Bodies mc fc level (CausalTable.lookup x init f.chosen) f) :=
  let i := visit.input
  if shape : i.board = f.board ∧ f.pv = false ∧ i.alpha.raw.toInt = f.alpha ∧ i.beta.raw.toInt = f.beta ∧
      i.depth = i.empties.count ∧ i.depth.toNat ≤ 12 ∧ i.selectivity = level ∧
      i.parallel = DeepLeafMir.parallelPool p.size ∧
      (-32767 ≤ i.alpha.raw.toInt ∧ i.alpha.raw.toInt < i.beta.raw.toInt ∧ i.beta.raw.toInt ≤ 32767) ∧
      i.beta.raw.toInt = i.alpha.raw.toInt+1 then
    if scored : (i.run mc fc p.address (trace.readAt mc fc visit.worker visit.first).history).score = some f.score then
      if local_ : i.lookup = some none then .proved (by
        obtain ⟨board,pv,lo,hi,count,size,_,pool,window,null⟩ := shape
        have result := local_bodies p trace visit.worker visit.first i count size local_ pool
          ⟨window.1,window.2.1,window.2.2⟩ null scored
        constructor
        · intro _ prior
          simpa only [←board,pv,←lo,←hi] using result.exact prior (CausalTable.lookup x init f.chosen)
        · intro prior
          simpa only [←board,pv,←lo,←hi] using result.weak prior (CausalTable.lookup x init f.chosen))
      else
        let read := probes f.chosen i.board
        let physical := JointReads.attempts i.board read
        let probe := fun (attempts : List ProbeRetry.RawAttempt) =>
          ProbeRetry.probe (attempts.map (ProbeRetry.observe (Cutoffs.key i.board)))
        if issued : probe i.attempts = probe physical ∧ 6 < i.empties.count.toNat then .proved (by
          obtain ⟨board,pv,lo,hi,count,size,tag,pool,window,null⟩ := shape
          have seen : ∀ slot e, probe i.attempts = .hit slot e → CausalTable.lookup x init f.chosen i.board e :=
            fun slot e hit => (JointReads.physical (mc := mc) (fc := fc) init i.board read []).1 slot e (issued.1.symm.trans hit)
          refine ⟨fun full prior => ?_,fun prior => ?_⟩ <;> rw [←board,pv,←lo,←hi]
          · apply Nested.Body.boundary (size := p.size) p trace visit.worker visit.first i.alpha i.beta i.depth
              (i.state (trace.readAt mc fc visit.worker visit.first).history) i.attempts rfl i.aligned seen
            rw [←pool]
            exact (EndgameSearch.CacheHistory.deep_score i count issued.2 (count ▸ size) (tag.trans full)).symm.trans scored
          · exact .leaf p trace visit.worker visit.first i rfl count size ⟨window.1,window.2.1,window.2.2⟩ null seen scored)
        else .rejected
    else .rejected
  else .rejected

private def enteredAt (events : Joint.Tape Op) (worker opening : Nat) : Option (Nat × Nat × Input) :=
  events.zipIdx.findSome? fun (e,j) => match e with
    | .engine (.localCall w (.entry t site .boundary _ (some op)) _) =>
      if w = worker ∧ t = opening then some (j,site,op.input) else none
    | _ => none

private def leftAt (events : Joint.Tape Op) (worker opening : Nat) : Option Nat :=
  events.zipIdx.findSome? fun (e,j) => match e with
    | .engine (.localCall w (.leave _ t .boundary ..) _) =>
      if w = worker ∧ t = opening then some j else none
    | _ => none

/-- The depth-twelve probe must open in this particular boundary activation.
    This also checks misses and rejected table entries before local search. -/
private def probeMatches (probes : Probes m) (worker start stop consumer : Nat) (i : Input) : Bool :=
  let starts := es.zipIdx.filterMap fun (ev,j) => match ev with
    | .enter w _ => if w = worker ∧ start < j ∧ j < stop then some j else none
    | _ => none
  if i.depth.toNat = 12 ∧ i.parallel then
    match starts,probes consumer i.board with
    | [opening],some r =>
      let observed := ProbeRetry.probe ((JointReads.attempts i.board (some r)).map (ProbeRetry.observe (Cutoffs.key i.board)))
      let register := ProbeRetry.probe (i.attempts.map (ProbeRetry.observe (Cutoffs.key i.board)))
      opening == r.opening && decide (r.finish < stop) && observed == register &&
        match es[r.finish]? with
        | some (.leave w (.probe value)) => worker == w && value == register
        | _ => false
    | _,_ => false
  else starts.isEmpty

private def samePool (a b : Pool) : Bool :=
  a.base == b.base && a.size == b.size && a.width == b.width

def noPoolHeader (es : Joint.Tape Op) : Bool :=
  es.all fun ev => match ev with | .poolHistory .. => false | _ => true

def poolHeader (p : Pool) (es : Joint.Tape Op) : Bool :=
  match es with
  | .poolHistory base size _ :: tail => base == p.base && size == p.size && noPoolHeader tail
  | _ => noPoolHeader es

/-- Preserve earlier local-cache effects across a search boundary. Their sample
    stream uses the same native worker clock and decoder as the current search. -/
def cacheSamples (p : Pool) (es : Joint.Tape Op) : Option (List NativeFrames.EntryStream.Sample) := do
  guard (poolHeader p es)
  match es with
  | .poolHistory _ _ past :: tail => return past ++ tail.filterMap Joint.Event.frameSample
  | _ => return es.filterMap Joint.Event.frameSample

/-- Start a new table journal on the same pool. Its writer operand type may
    change with the table image. Retain local calls from every preceding search. -/
def resumePool (p : Pool) (earlier : Joint.Tape Old) (current : Joint.Tape New) : Option (Joint.Tape New) := do
  let past ← cacheSamples p earlier
  guard (noPoolHeader current)
  return .poolHistory p.base p.size past :: current

def capture (p : Pool) (events : Joint.Tape Op) :=
  (cacheSamples p events).bind (NativeFrames.EntryStream.capture p)

private theorem deep_falls_through (i : Input) (deep : i.arm = .deep i.active) : i.lookup = some none := by
  cases result : i.lookup with
  | none => simp [Input.arm,result] at deep
  | some hit => cases hit with
    | none => rfl
    | some _ => simp [Input.arm,result] at deep

private def saved (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (reads : Lookups)
    (p : Pool) (trace : Timeline p) (worker opening : Nat) (i : Input) (before after : BitVec 64)
    (level : Selectivity) (f : Frame) : Check (Bodies mc fc level reads f) :=
  if valid : CallJournal.BoundaryChecks i before after = true ∧ i.board = f.board ∧ f.pv = false ∧
      i.alpha.raw.toInt = f.alpha ∧ i.beta.raw.toInt = f.beta ∧ level = Cutoffs.exactSelectivity ∧
      i.parallel = DeepLeafMir.parallelPool p.size ∧
      (i.run mc fc p.address (trace.readAt mc fc worker opening).history).score = some f.score then .proved (by
    obtain ⟨checks,board,pv,lo,hi,_,pool,score⟩ := valid
    have original := checks
    simp only [CallJournal.BoundaryChecks,Bool.and_eq_true,beq_iff_eq,Bool.not_eq_true',decide_eq_true_eq] at checks
    obtain ⟨⟨⟨⟨⟨⟨depth,count⟩,_⟩,_⟩,_⟩,arm⟩,low,mid,high,narrow⟩ := checks
    have local_ := local_bodies p trace worker opening i count (by omega)
      (deep_falls_through i (CallJournal.deep_arm i arm)) pool ⟨low,mid,high⟩ narrow score
    constructor
    · intro _ prior
      rw [←board,pv,←lo,←hi]
      exact (CallJournal.boundaryBodies (mc := mc) (fc := fc)
        (prior := prior) (reads := reads) (earlier := fun _ _ _ _ _ => False) (seen := reads)
        p trace i worker opening original score).1
    · intro prior
      simpa only [←board,pv,←lo,←hi] using local_.weak prior reads)
  else .rejected

def check (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig)
    (init : ∀ a, NativeTableWrites.Journal.Even (x.memory a)) (probes : Probes m) (pool : Pool)
    (state : CallJournal.State Op) (f : Frame) : Check (Bodies mc fc (state.level f) (CausalTable.lookup x init f.chosen) f) :=
  match state.choices.find? (fun choice => choice.stamp == f.chosen) with
  | none => .rejected
  | some owner =>
    match es[f.chosen]? with
    | some (.engine (.localCall worker (.scored pc word) _)) =>
      if owner.endgame && !owner.pv && decide (owner.depth.toNat ≤ 12) then
        if worker != owner.worker || word.raw != owner.word.raw || word.raw.toInt != f.score ||
            owner.rootLine.isSome then .rejected else
        match capture pool es with
        | none => .rejected
        | some cap =>
          let id := NativeFrames.frameKey (EndgameSearch.CacheOwner.poolPlan pool.size).length worker owner.opening
          match cap.timeline.visits.find? (fun v => v.worker == worker && v.frame == id && v.scoredAt == some pc) with
          | none => .rejected
          | some v =>
            match enteredAt es worker v.first,leftAt es worker v.first with
            | some (start,site,_),some stop =>
              if site == DeepLeafMir.boundaryPvsSite && decide (start < stop ∧ stop < f.chosen) &&
                  probeMatches probes worker start stop f.chosen v.input && v.input.depth == owner.depth &&
                  v.input.board == owner.board && v.input.alpha.raw == owner.alpha.raw &&
                  v.input.beta.raw == owner.beta.raw && v.input.selectivity == owner.level then
                evaluated mc fc init probes pool cap.timeline v (state.level f) f
              else .rejected
            | _,_ => .rejected
      else .outside
    | some (.engine (.boundaryScore worker p _ before after word)) =>
      if worker != owner.worker || word.raw.toInt != f.score || !samePool p pool then .rejected else
        match enteredAt es worker owner.opening, capture pool es with
        | some (start,site,i), some cap =>
          if (site == 0 || site == DeepLeafMir.boundaryPvsSite) && decide (start < f.chosen) &&
              probeMatches probes worker start f.chosen f.chosen i then
            saved mc fc (CausalTable.lookup x init f.chosen) pool cap.timeline worker owner.opening i
              before after (state.level f) f
          else .rejected
        | _,_ => .rejected
    | _ => .rejected

abbrev SeedObligation (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig)
    (state : CallJournal.State Op) (f : Frame) (reads : Lookups) :=
  SeedHistory.Body mc fc (state.level f)
    (CausalTable.returned (· < ·) (state.frames.filter (fun g => state.level g == state.level f)) f.chosen)
    reads f.board f.pv f.alpha f.beta f.score

abbrev WeakObligation (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig)
    (state : CallJournal.State Op) (f : Frame) (reads : Lookups) :=
  WeakFrame.Frame mc fc (CausalTable.returned (· < ·) state.weakChildren f.chosen)
    reads f.board f.pv f.alpha f.beta f.score

abbrev ExactObligation (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig)
    (state : CallJournal.State Op) (f : Frame) (reads : Lookups) :=
  Nested.Body mc fc (CausalTable.returned (· < ·) state.exactFrames f.chosen)
    reads f.board f.pv f.alpha f.beta f.score

structure Remaining (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (init : ∀ a, NativeTableWrites.Journal.Even (x.memory a))
    (probes : Probes m) (pool : Pool) (state : CallJournal.State Op) : Prop where
  seed : ∀ f ∈ state.frames, (check mc fc init probes pool state f).requires
    (SeedObligation mc fc state f (CausalTable.lookup x init f.chosen))
  weak : ∀ f ∈ state.weakChildren, (check mc fc init probes pool state f).requires
    (WeakObligation mc fc state f (CausalTable.lookup x init f.chosen))
  exact : ∀ f ∈ state.exactFrames, (check mc fc init probes pool state f).requires
    (ExactObligation mc fc state f (CausalTable.lookup x init f.chosen))

theorem histories {x : NativeTableWrites.Journal.Execution c endian rs}
    (init : ∀ a, NativeTableWrites.Journal.Even (x.memory a))
    {events : Joint.Tape (CallJournal.Operand c endian (CallJournal.observed x init))}
    {m : Joint.Atomics x events} (probes : Probes m) (pool : Pool)
    (state : CallJournal.State (CallJournal.Operand c endian (CallJournal.observed x init)))
    (rest : Remaining mc fc init probes pool state) :
    CallJournal.SeedBodies mc fc state ∧ CallJournal.WeakBodies mc fc state ∧ CallJournal.ExactBodies mc fc state := by
  refine ⟨fun f member => ?_,fun f member => ?_,fun f member => ?_⟩
  · exact (check mc fc init probes pool state f).derive (fun body => .internal (body.weak _)) (rest.seed f member)
  · exact (check mc fc init probes pool state f).derive (fun body => body.weak _) (rest.weak f member)
  · exact (check mc fc init probes pool state f).derive
      (fun body => body.exact (beq_iff_eq.mp (List.mem_filter.mp member).2) _) (rest.exact f member)

/-- Assemble the public execution directly from raw journal data; the boundary
    frames are replayed before constructing any of the three body fields. -/
def execution (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig)
    (memory : NativeTableWrites.Journal.Execution c endian rs)
    (init : ∀ a, NativeTableWrites.Journal.Even (memory.memory a))
    (resident : ∀ a, Entry.Resident mc fc endian ((memory.slot a (init a)).payloadAt 0))
    (events : Joint.Tape (CallJournal.Operand c endian (CallJournal.observed memory init)))
    (state : CallJournal.State (CallJournal.Operand c endian (CallJournal.observed memory init)))
    (replay : CallJournal.decode mc fc c endian (CallJournal.observed memory init) (events.map Joint.Event.engineEvent) = some state)
    (recorded : state.records = rs) (order : ∀ a i j, (memory.memory a).hb i j → i < j)
    (atomics : Joint.Atomics memory events) (probes : Probes atomics) (pool : Pool)
    (remaining : Remaining mc fc init probes pool state) : JournalEntry.Execution mc fc :=
  let bodies := histories init probes pool state remaining
  ⟨c,endian,rs,memory,init,resident,events,state,replay,recorded,order,bodies.1,bodies.2.1,bodies.2.2,atomics⟩

namespace Fixture

set_option maxRecDepth 8192

private def pool : Pool := ⟨2,64,by decide,fun _ => 0,0,⟨100,101,102⟩,⟨110,111,112⟩⟩
example : samePool pool pool := by decide
example : samePool pool {pool with base := pool.base+1} = false := by decide
private def worker : EndgameSearch.CacheHistory.NativeTrace.Worker pool := ⟨1,.worker,by decide⟩
private abbrev mc : Bitboard.Moves.X86Config := ⟨false,false⟩
private abbrev fc : Cutoffs.FlipConfig := ⟨false,false,false⟩

private def board : Position := ⟨⟨~~~(0:BitVec 64),0⟩,by simp [Board.wellFormed]⟩

private def input (b : Position) (level : Selectivity) : Input :=
  let mask := Board.emptySquares b.val
  let empty : EndgameSearch.Cancellation.EmptyState :=
    ⟨Empties.encode mask,Board.emptyCount b.val⟩
  ⟨b,empty,EndgameSearch.emptyState_initial b,⟨BitVec.ofInt 32 (-1)⟩,⟨0⟩,empty.count,level,[],
    ⟨⟨0,false⟩,fun _ => ⟨0,[]⟩,0⟩,true,false,false,true,.other,false,false⟩

private def tape (level : Selectivity) (word : Score) : Joint.Tape Unit :=
  let i := input board level
  let enter := fun t node => NativeFrames.Event.entry t 0 node worker.root none
  let code := NativeFrames.CacheCalls.boundary worker 1 3 i (some 8) .nil .nil
  let steps := [enter 0 .active,enter 1 .dispatch,enter 2 .splitSearch,enter 3 .pvs] ++
    code.events id 4 ++ [.scored 8 word,.leave 9 3 .pvs none (some word)]
  steps.map fun e => .engine (.localCall 1 e (if e.step = 3 then
    some ⟨i.board.val,i.depth,i.alpha,i.beta,true,level,false,false,none⟩ else none))

private def pastSamples := (tape 5 ⟨16384⟩).filterMap Joint.Event.frameSample
private def withPast (base size : Nat) : Joint.Tape Unit := [.poolHistory base size pastSamples]

example : (capture pool (withPast pool.base pool.size)).map (fun c => c.timeline.visits.length) = some 1 := by decide +kernel
example : (capture pool (withPast (pool.base+1) pool.size)).isNone := by decide +kernel
example : (capture pool (withPast pool.base (pool.size+1))).isNone := by decide +kernel
example : (capture pool (.engine (.other 0) :: withPast pool.base pool.size)).isNone := by decide +kernel
example : (capture pool (withPast pool.base pool.size ++ withPast pool.base pool.size)).isNone := by decide +kernel

example : ((resumePool pool (withPast pool.base pool.size) ([] : Joint.Tape Bool)).bind (capture pool)).map
    (fun c => c.timeline.visits.length) = some 1 := by decide +kernel

private def replay (level : Selectivity) (word : Score) (x : NativeTableWrites.Journal.Execution c endian rs)
    (init : ∀ a, NativeTableWrites.Journal.Even (x.memory a)) (m : Joint.Atomics x (tape level word)) : Option Bool := do
  let s ← CallJournal.run mc fc c endian ((tape level word).map Joint.Event.engineEvent)
  let f ← s.frames.head?
  match check mc fc init (m := m) (fun _ _ => none) pool s f with
  | .proved _ => some true
  | .rejected => some false
  | .outside => none

/-- The complete board has 64 player stones, giving the native score word
    16384. A one-unit mutation at both score and return passes the stack
    matching checks and is rejected by the independent boundary computation. -/
example (x : NativeTableWrites.Journal.Execution c endian rs) (init : ∀ a, NativeTableWrites.Journal.Even (x.memory a))
    (exact : Joint.Atomics x (tape 5 ⟨16384⟩)) (selective : Joint.Atomics x (tape 2 ⟨16384⟩))
    (mutated : Joint.Atomics x (tape 5 ⟨16383⟩)) :
    replay 5 ⟨16384⟩ x init exact = some true ∧ replay 2 ⟨16384⟩ x init selective = some true ∧
      replay 5 ⟨16383⟩ x init mutated = some false := by
  exact ⟨rfl,rfl,rfl⟩

end Fixture

end NeuralReversi.PvsParallel.Interleaving.JournalBodies
