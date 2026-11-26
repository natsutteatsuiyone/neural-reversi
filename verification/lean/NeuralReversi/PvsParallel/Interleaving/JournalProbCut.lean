import NeuralReversi.PvsParallel.Interleaving.JournalScout

namespace NeuralReversi.PvsParallel.Interleaving.JournalProbCut

open TranspositionTable hiding Score
open PvsBoardTree (Position legalSquares pass)
open ScoreRefresh (Score)
open CausalTable (Frame Lookups)
open SerialTable.SelectiveFrame (Estimates probCut margins)
open JournalBodies (Bodies Check)

abbrev Trace (Op : Type) := List (Joint.Event Op × Nat)

inductive Progress (Op : Type) where
  | continued (rest : Trace Op)
  | cut (evaluation verify : BitVec 32) (alpha beta : Int)

/-- The verification routine is observed once at its actual arguments. -/
def estimates (eval word : BitVec 32) (a z : Int) : Estimates :=
  ⟨eval,fun lo hi => if lo = a ∧ hi = z then word else 0⟩

private def noMore (tail : Trace Op) := match tail.head? with | some (.probWord ..,_) => false | _ => true

/-- Parse the dedicated register sites on one worker. A rejected screen issues
    no depth-two call. A failed verification leaves the entire search suffix
    available to the non-PV owner. -/
def consume (b : Position) (d : Depth) (level : Selectivity) (cut : Bool) (beta : Score) (worker : Nat)
    (tape : Trace Op) : Option (Progress Op) := do
  if d.toNat < 13 || level == Cutoffs.exactSelectivity then
    guard (noMore tape)
    return .continued tape
  guard (1 ≤ level.val ∧ level.val < 5)
  let (.probWord w (.estimate board main eval),_) :: (.probWord v (.window depth selectivity cutNode raw seen pc screen),_) :: rest := tape | none
  guard (worker = w ∧ worker = v ∧ board = b.val ∧ main = true ∧ depth = d ∧ selectivity = level ∧
    cutNode = cut ∧ raw = beta.raw ∧ seen = eval)
  let bits := margins d level cut beta.raw eval
  let thresholds := EndgameProbCut.nativeThresholds beta.raw.toInt bits.1 bits.2
  guard (pc.toInt = thresholds.verification ∧ screen.toInt = thresholds.screen)
  match EndgameProbCut.verificationWindow eval.toInt thresholds with
  | none => guard (noMore rest); return .continued rest
  | some (a,z) =>
    let (.probWord w (.verify board main lo hi score),_) :: tail := rest | none
    guard (w = worker ∧ board = b.val ∧ main = true ∧ lo.toInt = a ∧ hi.toInt = z)
    if z ≤ score.toInt then
      guard tail.isEmpty
      return .cut eval score a z
    else guard (noMore tail); return .continued tail

variable {x : NativeTableWrites.Journal.Execution c endian rs} {es : Joint.Tape Op} {m : Joint.Atomics x es}

/-- The depth-thirteen arm has no enhanced child lookups. At fourteen empties
    and above, consume all misses from the generated moves before ProbCut. -/
def afterEtc (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : Position) (query : PvsEnhancedCutoff.Query)
    (init : ∀ a, NativeTableWrites.Journal.Even (x.memory a)) (look : JournalScout.ChildReads m) (consumer : Nat)
    (parent : JointReads.Call m consumer (.probe (Cutoffs.key b))) (tape : Trace Op) : Option (Option (Trace Op)) := do
  if query.depth.toNat < 14 then return some tape
  let (windows,hit) ← JournalScout.scan mc fc b query (legalSquares mc b.val) tape
  let resolved ← windows.mapM (JournalScout.child mc fc b init look consumer parent)
  guard (PvsEnhancedCutoff.scan query ((resolved.map (JointReads.Child.row init)).map TableReads.ChildLookupFrom.observation) = hit)
  if hit.isSome then return none
  return some (match windows.getLast? with | none => tape | some last => tape.dropWhile (fun row => decide (row.2 ≤ last.finish)))

private def eligible (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : Position) (d : Depth) (a : Score) :=
  let generated := PvEntryMir.generated mc fc b false (fun _ => none)
  let stable := EndgameStability.tryCutoff b.val d.toNat ⟨(ScoreRefresh.toDiscDiff a).raw.toInt⟩
  decide (GeneralNonPvMir.dispatch d.toNat stable.isSome (generated.len.getD 0 != 0)
    (Board.hasLegalMoves mc (pass b).val) generated.wipeout.join.isSome false false = .rest)

private def decideCut (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (reads : Lookups)
    (f : Frame) (owner : CallJournal.Choice) (level : Selectivity) (tail : Trace Op) :
    Check (Bodies mc fc level reads f) :=
  match consume f.board owner.depth level owner.cut owner.beta owner.worker tail with
  | none => .rejected
  | some (.continued _) => .outside
  | some (.cut eval score a z) =>
    let issued := estimates eval score a z
    if ok : level ≠ Cutoffs.exactSelectivity ∧ f.pv = false ∧ f.beta = owner.beta.raw.toInt ∧
        -32767 ≤ f.alpha ∧ f.alpha < f.beta ∧ f.beta ≤ 32767 ∧
        f.score = owner.beta.raw.toInt ∧ probCut owner.depth level owner.cut owner.beta issued = some f.score then
      .proved (by
        obtain ⟨disabled,pv,beta,low,mid,high,_,ran⟩ := ok
        refine ⟨fun full => (disabled full).elim,fun prior => ?_⟩
        rw [pv,beta]
        exact WeakFrame.Frame.probability owner.beta issued ⟨low,beta ▸ mid,beta ▸ high⟩ ran)
    else .rejected

def check (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig)
    (init : ∀ a, NativeTableWrites.Journal.Even (x.memory a)) (probes : JournalBodies.Probes m) (look : JournalScout.ChildReads m)
    (checked : Bool) (state : CallJournal.State Op) (f : Frame) :
    Check (Bodies mc fc (state.level f) (CausalTable.lookup x init f.chosen) f) :=
  match state.choices.find? (fun o => o.stamp == f.chosen) with
  | none => .rejected
  | some owner =>
    if !owner.endgame || owner.pv || owner.rootLine.isSome ||
        !eligible mc fc f.board owner.depth owner.alpha then .outside else
    if owner.board != f.board || owner.depth != Board.emptyCount f.board.val || owner.level != state.level f ||
        f.alpha != owner.alpha.raw.toInt || f.beta != owner.beta.raw.toInt || f.beta != f.alpha+1 then .rejected else
    match CallReturns.captureTimed (es.map Joint.Event.frameSample) with
    | none => .rejected
    | some calls =>
      match JournalPv.parent calls owner f with
      | none => .rejected
      | some (_,start) =>
        match JournalPrefix.parentAt es owner.worker start f.chosen f.board,probes f.chosen f.board with
        | some (opening,stop,raw),some read =>
          if read.opening != opening || read.finish != stop ||
              ProbeRetry.probe ((JointReads.attempts f.board (some read)).map (ProbeRetry.observe (Cutoffs.key f.board))) != raw then .rejected else
          match JournalScout.parentCuts checked owner.depth owner.level owner.beta raw with
          | .error _ => .rejected
          | .ok true => .outside
          | .ok false =>
            let q : PvsEnhancedCutoff.Query := ⟨owner.depth,owner.level,true,⟨owner.alpha.raw⟩⟩
            match afterEtc mc fc f.board q init look f.chosen read (JournalScout.events es owner.worker stop f.chosen) with
            | none => .rejected
            | some none => .outside
            | some (some tail) => decideCut mc fc _ f owner (state.level f) tail
        | _,_ => .rejected

def pending (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig)
    (init : ∀ a, NativeTableWrites.Journal.Even (x.memory a)) (probes : JournalBodies.Probes m) (look : JournalScout.ChildReads m)
    (pool : EndgameSearch.CacheHistory.Pool) (checked : Bool) (s : CallJournal.State Op) (op : JournalPv.Options mc)
    (f : Frame) (frames : List Frame) (body : Prop) :=
  JournalScout.pending mc fc init probes look pool checked s op f frames ((check mc fc init probes look checked s f).requires body)

structure Remaining (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig)
    (init : ∀ a, NativeTableWrites.Journal.Even (x.memory a)) (probes : JournalBodies.Probes m) (look : JournalScout.ChildReads m)
    (pool : EndgameSearch.CacheHistory.Pool) (checked : Bool) (s : CallJournal.State Op) (op : JournalPv.Options mc) : Prop where
  seed : ∀ f ∈ s.frames, pending mc fc init probes look pool checked s op f (JournalPass.seedRows s f)
    (JournalBodies.SeedObligation mc fc s f (CausalTable.lookup x init f.chosen))
  weak : ∀ f ∈ s.weakChildren, pending mc fc init probes look pool checked s op f s.weakChildren
    (JournalBodies.WeakObligation mc fc s f (CausalTable.lookup x init f.chosen))
  exact : ∀ f ∈ s.exactFrames, pending mc fc init probes look pool checked s op f s.exactFrames
    (JournalBodies.ExactObligation mc fc s f (CausalTable.lookup x init f.chosen))

private theorem through (test : Check P) (use : A → B) : test.requires A → test.requires B := by
  cases test with | outside => exact use | rejected => exact id | proved _ => exact id

private theorem discharge (body : Bodies mc fc (s.level f) (CausalTable.lookup x init f.chosen) f → Q)
    (h : pending mc fc init probes look pool checked s op f frames Q) :
    JournalScout.pending mc fc init probes look pool checked s op f frames Q :=
  through _ (through _ (through _ (through _ (through _ (Check.derive _ body))))) h

theorem Remaining.etc (h : Remaining mc fc init probes look pool checked s op) :
    JournalScout.Remaining mc fc init probes look pool checked s op where
  seed f mem := discharge (fun r => .internal (r.weak _)) (h.seed f mem)
  weak f mem := discharge (fun r => r.weak _) (h.weak f mem)
  exact f mem := discharge (fun r => r.exact (beq_iff_eq.mp (List.mem_filter.mp mem).2) _) (h.exact f mem)

namespace Regression

private def mc : Bitboard.Moves.X86Config := ⟨false,false⟩
private def fc : Cutoffs.FlipConfig := ⟨false,false,false⟩
private def board : Position := ⟨⟨0x147c63536745644e,0x09821cac18b81830⟩,rfl⟩
private def word (n : Int) := BitVec.ofInt 32 n

structure Registers where
  depth : Depth := 13
  lvl : Selectivity := 2
  beta : Int := 1
  eval : Int := 6144
  pc : Int := 917
  screen : Int := -3623
  value : Option Int := some 917
  checkBeta : Int := 0
  staticMain : Bool := true
  verifyMain : Bool := true
  verifyWorker : Nat := 4
  wrongBoard : Bool := false
  wrongWindow : Bool := false
  suffix : Bool := false

private def sample (r : Registers) (position : Position := board) : Trace Unit :=
  let event := fun w op => Joint.Event.probWord (Operand := Unit) w op
  ([(event 4 (.estimate position.val r.staticMain (word r.eval))),
    (event 4 (.window r.depth r.lvl true (word (r.beta+r.checkBeta)) (word r.eval) (word r.pc) (word r.screen)))] ++
    (r.value.map (fun v => event r.verifyWorker (.verify (if r.wrongBoard then (pass position).val else position.val) r.verifyMain
      (word (r.pc - (if r.wrongWindow then 2 else 1))) (word r.pc) (word v)))).toList ++
    (if r.suffix then [Joint.Event.engine (Operand := Unit) (.localCall 4 (.entry 6 4 .pvs .none none) none)] else [])).zipIdx

private def continuation (r : Registers) : Option (Bool × Nat) := do
  match ← consume board r.depth r.lvl true ⟨word r.beta⟩ 4 (sample r) with
  | .continued rest => return (false,rest.length)
  | .cut .. => return (true,0)

private def worker : NativeTableWrites.Journal.Calls := fun _ => ⟨⟨0,0⟩,⟨0,rfl⟩,.clear,false,id⟩

/-- Decode the actual PVS selection and parent reader location before admitting
    the cut. Thirteen empties bypass ETC. The board follows the forced pass and
    a8 in the fourteen-empty JournalPass fixture. -/
private def status (position : Position) (r : Registers) (chosen : Int) : Option (Option Bool) := do
  let a : Score := ⟨word (r.beta-1)⟩
  let z : Score := ⟨word r.beta⟩
  let op : NativeFrames.Args := ⟨position.val,r.depth,a,z,true,r.lvl,false,true,none⟩
  let events : Joint.Tape Unit := [.engine (.localCall 4 (.entry 2 6 .pvs .none none) (some op)),
    .enter 4 ⟨.probe (Cutoffs.key position),0,64,Board.Hash.hash false position.val⟩, .leave 4 (.probe (.miss 0))] ++
    (sample r position).map Prod.fst ++ [.engine (.localCall 4 (.scored 10 ⟨word chosen⟩) none)]
  let s ← CallJournal.run mc fc worker false (events.map Joint.Event.engineEvent)
  let owner ← s.choices.head?
  let f ← s.frames.head?
  let frames ← CallReturns.captureTimed (events.map Joint.Event.frameSample)
  let (_,opening) ← JournalPv.parent frames owner f
  let (_,close,raw) ← JournalPrefix.parentAt events owner.worker opening f.chosen position
  guard (owner.depth = Board.emptyCount position.val ∧ eligible mc fc position r.depth a)
  let .false ← (JournalScout.parentCuts true r.depth (s.level f) z raw).toOption | none
  match decideCut mc fc (fun _ _ => False) f owner (s.level f) (JournalScout.events events owner.worker close f.chosen) with
  | .outside => some none | .rejected => some (some false) | .proved _ => some (some true)

private def pickedAt (b : Position) (r : Registers) (chosen : Int) := (status b r chosen).join

private abbrev picked := pickedAt board

-- The thresholds here were independently computed by the Rust implementation.
#guard picked {} 1 = some true
#guard picked {} 917 = some false
#guard continuation {value := some 916,suffix := true} = some (false,1)
#guard continuation {eval := -16384,pc := 4740,screen := 200,value := none,suffix := true} = some (false,1)
#guard continuation {eval := -16384,pc := 4740,screen := 200} = none
#guard continuation {beta := 16380,eval := 16384,pc := 16384,screen := 12625,value := none} = some (false,0)
#guard continuation {pc := 918} = none
#guard continuation {checkBeta := 1} = none
#guard continuation {wrongWindow := true} = none
#guard continuation {wrongBoard := true} = none
#guard continuation {verifyWorker := 1} = none
#guard continuation {verifyMain := false} = none
#guard continuation {staticMain := false} = none
#guard continuation {suffix := true} = none
#guard continuation {lvl := 5} = none
#guard continuation {depth := 12} = none

-- Actual main-network and evaluate_depth2 return words from two native
-- executions. The passing verifier produced 10059; the chosen frame kept 9216.
private def missed : Position := ⟨⟨426633433020562559,4576335670345856⟩,rfl⟩
private def passed : Position := ⟨⟨17545388337,3241606549595363916⟩,rfl⟩
private def fail : Registers := {lvl := 4,beta := -5120,eval := -3927,pc := -2035,screen := -6575,value := some (-2038)}
private def hit : Registers := {lvl := 2,beta := 9216,eval := 10652,pc := 10037,screen := 5497,value := some 10059}
#guard status missed fail (-5120) = some none
#guard pickedAt passed hit 9216 = some true
#guard pickedAt passed hit 10037 = some false
#guard pickedAt passed hit 10059 = some false

-- Independently instrumented registers on the board following a8.
#guard continuation {beta := -15000,eval := -15188,pc := -13553,screen := -16384,value := some (-14568),suffix := true} = some (false,1)

end Regression

end NeuralReversi.PvsParallel.Interleaving.JournalProbCut
