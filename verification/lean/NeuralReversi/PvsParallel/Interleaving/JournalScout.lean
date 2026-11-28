module
public import NeuralReversi.PvsParallel.Interleaving.JournalPv
meta import NeuralReversi.TranspositionTable.Proofs
meta import NeuralReversi.PvsParallel.Interleaving.JournalPv

@[expose] public section

namespace NeuralReversi.PvsParallel.Interleaving.JournalScout

open TranspositionTable hiding Score
open ScoreRefresh (Score)
open PvsBoardTree (Position GeneratedSquare legalSquares play)
open CausalTable (Frame Lookups)
open JournalBodies (Check Bodies)

def events (es : Joint.Tape Op) (worker start finish : Nat) :=
  es.zipIdx.filter fun (e,t) => e.worker == worker && decide (start < t ∧ t < finish) &&
    match e with | .engine (.other _) => false | _ => true

structure Window (M : Type) where
  move : M
  opening : Nat
  finish : Nat
  word : Option EntryData

def consume (b : Position) (tape : List (Joint.Event Op × Nat)) : Option ((Nat × Nat × Option EntryData) × List (Joint.Event Op × Nat)) := do
  let (.enter _ args,opening) :: tail := tape | none
  let .lookup actual := args.method | none
  guard (actual = b)
  let (.leave _ (.lookup word),finish) :: rest := (tail.dropWhile fun row => match row.1 with | .issue .. => true | _ => false) | none
  return ((opening,finish,word),rest)

/-- Walk the unsorted generated move list, stopping at the first native ETC candidate.
    Every same-worker reader, child or writer marker is significant before score selection. -/
def scan (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : Position) (q : PvsEnhancedCutoff.Query) :
    List (GeneratedSquare mc b.val) → List (Joint.Event Op × Nat) → Option (List (Window (GeneratedSquare mc b.val)) × Option PvsEnhancedCutoff.Hit)
  | [], _ => some ([],none)
  | m :: ms, tape => do
    let ((opening,finish,word),tail) ← consume (play mc fc b m).val tape
    let w := Window.mk m opening finish word
    match PvsEnhancedCutoff.candidate q (m.val,word) with
    | some hit => guard tail.isEmpty; return ([w],some hit)
    | none => let (later,hit) ← scan mc fc b q ms tail; return (w :: later,hit)

variable {x : NativeTableWrites.Journal.Execution c endian rs} {es : Joint.Tape Op} {m : Joint.Atomics x es}

abbrev ChildReads (m : Joint.Atomics x es) := (consumer opening : Nat) → (b : Position) → Option (JointReads.Call m consumer (.lookup b))

def child (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : Position)
    (init : ∀ a, NativeTableWrites.Journal.Even (x.memory a)) (look : ChildReads m) (consumer : Nat)
    (parent : JointReads.Call m consumer (.probe (Cutoffs.key b)))
    (w : Window (GeneratedSquare mc b.val)) : Option (JointReads.Child m consumer mc fc b) := do
  let next := (play mc fc b w.move).val
  let got ← look consumer w.opening next
  let args : ReaderWindow.Args := ⟨.lookup next,got.allocation,got.clusters,got.key⟩
  guard (parent.key == ReaderWindow.engineKey endian (.probe (Cutoffs.key b)) && args.inTable endian parent.allocation parent.clusters)
  let bound : JointReads.Child m consumer mc fc b := ⟨w.move,got⟩
  guard (got.opening = w.opening ∧ got.finish = w.finish ∧ (bound.row init).observation.2 = w.word)
  return bound

def noCalls : NativeNodes.Returns := ⟨[],fun _ => 0,fun _ => false,fun _ _ => false,0,fun _ _ _ _ _ _ => none⟩
theorem noCalls_inside (prior : CausalTable.Values) : noCalls.InPrefix prior := by
  intro b p a z v h
  obtain ⟨_,no,_⟩ := h
  contradiction

def checkedBodies (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : Position)
    (a z : Score) (d : Depth) (checked cut : Bool) (level : Selectivity) (score : Int) (reads : Lookups)
    (attempts : List ProbeRetry.RawAttempt) (rows : List (TableReads.ChildLookupFrom mc fc b (fun _ => True)))
    (seen : NativeNodes.Probed mc fc b reads attempts rows) (full : ExactDepth.Full b d)
    (null : z.raw.toInt = a.raw.toInt+1) (hit : PvsEnhancedCutoff.Hit)
    (observed : PvsEnhancedCutoff.scan ⟨d,level,true,⟨a.raw⟩⟩ (rows.map TableReads.ChildLookupFrom.observation) = some hit)
    (word : score = hit.2) :
    Check ((level = Cutoffs.exactSelectivity → ∀ prior, Nested.Body mc fc prior reads b false a.raw.toInt z.raw.toInt score) ∧
      ∀ prior, WeakFrame.Frame mc fc prior reads b false a.raw.toInt z.raw.toInt score) := by
  have weak (prior) : WeakFrame.Frame mc fc prior reads b false a.raw.toInt z.raw.toInt score := by
    rw [word,null]
    apply WeakFrame.Frame.enhanced _ _ _ _ observed
    intro probe present e value
    obtain ⟨row,member,rfl⟩ := List.mem_map.mp present
    exact ⟨row.move,rfl,seen.child row member e value⟩
  let mode := fun (_ : Bitboard.Square) => EndgameOrderMir.Heuristic.skip
  let tie := fun (sq : GeneratedSquare mc b.val) => sq.val
  let sched : GeneralOwnerMir.Schedule (GeneratedSquare mc b.val) := ⟨fun _ _ => false,fun _ _ => false⟩
  let splits : Nested.Splits (BoardFrame.Children mc fc b noCalls.prior) false cut d.toNat 7 z.raw.toInt := fun _ _ _ => none
  if exact_ : level = Cutoffs.exactSelectivity then
    if ran : GeneralNonPvMir.entryAction mc fc b d a z false checked (fun _ => none) attempts rows = .enhanced ∧
        GeneralOwnerMir.scheduled mc fc b d a z false checked cut (fun _ => none) attempts rows
          (EndgameOrderMir.framePriority mc fc b mode tie) noCalls.scout sched (Nested.scoutSplit splits) = some score then
      exact .proved ⟨fun _ prior => .scout d a z false checked cut (fun _ => none) attempts rows noCalls (noCalls_inside prior)
        mode tie sched splits full (by decide) seen ran.2,weak⟩
    else exact .rejected
  else exact .proved ⟨fun ok => (exact_ ok).elim,weak⟩

def opened (es : Joint.Tape Op) (owner : CallJournal.Choice) (f : Frame) : Option Nat := do
  let calls ← CallReturns.captureTimed (es.map Joint.Event.frameSample)
  let (act,id) ← calls.calls.toList.zipIdx.find? fun (a,_) =>
    a.worker == owner.worker && a.first == owner.opening && a.callee == .pvs
  let r ← act.row id
  guard (r.chosen = f.chosen ∧ r.returned = f.returned ∧ r.entry.endgame = true ∧ r.entry.rootLine = none ∧
    r.entry.level = owner.level.val ∧ r.entry.request = ⟨f.board,owner.depth,false,owner.cut,f.alpha,f.beta⟩ ∧
    r.score = f.score ∧ r.score = owner.word.raw.toInt)
  return act.entered

def parentCuts (checked : Bool) (d : Depth) (level : Selectivity) (beta : Score) : ProbeRetry.Result → Except String Bool
  | .miss _ => .ok false
  | .hit _ e => TranspositionTable.canCut checked e ⟨beta.raw⟩ d level true

/-- A zero-move, wipeout, stable or shallow frame returns before the ETC loop. -/
def reachesEtc (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : Position) (d : Depth) (a : Score) : Bool :=
  let generated := PvEntryMir.generated mc fc b false (fun _ => none)
  let stable := EndgameStability.tryCutoff b.val d.toNat ⟨(ScoreRefresh.toDiscDiff a).raw.toInt⟩
  decide (GeneralNonPvMir.dispatch d.toNat stable.isSome (generated.len.getD 0 != 0)
    (Board.hasLegalMoves mc (PvsBoardTree.pass b).val) generated.wipeout.join.isSome false true = .enhanced)

/-- Resolve the parent's probe and each prefix lookup against their individual instruction
    windows. The selected word is available to the writer before this activation has returned. -/
def check (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig)
    (init : ∀ a, NativeTableWrites.Journal.Even (x.memory a)) (probes : JournalBodies.Probes m) (look : ChildReads m)
    (checked : Bool) (s : CallJournal.State Op) (f : Frame) :
    Check (Bodies mc fc (s.level f) (CausalTable.lookup x init f.chosen) f) :=
  match s.choices.find? (fun owner => owner.stamp == f.chosen) with
  | none => .rejected
  | some owner =>
    if !owner.endgame || owner.pv || owner.rootLine.isSome || !reachesEtc mc fc f.board owner.depth owner.alpha then .outside else
    if shape : owner.board = f.board ∧ owner.stamp = f.chosen ∧ owner.level = s.level f ∧ owner.depth = Board.emptyCount f.board.val ∧
        f.pv = false ∧ f.alpha = owner.alpha.raw.toInt ∧ f.beta = owner.beta.raw.toInt ∧
        -32767 ≤ f.alpha ∧ f.alpha < f.beta ∧ f.beta ≤ 32767 ∧ owner.beta.raw.toInt = owner.alpha.raw.toInt+1 then
      match opened es owner f with
      | none => .rejected
      | some start => match JournalPrefix.parentAt es owner.worker start f.chosen f.board,probes f.chosen f.board with
        | some (opening,close,raw),some p =>
          let attempts := JointReads.attempts f.board (some p)
          if p.opening != opening || p.finish != close || ProbeRetry.probe (attempts.map (ProbeRetry.observe (Cutoffs.key f.board))) != raw then .rejected else
          match parentCuts checked owner.depth (s.level f) owner.beta raw with
          | .error _ => .rejected | .ok true => .outside
          | .ok false =>
            let query : PvsEnhancedCutoff.Query := ⟨owner.depth,s.level f,true,⟨owner.alpha.raw⟩⟩
            match scan mc fc f.board query (legalSquares mc f.board.val) (events es owner.worker close f.chosen) with
            | none => .rejected
            | some (windows,result) => match windows.mapM (child mc fc f.board init look f.chosen p) with
              | none => .rejected
              | some resolved =>
                let rows := resolved.map (JointReads.Child.row init)
                match result with
                | none => .outside
                | some hit =>
                  if ok : PvsEnhancedCutoff.scan query (rows.map TableReads.ChildLookupFrom.observation) = some hit ∧ f.score = hit.2 then by
                    obtain ⟨_,_,_,full,pv,lo,hi,_,_,_,null⟩ := shape
                    have got := checkedBodies mc fc f.board owner.alpha owner.beta owner.depth checked owner.cut (s.level f) f.score _
                      attempts rows (JointReads.physical init f.board (some p) resolved) full null hit ok.1 ok.2
                    cases got with
                    | outside => exact .outside
                    | rejected => exact .rejected
                    | proved bodies =>
                      apply Check.proved
                      constructor <;> rw [pv,lo,hi]
                      · exact bodies.1
                      · exact bodies.2
                  else .rejected
        | _,_ => .rejected
    else .rejected

def pending (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig)
    (init : ∀ a, NativeTableWrites.Journal.Even (x.memory a)) (probes : JournalBodies.Probes m) (look : ChildReads m)
    (pool : EndgameSearch.CacheHistory.Pool) (checked : Bool) (s : CallJournal.State Op) (op : JournalPv.Options mc)
    (f : Frame) (frames : List Frame) (body : Prop) :=
  JournalPv.pending mc fc init probes pool checked s op f frames ((check mc fc init probes look checked s f).requires body)

structure Remaining (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig)
    (init : ∀ a, NativeTableWrites.Journal.Even (x.memory a)) (probes : JournalBodies.Probes m) (look : ChildReads m)
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
    JournalPv.pending mc fc init probes pool checked s op f frames Q :=
  through _ (through _ (through _ (through _ (Check.derive _ body)))) h

theorem Remaining.owners (h : Remaining mc fc init probes look pool checked s op) :
    JournalPv.Remaining mc fc init probes pool checked s op where
  seed f mem := discharge (fun r => .internal (r.weak _)) (h.seed f mem)
  weak f mem := discharge (fun r => r.weak _) (h.weak f mem)
  exact f mem := discharge (fun r => r.exact (beq_iff_eq.mp (List.mem_filter.mp mem).2) _) (h.exact f mem)

namespace Regression

private def mc : Bitboard.Moves.X86Config := ⟨false,false⟩
private def fc : Cutoffs.FlipConfig := ⟨false,false,false⟩
private def board : Position := ⟨⟨0x100000001,0x200000002⟩,rfl⟩
private def entry (level : Selectivity) (bound : Bound) (value : Int) := Packing.newData ⟨BitVec.ofInt 32 value⟩ none bound 59 level 0 true false
private def round (w : Nat) (b : Position) (word : Option EntryData) : Joint.Tape Unit :=
  [.enter w ⟨.lookup b,0,16,0⟩,.engine (.other w),.leave w (.lookup word)]

structure Changes where
  swap : Bool := false
  skip : Bool := false
  extra : Bool := false
  ready : Bool := false
  wrongWorker : Bool := false
  childLevel : Selectivity := 5
  bound : Bound := .upper
  score : Int := 4

private def slotRow (w : Window (GeneratedSquare mc board.val)) : TableReads.ChildLookupFrom mc fc board (fun _ => True) := by
  let b := (play mc fc board w.move).val
  match e : w.word with
  | none => exact ⟨w.move,none,none,.miss,.miss⟩
  | some v =>
    let slot : Clear.SlotState := ⟨0,b.val.player,b.val.opponent,Packing.pack false v⟩
    let raw := SerialTable.StableSample.raw false slot (by change (0 : BitVec 64) &&& 1 = 0; decide)
    if occupied : v.bound ≠ 0 then
      apply TableReads.ChildLookupFrom.mk w.move (some v) none _ .miss
      apply TableReads.SlotLookupFrom.hit raw v trivial
      apply (LookupCut.readContract_proven ..).mpr
      exact ⟨by change (0 : BitVec 64) &&& 1 = 0; decide,rfl,rfl,rfl,Packing.unpack_pack false v,occupied⟩
    else exact ⟨w.move,none,none,.miss,.miss⟩

/-- The first child misses and the second returns -4. A selected word of +4
    suffices before the writer's first store or the caller's eventual return. -/
private def trial (level : Selectivity) (change : Changes) : Option Bool := do
  let [m,n] := legalSquares mc board.val | none
  let b := (play mc fc board m).val
  let c := (play mc fc board n).val
  let q : PvsEnhancedCutoff.Query := ⟨60,level,true,⟨0⟩⟩
  let w := some (entry change.childLevel change.bound (-4))
  let tape : Joint.Tape Unit := [.engine (.other 0)] ++
    (if change.skip then [] else round 0 (if change.swap then c else b) (if change.ready then w else none)) ++
    [.engine (.other 1)] ++ round (if change.wrongWorker then 1 else 0) c w ++
    (if change.extra then round 0 c w else [])
  let some (windows,hit) := scan mc fc board q [m,n] (events tape 0 0 tape.length) | return false
  let some actual := hit | none
  let rows := windows.map slotRow
  if ok : PvsEnhancedCutoff.scan q (rows.map TableReads.ChildLookupFrom.observation) = some actual ∧ change.score = actual.2 then
    match checkedBodies mc fc board ⟨0⟩ ⟨1⟩ 60 true false level change.score (fun _ _ => True) [] rows
        ⟨fun _ _ _ => trivial,fun _ _ _ _ => trivial⟩ (by unfold ExactDepth.Full; decide) rfl actual ok.1 ok.2 with
    | .proved _ => return true | .rejected => return false | .outside => none
  else return false

#guard trial 5 {} = some true
#guard trial 2 {childLevel := 2} = some true
#guard trial 5 {score := -4} = some false
#guard trial 5 {swap := true} = some false
#guard trial 5 {skip := true} = some false
#guard trial 5 {extra := true} = some false
#guard trial 5 {ready := true} = some false
#guard trial 5 {wrongWorker := true} = some false
#guard trial 5 {childLevel := 2} = none
#guard trial 5 {bound := .lower} = none

end Regression

end NeuralReversi.PvsParallel.Interleaving.JournalScout
