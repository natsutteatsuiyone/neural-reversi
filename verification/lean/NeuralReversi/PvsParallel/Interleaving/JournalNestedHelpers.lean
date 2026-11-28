module
public import NeuralReversi.PvsParallel.Interleaving.CutoffPollReplay
public import NeuralReversi.PvsParallel.Interleaving.JournalHelpers
meta import NeuralReversi.PvsParallel.Interleaving.CutoffPollReplay
meta import NeuralReversi.PvsParallel.Interleaving.JournalHelpers

@[expose] public section

namespace NeuralReversi.PvsParallel.Interleaving.JournalNestedHelpers

open Threading SplitWorkerMir
open PvsBoardTree (Position GeneratedSquare play)
open JournalHelpers (Receipt Uses)
open CallReturns (Scope Row)

def prior (sc : Scope) := CausalTable.returned (fun i j => CallReturns.earlier i j = true) (sc.rows.map Row.frame) sc.consumer

example (sc : Scope) (plan : CallReturns.Plan Position) : (sc.nativePlan plan).prior = prior sc := rfl

def one (sc : Scope) (r : Receipt sc.rows) (next : Position) (wide : Bool) (lo hi score : Int) :
    Option (PLift (prior sc next wide lo hi score)) := do
  let q := if wide then wideQuery r.input else scoutQuery r.input
  let req := CallReturns.queryRequest next q
  let n := if wide then r.first+1 else r.first
  if shape : r.scope.consumer = sc.consumer ∧ req.pv = wide ∧ req.alpha = lo ∧ req.beta = hi then
    if found : ((r.scope.atSite n .splitSearch (if wide then 1 else 0) req).map Row.score) = some score then
      return ⟨by
        obtain ⟨row,atSite,word⟩ := Option.map_eq_some_iff.mp found
        have actual := (r.scope.ordinal_hit (r.scope.atSite_hit atSite).1).2.2.2.2
        rw [r.registry,shape.1,shape.2.1,shape.2.2.1,shape.2.2.2,word] at actual
        exact actual⟩
    else none
  else none

abbrev Indexed (sc : Scope) (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig)
    (b : Position) (moves : List (GeneratedSquare mc b.val)) : PvsMoveLoop.ChildRelation Nat :=
  fun i pv a z v => ∃ m, moves[i]? = some m ∧ BoardFrame.Children mc fc b (prior sc) m pv a z v

def packet (es : Joint.Tape Op) (sc : Scope) (bank level : Nat) (mark : AtomicWord.Event)
    (t : NativeSplit.Tickets rc pv beta) (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : Position)
    (moves : List (GeneratedSquare mc b.val)) (root : Bool) (k : Nat) (p : FieldStores.Raw) :
    Option (PLift (Uses (Indexed sc mc fc b moves) beta p.input)) := do
  let i := p.input
  let actor ← t.state.writes.find? (fun op => op.old.toNat == i.index)
  let r ← JournalHelpers.report es sc.rows sc.parent bank level mark t k actor.worker p mc fc b moves root
  match atMove : moves[i.index]? with
  | none => none
  | some m => do
    let next := (play mc fc b m).val
    let scout ← one sc r next false (-(i.alpha.toInt+1)) (-i.alpha.toInt) i.scout.toInt
    if again : retry i = SplitWindow.reSearch i.pv i.alpha.toInt i.scout.toInt then
      if used : retry i = true then
        let wide ← one sc r next true (-beta) (-i.fresh.toInt) i.wide.toInt
        return ⟨⟨⟨m,atMove,scout.down⟩,fun _ => ⟨m,atMove,wide.down⟩,again⟩⟩
      else return ⟨⟨⟨m,atMove,scout.down⟩,fun yes => (used yes).elim,again⟩⟩
    else none

/-- Both owner paths use the same finite check. Each caller's child relation is obtained by
    substituting the move actually indexed in the scheduler, including at a root chosen score. -/
def trace (es : Joint.Tape Op) (sc : Scope) (level : Nat) (mc : Bitboard.Moves.X86Config)
    (fc : Cutoffs.FlipConfig) (b : Position) (moves : List (GeneratedSquare mc b.val))
    {t : NativeSplit.Tickets rc pv beta} (raw : AlphaOrigin.Capture t root cut depth (fun _ _ _ _ _ => True))
    (castMove : ∀ i wide a z v, Indexed sc mc fc b moves i wide a z v → done_ i wide a z v) :
    Option (ObservedJoin.Trace (ScoreOrigin.task t root cut depth) done_ rc.length t.start) := do
  guard (JournalHelpers.ownerExit es sc b level raw)
  let test := fun k p => (packet es sc raw.bankIndex level (raw.mask.get raw.generation.mark).event
    t mc fc b moves root k p).map fun (calls : PLift (Uses _ beta p.input)) =>
      (⟨⟨castMove _ _ _ _ _ calls.down.scout,fun yes => castMove _ _ _ _ _ (calls.down.wide yes),calls.down.again⟩⟩ :
        PLift (Uses done_ beta p.input))
  if checked : t.calls.zipIdx.all (fun (p,k) => p.stopped || (test k p).isSome) = true then
    let completed := raw.withReports (JournalHelpers.reportsUsing (raw := raw) test checked)
    return .physical completed.score.frame.physical
  else none

/-- Physical scheduler operands. Each indexed constructor fixes the move-list length, split
    cursor, node kind and window. Child scores are reconstructed from the common event tape. -/
inductive Capture (es : Joint.Tape Op) (sc : Scope) (level : Nat) (mc : Bitboard.Moves.X86Config)
    (fc : Cutoffs.FlipConfig) (b : Position) (moves : List (GeneratedSquare mc b.val)) :
    Schedule.Task → Nat → {w : Nat} → BitVec w → Type where
  | native (fallback : GeneratedSquare mc b.val) (t : NativeSplit.Tickets rc pv beta) (cut : Bool) (depth : Nat)
      (raw : AlphaOrigin.Capture t false cut depth (fun _ _ _ _ _ => True)) :
      Capture es sc level mc fc b moves (ScoreOrigin.task t false cut depth) rc.length t.start

def Capture.fromTape (es : Joint.Tape Op) (sc : Scope) (level : Nat)
    (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : Position)
    (moves : List (GeneratedSquare mc b.val)) (fallback : GeneratedSquare mc b.val)
    (t : NativeSplit.Tickets rc pv beta) (cut : Bool) (depth : Nat)
    (data : AlphaOrigin.SetupOperands t) (stops : CutoffPollReplay.Context data es)
    (before : Option (Threading.AtomicWord.Event → Threading.AtomicWord.Event → Bool) := none) :
    Option (Capture es sc level mc fc b moves (ScoreOrigin.task t false cut depth) rc.length t.start) := do
  return .native fallback t cut depth (← CutoffPollReplay.fromTape data false cut depth es stops before)

theorem Capture.fromTape_dispatches
    (ok : Capture.fromTape es sc level mc fc b moves fallback t cut depth data stops before = some out) :
    ∀ d ∈ t.dispatches, ∃ sd : ActivePointer.SplitDispatch data.memory.history
        (CutoffPollReplay.slots data) es (CutoffPollReplay.generation data) d.entry,
      ActivePointer.splitDispatch stops.chronology (CutoffPollReplay.slots data) es
        (CutoffPollReplay.generation data) d.entry = some sd ∧
      sd.origin.generation = CutoffPollReplay.generation data ∧
      Threading.AtomicWord.SequencedBefore sd.origin.drop d.entry := by
  obtain ⟨cap,replayed,_⟩ := Option.bind_eq_some_iff.mp ok
  intro d mem
  obtain ⟨sd,found,gen,_,after⟩ := CutoffPollReplay.fromTape_dispatches t data false cut depth es stops before cap replayed d mem
  exact ⟨sd,found,gen,after⟩

abbrev Feed (es : Joint.Tape Op) (sc : Scope) (level : Nat) (mc : Bitboard.Moves.X86Config)
    (fc : Cutoffs.FlipConfig) (b : Position) (pv cut : Bool) (depth width : Nat) (beta : Int) :=
  ∀ (moves : List (GeneratedSquare mc b.val)) (idx : Nat) (entry : SplitWindow.State),
    Option (Capture es sc level mc fc b moves ⟨pv,false,cut,depth,beta,entry⟩ moves.length (BitVec.ofNat width idx))

def bind (cap : Capture es sc level mc fc b moves task len cursor) :
    Option ((fallback : GeneratedSquare mc b.val) × ObservedJoin.Trace task
      (fun i => BoardFrame.Children mc fc b (prior sc) (moves[i]?.getD fallback)) len cursor) :=
  match cap with
  | .native fallback t cut depth raw => do
    let physical ← trace es sc level mc fc b moves raw (done_ := fun i => BoardFrame.Children mc fc b (prior sc)
      (moves[i]?.getD fallback)) (fun _ _ _ _ _ ⟨m,hit,returned⟩ => by simpa only [hit,Option.getD_some] using returned)
    return ⟨fallback,physical⟩

def splits (feed : Feed es sc level mc fc b pv cut depth width beta) :
    Nested.Splits (BoardFrame.Children mc fc b (prior sc)) pv cut depth width beta := fun moves idx entry => do
  let cap ← feed moves idx entry
  let ⟨fallback,trace⟩ ← bind cap
  return ⟨fallback,trace⟩

/-- A root writer can publish after choosing its score and before returning to its caller.
    Its split register and helper completions use only events available by that choice. -/
inductive RootCapture (es : Joint.Tape Op) (sc : Scope) (level : Nat) (mc : Bitboard.Moves.X86Config)
    (fc : Cutoffs.FlipConfig) (b : Position) (r : RootOwnerMir.Result (GeneratedSquare mc b.val)) :
    Schedule.Task → Nat → {w : Nat} → BitVec w → Type where
  | native (t : NativeSplit.Tickets rc true beta) (depth : Nat)
      (raw : AlphaOrigin.Capture t true false depth (fun _ _ _ _ _ => True)) :
      RootCapture es sc level mc fc b r (ScoreOrigin.task t true false depth) rc.length t.start

def RootCapture.fromTape (es : Joint.Tape Op) (sc : Scope) (level : Nat)
    (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : Position)
    (r : RootOwnerMir.Result (GeneratedSquare mc b.val)) (t : NativeSplit.Tickets rc true beta) (depth : Nat)
    (data : AlphaOrigin.SetupOperands t) (stops : CutoffPollReplay.Context data es)
    (before : Option (Threading.AtomicWord.Event → Threading.AtomicWord.Event → Bool) := none) :
    Option (RootCapture es sc level mc fc b r (ScoreOrigin.task t true false depth) rc.length t.start) := do
  return .native t depth (← CutoffPollReplay.fromTape data true false depth es stops before)

theorem RootCapture.fromTape_dispatches
    (ok : RootCapture.fromTape es sc level mc fc b r t depth data stops before = some out) :
    ∀ d ∈ t.dispatches, ∃ sd : ActivePointer.SplitDispatch data.memory.history
        (CutoffPollReplay.slots data) es (CutoffPollReplay.generation data) d.entry,
      ActivePointer.splitDispatch stops.chronology (CutoffPollReplay.slots data) es
        (CutoffPollReplay.generation data) d.entry = some sd ∧
      sd.origin.generation = CutoffPollReplay.generation data ∧
      Threading.AtomicWord.SequencedBefore sd.origin.drop d.entry := by
  obtain ⟨cap,replayed,_⟩ := Option.bind_eq_some_iff.mp ok
  intro d mem
  obtain ⟨sd,found,gen,_,after⟩ := CutoffPollReplay.fromTape_dispatches t data true false depth es stops before cap replayed d mem
  exact ⟨sd,found,gen,after⟩

abbrev RootFeed (es : Joint.Tape Op) (sc : Scope) (level : Nat) (mc : Bitboard.Moves.X86Config)
    (fc : Cutoffs.FlipConfig) (b : Position) (width : Nat) (beta : Int) :=
  ∀ (r : RootOwnerMir.Result (GeneratedSquare mc b.val)), Option
    (RootCapture es sc level mc fc b r (RootSchedule.task r (Board.emptyCount b.val).toNat beta false)
      (RootTrial.labels r).length (BitVec.ofNat width r.remaining.idx))

def rootTrace {world : CausalTable.Values} (cap : RootCapture es sc level mc fc b r task len cursor)
    (included : ∀ b p a z v, prior sc b p a z v → world b p a z v) :
    Option (ObservedJoin.Trace task (RootSchedule.indexed (BoardFrame.Children mc fc b world) r) len cursor) :=
  match cap with
  | .native t depth raw => trace es sc level mc fc b (r.visited ++ r.remaining.pending) raw
      (fun _ _ _ _ _ ⟨m,atMove,returned⟩ => ⟨m,atMove,included _ _ _ _ _ returned⟩)

def rootSplit {world : CausalTable.Values} (feed : RootFeed es sc level mc fc b width beta)
    (included : ∀ b p a z v, prior sc b p a z v → world b p a z v) (r : RootOwnerMir.Result (GeneratedSquare mc b.val)) :
    Option (ObservedJoin.Trace (RootSchedule.task r (Board.emptyCount b.val).toNat beta false)
      (RootSchedule.indexed (BoardFrame.Children mc fc b world) r) (RootTrial.labels r).length (BitVec.ofNat width r.remaining.idx)) :=
  (feed r).bind (fun cap => rootTrace cap included)

namespace Regression

/-- Two active split searches make the same null-window request. Only the nominated helper
    supplies its return, while the owner's registry can contain both workers' completed frames. -/
private def sameRequest (parent : Nat) (score : Int) (early : Bool) : Bool := (show Option Unit from do
  let b : Position := ⟨⟨0x1000000000,0x81c000000⟩,rfl⟩
  let i : Input := ⟨1,60,true,false,true,8,0,2,BitVec.ofInt 32 score,0⟩
  let q := scoutQuery i
  let args : NativeFrames.Args := ⟨b.val,59,⟨q.alpha⟩,⟨q.beta⟩,true,5,false,q.cut,none⟩
  let enter := fun w t node input => Joint.Event.engine (CallJournal.Event.localCall w (.entry t 0 node .none none) input)
  let push := fun w v => Joint.Event.engine (.localCall w (.scored 2 ⟨BitVec.ofInt 32 v⟩) none)
  let leave := fun w v => Joint.Event.engine (.localCall w (.leave 3 1 .pvs none (some ⟨BitVec.ofInt 32 v⟩)) none)
  let es : Joint.Tape Unit := [enter 1 0 .splitSearch none,enter 2 0 .splitSearch none,
    enter 1 1 .pvs (some args),enter 2 1 .pvs (some args),push 2 (-5),leave 2 (-5),push 1 (-4),leave 1 (-4)]
  let state ← CallReturns.captureTimed (es.map Joint.Event.frameSample)
  let time := if early then 6 else es.length
  let sc : Scope := ⟨state.rows true,17,time,state.calls⟩
  let helper : Scope := ⟨sc.rows,parent,time,state.calls⟩
  let _ ← one sc ⟨helper,0,i,rfl⟩ b false (-1) 0 score
  return ()).isSome

#guard sameRequest 0 (-4) false
#guard sameRequest 1 (-5) false
#guard !sameRequest 0 (-5) false
#guard !sameRequest 1 (-4) false
#guard !sameRequest 0 (-4) true
#guard sameRequest 1 (-5) true

end Regression

end NeuralReversi.PvsParallel.Interleaving.JournalNestedHelpers
