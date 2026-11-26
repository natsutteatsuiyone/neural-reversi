import NeuralReversi.PvsParallel.Interleaving.JournalPrefix
import NeuralReversi.PvsParallel.Interleaving.CallReturns

namespace NeuralReversi.PvsParallel.Interleaving.JournalPass

open PvsBoardTree (Position pass)
open TranspositionTable hiding Score
open ScoreRefresh (Score)
open JournalBodies (Check)
open CausalTable (Frame Lookups)
open NativeNodes (Returns)
open CallReturns (Row Activation)

private abbrev Prior (frames : List Frame) (f : Frame) := CausalTable.returned (· < ·) frames f.chosen

structure Result (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (frames : List Frame)
    (reads : Lookups) (f : Frame) : Prop where
  exact : Nested.Body mc fc (Prior frames f) reads f.board f.pv f.alpha f.beta f.score
  weak : WeakFrame.Frame mc fc (Prior frames f) reads f.board f.pv f.alpha f.beta f.score

private def oneReturn (child : Frame) (at_ : Nat) (depth : Depth) (cut : Bool) : Returns :=
  ⟨[child],fun _ => depth,fun _ => cut,fun i j => decide (i < j),at_,fun _ _ _ _ _ _ => some 0⟩

private theorem inside (frames : List Frame) (member : child ∈ frames) :
    (oneReturn child at_ depth cut).InPrefix (CausalTable.returned (· < ·) frames at_) := by
  intro b pv a z v used
  obtain ⟨g,h,t,ret,lt,vals⟩ := used
  have same : g = child := List.mem_singleton.mp h
  subst g
  exact ⟨child,member,t,ret,of_decide_eq_true lt,vals⟩

/-- Replay the native owner with exactly one available return. A move or split
    branch cannot borrow a second result from an unrelated frame. -/
private def verify (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (fs : List Frame)
    (reads : Lookups) (f : Frame) (g : Frame) (mem : g ∈ fs) (d : Depth) (a z : Score)
    (cut checked : Bool) (full : ExactDepth.Full f.board d)
    (slots : f.alpha = a.raw.toInt ∧ f.beta = z.raw.toInt)
    (stuck : Board.hasLegalMoves mc f.board.val = false) (mobile : Board.hasLegalMoves mc (pass f.board).val = true)
    (used : Prior fs f (pass f.board) f.pv (-f.beta) (-f.alpha) g.score) (value : f.score = -g.score) :
    Check (Result mc fc fs reads f) := by
  let x := oneReturn g f.chosen d (!f.pv && !cut)
  have weak : WeakFrame.Frame mc fc (Prior fs f) reads f.board f.pv f.alpha f.beta f.score := by
    rw [value]
    exact .pass (PvsBoardTree.nil_of_stuck stuck) mobile used
  let mode := fun (_ : Bitboard.Square) => EndgameOrderMir.Heuristic.skip
  let tie := fun (m : PvsBoardTree.GeneratedSquare mc f.board.val) => m.val
  let blank : MoveList.Initialization.Memory Move.Move := fun _ => none
  cases pv : f.pv with
  | true =>
    let split : Nested.Splits (BoardFrame.Children mc fc f.board x.prior) true cut d.toNat 7 f.beta := fun _ _ _ => none
    let schedule := fun (_ : Nat) => (false,false)
    exact if run : PvSerialMir.scheduled mc fc f.board d false checked blank f.alpha f.beta
        (fun b cd lo hi => x.reply b cd true false lo hi) (EndgameOrderMir.pvOrder mc fc f.board mode tie none)
        (x.pvReplies mc fc f.board d cut) schedule (Nested.pvSplit split) = some f.score then
      .proved ⟨pv ▸ Nested.Body.pv d false checked cut blank x (inside fs mem) mode tie none schedule split
        full (by decide) run,weak⟩ else .rejected
  | false =>
    let split : Nested.Splits (BoardFrame.Children mc fc f.board x.prior) false cut d.toNat 7 z.raw.toInt := fun _ _ _ => none
    let schedule : GeneralOwnerMir.Schedule (PvsBoardTree.GeneratedSquare mc f.board.val) := ⟨fun _ _ => false,fun _ _ => false⟩
    let rows : List (TableReads.ChildLookupFrom mc fc f.board (fun _ => False)) := []
    exact if run : GeneralOwnerMir.scheduled mc fc f.board d a z false checked cut blank [] rows
        (EndgameOrderMir.framePriority mc fc f.board mode tie) x.scout schedule (Nested.scoutSplit split) = some f.score then
      .proved ⟨by
        rw [pv,slots.1,slots.2]
        exact .scout d a z false checked cut blank [] rows x (inside fs mem) mode tie schedule split full (by decide)
          ⟨fun _ _ hit => by contradiction,fun _ h => (List.not_mem_nil h).elim⟩ run,weak⟩ else .rejected

private def quiet (es : Joint.Tape Op) (worker start stop : Nat) :=
  es.zipIdx.all fun (e,t) =>
    e.worker != worker || !(decide (start < t ∧ t < stop)) || match e with | .engine (.other _) => true | _ => false

/-- Use every activation under the parent, including abandoned calls. The
    worker can issue arbitrary instructions inside the child's lifetime. -/
private def physical (es : Joint.Tape Op) (state : CallReturns.State) (owner : CallJournal.Choice) :
    Option (CallReturns.Entry × CallReturns.Row) := do
  let (a,id) ← state.calls.toList.zipIdx.find? fun (a,_) =>
    a.worker == owner.worker && a.first == owner.opening && a.callee == .pvs
  let parent ← a.input
  let (time,score) ← a.chosen
  guard (time == owner.stamp && score.raw == owner.word.raw)
  let [(child,cid)] := state.calls.toList.zipIdx.filter (fun (c,_) => c.parent == some id && decide (c.entered < time)) | none
  let row ← child.row cid
  let returned ← row.returned
  guard (child.callee == .pvs && child.site == NativeFrames.OwnerSites.Role.passChild.site &&
    child.worker == a.worker && decide (a.entered < child.entered ∧ child.entered < row.chosen ∧ row.chosen < returned ∧ returned < time) &&
    quiet es a.worker a.entered child.entered && quiet es a.worker returned time)
  return (parent,row)

/-- Consume the pass instruction on the observed call stack and resolve its only
    returned child in the score registry belonging to this history. -/
def check (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (es : Joint.Tape Op)
    (s : CallJournal.State Op) (fs : List Frame) (reads : Lookups) (f : Frame) (checked : Bool) : Check (Result mc fc fs reads f) :=
  match s.choices.find? (fun parent => parent.stamp == f.chosen) with
  | none => .rejected
  | some owner =>
    if !owner.endgame || owner.rootLine.isSome || (!owner.pv && decide (owner.depth.toNat ≤ 12)) then .outside else
    if !owner.pv && (EndgameStability.tryCutoff f.board.val owner.depth.toNat
        ⟨(ScoreRefresh.toDiscDiff owner.alpha).raw.toInt⟩).isSome then .outside else
    if stuck : Board.hasLegalMoves mc f.board.val = false then
      if mobile : Board.hasLegalMoves mc (pass f.board).val = true then
        match CallReturns.captureTimed (es.map Joint.Event.frameSample) with
        | none => .rejected
        | some calls =>
          match physical es calls owner with
          | none => .rejected
          | some (parent,child) =>
            let q := parent.request
            let actual := child.entry.request
            let field := fs.find? fun g => g.chosen == child.chosen
            match found : field with
            | none => .rejected
            | some g =>
              if same : owner.stamp = f.chosen ∧ owner.board = f.board ∧ owner.level = s.level f ∧
                  owner.depth = Board.emptyCount f.board.val ∧ owner.pv = f.pv ∧
                  owner.word.raw.toInt = f.score ∧ f.alpha = owner.alpha.raw.toInt ∧ f.beta = owner.beta.raw.toInt ∧
                  -32767 ≤ f.alpha ∧ f.alpha < f.beta ∧ f.beta ≤ 32767 ∧ (f.pv = false → f.beta = f.alpha+1) ∧
                  q = ⟨f.board,owner.depth,f.pv,owner.cut,f.alpha,f.beta⟩ ∧ parent.endgame = true ∧
                  parent.rootLine = none ∧ parent.level = owner.level.val ∧
                  child.entry.endgame = true ∧ child.entry.rootLine = none ∧ child.entry.level = parent.level ∧
                  actual = ⟨pass f.board,owner.depth,f.pv,!f.pv && !owner.cut,-f.beta,-f.alpha⟩ ∧
                  g.board = actual.board ∧ g.pv = actual.pv ∧ g.alpha = actual.alpha ∧ g.beta = actual.beta ∧
                  g.score = child.score ∧ g.returned = child.returned ∧ f.score = -g.score then
                match ht : g.returned with
                | none => .rejected
                | some t =>
                  if lt : t < f.chosen then by
                    obtain ⟨_,_,_,full,_,_,lo,hi,_,_,_,_,_,_,_,_,_,_,_,args,b,p,a,z,_,_,v⟩ := same
                    have member : g ∈ fs := List.mem_of_find?_eq_some found
                    rw [args] at b p a z
                    exact verify mc fc fs reads f g member owner.depth owner.alpha owner.beta owner.cut checked full
                      ⟨lo,hi⟩ stuck mobile ⟨g,member,t,ht,lt,b,p,a,z,rfl⟩ v
                  else .rejected
              else .rejected
      else .outside
    else .outside

variable {x : NativeTableWrites.Journal.Execution c endian rs} {es : Joint.Tape Op} {m : Joint.Atomics x es}

abbrev seedRows (s : CallJournal.State Op) (f : Frame) := s.frames.filter (fun g => s.level g == s.level f)

def pending (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig)
    (init : ∀ a, NativeTableWrites.Journal.Even (x.memory a)) (probes : JournalBodies.Probes m)
    (pool : EndgameSearch.CacheHistory.Pool) (checked : Bool) (s : CallJournal.State Op)
    (f : Frame) (frames : List Frame) (body : Prop) :=
  (JournalBodies.check mc fc init probes pool s f).requires ((JournalPrefix.check mc fc init probes checked s f).requires
    ((check mc fc es s frames (CausalTable.lookup x init f.chosen) f checked).requires body))

structure Remaining (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig)
    (init : ∀ a, NativeTableWrites.Journal.Even (x.memory a)) (probes : JournalBodies.Probes m)
    (pool : EndgameSearch.CacheHistory.Pool) (checked : Bool) (s : CallJournal.State Op) : Prop where
  seed : ∀ f ∈ s.frames, pending mc fc init probes pool checked s f (seedRows s f)
    (JournalBodies.SeedObligation mc fc s f (CausalTable.lookup x init f.chosen))
  weak : ∀ f ∈ s.weakChildren, pending mc fc init probes pool checked s f s.weakChildren
    (JournalBodies.WeakObligation mc fc s f (CausalTable.lookup x init f.chosen))
  exact : ∀ f ∈ s.exactFrames, pending mc fc init probes pool checked s f s.exactFrames
    (JournalBodies.ExactObligation mc fc s f (CausalTable.lookup x init f.chosen))

private theorem through (test : Check P) (use : A → B) : test.requires A → test.requires B := by
  cases test with
  | outside => exact use
  | rejected => exact id
  | proved _ => exact id

theorem Remaining.prefixes (h : Remaining mc fc init probes pool checked s) : JournalPrefix.Remaining mc fc init probes pool checked s where
  seed f mem := through _ (through _ (Check.derive _ (fun r => .internal r.weak))) (h.seed f mem)
  weak f mem := through _ (through _ (Check.derive _ (fun r => r.weak))) (h.weak f mem)
  exact f mem := through _ (through _ (Check.derive _ (fun r => r.exact))) (h.exact f mem)

namespace Fixture

-- This position has fourteen empties. From the initial board it is reached by
-- c4c3d3c5f6e3e2f5e6f1g6f2b4b5c2c1b6c6b1a4f3g2g3e7e8b7g1g5d1d7g4h7f4h5c7f7a6e1c8d6a5d8a3h3g7d2.
private def board : Position := ⟨⟨0x147e67536745644e,0x088018ac18b81830⟩,rfl⟩
private def mc : Bitboard.Moves.X86Config := ⟨false,false⟩
private def fc : Cutoffs.FlipConfig := ⟨false,false,false⟩
private def calls : NativeTableWrites.Journal.Calls := fun _ => ⟨⟨0,0⟩,⟨0,rfl⟩,.clear,false,id⟩

private structure Mutation where
  site : Nat := 2
  cut : Bool := false
  depth : Depth := 14
  alpha : Int := 0
  wrongBoard : Bool := false
  score : Int := -15872
  hole : Bool := false
  crossThread : Bool := false
  level : Option Selectivity := none
  output : Bool := true

private def tape (pv cut : Bool) (level : Selectivity) (m : Mutation)
    (b : Position := board) (childScore : Int := 15872) : Joint.Tape Unit :=
  let lo : Int := if pv then -16000 else -15872
  let hi : Int := if pv then -15500 else -15871
  let score := fun (v : Int) => Score.mk (BitVec.ofInt 32 v)
  let parent : NativeFrames.Args := ⟨b.val,Board.emptyCount b.val,score lo,score hi,true,level,pv,cut,none⟩
  let child : NativeFrames.Args := ⟨(if m.wrongBoard then b else pass b).val,m.depth,score (-hi+m.alpha),score (-lo),true,m.level.getD level,
    pv,if m.cut then !(!pv && !cut) else (!pv && !cut),none⟩
  let on := if m.crossThread then 1 else 0
  let enter := fun w t site args => Joint.Event.engine (.localCall w (.entry t site .pvs .none none) (some args))
  let scored := fun w t v => Joint.Event.engine (.localCall w (.scored t (score v)) none)
  let leave := fun w t op out => Joint.Event.engine (.localCall w (.leave t op .pvs none out) none)
  [enter 0 0 6 parent] ++
    (if m.hole then [enter 0 1 2 child,leave 0 2 1 none] else []) ++
    [enter on 3 m.site child,.engine (.other 1),scored on 4 childScore,leave on 5 3 (if m.output then some (score childScore) else none),
     scored 0 6 m.score]

private def result (pv cut : Bool) (level : Selectivity) (m : Mutation)
    (b : Position := board) (childScore : Int := 15872) : Option (List Bool) := do
  let events := tape pv cut level m b childScore
  let s ← CallJournal.run mc fc calls false (events.map Joint.Event.engineEvent)
  let f ← s.frames.head?
  let check := fun fs => match check mc fc events s fs (fun _ _ => False) f true with
    | .outside => none | .rejected => some false | .proved _ => some true
  [seedRows s f,if level = Cutoffs.exactSelectivity then s.exactFrames else s.weakChildren].mapM check

#guard result true false 5 {} = some [true,true]
#guard result false false 5 {} = some [true,true]
#guard result false true 2 {} = some [true,true]
#guard result true true 2 {} = some [true,true]
#guard result false false 5 {site := 6} = some [false,false]
#guard result false false 5 {cut := true} = some [false,false]
#guard result true false 5 {depth := 13} = some [false,false]
#guard result true false 5 {alpha := 1} = some [false,false]
#guard result false false 5 {wrongBoard := true} = some [false,false]
#guard result true false 5 {score := -15871} = some [false,false]
#guard result true false 5 {hole := true} = some [false,false]
#guard result true false 5 {crossThread := true} = some [false,false]
#guard result false false 2 {level := some 3} = some [false,false]
#guard result true false 5 {output := false} = some [false,false]

-- Rust's forced_pass_preserves_depth_and_reflects_the_pv_window board.
private def lastSquare : Position := ⟨⟨0xfffe,0xffffffffffff0000⟩,rfl⟩
#guard result true false 5 {depth := 1,score := -9728} lastSquare 9728 = some [true,true]
#guard result true false 5 {depth := 0,score := -9728} lastSquare 9728 = some [false,false]

end Fixture

end NeuralReversi.PvsParallel.Interleaving.JournalPass
