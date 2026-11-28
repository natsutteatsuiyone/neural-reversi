module
public import NeuralReversi.PvsParallel.Interleaving.JournalProbCut
public import NeuralReversi.PvsParallel.Interleaving.JournalNestedHelpers
meta import NeuralReversi.PvsParallel.Interleaving.CallReturns
meta import NeuralReversi.PvsParallel.Interleaving.JournalNestedHelpers
meta import NeuralReversi.PvsParallel.Interleaving.JournalProbCut

@[expose] public section

namespace NeuralReversi.PvsParallel.Interleaving.JournalScoutOwner

open PvsBoardTree (Position GeneratedSquare legalSquares play)
open TranspositionTable hiding Score
open ScoreRefresh (Score)
open CausalTable (Frame Lookups Values)
open NativeFrames.OwnerSites (Role)
open GeneralNonPvMir (presearch order)
open EndgameOrderMir (Heuristic framePriority)
open JournalBodies (Check)

structure Options (mc : Bitboard.Moves.X86Config) (b : Position) where
  mode : Bitboard.Square → Heuristic
  tie : GeneratedSquare mc b.val → Fin 64
  schedule : GeneralOwnerMir.Schedule (GeneratedSquare mc b.val)

structure Owner (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : Position)
    (x : NativeNodes.Returns) (a z : Int) where
  issued : List (GeneratedSquare mc b.val × Nat)
  handedOff : Bool
  score : Int
  weak : ∀ prior reads, x.InPrefix prior → WeakFrame.Frame mc fc prior reads b false a z score

def path (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : Position) (d : Depth)
    (a z : Int) (null : z = a+1) (cut : Bool) (op : Options mc b) (x : NativeNodes.Returns)
    (first : GeneratedSquare mc b.val) (rest : List (GeneratedSquare mc b.val)) (gen : legalSquares mc b.val = first :: rest)
    (hint : Option (GeneratedSquare mc b.val)) (width : Nat)
    (splits : Nested.Splits (BoardFrame.Children mc fc b x.prior) false cut d.toNat width z) : Option (Owner mc fc b x a z) := by
  let key := framePriority mc fc b op.mode op.tie
  let ordered := order key hint (first :: rest)
  let move := hint.getD first
  let used := (first :: rest).length = 1 ∨ hint.isSome = true
  let front := if used then [(move,Role.presearch.site)] else []
  let search := GeneralNonPvMir.observe mc fc b d cut x.scout
  let reply := GeneralOwnerMir.scoutReply mc fc b d cut x.scout a
  let pre := presearch (first :: rest).length hint first a z (fun m => search m false)
  have legal (m) (_ : hint = some m) : m ∈ first :: rest := by
    obtain ⟨n,in_,eq⟩ := (PvsBoardTree.legalSquares_complete mc b.val m.val).mpr ((Board.getMoves_exact mc b.val _).mp m.property)
    simpa [←gen,show n = m from Subtype.ext eq] using in_
  have low := fun m sq (_ : hint = some sq) (_ : m ∈ first :: rest) (_ : m ≠ sq) => EndgameOrderMir.framePriority_low mc fc b op.mode op.tie m
  have perm := GeneralNonPvMir.orderPerm key (first :: rest) hint legal low
  have witness (prior) (inside : x.InPrefix prior) (m) (hit : (reply m).isSome) :
      BoardFrame.Children mc fc b prior m false (-(a+1)) (-a) (search m false (-(a+1)) (-a)) := by
    obtain ⟨v,got⟩ := Option.isSome_iff_exists.mp hit
    change prior _ false _ _ ((reply m).getD 0)
    rw [got]
    exact inside _ _ _ _ _ (x.hit got)
  if valid : used → (reply move).isSome = true ∧ search move false (-(a+1)) (-a) < PvsMoveLoop.scoreInf then
    cases choice : pre with
    | storeLower sq score =>
      exact some ⟨front,false,score,fun prior reads inside => by
        have suffix : GeneralNonPvMir.PresearchSuffix (BoardFrame.Children mc fc b prior) search a z ordered (.returned score) pre := by rw [choice]; rfl
        obtain ⟨_,_,ran⟩ := GeneralNonPvMir.presearchRun key _ search first rest hint a z null legal low
          (fun u => witness prior inside move (valid u).1) (fun u => (valid u).2) _ suffix
        exact .moves first rest ordered gen perm (.returned search ran)⟩
    | owner idx best =>
      let q : PvsPostpone.Queue _ := ⟨idx,ordered.drop idx,[]⟩
      match found : GeneralOwnerMir.run op.schedule d.toNat cut search (q.budget+1) 0 q ⟨a,best⟩ none with
      | none => exact none
      | some r =>
        if all : r.visited.all (fun m => (reply m).isSome) then
          let visits := front ++ r.visited.map (fun m => (m,Role.scout.site))
          have traced (prior) (inside : x.InPrefix prior) :=
            GeneralOwnerMir.seeded key _ search first rest hint a z null legal low
              (fun u => witness prior inside move (valid u).1) (fun u => (valid u).2) choice ordered rfl
              (GeneralOwnerMir.run_trace op.schedule d.toNat cut (BoardFrame.Children mc fc b prior) search z
                (q.budget+1) 0 q ⟨a,best⟩ none r found (fun m h => witness prior inside m (List.all_eq_true.mp all m h)))
          cases out : r.outcome with
          | returned score =>
            exact some ⟨visits,false,score,fun prior _ inside => by
              have ran := traced prior inside
              rw [out] at ran
              exact .moves first rest ordered gen perm (.returned search ran)⟩
          | handoff entry =>
            let labels := (ordered.take idx ++ r.visited) ++ r.remaining.pending
            match splits labels r.remaining.idx entry with
            | none => exact none
            | some split =>
              match value : split.trace.answer with
              | none => exact none
              | some score =>
                if ok : labels.length < 2^width ∧ -32767 ≤ entry.alpha ∧ entry.alpha < z ∧ z ≤ 32767 then
                  exact some ⟨visits,true,score,fun prior _ inside => by
                    have ran := traced prior inside
                    rw [out] at ran
                    obtain ⟨j,returned,_⟩ := split.trace.exit.observes entry.alpha
                      ⟨ok.2.1,ok.2.2.1,ok.2.2.2⟩ (Int.le_max_left ..)
                    let j' := j.mapCalls (q := fun i => BoardFrame.Children mc fc b prior (labels[i]?.getD split.fallback))
                      (fun i p lo hi word used => inside _ p lo hi word used)
                    exact .moves first rest ordered gen perm
                      (.joined search split.fallback false cut d.toNat width ran ok.1 j' (returned.trans value))⟩
                else exact none
        else exact none
  else exact none

def issued (scope : CallReturns.Scope) (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : Position)
    (d : Depth) (a : Int) (cut : Bool) (moves : List (GeneratedSquare mc b.val × Nat)) : Bool :=
  let expected := moves.map fun (m,site) => (CallReturns.pvRequest mc fc b d cut m false (-(a+1)) (-a),site)
  let actual := scope.slots.mapM fun slot => slot.result.map (fun r => (r.entry.request,r.site))
  actual == some expected

def finish (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (s : CallJournal.State Op) (f : Frame)
    (reads : Lookups) (checked : Bool) (a z : Score) (d : Depth) (cut : Bool) (op : Options mc f.board)
    (acts : CallReturns.State) (id : Nat) (attempts : List ProbeRetry.RawAttempt)
    (rows : List (TableReads.ChildLookupFrom mc fc f.board (fun _ => True))) (seen : NativeNodes.Probed mc fc f.board reads attempts rows)
    (full : ExactDepth.Full f.board d) (null : z.raw.toInt = a.raw.toInt+1) (pv : f.pv = false)
    (lo : f.alpha = a.raw.toInt) (hi : f.beta = z.raw.toInt)
    (width : Nat := 7) (wide : 7 ≤ width := by decide) (handedOff : Bool := false)
    (splits : Nested.Splits (BoardFrame.Children mc fc f.board ((JournalPv.scope s f acts id).native .unique).prior)
      false cut d.toNat width z.raw.toInt := fun _ _ _ => none) : Check (JournalPv.Bodies mc fc s f reads) := by
  let sc := JournalPv.scope s f acts id
  let x := sc.native .unique
  let raw := GeneralNonPvMir.probeHint f.board attempts
  let hint := GeneralNonPvMir.checkedHint mc f.board raw
  if hint.isSome != raw.isSome then exact .rejected else
    cases gen : legalSquares mc f.board.val with
    | nil => exact .rejected
    | cons first rest =>
      match path mc fc f.board d a.raw.toInt z.raw.toInt null cut op x first rest gen hint width splits with
      | none => exact .rejected
      | some ran =>
        if ran.handedOff != handedOff then exact .rejected else
        if ok : ran.score = f.score ∧ issued sc mc fc f.board d a.raw.toInt cut ran.issued = true then
          obtain ⟨seed,reg⟩ := JournalPv.projections s f acts id []
          have weak (prior) (inside : x.InPrefix prior) : WeakFrame.Frame mc fc prior reads f.board f.pv f.alpha f.beta f.score := by
            rw [pv,lo,hi,←ok.1]
            exact ran.weak prior reads inside
          have finite_ : JournalBodies.SeedObligation mc fc s f reads := .internal (weak _ seed)
          have weak_ (level : s.level f ≠ Cutoffs.exactSelectivity) : JournalBodies.WeakObligation mc fc s f reads :=
            by simpa only [JournalBodies.WeakObligation,JournalPv.registry,level,↓reduceIte] using weak _ reg
          if level : s.level f = Cutoffs.exactSelectivity then
            if compiled : GeneralNonPvMir.entryAction mc fc f.board d a z false checked (fun _ => none) attempts rows = .rest ∧
                GeneralOwnerMir.scheduled mc fc f.board d a z false checked cut (fun _ => none) attempts rows
                  (framePriority mc fc f.board op.mode op.tie) x.scout op.schedule (Nested.scoutSplit splits) = some f.score then
              exact .proved ⟨finite_,weak_,fun _ => by
                unfold JournalBodies.ExactObligation
                rw [pv,lo,hi]
                apply Nested.Body.scout d a z false checked cut (fun _ => none) attempts rows x _ op.mode op.tie op.schedule splits full wide seen compiled.2
                have inReg : x.InPrefix (CausalTable.returned (· < ·) (JournalPv.registry s f) f.chosen) := reg
                simpa only [JournalPv.registry,level,↓reduceIte] using inReg⟩
            else exact .rejected
          else exact .proved ⟨finite_,weak_,fun same => (level same).elim⟩
        else exact .rejected

def beforeProbe (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : Position) (d : Depth) (a : Score) :=
  let generation := PvEntryMir.generated mc fc b false (fun _ => none)
  GeneralNonPvMir.dispatch d.toNat (EndgameStability.tryCutoff b.val d.toNat ⟨(ScoreRefresh.toDiscDiff a).raw.toInt⟩).isSome
    (generation.len.getD 0 != 0) (Board.hasLegalMoves mc (PvsBoardTree.pass b).val) generation.wipeout.join.isSome false false

variable {x : NativeTableWrites.Journal.Execution c endian rs} {es : Joint.Tape Op} {m : Joint.Atomics x es}

structure SplitOperands (es : Joint.Tape Op) (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig)
    (sc : CallReturns.Scope) (b : Position) (level : Nat) (cut : Bool) (d : Depth) (beta : Int) where
  width : Nat
  supported : 7 ≤ width
  feed : JournalNestedHelpers.Feed es sc level mc fc b false cut d.toNat width beta

abbrev SplitFor (es : Joint.Tape Op) (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : Position) :=
  (sc : CallReturns.Scope) → (level : Nat) → (cut : Bool) → (d : Depth) → (beta : Int) →
    Option (SplitOperands es mc fc sc b level cut d beta)

/-- A successful split leaves the owner loop. A later recursive child or another
    handoff under that same owner cannot belong to the reported result. -/
def splitSuffix (direct : List CallReturns.Activation) : Bool :=
  match direct.dropWhile (fun t => t.callee == .pvs) with
  | [last] => last.callee == .split && last.site == 6
  | _ => false

/-- The first direct recursive search distinguishes a general selective owner from the earlier
    ProbCut exit. Its presearch, all normal scouts, and the entire visited order must match. -/
def checkWith (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig)
    (init : ∀ a, NativeTableWrites.Journal.Even (x.memory a)) (probes : JournalBodies.Probes m) (look : JournalScout.ChildReads m)
    (checked : Bool) (s : CallJournal.State Op) (f : Frame) (op : Options mc f.board)
    (handoff : Option (SplitFor es mc fc f.board)) : Check (JournalPv.Bodies mc fc s f (CausalTable.lookup x init f.chosen)) :=
  match s.choices.find? (fun owner => owner.stamp == f.chosen) with
  | none => .rejected
  | some owner =>
    if !owner.endgame || owner.pv || owner.rootLine.isSome || decide (beforeProbe mc fc f.board owner.depth owner.alpha ≠ .rest) then .outside else
    if shape : owner.level = s.level f ∧ owner.depth = Board.emptyCount f.board.val ∧ f.pv = false ∧
        f.alpha = owner.alpha.raw.toInt ∧ f.beta = owner.beta.raw.toInt ∧ -32767 ≤ f.alpha ∧ f.alpha < f.beta ∧ f.beta ≤ 32767 ∧
        owner.beta.raw.toInt = owner.alpha.raw.toInt+1 then
      match CallReturns.captureTimed (es.map Joint.Event.frameSample) with
      | none => .rejected
      | some acts => match JournalPv.parent acts owner f with
        | none => .rejected
        | some (id,start) => match JournalPrefix.parentAt es owner.worker start f.chosen f.board,probes f.chosen f.board with
          | some (opening,close,raw),some p =>
            let attempts := JointReads.attempts f.board (some p)
            if p.opening != opening || p.finish != close || ProbeRetry.probe (attempts.map (ProbeRetry.observe (Cutoffs.key f.board))) != raw then .rejected else
            match JournalScout.parentCuts checked owner.depth (s.level f) owner.beta raw with
            | .error _ => .rejected | .ok true => .outside
            | .ok false =>
              let needed := if 14 ≤ owner.depth.toNat then legalSquares mc f.board.val else []
              let q : PvsEnhancedCutoff.Query := ⟨owner.depth,s.level f,true,⟨owner.alpha.raw⟩⟩
              match JournalScout.scan mc fc f.board q needed (JournalScout.events es owner.worker close f.chosen) with
              | none => .rejected | some (_,some _) => .outside
              | some (windows,none) => match windows.mapM (JournalScout.child mc fc f.board init look f.chosen p) with
                | none => .rejected
                | some resolved =>
                  let after := (windows.getLast?.map (·.finish)).getD close
                  let suffix := JournalScout.events es owner.worker after f.chosen
                  match JournalProbCut.consume f.board owner.depth owner.level owner.cut owner.beta owner.worker suffix with
                  | none => .rejected
                  | some (.cut ..) => .outside
                  | some (.continued tail) =>
                    let direct := acts.calls.toList.filter fun t => t.parent == some id && decide (t.entered < f.chosen)
                    if direct.isEmpty || direct.any (fun t => t.callee != .pvs && t.callee != .split) then .outside else
                    let hasSplit := direct.any (fun t => t.callee == .split)
                    if hasSplit != handoff.isSome then .outside else
                    if hasSplit && !splitSuffix direct then .rejected else
                    match tail.head? with
                    | none => .rejected
                    | some (_,first) =>
                      if direct.any (fun t => decide (t.entered < first)) then .rejected else
                      let body := fun width (wide : 7 ≤ width) handedOff
                          (splits : Nested.Splits (BoardFrame.Children mc fc f.board ((JournalPv.scope s f acts id).native .unique).prior)
                            false owner.cut owner.depth.toNat width owner.beta.raw.toInt) => finish mc fc s f _ checked owner.alpha owner.beta owner.depth owner.cut op acts id attempts _
                        (JointReads.physical init f.board (some p) resolved) shape.2.1 shape.2.2.2.2.2.2.2.2 shape.2.2.1 shape.2.2.2.1 shape.2.2.2.2.1 width wide handedOff splits
                      match handoff with
                      | none => body 7 (by decide) false (fun _ _ _ => none)
                      | some get =>
                        match get (JournalPv.scope s f acts id) (s.level f).val owner.cut owner.depth owner.beta.raw.toInt with
                        | none => .rejected
                        | some raw => body raw.width raw.supported true (JournalNestedHelpers.splits raw.feed)
          | _,_ => .rejected
    else .rejected

def check (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig)
    (init : ∀ a, NativeTableWrites.Journal.Even (x.memory a)) (probes : JournalBodies.Probes m) (look : JournalScout.ChildReads m)
    (checked : Bool) (s : CallJournal.State Op) (f : Frame) (op : Options mc f.board) :=
  checkWith mc fc init probes look checked s f op none

abbrev Inputs (mc : Bitboard.Moves.X86Config) := (f : Frame) → Options mc f.board

def pending (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig)
    (init : ∀ a, NativeTableWrites.Journal.Even (x.memory a)) (probes : JournalBodies.Probes m) (look : JournalScout.ChildReads m)
    (pool : EndgameSearch.CacheHistory.Pool) (checked : Bool) (s : CallJournal.State Op) (pv : JournalPv.Options mc) (op : Inputs mc)
    (f : Frame) (frames : List Frame) (body : Prop) :=
  JournalProbCut.pending mc fc init probes look pool checked s pv f frames ((check mc fc init probes look checked s f (op f)).requires body)

structure Remaining (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig)
    (init : ∀ a, NativeTableWrites.Journal.Even (x.memory a)) (probes : JournalBodies.Probes m) (look : JournalScout.ChildReads m)
    (pool : EndgameSearch.CacheHistory.Pool) (checked : Bool) (s : CallJournal.State Op) (pv : JournalPv.Options mc) (op : Inputs mc) : Prop where
  seed : ∀ f ∈ s.frames, pending mc fc init probes look pool checked s pv op f (JournalPass.seedRows s f)
    (JournalBodies.SeedObligation mc fc s f (CausalTable.lookup x init f.chosen))
  weak : ∀ f ∈ s.weakChildren, pending mc fc init probes look pool checked s pv op f s.weakChildren
    (JournalBodies.WeakObligation mc fc s f (CausalTable.lookup x init f.chosen))
  exact : ∀ f ∈ s.exactFrames, pending mc fc init probes look pool checked s pv op f s.exactFrames
    (JournalBodies.ExactObligation mc fc s f (CausalTable.lookup x init f.chosen))

private theorem through (test : Check P) (use : A → B) : test.requires A → test.requires B := by
  cases test with | outside => exact use | rejected => exact id | proved _ => exact id

private theorem discharge (body : JournalPv.Bodies mc fc s f (CausalTable.lookup x init f.chosen) → Q)
    (h : pending mc fc init probes look pool checked s pv op f frames Q) :
    JournalProbCut.pending mc fc init probes look pool checked s pv f frames Q :=
  through _ (through _ (through _ (through _ (through _ (through _ (Check.derive _ body)))))) h

theorem Remaining.probCuts (h : Remaining mc fc init probes look pool checked s pv op) :
    JournalProbCut.Remaining mc fc init probes look pool checked s pv where
  seed f mem := discharge (fun r => r.seed) (h.seed f mem)
  weak f mem := discharge (fun r => r.weak (bne_iff_ne.mp (Bool.and_eq_true_iff.mp (List.mem_filter.mp mem).2).1)) (h.weak f mem)
  exact f mem := discharge (fun r => r.exact (beq_iff_eq.mp (List.mem_filter.mp mem).2)) (h.exact f mem)

namespace Regression

private def mc : Bitboard.Moves.X86Config := ⟨false,false⟩
private def fc : Cutoffs.FlipConfig := ⟨false,false,false⟩
private def two : Position := ⟨⟨0x100000001,0x200000002⟩,rfl⟩
private def one : Position := ⟨⟨1,0x40000000000002⟩,rfl⟩
private def options (b : Position) : Options mc b := ⟨fun _ => .skip,fun sq => sq.val,⟨fun _ _ => false,fun _ _ => false⟩⟩
private def word (v : Int) : Score := ⟨BitVec.ofInt 32 v⟩
private def enter (t site : Nat) (b : Position) (level : Selectivity) (d : Depth) (cut : Bool) (a z : Int) : Joint.Event Unit :=
  .engine (.localCall 0 (.entry t site .pvs .none none) (some ⟨b.val,d,word a,word z,true,level,false,cut,none⟩))
private def calls : NativeTableWrites.Journal.Calls := fun _ => ⟨⟨0,0⟩,⟨0,rfl⟩,.clear,false,id⟩

private def prefixChild : CallReturns.Activation := ⟨0,1,7,.pvs,some 0,none,none,none,1⟩
private def handoff : CallReturns.Activation := {prefixChild with site := 6, callee := .split}

#guard splitSuffix [prefixChild,handoff]
#guard !splitSuffix []
#guard !splitSuffix [prefixChild]
#guard !splitSuffix [prefixChild,{handoff with site := 5}]
#guard !splitSuffix [handoff,prefixChild]
#guard !splitSuffix [handoff,handoff]

structure Change where
  reversed : Bool := false
  shifted : Bool := false
  moveSite : Option Nat := none
  childLevel : Option Selectivity := none
  extra : Bool := false
  returned : Bool := true

/-- A selected score is checked before the parent returns. Exact and selective children
    come from their actual same-level completed returns and the same activation IDs. -/
private def trial (b : Position) (level : Selectivity) (scores : List Int) (score : Int) (c : Change) : Option Bool := do
  let sorted := order (framePriority mc fc b (options b).mode (options b).tie) none (legalSquares mc b.val)
  let ms := if c.reversed then sorted.reverse else sorted
  let pairs := ms.zip scores
  let calls_ := if c.extra then pairs ++ pairs.take 1 else pairs
  let children := calls_.zipIdx.flatMap fun ((m,raw),i) =>
    let t := 1+3*i
    let next := (play mc fc b m).val
    let site := c.moveSite.getD (if ms.length = 1 then Role.presearch.site else Role.scout.site)
    [enter t site next (c.childLevel.getD level) (Board.emptyCount next.val) true (if c.shifted then -2 else -1) 0,
      .engine (.localCall 0 (.scored (t+1) (word raw)) none),.engine (.localCall 0 (.leave (t+2) t .pvs none (if c.returned then some (word raw) else none)) none)]
  let es := [enter 0 44 b level (Board.emptyCount b.val) false 0 1] ++ children ++ [.engine (.localCall 0 (.scored 32 (word score)) none)]
  let s ← CallJournal.run mc fc calls false (es.map Joint.Event.engineEvent)
  let acts ← CallReturns.captureTimed (es.map Joint.Event.frameSample)
  let f ← s.frames.head?
  let parent ← s.choices.head?
  let (id,_) ← JournalPv.parent acts parent f
  if ok : f.pv = false ∧ f.alpha = 0 ∧ f.beta = 1 then
    match finish mc fc s f (fun _ _ => True) true ⟨0⟩ ⟨1⟩ (Board.emptyCount f.board.val) false (options f.board) acts id [] []
        ⟨fun _ _ _ => trivial,fun _ _ _ _ => trivial⟩ rfl rfl ok.1 ok.2.1 ok.2.2 with
    | .outside => none | .rejected => return false | .proved _ => return true
  else return false

#guard trial two 5 [3,1] (-1) {} = some true
#guard trial two 2 [3,1] (-1) {} = some true
#guard trial two 5 [-4] 4 {} = some true
#guard trial one 5 [-4] 4 {} = some true
#guard trial one 2 [3] (-3) {} = some true
#guard trial two 5 [3,1] (-1) {reversed := true} = some false
#guard trial two 2 [3,1] 1 {} = some false
#guard trial two 5 [3] (-3) {} = some false
#guard trial two 5 [-4] 4 {extra := true} = some false
#guard trial one 2 [-4] 4 {moveSite := some 7} = some false
#guard trial two 5 [3,1] (-1) {shifted := true} = some false
#guard trial two 5 [3,1] (-1) {childLevel := some 2} = some false
#guard trial one 5 [-4] 4 {returned := false} = some false

end Regression

end NeuralReversi.PvsParallel.Interleaving.JournalScoutOwner
