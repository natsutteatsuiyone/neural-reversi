module
public import NeuralReversi.PvsParallel.Interleaving.BoardFrame
public import NeuralReversi.PvsParallel.Interleaving.CausalReads
public import NeuralReversi.PvsParallel.DeepLeafMir
public import NeuralReversi.EndgameSearch.Direct.Mir
public import NeuralReversi.PvsParallel.GeneralOwnerMir
public import NeuralReversi.PvsParallel.SerialPvCalls

@[expose] public section

namespace NeuralReversi.PvsParallel.Interleaving.NativeNodes

open PvsBoardTree (Position GeneratedSquare legalSquares play)
open PvsMoveLoop (ChildRelation ChildSearch scoreInf)
open TranspositionTable (Depth)
open PvsPostpone (Queue)
open Threading.SplitWindow (State)
open Prefix.Serial (Run Outcome)

/-- Bind a native child call to the earlier return selected by the call-site trace.
    Match the callee operands before reading its score. -/
structure Returns where
  frames : List CausalTable.Frame
  depth : Nat → Depth
  cut : Nat → Bool
  before : Nat → Nat → Bool
  consumer : Nat
  select : Position → Depth → Bool → Bool → Int → Int → Option Nat

def Returns.prior (x : Returns) := CausalTable.returned (fun i j => x.before i j) x.frames x.consumer

def Returns.reply (x : Returns) (b : Position) (d : Depth) (pv cut : Bool) (a z : Int) : Option Int := do
  let i ← x.select b d pv cut a z
  let f ← x.frames[i]?
  let t ← f.returned
  guard (x.before t x.consumer ∧ x.depth i = d ∧ x.cut i = cut ∧ f.board = b ∧ f.pv = pv ∧ f.alpha = a ∧ f.beta = z)
  return f.score

theorem Returns.hit (x : Returns) {pv cut : Bool} (got : x.reply b d pv cut a z = some v) : x.prior b pv a z v := by
  unfold reply at got
  obtain ⟨i,_,rest⟩ := Option.bind_eq_some_iff.mp got
  obtain ⟨f,found,rest⟩ := Option.bind_eq_some_iff.mp rest
  obtain ⟨t,returned,rest⟩ := Option.bind_eq_some_iff.mp rest
  simp [guard] at rest
  split at rest
  next good =>
    obtain ⟨earlier,_,_,board,kind,lo,hi⟩ := good
    exact ⟨f,List.mem_of_getElem? found,t,returned,earlier,board,kind,lo,hi,Option.some.inj rest⟩
  next => cases rest

/-- A native split call receives the actual post-postponement move slice and cursor. -/
structure Split (done_ : ChildRelation M) (pv cut : Bool) (depth width : Nat) (beta : Int)
    (moves : List M) (idx : Nat) (entry : State) where
  fallback : M
  join : Join ⟨pv,false,cut,depth,beta,entry⟩ (fun i => done_ (moves[i]?.getD fallback))
    moves.length (BitVec.ofNat width idx)

abbrev Splits (done_ : ChildRelation M) (pv cut : Bool) (depth width : Nat) (beta : Int) :=
  ∀ moves idx entry, Option (Split done_ pv cut depth width beta moves idx entry)

variable {M : Type} {done_ : ChildRelation M}

def call (s : Splits done_ pv cut depth width beta) (moves : List M) (idx : Nat) (entry : State) : Option Int :=
  (s moves idx entry).bind (fun split => split.join.answer)

private theorem handoff (s : Splits done_ pv cut depth width beta) (reply : ChildSearch M)
    (trace : Run done_ reply pv beta ⟨0,moves,[]⟩ ⟨alpha,-scoreInf⟩ visited remaining (.handoff entry))
    (size : moves.length < 2^width)
    (answer : call s (visited ++ remaining.pending) remaining.idx entry = some score) :
    Turn done_ moves pv alpha beta score := by
  obtain ⟨split,_,ok⟩ := Option.bind_eq_some_iff.mp answer
  have same : (visited ++ remaining.pending).Perm moves := by
    simpa [Queue.pending] using trace.reproduced.1
  exact .joined reply split.fallback false cut depth width trace (same.length_eq ▸ size) split.join ok

def pvSplit (s : Splits done_ true cut depth width beta) (exit : RootOwnerMir.Slice.Exit M) : Option Int :=
  call s exit.moves exit.idx exit.snapshot.state

/-- The indexed PV owner performs its own handoff decision, including every busy move. -/
theorem pvTurn (square : M → Bitboard.Square) (replies : RootOwnerMir.Replies M)
    (witness : ∀ m p a z v, replies m p a z = some v → done_ m p a z v)
    (schedule : Nat → Bool × Bool) (splits : Splits done_ true cut depth width beta)
    (size : moves.length < 2^width)
    (answer : (do
      let owner ← RootOwnerMir.Slice.run square replies schedule depth limit beta 0 0 moves.length moves (RootReplay.start alpha)
      if owner.handoff then pvSplit splits owner else some owner.snapshot.state.best) = some score) :
    Turn done_ moves true alpha beta score := by
  rw [RootOwnerMir.Slice.fromMoves_eq] at answer
  cases ran : RootOwnerMir.fromMoves square replies schedule depth limit alpha beta moves with
  | none => simp [ran] at answer
  | some r =>
    obtain ⟨trace,_,_⟩ := RootOwnerMir.refines _ _ _ witness _ _ _ _ _ _ _ _ _ _ (by simp) rfl ran
    cases h : r.handoff with
    | false =>
      have eq : r.snapshot.state.best = score := by simpa [ran,RootOwnerMir.Slice.layout,h] using answer
      rw [show r.outcome = .returned score by simp [RootOwnerMir.Result.outcome,h,eq]] at trace
      exact .returned _ trace
    | true =>
      rw [show r.outcome = .handoff r.snapshot.state by simp [RootOwnerMir.Result.outcome,h]] at trace
      exact handoff splits _ trace size (by simpa [ran,RootOwnerMir.Slice.layout,h,pvSplit] using answer)

def scoutSplit (s : Splits done_ false cut depth width beta) : GeneralOwnerMir.SplitReturn M :=
  fun ordered idx r => match r.outcome with
    | .returned _ => none
    | .handoff entry => call s (ordered.take idx ++ r.visited ++ r.remaining.pending) r.remaining.idx entry

open GeneralNonPvMir (presearch order)

private theorem scoutTurn [DecidableEq M] (priority : M → Int) (done_ : ChildRelation M) (search : ChildSearch M)
    (reply : M → Option Int) (witness : ∀ m, (reply m).isSome → done_ m false (-(a+1)) (-a) (search m false (-(a+1)) (-a)))
    (first : M) (rest : List M) (hint : Option M) (null : beta = a+1)
    (legal : ∀ m, hint = some m → m ∈ first :: rest)
    (low : ∀ m sq, hint = some sq → m ∈ first :: rest → m ≠ sq → priority m < MoveList.Ordering.ttMoveValue)
    (front : (first :: rest).length = 1 ∨ hint.isSome → let m := hint.getD first
      done_ m false (-(a+1)) (-a) (search m false (-(a+1)) (-a)) ∧ search m false (-(a+1)) (-a) < scoreInf)
    (start : presearch (first :: rest).length hint first a beta (fun m => search m false) = .owner idx best)
    (schedule : GeneralOwnerMir.Schedule M) (splits : Splits done_ false cut depth width beta)
    (size : (first :: rest).length < 2^width)
    (answer : GeneralOwnerMir.recordedOwner schedule depth cut search reply (scoutSplit splits) a
      (order priority hint (first :: rest)) idx best = some score) :
    Turn done_ (order priority hint (first :: rest)) false a beta score := by
  let ordered := order priority hint (first :: rest)
  let q : Queue M := ⟨idx, ordered.drop idx, []⟩
  obtain ⟨r,normal⟩ := GeneralOwnerMir.run_exists schedule depth cut search _ 0 q ⟨a,best⟩ none (Nat.lt_succ_self _)
  unfold GeneralOwnerMir.recordedOwner at answer
  change (do
    let r ← GeneralOwnerMir.run schedule depth cut search (q.budget + 1) 0 q ⟨a,best⟩ none
    if r.visited.all (fun m => (reply m).isSome) then
      match r.outcome with
      | .returned value => some value
      | .handoff _ => scoutSplit splits ordered idx r
    else none) = some score at answer
  rw [normal] at answer
  simp at answer
  have suffix := GeneralOwnerMir.run_trace schedule depth cut done_ search beta _ 0 _ _ _ _ normal
    (fun m mem => witness m (answer.1 m mem))
  have trace := GeneralOwnerMir.seeded priority done_ search first rest hint a beta null legal low
    (fun used => (front used).1) (fun used => (front used).2) start ordered rfl suffix
  cases h : r.outcome with
  | returned v =>
    have eq : v = score := by simpa [h] using answer.2
    rw [h,eq] at trace
    exact .returned _ trace
  | handoff entry =>
    rw [h] at trace
    exact handoff splits _ trace ((GeneralNonPvMir.orderPerm priority _ hint legal low).length_eq ▸ size)
      (by simpa [h,scoutSplit] using answer.2)

open EndgameOrderMir (Heuristic pvOrder framePriority)
open ScoreRefresh (Score)
open TranspositionTable.ProbeRetry (RawAttempt)
open BoardFrame (Frame Children)
open GeneralNonPvMir (presearch observe checkedHint probeHint)
open PvsBoardTree (pass)
open Aspiration (Window)

private theorem generated_mem (m : GeneratedSquare mc b.val) : m ∈ legalSquares mc (b : Position).val := by
  obtain ⟨v,present,eq⟩ := (PvsBoardTree.legalSquares_complete mc b.val m.val).mpr
    ((Board.getMoves_exact mc b.val _).mp m.property)
  rwa [show v = m from Subtype.ext eq] at present

private theorem fits (mc : Bitboard.Moves.X86Config) (b : Position) (widthOK : 7 ≤ width) :
    (legalSquares mc b.val).length < 2^width := by
  have size := (EndgameOrderMir.generated_layout mc b).2
  have high := Nat.pow_le_pow_right (by decide : 0 < 2) widthOK
  omega

def Returns.pvReplies (x : Returns) (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : Position)
    (d : Depth) (cut : Bool) : RootOwnerMir.Replies (GeneratedSquare mc b.val) :=
  fun m pv a z => x.reply (play mc fc b m).val (d-1) pv (!pv && !cut) a z

def Returns.scout (x : Returns) : GeneralNonPvMir.Child := fun b d cut a z => x.reply b d false cut a z

private theorem pvExit (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : Position) (d : Depth)
    (paired checked : Bool) (mem : MoveList.Initialization.Memory Move.Move) (a z : Int) (x : Returns)
    (full : ExactDepth.Full b d) (answer : PvEntryMir.run mc fc b d paired checked mem a z
      (fun next cd lo hi => x.reply next cd true false lo hi) = some score) : Frame mc fc x.prior reads b true a z score := by
  simp only [PvEntryMir.run,PvEntryMir.generated_count,Option.getD_some,PvEntryMir.dispatch] at answer
  by_cases zero : d.toNat = 0
  · have word : d = 0 := BitVec.eq_of_toNat_eq zero
    obtain ⟨stuck,other,value⟩ := ExactDepth.zeroDepthContract_proven mc fc b checked (word ▸ full)
    cases eval : NonPvMir.pvZeroScore checked b.val with
    | error reason => simp [zero,eval,Except.toOption] at answer
    | ok s =>
      simp [zero,eval,Except.toOption] at answer
      change ScoreRefresh.BoardBridge.finalScore checked b.val = .ok s at eval
      simp [eval,Except.map] at value
      rw [←answer,value,PvsBoardTree.minimax,PvsBoardTree.terminal_tree mc fc b (PvsBoardTree.nil_of_stuck stuck) other,
        PvsMoveLoop.minimax.eq_def]
      exact .terminal (PvsBoardTree.nil_of_stuck stuck) other
  · cases moves : legalSquares mc b.val with
    | nil =>
      cases other : Board.hasLegalMoves mc (pass b).val with
      | false =>
        cases eval : ScoreRefresh.BoardBridge.solve checked b.val d with
        | error reason => simp [moves,other,zero,eval,Except.toOption] at answer
        | ok raw =>
          have value := PvEntryMir.terminal_solve b checked
          rw [← full,eval] at value
          simp [Except.map] at value
          simp [moves,other,zero,eval,Except.toOption] at answer
          rw [←answer,value]
          exact .terminal moves other
      | true =>
        cases reply : x.reply (pass b) d true false (-z) (-a) with
        | none => simp [moves,other,zero,reply] at answer
        | some v =>
          simp [moves,other,zero,reply] at answer
          rw [←answer]
          exact .pass moves other (x.hit reply)
    | cons first rest =>
      cases cache : (PvEntryMir.generated mc fc b paired mem).wipeout with
      | none => simp [moves,zero,cache] at answer
      | some item =>
        cases item with
        | none => simp [moves,zero,cache] at answer
        | some sq =>
          simp [moves,zero,cache] at answer
          obtain ⟨m,legal,hit⟩ := PvEntryMir.cached_wipe mc fc b paired mem cache
          rw [←answer]
          exact .wipe m legal hit

theorem pv (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : Position) (d : Depth)
    (paired checked cut : Bool) (mem : MoveList.Initialization.Memory Move.Move) (a z : Int) (x : Returns)
    (mode : Bitboard.Square → Heuristic) (tie : GeneratedSquare mc b.val → Fin 64) (hint : Option Bitboard.Square)
    (schedule : Nat → Bool × Bool) (splits : Splits (Children mc fc b x.prior) true cut d.toNat width z)
    (full : ExactDepth.Full b d) (widthOK : 7 ≤ width)
    (answer : PvSerialMir.scheduled mc fc b d paired checked mem a z
      (fun next cd lo hi => x.reply next cd true false lo hi) (pvOrder mc fc b mode tie hint)
      (x.pvReplies mc fc b d cut) schedule (pvSplit splits) = some score) :
    Frame mc fc x.prior reads b true a z score := by
  dsimp only [PvSerialMir.scheduled] at answer
  split at answer
  next which =>
    have perm := EndgameOrderMir.pvOrder_perm mc fc b mode tie hint
    cases moves : legalSquares mc b.val with
    | nil =>
      simp only [PvEntryMir.generated_count,Option.getD_some,moves,List.length_nil] at which
      unfold PvEntryMir.dispatch at which
      repeat' split at which <;> simp_all
    | cons first rest =>
      exact .moves first rest _ moves (moves ▸ perm) (pvTurn _ _ (fun _ _ _ _ _ got => x.hit got) _ splits
        (perm.length_eq ▸ fits mc b widthOK) answer)
  next => exact pvExit mc fc b d paired checked mem a z x full answer

open GeneralNonPvMir (resumeOwner entryAction presearchRun)
open GeneralOwnerMir (scoutReply)

private theorem scoutRest (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : Position)
    (d : Depth) (alpha beta : Score) (paired checked cut : Bool) (mem : MoveList.Initialization.Memory Move.Move)
    (attempts : List RawAttempt) (rows : List (TableReads.ChildLookupFrom mc fc b src)) (x : Returns)
    (mode : Bitboard.Square → Heuristic) (tie : GeneratedSquare mc b.val → Fin 64)
    (schedule : GeneralOwnerMir.Schedule (GeneratedSquare mc b.val))
    (splits : Splits (Children mc fc b x.prior) false cut d.toNat width beta.raw.toInt)
    (widthOK : 7 ≤ width) (window : Window alpha.raw.toInt beta.raw.toInt) (null : beta.raw.toInt = alpha.raw.toInt+1)
    (earlier : CausalTable.Correct mc fc x.prior)
    (continue_ : entryAction mc fc b d alpha beta paired checked mem attempts rows = .rest)
    (answer : GeneralOwnerMir.scheduled mc fc b d alpha beta paired checked cut mem attempts rows
      (framePriority mc fc b mode tie) x.scout schedule (scoutSplit splits) = some score) :
    Frame mc fc x.prior reads b false alpha.raw.toInt beta.raw.toInt score := by
  simp only [GeneralOwnerMir.scheduled,GeneralNonPvMir.frame,continue_] at answer
  cases moves : legalSquares mc b.val with
  | nil => simp [moves] at answer
  | cons first rest =>
    let a := alpha.raw.toInt
    let z := beta.raw.toInt
    let pr := framePriority mc fc b mode tie
    let hint := checkedHint mc b (probeHint b attempts)
    let m := hint.getD first
    let search := observe mc fc b d cut x.scout
    let reply := scoutReply mc fc b d cut x.scout a
    let ordered := order pr hint (first :: rest)
    let selected := presearch (first :: rest).length hint first a z (fun m => search m false)
    let owner := GeneralOwnerMir.recordedOwner schedule d.toNat cut search reply (scoutSplit splits) a ordered
    rw [moves] at answer
    by_cases seen : hint.isSome = (probeHint b attempts).isSome
    case neg => simp [hint] at seen; simp [guard,seen] at answer; cases answer
    simp only [guard,show (checkedHint mc b (probeHint b attempts)).isSome = _ from seen,
      ↓reduceIte,pure,bind,Option.bind] at answer
    change (do
      if (first :: rest).length = 1 || (probeHint b attempts).isSome then let _ ← reply m
      resumeOwner owner selected) = some score at answer
    rw [←seen] at answer
    have consumed (used : (first :: rest).length = 1 ∨ hint.isSome) : (reply m).isSome := by
      cases eq : reply m with
      | none => simp only [Bool.or_eq_true,decide_eq_true_eq,ite_eq_left used,eq] at answer; cases answer
      | some => rfl
    have returned : resumeOwner owner selected = some score := by
      by_cases used : (first :: rest).length = 1 ∨ hint.isSome
      · obtain ⟨v,eq⟩ := Option.isSome_iff_exists.mp (consumed used)
        simp only [Bool.or_eq_true,decide_eq_true_eq,ite_eq_left used,eq] at answer; simpa using answer
      · simpa only [Bool.or_eq_true,decide_eq_true_eq,ite_eq_right used] using answer
    have witness (m) (present : (reply m).isSome) :
        Children mc fc b x.prior m false (-(a+1)) (-a) (search m false (-(a+1)) (-a)) := by
      obtain ⟨v,eq⟩ := Option.isSome_iff_exists.mp present
      change x.prior _ false _ _ ((reply m).getD 0)
      simpa [eq] using x.hit eq
    have front (used : (first :: rest).length = 1 ∨ hint.isSome) :
        Children mc fc b x.prior m false (-(a+1)) (-a) (search m false (-(a+1)) (-a)) ∧
        search m false (-(a+1)) (-a) < scoreInf := by
      have child := witness m (consumed used)
      have win : Window (-(a+1)) (-a) := by simpa [a,←null] using PvsMoveLoop.reflectedWindow window
      exact ⟨child,(earlier _ _ _ _ _ win (fun _ => by omega) child).upper⟩
    have legal := fun (m : GeneratedSquare mc b.val) (_ : hint = some m) => moves ▸ generated_mem (b := b) m
    have low := fun m sq (_ : hint = some sq) (_ : m ∈ first :: rest) (_ : m ≠ sq) =>
      EndgameOrderMir.framePriority_low mc fc b mode tie m
    apply Frame.moves first rest ordered moves (GeneralNonPvMir.orderPerm pr _ hint legal low)
    cases h : selected with
    | owner idx best =>
      apply scoutTurn pr _ search reply witness first rest hint null legal low front h schedule splits
        (moves ▸ fits mc b widthOK)
      simpa [h,resumeOwner] using returned
    | storeLower sq v =>
      have same : v = score := by simpa [h,resumeOwner] using returned
      have suffix : GeneralNonPvMir.PresearchSuffix (Children mc fc b x.prior) search a z ordered (.returned score) selected := by
        rw [h,same]; rfl
      obtain ⟨visited,remaining,trace⟩ := presearchRun pr _ search first rest hint a z null legal low
        (fun used => (front used).1) (fun used => (front used).2) _ suffix
      exact .returned search trace

open TranspositionTable hiding Score
open TableReads (ChildLookupFrom)

/-- The operands of the parent probe and of every ETC lookup are the results of the frame's reads. -/
structure Probed (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : Position) (reads : CausalTable.Lookups)
    (attempts : List RawAttempt) (rows : List (ChildLookupFrom mc fc b src)) : Prop where
  parent : ∀ slot e, ProbeRetry.probe (attempts.map (ProbeRetry.observe (Cutoffs.key b))) = .hit slot e → reads b e
  child : ∀ row ∈ rows, ∀ e, row.front.or row.back = some e → reads (play mc fc b row.move).val e

theorem Probed.legalHint (seen : Probed mc fc b reads attempts rows)
    (prior : ∀ board e, reads board e → Table.EntrySafe mc fc board e) :
    (checkedHint mc b (probeHint b attempts)).isSome = (probeHint b attempts).isSome := by
  apply (GeneralNonPvMir.checkedHint_present mc b _).mpr
  intro sq found
  cases result : ProbeRetry.probe (attempts.map (ProbeRetry.observe (Cutoffs.key b))) with
  | miss slot => simp [probeHint,result,TranspositionTable.ReaderMir.bestMove] at found
  | hit slot e =>
    have decoded := found
    simp only [probeHint,result,TranspositionTable.ReaderMir.bestMove] at decoded
    obtain ⟨m,_,rfl⟩ := ((prior _ _ (seen.parent slot e result)).2).generated (mc := mc) decoded
    exact m.property

private def tableCut (b : Position) (d : Depth) (beta : Score) (checked : Bool) (attempts : List RawAttempt) :=
  do
    let .hit _ e := ProbeRetry.probe (attempts.map (ProbeRetry.observe (Cutoffs.key b))) | none
    let eligible ← (TranspositionTable.canCut checked e ⟨beta.raw⟩ d Cutoffs.exactSelectivity true).toOption
    if eligible then some e.score.toInt else none

private theorem prefixFrame (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : Position)
    (d : Depth) (alpha beta : Score) (paired checked cut : Bool) (mem : MoveList.Initialization.Memory Move.Move)
    (attempts : List RawAttempt) (rows : List (ChildLookupFrom mc fc b src)) (x : Returns)
    (seen : Probed mc fc b reads attempts rows) (full : ExactDepth.Full b d)
    (null : beta.raw.toInt = alpha.raw.toInt+1)
    (answer : GeneralNonPvMir.runPrefix mc fc b d alpha beta paired checked cut mem attempts rows x.scout = some score) :
    Frame mc fc x.prior reads b false alpha.raw.toInt beta.raw.toInt score := by
  let stable := EndgameStability.tryCutoff b.val d.toNat ⟨(ScoreRefresh.toDiscDiff alpha).raw.toInt⟩
  let g := PvEntryMir.generated mc fc b paired mem
  let q : PvsEnhancedCutoff.Query := ⟨d,Cutoffs.exactSelectivity,true,⟨alpha.raw⟩⟩
  let probes := rows.map ChildLookupFrom.observation
  let enhanced := (PvsEnhancedCutoff.scan q probes).map Prod.snd
  change (match GeneralNonPvMir.dispatch d.toNat stable.isSome (g.len.getD 0 != 0) (Board.hasLegalMoves mc (pass b).val)
      g.wipeout.join.isSome (tableCut b d beta checked attempts).isSome enhanced.isSome with
    | .stability => (do
      let res ← stable
      return (← (ScoreRefresh.fromDiscDiff checked ⟨BitVec.ofInt 32 res.value⟩).toOption).raw.toInt)
    | .terminal => (do return (← (ScoreRefresh.BoardBridge.solve checked b.val d).toOption).raw.toInt)
    | .pass => (do return -(← x.scout (pass b) d (!cut) (-beta.raw.toInt) (-alpha.raw.toInt)))
    | .wipeout => some PvsMoveLoop.scoreMax
    | .probe => tableCut b d beta checked attempts
    | .enhanced => enhanced
    | .cache | .rest | .shallow => none) = some score at answer
  have count : g.len = some (legalSquares mc b.val).length := PvEntryMir.generated_count ..
  simp only [count,Option.getD_some,GeneralNonPvMir.dispatch] at answer
  by_cases small : d.toNat ≤ 12
  · simp [small] at answer
  · simp only [ite_eq_right small] at answer
    cases st : stable with
    | some res =>
      cases converted : ScoreRefresh.fromDiscDiff checked ⟨BitVec.ofInt 32 res.value⟩ with
      | error reason => simp [st,converted,Except.toOption] at answer
      | ok raw =>
        simp [st,converted,Except.toOption] at answer
        rw [←answer,null]
        exact .stable alpha res raw checked (full ▸ st) converted
    | none =>
      cases moves : legalSquares mc b.val with
      | nil =>
        cases other : Board.hasLegalMoves mc (pass b).val with
        | false =>
          cases solved : ScoreRefresh.BoardBridge.solve checked b.val d with
          | error reason => simp [st,moves,other,solved,Except.toOption] at answer
          | ok raw =>
            have v := PvEntryMir.terminal_solve b checked
            rw [←full,solved] at v
            simp [Except.map] at v
            simp [st,moves,other,solved,Except.toOption] at answer
            rw [←answer,v]
            exact .terminal moves other
        | true =>
          cases child : x.scout (pass b) d (!cut) (-beta.raw.toInt) (-alpha.raw.toInt) with
          | none => simp [st,moves,other,child] at answer
          | some v =>
            simp [st,moves,other,child] at answer
            rw [←answer]
            exact .pass moves other (x.hit child)
      | cons first rest =>
        cases wiped : g.wipeout.join with
        | some sq =>
          obtain ⟨m,member,yes⟩ := PvEntryMir.cached_wipe mc fc b paired mem (square := sq) (by
            change g.wipeout = _
            cases cache : g.wipeout <;> simp_all)
          simp [st,moves,wiped] at answer
          rw [←answer]
          exact .wipe m member yes
        | none =>
          cases parent : tableCut b d beta checked attempts with
          | some v =>
            simp [st,moves,wiped,parent] at answer
            unfold tableCut at parent
            cases hit : ProbeRetry.probe (attempts.map (ProbeRetry.observe (Cutoffs.key b))) with
            | miss slot => simp [hit] at parent
            | hit slot e =>
              cases gate : TranspositionTable.canCut checked e ⟨beta.raw⟩ d Cutoffs.exactSelectivity true with
              | error err => simp [hit,gate,Except.toOption] at parent
              | ok eligible =>
                cases eligible <;> simp [hit,gate,Except.toOption] at parent
                rw [←answer,←parent]
                exact .table ⟨beta.raw⟩ (seen.parent _ _ hit) gate
          | none =>
            by_cases large : 14 ≤ d.toNat
            · cases hit : PvsEnhancedCutoff.scan q probes with
              | none => simp [st,moves,wiped,parent,large,enhanced,hit] at answer
              | some result =>
                simp [st,moves,wiped,parent,large,enhanced,hit] at answer
                rw [←answer,null]
                apply Frame.enhanced q probes result rfl rfl _ hit
                intro probe member e present
                obtain ⟨row,listed,rfl⟩ := List.mem_map.mp (show probe ∈ rows.map ChildLookupFrom.observation from member)
                exact ⟨row.move,generated_mem row.move,rfl,seen.child row listed e present⟩
            · simp [st,moves,wiped,parent,large] at answer

/-- A normal general non-PV return, including entry cutoffs, the TT presearch and worker handoff. -/
theorem scout (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : Position)
    (d : Depth) (alpha beta : Score) (paired checked cut : Bool) (mem : MoveList.Initialization.Memory Move.Move)
    (attempts : List RawAttempt) (rows : List (ChildLookupFrom mc fc b src)) (x : Returns)
    (mode : Bitboard.Square → Heuristic) (tie : GeneratedSquare mc b.val → Fin 64)
    (schedule : GeneralOwnerMir.Schedule (GeneratedSquare mc b.val))
    (splits : Splits (Children mc fc b x.prior) false cut d.toNat width beta.raw.toInt)
    (full : ExactDepth.Full b d) (widthOK : 7 ≤ width) (window : Window alpha.raw.toInt beta.raw.toInt)
    (null : beta.raw.toInt = alpha.raw.toInt+1) (earlier : CausalTable.Correct mc fc x.prior)
    (seen : Probed mc fc b reads attempts rows)
    (answer : GeneralOwnerMir.scheduled mc fc b d alpha beta paired checked cut mem attempts rows
      (framePriority mc fc b mode tie) x.scout schedule (scoutSplit splits) = some score) :
    Frame mc fc x.prior reads b false alpha.raw.toInt beta.raw.toInt score := by
  by_cases more : entryAction mc fc b d alpha beta paired checked mem attempts rows = .rest
  · exact scoutRest mc fc b d alpha beta paired checked cut mem attempts rows x mode tie schedule splits
      widthOK window null earlier more answer
  · apply prefixFrame mc fc b d alpha beta paired checked cut mem attempts rows x seen full null
    unfold GeneralOwnerMir.scheduled GeneralNonPvMir.frame at answer
    split at answer <;> first | contradiction | exact answer

open NativeTableWrites.Journal (Even)

structure Addressed where
  allocation : Nat
  clusters : SeqLock.Word
  key : SeqLock.Word
  opening : Nat

def Addressed.probe (p : Addressed) (x : NativeTableWrites.Journal.Execution c endian rs) (r : CausalReads.Reader) (b : Position) :=
  FrameTableReads.Operations.probe (r.supply x) p.allocation p.clusters p.key endian (Cutoffs.key b) p.opening

def Addressed.cluster (p : Addressed) (x : NativeTableWrites.Journal.Execution c endian rs) (r : CausalReads.Reader) (b : Position) :=
  FrameTableReads.Operations.lookup (r.supply x) p.allocation p.clusters p.key endian b p.opening

/-- Populate the probe and ETC operands from executed MIR load sites. The address and
    opening register may vary at each child. A skipped parent probe contributes no rounds. -/
theorem Probed.physical (x : NativeTableWrites.Journal.Execution c endian rs) (init : ∀ a, Even (x.memory a)) (r : CausalReads.Reader)
    (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : Position) (consumer : Nat)
    (parent : Option Addressed) (attempts : List RawAttempt) (rows : List (ChildLookupFrom mc fc b src))
    (words : attempts = parent.toList.flatMap (fun p => (p.probe x r b).rounds.map (fun t => t.read (r.supply x) endian)))
    (steps : ∀ p ∈ parent, ∀ t ∈ (p.probe x r b).rounds, CausalReads.Issued x r consumer t.front (FrameTableReads.snapCode endian) ∧
      ∀ back ∈ t.back, CausalReads.Issued x r consumer back (FrameTableReads.snapCode endian))
    (child : ∀ row ∈ rows, let next := (play mc fc b row.move).val
      ∃ p : Addressed, let out := p.cluster x r next
        (row.front,row.back) = out.pair (r.supply x) endian next ∧
        CausalReads.Issued x r consumer out.front (FrameTableReads.lookupCode endian next) ∧
        ∀ back ∈ out.back, CausalReads.Issued x r consumer back (FrameTableReads.lookupCode endian next)) :
    Probed mc fc b (CausalTable.lookup x init consumer) attempts rows := by
  constructor
  · intro slot e hit
    cases parent with
    | none => simp [words,ProbeRetry.probe,ProbeRetry.scan] at hit
    | some p =>
      simp only [words,Option.toList_some,List.flatMap_cons,List.flatMap_nil,List.append_nil,List.map_map,Function.comp_def] at hit
      have val := FrameTableReads.Operations.probe_result (r.supply x) p.allocation p.clusters p.key endian (Cutoffs.key b) p.opening
      exact CausalReads.probe_hit x init r consumer p.allocation p.clusters p.key b p.opening
        (steps p rfl) (val.symm.trans hit)
  · intro row listed e hit
    obtain ⟨p,eq,front,back⟩ := child row listed
    apply CausalReads.cluster_hit x init r consumer p.allocation p.clusters p.key _ p.opening front back
    dsimp only [Addressed.cluster] at eq
    rwa [←eq]

/-- A selected return belongs to this history prefix. For the event-indexed history
    this is the identity inclusion when board, time and frame list are shared. -/
def Returns.InPrefix (x : Returns) (prior : CausalTable.Values) : Prop :=
  ∀ b pv a z v, x.prior b pv a z v → prior b pv a z v

private theorem Returns.correct {x : Returns} {prior : CausalTable.Values} (inside : x.InPrefix prior) (valid : CausalTable.Correct mc fc prior) :
    CausalTable.Correct mc fc x.prior :=
  fun b p a z v w n used => valid b p a z v w n (inside _ _ _ _ _ used)

/-- The large-frame cases consume the actual scheduled owner result. The shallow
    cases consume the fixed solver or the worker's cancellation-aware cache run. -/
inductive Body (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (prior : CausalTable.Values)
    (reads : CausalTable.Lookups) (b : Position) : Bool → Int → Int → Int → Prop where
  | pv (d : Depth) (paired checked cut : Bool) (mem : MoveList.Initialization.Memory Move.Move) (x : Returns)
      (inside : x.InPrefix prior) (mode : Bitboard.Square → Heuristic) (tie : GeneratedSquare mc b.val → Fin 64)
      (hint : Option Bitboard.Square) (schedule : Nat → Bool × Bool)
      (splits : Splits (Children mc fc b x.prior) true cut d.toNat width beta)
      (full : ExactDepth.Full b d) (wide : 7 ≤ width)
      (run : PvSerialMir.scheduled mc fc b d paired checked mem alpha beta
        (fun next cd lo hi => x.reply next cd true false lo hi) (pvOrder mc fc b mode tie hint)
        (x.pvReplies mc fc b d cut) schedule (pvSplit splits) = some v) : Body mc fc prior reads b true alpha beta v
  | scout (d : Depth) (alpha beta : Score) (paired checked cut : Bool) (mem : MoveList.Initialization.Memory Move.Move)
      (attempts : List RawAttempt) (rows : List (ChildLookupFrom mc fc b src)) (x : Returns)
      (inside : x.InPrefix prior) (mode : Bitboard.Square → Heuristic) (tie : GeneratedSquare mc b.val → Fin 64)
      (schedule : GeneralOwnerMir.Schedule (GeneratedSquare mc b.val))
      (splits : Splits (Children mc fc b x.prior) false cut d.toNat width beta.raw.toInt)
      (full : ExactDepth.Full b d) (wide : 7 ≤ width) (probed : Probed mc fc b reads attempts rows)
      (run : GeneralOwnerMir.scheduled mc fc b d alpha beta paired checked cut mem attempts rows
        (framePriority mc fc b mode tie) x.scout schedule (scoutSplit splits) = some v) :
      Body mc fc prior reads b false alpha.raw.toInt beta.raw.toInt v
  | direct (alpha beta raw : Score) (aligned : EndgameSearch.EmptyStateMatches b mask count)
      (small : count.toNat ≤ 4)
      (result : EndgameSearch.Direct.Mir.run target bmi2 b.val ⟨mask,count⟩ alpha checked = some raw) :
      Body mc fc prior reads b false alpha.raw.toInt beta.raw.toInt raw.raw.toInt
  | boundary (p : EndgameSearch.CacheHistory.Pool) (timeline : EndgameSearch.CacheHistory.Timeline p) (worker atStep : Nat)
      (alpha beta : Score) (d : Depth) (state : EndgameSearch.Cancellation.State) (attempts : List RawAttempt)
      (history : state.history = (timeline.readAt mc fc worker atStep).history) (aligned : state.empties.matches b)
      (parent : ∀ slot e, ProbeRetry.probe (attempts.map (ProbeRetry.observe (Cutoffs.key b))) = .hit slot e → reads b e)
      (result : DeepLeafMir.sourceRun target bmi2 paired mc fc p.address b alpha beta d state attempts
        (DeepLeafMir.parallelPool size) active checked aborted = some v) :
      Body mc fc prior reads b false alpha.raw.toInt beta.raw.toInt v
  | boundarySave (p : EndgameSearch.CacheHistory.Pool) (timeline : EndgameSearch.CacheHistory.Timeline p)
      (worker opening : Nat) (i : EndgameSearch.CacheHistory.Input) (board : i.board = b)
      (depth : i.depth.toNat = 12) (count : i.depth = i.empties.count)
      (parallel : i.parallel = true) (running : i.aborted = false)
      (enough : DeepLeafMir.enoughNodes i.checked before after = true)
      (localRun : i.arm = .deep i.active)
      (result : (i.run mc fc p.address (timeline.readAt mc fc worker opening).history).score = some v) :
      Body mc fc prior reads b false i.alpha.raw.toInt i.beta.raw.toInt v
  | fixed (result : ScoreRefresh.DiscScore) (alpha beta raw : Score)
      (solved : EndgameSearch.Direct.FixedDisc target bmi2 b (ScoreRefresh.toDiscDiff alpha).raw result.raw)
      (converted : ScoreRefresh.fromDiscDiff checked result = .ok raw) :
      Body mc fc prior reads b false alpha.raw.toInt beta.raw.toInt raw.raw.toInt
  | shallow (p : EndgameSearch.CacheHistory.Pool) (timeline : EndgameSearch.CacheHistory.Timeline p) (worker atStep : Nat)
      (alpha beta raw : Score) (lo : 5 ≤ (Board.emptyCount b.val).toNat) (hi : (Board.emptyCount b.val).toNat ≤ 6)
      (solved : let h := (timeline.readAt mc fc worker atStep).history
        let result := EndgameSearch.run mc fc (EndgameSearch.orderedOptions target bmi2 mc fc p.address) .shallow b
          (ScoreRefresh.toDiscDiff alpha).raw.toInt h
        ScoreRefresh.fromDiscDiff checked ⟨BitVec.ofInt 32 result.score⟩ = .ok raw) :
      Body mc fc prior reads b false alpha.raw.toInt beta.raw.toInt raw.raw.toInt
  | cached (p : EndgameSearch.CacheHistory.Pool) (timeline : EndgameSearch.CacheHistory.Timeline p) (worker atStep : Nat)
      (alpha beta raw : Score) (s : EndgameSearch.Cancellation.State)
      (localState : s.history = (timeline.readAt mc fc worker atStep).history) (aligned : s.empties.matches b)
      (solved : let opts := EndgameSearch.orderedOptions target bmi2 mc fc p.address
        let out := EndgameSearch.Cancellation.run mc fc opts b (ScoreRefresh.toDiscDiff alpha).raw.toInt s
        ScoreRefresh.fromDiscDiff checked ⟨BitVec.ofInt 32 (out.score.getD 0)⟩ = .ok raw ∧
          EndgameSearch.Cancellation.publish out aborted raw = some raw) :
      Body mc fc prior reads b false alpha.raw.toInt beta.raw.toInt raw.raw.toInt

theorem preserves : CausalTable.Preserves mc fc (Body mc fc) := by
  intro prior reads valid observed b pv a z v window narrow run
  cases run with
  | pv d paired checked cut mem x inside mode tie hint schedule splits full wide run =>
    exact BoardFrame.preserves _ _ (Returns.correct inside valid) observed _ _ _ _ _ window narrow
      (pv mc fc b d paired checked cut mem a z x mode tie hint schedule splits full wide run)
  | scout d alpha beta paired checked cut mem attempts rows x inside mode tie schedule splits full wide probed run =>
    have priorOK := Returns.correct inside valid
    exact BoardFrame.preserves _ _ priorOK observed _ _ _ _ _ window narrow
      (scout mc fc b d alpha beta paired checked cut mem attempts rows x mode tie schedule splits full wide
        window (narrow rfl) priorOK probed run)
  | direct a z raw aligned small result =>
    obtain ⟨s,ran,bound⟩ := EndgameSearch.Direct.Mir.scoreContract_proven _ _ mc fc b _ _ a z _ aligned small (narrow rfl)
    cases Option.some.inj (ran.symm.trans result)
    exact bound_sound _ window bound
  | boundary p timeline worker atStep a z d state attempts history aligned parent result =>
    have cache := history ▸ (timeline.reads mc fc worker atStep).1
    rw [DeepLeafMir.sourceRun_eq _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ aligned cache] at result
    apply bound_sound _ window
    apply DeepLeafMir.bound_from_probe _ _ mc fc p.address b a z d state attempts _ _ _ _ _ aligned
      (history ▸ (timeline.reads mc fc worker atStep).1) (narrow rfl) result
    intro slot e hit cut
    exact PvsMoveLoop.Bound.ofSoundResult ((observed b e (parent slot e hit)).cut _ _ _ (narrow rfl) cut)
  | boundarySave p timeline worker opening i board _ _ _ _ _ arm result =>
    subst b
    unfold EndgameSearch.CacheHistory.Input.run at result
    rw [arm] at result
    exact bound_sound _ window (DeepLeafMir.local_bound i.target i.bmi2 i.paired mc fc p.address i.board
      i.alpha i.beta (i.state _) i.active i.checked i.aborted i.aligned (timeline.reads mc fc worker opening).1
      (narrow rfl) result)
  | fixed result alpha beta raw solved converted =>
    exact BoardFrame.preserves _ _ valid observed _ _ _ _ _ window narrow (.fixed result alpha beta raw solved converted)
  | shallow p timeline worker atStep alpha beta raw lo hi solved =>
    exact BoardFrame.preserves _ _ valid observed _ _ _ _ _ window narrow
      (.shallow p timeline worker atStep alpha beta raw lo hi solved)
  | cached p timeline worker atStep alpha beta raw s localState aligned solved =>
    exact BoardFrame.preserves _ _ valid observed _ _ _ _ _ window narrow
      (.cached p timeline worker atStep alpha beta raw s localState aligned solved)

end NeuralReversi.PvsParallel.Interleaving.NativeNodes
