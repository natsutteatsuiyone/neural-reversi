import NeuralReversi.PvsParallel.SerialNonPv
import NeuralReversi.PvsParallel.SerialCalls.Sites
import NeuralReversi.PvsParallel.EndgameOrderMir

namespace NeuralReversi.PvsParallel.SerialNonPvCalls

open Bitboard.Moves (X86Config)
open Cutoffs (FlipConfig)
open PvsBoardTree (Position GeneratedSquare legalSquares play pass)
open TranspositionTable (Depth)
open TranspositionTable.ProbeRetry (RawAttempt)
open TableReads (ChildLookupFrom)
open SerialCalls (Program Request)
open GeneralNonPvMir (checkedHint probeHint order entryAction runPrefix presearch resumeOwner)
open GeneralOwnerMir (Receipt recordedOwner after)
open PvsPostpone (Queue)
open Threading.SplitWindow (State)

open SerialCalls.Program (ofOption)
private theorem eval_ofOption (result : Option α) (reply : Req → Option Int) :
    (ofOption result).eval reply = result := by cases result <;> rfl

/-- The serial NonPV owner keeps alpha fixed and stops the instant its update cuts. -/
def owner (request : Move → Req) (alpha : Int) : List Move → Int → Program Req Int
  | [], best => .pure best
  | m :: rest, best =>
    SerialCalls.Sites.ownerWith false false alpha (request m) (request m) fun _ _ score =>
      let update := after alpha best score
      if update.2.2 then .pure score else owner request alpha rest update.1

private def receiptScore (reply : Move → Option Int) (r : Receipt Move) : Option Int :=
  if r.visited.all (fun m => (reply m).isSome) then
    match r.outcome with
    | .returned v => some v
    | .handoff _ => none
  else none

private theorem receipt_cons (m : Move) (r : Receipt Move) (reply : Move → Option Int) :
    receiptScore reply {r with visited := m :: r.visited} =
      if (reply m).isSome then receiptScore reply r else none := by
  cases h : (reply m).isSome <;> simp [receiptScore, h]

private theorem eval_owner_run (request : Move → Req) (reply : Req → Option Int)
    (search : PvsMoveLoop.ChildSearch Move) (alpha : Int)
    (matched : ∀ m, search m false (-(alpha+1)) (-alpha) = (reply (request m)).getD 0)
    (depth : Nat) (cut : Bool) (moves : List Move)
    (idx best fuel i previous r)
    (ran : GeneralOwnerMir.run SerialNonPv.schedule depth cut search fuel i ⟨idx,moves,[]⟩
      ⟨alpha,best⟩ previous = some r) :
    (owner request alpha moves best).eval reply = receiptScore (reply ∘ request) r := by
  induction moves generalizing idx best fuel i previous r with
  | nil =>
    cases fuel with
    | zero => cases ran
    | succ fuel =>
      simp [GeneralOwnerMir.run, Queue.pending] at ran
      subst r
      rfl
  | cons m rest ih =>
    cases fuel with
    | zero => cases ran
    | succ fuel =>
      let word := (reply (request m)).getD 0
      let update := after alpha best (-word)
      let chosen := if update.2.1 then some m else previous
      simp only [GeneralOwnerMir.run, Queue.pending, List.append_nil, List.isEmpty_cons,
        Bool.false_eq_true, ite_false, SerialNonPv.schedule, GeneralOwnerMir.canSplit, Bool.and_false,
        PvsPostpone.step, PvsPostpone.shouldPostpone, Bool.false_and, matched] at ran
      change (if update.2.2 then some ⟨[m], ⟨idx+1,rest,[]⟩, .returned (-word), chosen⟩ else
        (GeneralOwnerMir.run SerialNonPv.schedule depth cut search fuel (i+1) ⟨idx+1, rest,[]⟩
          ⟨alpha,update.1⟩ chosen).bind
          (fun tail => some {tail with visited := m :: tail.visited})) = some r at ran
      by_cases stop : update.2.2
      · simp only [stop, ite_true, Option.some.injEq] at ran
        subst r
        cases got : reply (request m) with
        | none => simp [owner, SerialCalls.Sites.ownerWith, Program.eval, receiptScore, Function.comp_apply, got]
        | some value =>
          have w : word = value := by simp [word, got]
          have cuts : (after alpha best (-value)).2.2 = true := w ▸ stop
          simp [owner, SerialCalls.Sites.ownerWith, Program.eval, receiptScore, Function.comp_apply, got, cuts, w]
      · simp only [ite_eq_right stop] at ran
        obtain ⟨tail, inner, returned⟩ := Option.bind_eq_some_iff.mp ran
        cases Option.some.inj returned
        rw [receipt_cons]
        cases got : reply (request m) with
        | none => simp [owner, SerialCalls.Sites.ownerWith, Program.eval, Function.comp_apply, got]
        | some value =>
          have w : word = value := by simp [word, got]
          have next : ¬(after alpha best (-value)).2.2 = true := w ▸ stop
          have recurse := ih (idx+1) update.1 fuel (i+1) chosen tail inner
          simpa [owner, SerialCalls.Sites.ownerWith, Program.eval, Function.comp_apply, got, next, update, w] using recurse

private theorem eval_owner (request : Move → Req) (reply : Req → Option Int)
    (search : PvsMoveLoop.ChildSearch Move) (alpha : Int)
    (matched : ∀ m, search m false (-(alpha+1)) (-alpha) = (reply (request m)).getD 0)
    (depth : Nat) (cut : Bool) (ordered : List Move) (idx : Nat) (best : Int) :
    (owner request alpha (ordered.drop idx) best).eval reply =
      recordedOwner SerialNonPv.schedule depth cut search (reply ∘ request)
        (fun _ _ _ => none) alpha ordered idx best := by
  let q : Queue Move := ⟨idx, ordered.drop idx, []⟩
  obtain ⟨r, ran⟩ := GeneralOwnerMir.run_exists SerialNonPv.schedule depth cut search
    (q.budget+1) 0 q ⟨alpha,best⟩ none (Nat.lt_succ_self _)
  have scored := eval_owner_run request reply search alpha matched depth cut (ordered.drop idx)
    idx best _ _ _ r ran
  unfold recordedOwner
  change _ = (GeneralOwnerMir.run SerialNonPv.schedule depth cut search (q.budget+1) 0 q
    ⟨alpha,best⟩ none).bind _
  rw [ran]
  exact scored

def scout (mc : X86Config) (fc : FlipConfig) (b : Position) (depth : Depth) (cut : Bool)
    (alpha : Int) (m : GeneratedSquare mc b.val) : Request :=
  SerialCalls.Sites.scoutRequest (play mc fc b m).val depth cut alpha

/-- The TT presearch and normal NonPV scout use identical recursive operands. -/
theorem presearch_site (mc : X86Config) (fc : FlipConfig) (b : Position) (depth : Depth)
    (cut : Bool) (alpha : Int) (m : GeneratedSquare mc b.val) :
    SerialCalls.Sites.presearchWith (scout mc fc b depth cut alpha m) .pure =
      SerialCalls.Sites.presearch (play mc fc b m).val depth cut alpha := rfl

def passRequest (b : Position) (depth : Depth) (cut : Bool) (alpha beta : Int) : Request :=
  ⟨pass b, depth, false, !cut, -beta, -alpha⟩

def mobile [DecidableEq Move] (request : Move → Req) (priority : Move → Int)
    (hint : Option Move) (raw : Bool) (first : Move) (rest : List Move) (alpha beta : Int) : Program Req Int :=
  let list := first :: rest
  let ordered := order priority hint list
  if hint.isSome = raw then
    if list.length = 1 || raw then
      SerialCalls.Sites.presearchWith (request (hint.getD first)) fun score =>
        if (alpha < score) && (beta ≤ score) then .pure score
        else owner request alpha (ordered.drop 1) score
    else owner request alpha ordered (-32767)
  else .fail

def prelude (mc : X86Config) (fc : FlipConfig) (b : Position) (depth : Depth)
    (alpha beta : ScoreRefresh.Score) (paired checked cut : Bool)
    (mem : MoveList.Initialization.Memory Move.Move) (attempts : List RawAttempt)
    (reads : List (ChildLookupFrom mc fc b source)) : Program Request Int :=
  if entryAction mc fc b depth alpha beta paired checked mem attempts reads = .pass then
    SerialCalls.Sites.pass (pass b) depth false cut alpha.raw.toInt beta.raw.toInt
  else ofOption (runPrefix mc fc b depth alpha beta paired checked cut mem attempts reads (fun _ _ _ _ _ => none))

def frame (mc : X86Config) (fc : FlipConfig) (b : Position) (depth : Depth)
    (alpha beta : ScoreRefresh.Score) (paired checked cut : Bool)
    (mem : MoveList.Initialization.Memory Move.Move) (attempts : List RawAttempt)
    (reads : List (ChildLookupFrom mc fc b source)) (priority : GeneratedSquare mc b.val → Int) : Program Request Int :=
  if entryAction mc fc b depth alpha beta paired checked mem attempts reads = .rest then
    match legalSquares mc b.val with
    | [] => .fail
    | first :: rest =>
      let tt := probeHint b attempts
      mobile (scout mc fc b depth cut alpha.raw.toInt) priority (checkedHint mc b tt) tt.isSome
        first rest alpha.raw.toInt beta.raw.toInt
  else prelude mc fc b depth alpha beta paired checked cut mem attempts reads

def child (reply : Request → Option Int) : GeneralNonPvMir.Child :=
  fun next d cc a z => reply ⟨next,d,false,cc,a,z⟩

theorem eval_prelude (mc : X86Config) (fc : FlipConfig) (b : Position) (depth : Depth)
    (alpha beta : ScoreRefresh.Score) (paired checked cut : Bool)
    (mem : MoveList.Initialization.Memory Move.Move) (attempts : List RawAttempt)
    (reads : List (ChildLookupFrom mc fc b source)) (reply : Request → Option Int) :
    (prelude mc fc b depth alpha beta paired checked cut mem attempts reads).eval reply =
      runPrefix mc fc b depth alpha beta paired checked cut mem attempts reads (child reply) := by
  unfold prelude
  cases act : entryAction mc fc b depth alpha beta paired checked mem attempts reads <;>
    unfold entryAction at act <;>
    simp only [runPrefix, act, reduceCtorEq, ite_false, ite_true, eval_ofOption, child] <;> rfl

/-- Partial callbacks are allowed. An absent word makes the computation fail
    exactly when its call is issued; a cutoff never reads the remaining suffix. -/
theorem eval_frame (mc : X86Config) (fc : FlipConfig) (b : Position) (depth : Depth)
    (alpha beta : ScoreRefresh.Score) (paired checked cut : Bool)
    (mem : MoveList.Initialization.Memory Move.Move) (attempts : List RawAttempt)
    (reads : List (ChildLookupFrom mc fc b source)) (priority : GeneratedSquare mc b.val → Int)
    (reply : Request → Option Int) :
    (frame mc fc b depth alpha beta paired checked cut mem attempts reads priority).eval reply =
      SerialNonPv.runFrame mc fc b depth alpha beta paired checked cut mem attempts reads priority (child reply) := by
  unfold frame SerialNonPv.runFrame GeneralOwnerMir.scheduled GeneralNonPvMir.frame
  cases act : entryAction mc fc b depth alpha beta paired checked mem attempts reads <;>
    try exact eval_prelude mc fc b depth alpha beta paired checked cut mem attempts reads reply
  cases moves : legalSquares mc b.val with
  | nil => rfl
  | cons first rest =>
    let a := alpha.raw.toInt
    let z := beta.raw.toInt
    let request := scout mc fc b depth cut a
    let hint := checkedHint mc b (probeHint b attempts)
    let raw := (probeHint b attempts).isSome
    let ordered := order priority hint (first :: rest)
    let search := GeneralNonPvMir.observe mc fc b depth cut (child reply)
    let lookup := (reply ∘ request)
    let resume := recordedOwner SerialNonPv.schedule depth.toNat cut search lookup (fun _ _ _ => none) a ordered
    have owner_eq (idx : Nat) (best : Int) :
        (owner request a (ordered.drop idx) best).eval reply = resume idx best :=
      eval_owner request reply search a (fun _ => rfl) depth.toNat cut ordered idx best
    change (mobile request priority hint raw first rest a z).eval reply = (do
      guard (hint.isSome = raw)
      if (first :: rest).length = 1 || raw then let _ ← lookup (hint.getD first)
      resumeOwner resume (presearch (first :: rest).length hint first a z (fun m => search m false)))
    by_cases valid : hint.isSome = raw
    case neg => simp [mobile, valid, guard, Program.eval]; rfl
    simp only [mobile, valid, guard, ite_true, pure, bind, Option.bind]
    by_cases used : ((decide ((first :: rest).length = 1)) || raw) = true
    · simp only [used, ite_true, SerialCalls.Sites.presearchWith, Program.eval, presearch, valid]
      cases got : reply (request (hint.getD first)) with
      | none => rw [show lookup (hint.getD first) = none from got]; rfl
      | some word =>
        have haveWord : search (hint.getD first) false (-(a+1)) (-a) = word := by
          change (reply (request (hint.getD first))).getD 0 = word
          simp [got]
        change _ = (lookup (hint.getD first)).bind _
        rw [show lookup (hint.getD first) = some word from got]
        simp only [Option.bind_some, haveWord]
        change (if (a < -word) && (z ≤ -word) then Program.pure (-word)
          else owner request a (ordered.drop 1) (-word)).eval reply =
            resumeOwner resume (if (a < -word) && (z ≤ -word) then .storeLower _ (-word) else .owner 1 (-word))
        split <;> first | rfl | exact owner_eq 1 (-word)
    · simpa only [presearch, valid, ite_eq_right used, resumeOwner, List.drop_zero] using owner_eq 0 (-32767)

private theorem owner_prefix (request : Move → Req) (alpha : Int) (moves : List Move)
    (best : Int) {events value} (ran : Program.Runs (owner request alpha moves best) events value) :
    List.IsPrefix (events.map Prod.fst) (moves.map request) := by
  induction moves generalizing best events value with
  | nil => cases ran; exact ⟨[], rfl⟩
  | cons m tail ih =>
    simp only [owner, SerialCalls.Sites.ownerWith, Bool.false_and, Bool.false_eq_true, ite_false] at ran
    cases ran with
    | call more =>
      dsimp only at more
      split at more
      · cases more; exact ⟨tail.map request, rfl⟩
      · obtain ⟨suffix, eq⟩ := ih _ more
        exact ⟨suffix, congrArg (request m :: ·) eq⟩

private theorem mobile_prefix {Move : Type} [DecidableEq Move] (request : Move → Req) (priority : Move → Int)
    (hint : Option Move) (raw : Bool) (first : Move) (rest : List Move) (alpha beta : Int)
    (legal : ∀ m, hint = some m → m ∈ first :: rest)
    (low : ∀ m ∈ first :: rest, priority m < MoveList.Ordering.ttMoveValue)
    (ran : Program.Runs (mobile request priority hint raw first rest alpha beta) events value) :
    List.IsPrefix (events.map Prod.fst) ((order priority hint (first :: rest)).map request) := by
  unfold mobile at ran
  dsimp only at ran
  split at ran
  next present =>
    split at ran
    next used =>
      have head : (order priority hint (first :: rest)).head? = some (hint.getD first) := by
        cases hint with
        | some sq => exact (GeneralNonPvMir.orderHint priority _ sq (legal _ rfl) (fun m mem _ => low m mem)).2
        | none =>
          have empty : rest = [] := by simpa [← present] using used
          subst rest
          rfl
      cases ordered : order priority hint (first :: rest) with
      | nil => simp [ordered] at head
      | cons m tail =>
        have pick : m = hint.getD first := by simpa [ordered] using head
        subst m
        rw [ordered] at ran
        cases ran with
        | call more =>
          dsimp only at more
          split at more
          · cases more; exact ⟨tail.map request, rfl⟩
          · obtain ⟨suffix, eq⟩ := owner_prefix _ _ _ _ more
            exact ⟨suffix, congrArg (request (hint.getD first) :: ·) eq⟩
    next => exact owner_prefix _ _ _ _ ran
  next => cases ran

theorem forced_of_action {d : Nat} {stable mobile other wipes pc cc : Bool}
    (selected : GeneralNonPvMir.dispatch d stable mobile other wipes pc cc = .pass) :
    mobile = false ∧ other = true := by
  unfold GeneralNonPvMir.dispatch at selected
  repeat' split at selected
  all_goals first | contradiction | simp_all

/-- Each issued scout places one legal move. A pass occurs alone and requires
    the other side to be mobile. The parent observes at most one word per request. -/
theorem frame_children (mc : X86Config) (fc : FlipConfig) (b : Position) (depth : Depth)
    (alpha beta : ScoreRefresh.Score) (paired checked cut : Bool)
    (mem : MoveList.Initialization.Memory Move.Move) (attempts : List RawAttempt)
    (reads : List (ChildLookupFrom mc fc b source)) (priority : GeneratedSquare mc b.val → Int)
    (low : ∀ m ∈ legalSquares mc b.val, priority m < MoveList.Ordering.ttMoveValue)
    (ran : Program.Runs (frame mc fc b depth alpha beta paired checked cut mem attempts reads priority)
      events value) :
    (events.map Prod.fst).Nodup ∧ ∀ e ∈ events,
      (∃ m, e.1 = scout mc fc b depth cut alpha.raw.toInt m) ∨
      (Board.ForcedPass mc b.val ∧ e.1 = passRequest b depth cut alpha.raw.toInt beta.raw.toInt) := by
  unfold frame at ran
  split at ran
  · cases moves : legalSquares mc b.val with
    | nil => rw [moves] at ran; cases ran
    | cons first rest =>
      rw [moves] at ran low
      let hint := checkedHint mc b (probeHint b attempts)
      have legal (m : GeneratedSquare mc b.val) (_ : hint = some m) : m ∈ first :: rest := by
        obtain ⟨v, present, eq⟩ := (PvsBoardTree.legalSquares_complete mc b.val m.val).mpr
          ((Board.getMoves_exact mc b.val _).mp m.property)
        rwa [Subtype.ext eq, moves] at present
      have pref := mobile_prefix _ priority hint _ first rest _ _ legal low ran
      have perm := GeneralNonPvMir.orderPerm priority _ hint legal (fun m _ _ mem _ => low m mem)
      have nodup := perm.nodup_iff.mpr (moves ▸ (EndgameOrderMir.generated_layout mc b).1)
      refine ⟨pref.sublist.nodup (nodup.map _ ?_), ?_⟩
      · intro m n ne same
        exact ne (PvsBoardTree.play_injective mc fc b (congrArg (fun (r : Request) => r.board.val) same))
      · intro e mem
        obtain ⟨m, _, eq⟩ := List.mem_map.mp (pref.subset (List.mem_map_of_mem (f := Prod.fst) mem))
        exact .inl ⟨m, eq.symm⟩
  · unfold prelude at ran
    split at ran
    next selected =>
      have act := forced_of_action selected
      rw [PvEntryMir.generated_count] at act
      have stuck : legalSquares mc b.val = [] := by simpa using act.1
      cases ran with
      | call more =>
        cases more
        refine ⟨by simp, ?_⟩
        intro e mem
        rcases List.mem_singleton.mp mem with rfl
        exact .inr ⟨⟨PvsBoardTree.stuck_of_nil stuck, act.2⟩, rfl⟩
    next =>
      obtain ⟨_, rfl⟩ := Program.runs_ofOption ran
      simp

open Aspiration (Window)
open PvsBoardTree (build)
open PvsMoveLoop (scoutWindow reflectedWindow)

/-- The search window and depth in each suspended call are valid for that board. -/
theorem frame_edges (mc : X86Config) (fc : FlipConfig) (b : Position) (depth : Depth)
    (alpha beta : ScoreRefresh.Score) (paired checked cut : Bool)
    (mem : MoveList.Initialization.Memory Move.Move) (attempts : List RawAttempt)
    (reads : List (ChildLookupFrom mc fc b source)) (priority : GeneratedSquare mc b.val → Int)
    (low : ∀ m ∈ legalSquares mc b.val, priority m < MoveList.Ordering.ttMoveValue)
    (window : Window alpha.raw.toInt beta.raw.toInt) (null : beta.raw.toInt = alpha.raw.toInt + 1)
    (ran : Program.Runs (frame mc fc b depth alpha beta paired checked cut mem attempts reads priority)
      events value) :
    ∀ e ∈ events, ExactDepth.RecursiveCall mc fc b depth e.1.board e.1.depth ∧
      Window e.1.alpha e.1.beta ∧ e.1.beta = e.1.alpha + 1 ∧ e.1.pv = false := by
  intro e mem
  rcases (frame_children mc fc b depth alpha beta paired checked cut _ attempts reads priority low ran).2 e mem with
    ⟨m, same⟩ | ⟨forced, same⟩
  · rw [same]
    exact ⟨.ply b m depth, scoutWindow window, by dsimp [scout, SerialCalls.Sites.scoutRequest]; omega, rfl⟩
  · rw [same]
    exact ⟨.forcedPass b depth forced, reflectedWindow window, by dsimp [passRequest]; omega, rfl⟩

/-- The result of the native NonPV frame only depends on the children it issued.
    Their replay evidence may come from separate, state-dependent recursive runs. -/
theorem frame_replay (mc : X86Config) (fc : FlipConfig) (b : Position) (depth : Depth)
    (alpha beta : ScoreRefresh.Score) (paired checked cut : Bool)
    (mem : MoveList.Initialization.Memory Move.Move) (attempts : List RawAttempt)
    (reads : List (ChildLookupFrom mc fc b (·.publication history))) (priority : GeneratedSquare mc b.val → Int)
    (n : Nat) (full : ExactDepth.Full b depth)
    (window : Window alpha.raw.toInt beta.raw.toInt) (null : beta.raw.toInt = alpha.raw.toInt + 1)
    (observed : TableReads.ProbeReads history attempts) (published : ExactTableHistory.Publications mc fc history)
    (low : ∀ m ∈ legalSquares mc b.val, priority m < MoveList.Ordering.ttMoveValue)
    (ran : Program.Runs (frame mc fc b depth alpha beta paired checked cut mem attempts reads priority)
      events value)
    (children : ∀ e ∈ events, boundedReplay n (build mc fc e.1.board) e.1.pv e.1.alpha e.1.beta e.2) :
    boundedReplay (n+1) (build mc fc b) false alpha.raw.toInt beta.raw.toInt value := by
  have unique := (frame_children mc fc b depth alpha beta paired checked cut mem attempts reads priority low ran).1
  let reply (r : Request) := events.lookup r
  have answer := ran.eval reply (fun e present => Program.lookup_mem unique present)
  rw [eval_frame] at answer
  have replay (r : Request) (v : Int) (found : reply r = some v) :
      boundedReplay n (build mc fc r.board) r.pv r.alpha r.beta v := by
    obtain ⟨front, tail, eq, _⟩ := List.lookup_eq_some_iff.mp found
    exact children (r,v) (eq ▸ by simp)
  have execution : ExecutionWithUses (CausalReplay.PrefixExitFrom mc fc (·.publication history))
      (fun _ => boundedReplay n) (build mc fc b) false alpha.raw.toInt beta.raw.toInt value := by
    apply GeneralOwnerMir.scheduledExecution mc fc b depth alpha beta paired checked cut mem attempts reads
      priority (child reply) SerialNonPv.schedule (fun _ _ _ => none) full null observed low _ _ _ answer
    · intro _ _ v found
      exact replay (passRequest b depth cut alpha.raw.toInt beta.raw.toInt) v found
    · intro m v found
      have childReplay := replay (scout mc fc b depth cut alpha.raw.toInt m) v found
      refine ⟨childReplay, ?_⟩
      exact (replayContract_proven (build mc fc (play mc fc b m).val) false _ _ v n
        (scoutWindow window) (fun _ => by omega) childReplay).upper
    · intro _ _ _ _ _ _ impossible
      contradiction
  exact execution.weaken (fun _ _ _ _ _ _ h => h) (fun _ _ _ _ _ _ _ exit => exit.checked published)

end NeuralReversi.PvsParallel.SerialNonPvCalls
