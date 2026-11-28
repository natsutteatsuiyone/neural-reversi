module
public import NeuralReversi.PvsParallel.CausalReplay
public import NeuralReversi.EndgameSearch.DeepRecursionMir
public import NeuralReversi.PvsParallel.DeepLeafMir.Program

@[expose] public section

namespace NeuralReversi.PvsParallel.DeepLeafMir

def enoughNodes (checked : Bool) (before after : BitVec 64) : Bool :=
  !(checked && after.usubOverflow before) && decide (boundaryStoreMinNodes ≤ (after-before).toNat)

open Bitboard.Moves (X86Config)
open Flip.Dispatch (TargetConfig)
open Cutoffs (FlipConfig exactSelectivity key)
open PvsBoardTree (Position build minimax)
open TranspositionTable (EntryData Depth canCut)
open ScoreRefresh (Score toDiscDiff fromDiscDiff)
open EndgameSearch.Cancellation (State Outcome publish)
open EndgameTreeCache (Address soundHistory)
open ExactTableHistory (Publications)
open TableReads (ProbeReads)

def lookup (b : Position) (attempts : List TranspositionTable.ProbeRetry.RawAttempt)
    (beta : Score) (depth : Depth) (parallel checked : Bool) : Option (Option EntryData) := do
  if depth.toNat = 12 ∧ parallel then
    match TranspositionTable.ProbeRetry.probe (attempts.map (TranspositionTable.ProbeRetry.observe (key b))) with
    | .miss _ => return none
    | .hit _ entry =>
      let accepted ← (canCut checked entry ⟨beta.raw⟩ depth exactSelectivity true).toOption
      return if accepted then some entry else none
  else return none

private theorem lookup_exit {attempts : List TranspositionTable.ProbeRetry.RawAttempt}
    {b : Position} {depth : Depth} {parallel checked : Bool} (alpha beta : Score)
    (reads : TableReads.ProbeReadsFrom source attempts)
    (response : lookup b attempts beta depth parallel checked = some (some entry)) :
    CausalReplay.PrefixExitFrom mc fc source mc fc b false alpha.raw.toInt beta.raw.toInt entry.score.toInt := by
  unfold lookup at response
  split at response
  · cases hit : TranspositionTable.ProbeRetry.probe
        (attempts.map (TranspositionTable.ProbeRetry.observe (key b))) with
    | miss slot => simp [hit] at response
    | hit slot item =>
      cases cut : canCut checked item ⟨beta.raw⟩ depth exactSelectivity true with
      | error e => simp [hit, cut, Except.toOption] at response
      | ok value =>
        cases value <;> simp [hit, cut, Except.toOption] at response
        cases response
        exact .probe attempts slot _ ⟨alpha.raw⟩ ⟨beta.raw⟩ depth checked reads hit cut
  · simp at response

private theorem published_range (table : Publications mc fc history) :
    ∀ b e, (b, e) ∈ history → Cutoffs.ExactEndgame e → -32767 < e.score.toInt := by
  induction table with
  | initial => simp
  | append prior write ih =>
    intro b e member exact
    rcases List.mem_append.mp member with old | last
    · exact ih b e old exact
    · obtain ⟨rfl, rfl⟩ := Prod.mk.inj (List.mem_singleton.mp last)
      cases write with
      | other _ different => exact (different exact).elim
      | penalized entry present _ => exact ih _ entry present exact
      | searched pv alpha beta score move depth gen ttPv key previous written window result saved =>
        have same := (Cutoffs.storeContract_proven mc fc _ b pv alpha beta score move depth gen ttPv
          key previous _ prior.sound window result saved).1
        rw [same]
        exact result.lower

private theorem lookup_total (b : Position) (attempts : List TranspositionTable.ProbeRetry.RawAttempt)
    (beta : Score) (depth : Depth) (parallel checked : Bool)
    (table : Publications mc fc history) (reads : ProbeReads history attempts) :
    ∃ entry, lookup b attempts beta depth parallel checked = some entry := by
  unfold lookup
  split
  · cases hit : TranspositionTable.ProbeRetry.probe (attempts.map (TranspositionTable.ProbeRetry.observe (key b))) with
    | miss => exact ⟨_, rfl⟩
    | hit slot entry =>
      obtain ⟨tried, mem, read, occupied⟩ := TranspositionTable.ProbeRetry.contract_proven.2 _ _ _ _ hit
      have stored := reads.published.matches slot entry tried mem read occupied
      simp only
      rw [TranspositionTable.canCut_contract checked entry ⟨beta.raw⟩ depth exactSelectivity true]
      by_cases eligible : TranspositionTable.Eligible entry depth exactSelectivity true
      · have lower := published_range table b entry stored (Cutoffs.exactEndgame_of_eligible eligible)
        have ok : entry.score.toInt ≠ -32768 := by omega
        rw [ite_eq_left eligible]
        simp [ok, Except.toOption]
      · rw [ite_eq_right eligible]
        exact ⟨_, rfl⟩
  · exact ⟨_, rfl⟩

def pollInput (state : State) (active : Bool) : State :=
  if active then state else
    { state with polls := ⟨⟨0, false⟩, fun _ => ⟨0, []⟩, 0⟩ }

private theorem input_matches (aligned : state.empties.matches b) (active : Bool) :
    (pollInput state active).empties.matches b := by cases active <;> exact aligned

private theorem input_cache (sound : soundHistory mc fc state.history) (active : Bool) :
    soundHistory mc fc (pollInput state active).history := by cases active <;> exact sound

def accept (result : Outcome) (checked aborted : Bool) : Option Int := do
  let raw ← (fromDiscDiff checked ⟨BitVec.ofInt 32 (result.score.getD 0)⟩).toOption
  let answer ← publish result aborted raw
  return answer.raw.toInt

def acceptedNws (target : TargetConfig) (bmi2 : Bool) (mc : X86Config) (fc : FlipConfig)
    (address : Address) (b : Position) (alpha : Score) (state : State) (checked aborted : Bool) : Option Int := do
  let result := EndgameSearch.Cancellation.run mc fc (EndgameSearch.orderedOptions target bmi2 mc fc address)
    b (toDiscDiff alpha).raw.toInt state
  accept result checked aborted

private theorem acceptedNws_spec (target : TargetConfig) (bmi2 : Bool) (mc : X86Config)
    (fc : FlipConfig) (address : Address) (b : Position) (alpha beta : Score) (state : State)
    (checked aborted : Bool) (aligned : state.empties.matches b) (cache : soundHistory mc fc state.history)
    (null : beta.raw.toInt = alpha.raw.toInt + 1)
    (answer : acceptedNws target bmi2 mc fc address b alpha state checked aborted = some result) :
    ExecutionWithUses exit children (build mc fc b) false alpha.raw.toInt beta.raw.toInt result ∧
    PvsMoveLoop.Bound (minimax mc fc b) alpha.raw.toInt beta.raw.toInt result := by
  let computed := EndgameSearch.Cancellation.run mc fc (EndgameSearch.orderedOptions target bmi2 mc fc address)
    b (toDiscDiff alpha).raw.toInt state
  obtain ⟨_, _, _, raw, cast, _, valid⟩ := EndgameSearch.Cancellation.boundaryContract_proven
    target bmi2 mc fc address b alpha beta state checked aborted aligned cache null
  change (do let r ← (fromDiscDiff checked ⟨BitVec.ofInt 32 (computed.score.getD 0)⟩).toOption
             let v ← publish computed aborted r
             return v.raw.toInt) = some result at answer
  change fromDiscDiff checked ⟨BitVec.ofInt 32 (computed.score.getD 0)⟩ = .ok raw at cast
  rw [cast] at answer
  cases published : publish computed aborted raw with
  | none => simp [Except.toOption, published] at answer
  | some v =>
    have same : v = raw := by
      unfold publish at published
      split at published <;> simp_all
    subst v
    simp [Except.toOption, published] at answer
    subst result
    exact ⟨.shallow target bmi2 mc fc address b alpha beta raw state checked aborted aligned cache ⟨cast, published⟩,
      PvsMoveLoop.Bound.ofSoundResult (valid published)⟩

def withNws (b : Position) (beta : Score) (depth : Depth) (state : State)
    (attempts : List TranspositionTable.ProbeRetry.RawAttempt) (parallel active checked : Bool)
    (child : Bool → Option Int) : Option Int := do
  if depth ≠ state.empties.count then none else
    let entry ← lookup b attempts beta depth parallel checked
    match ← dispatch depth.toNat parallel entry.isSome active with
    | .probe => return (← entry).score.toInt
    | .activeNws => child true
    | .ignoreNws => child false

def run (target : TargetConfig) (bmi2 : Bool) (mc : X86Config) (fc : FlipConfig)
    (address : Address) (b : Position) (alpha beta : Score) (depth : Depth) (state : State)
    (attempts : List TranspositionTable.ProbeRetry.RawAttempt) (parallel active checked aborted : Bool) : Option Int :=
  withNws b beta depth state attempts parallel active checked fun poll =>
    acceptedNws target bmi2 mc fc address b alpha (pollInput state poll) checked aborted

private theorem runCases {P : Int → Prop} (target : TargetConfig) (bmi2 : Bool) (mc : X86Config) (fc : FlipConfig)
    (address : Address) (b : Position) (alpha beta : Score) (depth : Depth) (state : State)
    (attempts : List TranspositionTable.ProbeRetry.RawAttempt) (parallel active checked aborted : Bool)
    (probeCase : ∀ entry, lookup b attempts beta depth parallel checked = some (some entry) → P entry.score.toInt)
    (nwsCase : ∀ poll v, acceptedNws target bmi2 mc fc address b alpha (pollInput state poll) checked aborted =
      some v → P v)
    (answer : run target bmi2 mc fc address b alpha beta depth state attempts parallel active checked aborted =
      some result) : P result := by
  unfold run withNws at answer
  split at answer
  · simp at answer
  · cases found : lookup b attempts beta depth parallel checked with
    | none => simp [found] at answer
    | some item =>
      cases arm : dispatch depth.toNat parallel item.isSome active with
      | none => simp [found, arm] at answer
      | some branch =>
        cases branch with
        | probe =>
          cases item with
          | none =>
            change dispatch depth.toNat parallel false active = some .probe at arm
            simp [found, arm] at answer
          | some entry =>
            change dispatch depth.toNat parallel true active = some .probe at arm
            simp [found, arm] at answer
            subst result
            exact probeCase entry found
        | activeNws =>
          simp [found, arm] at answer
          exact nwsCase true _ answer
        | ignoreNws =>
          simp [found, arm] at answer
          exact nwsCase false _ answer

theorem observedExecution (target : TargetConfig) (bmi2 : Bool) (mc : X86Config) (fc : FlipConfig)
    (address : Address) (b : Position) (alpha beta : Score) (depth : Depth) (state : State)
    (attempts : List TranspositionTable.ProbeRetry.RawAttempt) (parallel active checked aborted : Bool)
    (reads : TableReads.ProbeReadsFrom source attempts)
    (aligned : state.empties.matches b) (cache : soundHistory mc fc state.history)
    (null : beta.raw.toInt = alpha.raw.toInt + 1)
    (answer : run target bmi2 mc fc address b alpha beta depth state attempts parallel active checked aborted =
      some result) :
    ExecutionWithUses (CausalReplay.PrefixExitFrom mc fc source) children (build mc fc b) false
      alpha.raw.toInt beta.raw.toInt result := by
  apply runCases target bmi2 mc fc address b alpha beta depth state attempts parallel active checked aborted _ _ answer
  · intro entry found
    exact .cutoff mc fc b (lookup_exit alpha beta reads found)
  · intro poll v found
    exact (acceptedNws_spec target bmi2 mc fc address b alpha beta _ _ _ (input_matches aligned poll)
      (input_cache cache poll) null found).1

/-- A physical parent probe contributes only the entry that actually passes
    `canCut`. Every other probe outcome continues into the local NWS arm. -/
theorem bound_from_probe (target : TargetConfig) (bmi2 : Bool) (mc : X86Config) (fc : FlipConfig)
    (address : Address) (b : Position) (alpha beta : Score) (depth : Depth) (state : State)
    (attempts : List TranspositionTable.ProbeRetry.RawAttempt) (parallel active checked aborted : Bool)
    (probe : ∀ slot e, TranspositionTable.ProbeRetry.probe (attempts.map (TranspositionTable.ProbeRetry.observe (key b))) = .hit slot e →
      canCut checked e ⟨beta.raw⟩ depth exactSelectivity true = .ok true →
      PvsMoveLoop.Bound (minimax mc fc b) alpha.raw.toInt beta.raw.toInt e.score.toInt)
    (aligned : state.empties.matches b) (cache : soundHistory mc fc state.history)
    (null : beta.raw.toInt = alpha.raw.toInt+1)
    (answer : run target bmi2 mc fc address b alpha beta depth state attempts parallel active checked aborted = some result) :
    PvsMoveLoop.Bound (minimax mc fc b) alpha.raw.toInt beta.raw.toInt result := by
  apply runCases target bmi2 mc fc address b alpha beta depth state attempts parallel active checked aborted _ _ answer
  · intro e found
    unfold lookup at found
    split at found
    · cases read : TranspositionTable.ProbeRetry.probe (attempts.map (TranspositionTable.ProbeRetry.observe (key b))) with
      | miss slot => simp [read] at found
      | hit slot item =>
        cases accepted : canCut checked item ⟨beta.raw⟩ depth exactSelectivity true with
        | error err => simp [read,accepted,Except.toOption] at found
        | ok ok =>
          cases ok <;> simp [read,accepted,Except.toOption] at found
          exact found ▸ probe slot item read accepted
    · cases found
  · intro poll v found
    exact (acceptedNws_spec (exit := fun _ _ _ _ _ _ _ => False) (children := fun _ _ _ _ _ _ => False)
      target bmi2 mc fc address b alpha beta _ _ _ (input_matches aligned poll) (input_cache cache poll) null found).2

theorem scoreContract_proven (target : TargetConfig) (bmi2 : Bool) (mc : X86Config) (fc : FlipConfig)
    (address : Address) (b : Position) (alpha beta : Score) (depth : Depth) (state : State)
    (attempts : List TranspositionTable.ProbeRetry.RawAttempt) (parallel active checked aborted : Bool)
    (table : Publications mc fc history) (reads : ProbeReads history attempts)
    (aligned : state.empties.matches b) (cache : soundHistory mc fc state.history)
    (null : beta.raw.toInt = alpha.raw.toInt + 1)
    (answer : run target bmi2 mc fc address b alpha beta depth state attempts parallel active checked aborted =
      some result) :
    ExecutionWithUses Cutoffs.Execution children (build mc fc b) false alpha.raw.toInt beta.raw.toInt result ∧
    PvsMoveLoop.Bound (minimax mc fc b) alpha.raw.toInt beta.raw.toInt result := by
  apply runCases target bmi2 mc fc address b alpha beta depth state attempts parallel active checked aborted _ _ answer
  · intro entry found
    have checked := (lookup_exit (mc := mc) (fc := fc) alpha beta reads found).checked table
    exact ⟨.cutoff mc fc b checked, Cutoffs.contract_proven mc fc b false _ _ _ (fun _ => null) checked⟩
  · intro poll v found
    exact acceptedNws_spec target bmi2 mc fc address b alpha beta _ _ _ (input_matches aligned poll)
      (input_cache cache poll) null found

def uncancelled (target : TargetConfig) (bmi2 : Bool) (mc : X86Config) (fc : FlipConfig)
    (address : Address) (b : Position) (alpha : Score) (state : State) (active : Bool) : Prop :=
  let outcome := EndgameSearch.Cancellation.run mc fc (EndgameSearch.orderedOptions target bmi2 mc fc address)
    b (toDiscDiff alpha).raw.toInt (pollInput state active)
  outcome.state.polls.cache.latched = false

theorem normalReturn (target : TargetConfig) (bmi2 : Bool) (mc : X86Config) (fc : FlipConfig)
    (address : Address) (b : Position) (alpha beta : Score) (depth : Depth) (state : State)
    (attempts : List TranspositionTable.ProbeRetry.RawAttempt) (parallel active checked : Bool)
    (table : Publications mc fc history) (reads : ProbeReads history attempts)
    (aligned : state.empties.matches b) (cache : soundHistory mc fc state.history)
    (full : ExactDepth.Full b depth) (size : 6 < depth.toNat ∧ depth.toNat ≤ 12)
    (null : beta.raw.toInt = alpha.raw.toInt + 1)
    (clean : uncancelled target bmi2 mc fc address b alpha state active) :
    ∃ result, run target bmi2 mc fc address b alpha beta depth state attempts parallel active checked false = some result ∧
      ExecutionWithUses Cutoffs.Execution children (build mc fc b) false alpha.raw.toInt beta.raw.toInt result ∧
      PvsMoveLoop.Bound (minimax mc fc b) alpha.raw.toInt beta.raw.toInt result := by
  obtain ⟨item, found⟩ := lookup_total b attempts beta depth parallel checked table reads
  have same : depth = state.empties.count := full.trans aligned.1.symm
  have hasNws : ∃ raw, acceptedNws target bmi2 mc fc address b alpha (pollInput state active) checked false =
      some raw := by
    obtain ⟨_, _, _, score, cast, _, _⟩ := EndgameSearch.Cancellation.boundaryContract_proven
      target bmi2 mc fc address b alpha beta (pollInput state active) checked false (input_matches aligned active)
      (input_cache cache active) null
    unfold uncancelled at clean
    exact ⟨score.raw.toInt, by simp [acceptedNws, accept, cast, publish, clean, Except.toOption]⟩
  have hasResult : ∃ result, run target bmi2 mc fc address b alpha beta depth state attempts parallel active
      checked false = some result := by
    unfold run withNws
    rw [ite_eq_right (fun neq => neq same), found]
    by_cases take : depth.toNat = 12 ∧ parallel ∧ item.isSome
    · cases item with
      | none => simp at take
      | some entry =>
        have arm : dispatch depth.toNat parallel (some entry).isSome active = some .probe := by
          unfold dispatch
          rw [ite_eq_left size, ite_eq_left take]
        exact ⟨entry.score.toInt, by change (dispatch depth.toNat parallel (some entry).isSome active).bind _ = _
                                     rw [arm]
                                     rfl⟩
    · have arm : dispatch depth.toNat parallel item.isSome active = some (if active then .activeNws else .ignoreNws) := by
        unfold dispatch
        rw [ite_eq_left size, ite_eq_right take]
      obtain ⟨raw, published⟩ := hasNws
      refine ⟨raw, ?_⟩
      cases active <;> simp_all
  obtain ⟨result, answer⟩ := hasResult
  exact ⟨result, answer, scoreContract_proven _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ table reads aligned cache null answer⟩

def sourceAccepted (target : TargetConfig) (bmi2 paired : Bool) (mc : X86Config) (fc : FlipConfig)
    (address : Address) (b : Position) (alpha : Score) (state : State) (checked aborted : Bool) : Option Int := do
  let result := EndgameSearch.DeepRecursionMir.run target bmi2 paired mc fc address b (toDiscDiff alpha).raw.toInt state
  accept result checked aborted

theorem local_bound (target : TargetConfig) (bmi2 paired : Bool) (mc : X86Config) (fc : FlipConfig)
    (address : Address) (b : Position) (alpha beta : Score) (state : State) (active checked aborted : Bool)
    (aligned : state.empties.matches b) (cache : soundHistory mc fc state.history)
    (null : beta.raw.toInt = alpha.raw.toInt + 1)
    (accepted : sourceAccepted target bmi2 paired mc fc address b alpha (pollInput state active) checked aborted = some value) :
    PvsMoveLoop.Bound (minimax mc fc b) alpha.raw.toInt beta.raw.toInt value := by
  unfold sourceAccepted at accepted
  rw [EndgameSearch.DeepRecursionMir.agrees target bmi2 paired mc fc address b _ _
    (input_matches aligned active) (EndgameSearch.ShallowMir.scoreThreshold alpha) (input_cache cache active)] at accepted
  exact (acceptedNws_spec (exit:=Cutoffs.Execution) (children:=fun _ _ _ _ _ _ => False) target bmi2 mc fc
    address b alpha beta _ checked aborted (input_matches aligned active) (input_cache cache active) null accepted).2

def sourceRun (target : TargetConfig) (bmi2 paired : Bool) (mc : X86Config) (fc : FlipConfig)
    (address : Address) (b : Position) (alpha beta : Score) (depth : Depth) (state : State)
    (attempts : List TranspositionTable.ProbeRetry.RawAttempt) (parallel active checked aborted : Bool) : Option Int :=
  withNws b beta depth state attempts parallel active checked fun poll =>
    sourceAccepted target bmi2 paired mc fc address b alpha (pollInput state poll) checked aborted

theorem sourceRun_eq (target : TargetConfig) (bmi2 paired : Bool) (mc : X86Config) (fc : FlipConfig)
    (address : Address) (b : Position) (alpha beta : Score) (depth : Depth) (state : State)
    (attempts : List TranspositionTable.ProbeRetry.RawAttempt) (parallel active checked aborted : Bool)
    (aligned : state.empties.matches b) (cache : soundHistory mc fc state.history) :
    sourceRun target bmi2 paired mc fc address b alpha beta depth state attempts parallel active checked aborted =
      run target bmi2 mc fc address b alpha beta depth state attempts parallel active checked aborted := by
  have same (poll : Bool) :
      sourceAccepted target bmi2 paired mc fc address b alpha (pollInput state poll) checked aborted =
        acceptedNws target bmi2 mc fc address b alpha (pollInput state poll) checked aborted := by
    unfold sourceAccepted acceptedNws
    rw [EndgameSearch.DeepRecursionMir.agrees target bmi2 paired mc fc address b _ _
      (input_matches aligned poll) (EndgameSearch.ShallowMir.scoreThreshold alpha) (input_cache cache poll)]
  unfold sourceRun run
  simp only [same]

end NeuralReversi.PvsParallel.DeepLeafMir
