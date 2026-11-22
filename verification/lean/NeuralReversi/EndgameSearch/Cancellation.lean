import NeuralReversi.EndgameSearch.EmptyState
import NeuralReversi.Threading.SplitCutoff

namespace NeuralReversi.EndgameSearch.Cancellation

open PvsBoardTree (Position GeneratedSquare legalSquares play pass minimax)
open Bitboard.Moves (X86Config)
open EndgameSearch (Options CorrectOptions SafeThreshold EmptyStateMatches)
open EndgameCache (SearchRange)
open EndgameTreeCache (Store Address discValue soundHistory)
open Solve2 (NWS)
open Flip.Dispatch (TargetConfig)
open ScoreRefresh (Score DiscScore fromDiscDiff toDiscDiff)

open Threading.SplitCutoff (Polling after_latched occurred_latched)

structure EmptyState where
  links : EmptyListLinks.Links
  parity : BitVec 8
  count : BitVec 32

def EmptyState.matches (state : EmptyState) (b : Position) : Prop :=
  EmptyStateMatches b state.links state.parity state.count

def EmptyState.remove (state : EmptyState) (sq : Bitboard.Square) : EmptyState :=
  ⟨EmptyListLinks.remove state.links sq, EmptyListParity.toggle state.parity sq, state.count - 1⟩

def EmptyState.restore (state : EmptyState) (sq : Bitboard.Square) : EmptyState :=
  ⟨EmptyListLinks.restore state.links sq, EmptyListParity.toggle state.parity sq, state.count + 1⟩

structure State where
  empties : EmptyState
  history : List Store
  polls : Polling

structure Outcome where
  score : Option Int
  state : State

private def State.afterPoll (state : State) : State :=
  { state with polls := state.polls.after }

private theorem State.afterPoll_latched (state : State) (latched : state.polls.cache.latched = true) :
    state.afterPoll = state := by
  cases state
  simp [State.afterPoll, after_latched latched]

private def Outcome.negate (result : Outcome) : Outcome :=
  { result with score := result.score.map Neg.neg }

private def moveFrame {mc : X86Config} {b : Position} (sq : GeneratedSquare mc b.val)
    (call : State → Outcome) (state : State) : Outcome :=
  if state.polls.occurred then ⟨none, state.afterPoll⟩
  else
    let result := (call { state.afterPoll with empties := state.empties.remove sq.val }).negate
    { result with state := { result.state with empties := result.state.empties.restore sq.val } }

private def storeDeep (b : Position) (alpha : Int) (result : Int × State) : Outcome :=
  let score := result.1
  let state := result.2
  if state.polls.occurred then ⟨none, state.afterPoll⟩
  else ⟨some score, { state.afterPoll with history := state.history ++ [⟨.deep, ⟨b.val, alpha, score⟩⟩] }⟩

private def scanRaw {Move : Type} (call : Move → State → Outcome) (alpha : Int) :
    List Move → Int → State → Int × State
  | [], best, state => (best, state)
  | move :: rest, best, state =>
    let result := call move state
    let score := result.score.getD 0
    if alpha + 1 ≤ score then (score, result.state)
    else scanRaw call alpha rest (max best score) result.state

private theorem scanRaw_stopped {Move : Type} (call : Move → State → Outcome)
    (skip : ∀ sq input, input.polls.cache.latched = true → call sq input = ⟨none, input⟩)
    (alpha : Int) (moves : List Move) (best : Int) (state : State) (cut : state.polls.cache.latched = true) :
    (scanRaw call alpha moves best state).2 = state := by
  induction moves generalizing best with
  | nil => rfl
  | cons sq rest ih =>
    rw [scanRaw, skip sq state cut]
    dsimp only
    split
    · rfl
    · exact ih _

def StoreGateContract : Prop :=
  ∀ (b : Position) (alpha score : Int) (state : State),
    let out := storeDeep b alpha (score, state)
    out.state.empties = state.empties ∧ out.state.polls = state.polls.after ∧
    (state.polls.occurred = true → out.score = none ∧ out.state.history = state.history) ∧
    (state.polls.occurred = false → out.score = some score ∧
      out.state.history = state.history ++ [⟨.deep, ⟨b.val, alpha, score⟩⟩])

theorem storeGateContract_proven : StoreGateContract := by
  intro b alpha score state
  cases stopped : state.polls.occurred <;> simp [storeDeep, State.afterPoll, stopped]


private def searchOrder (opts : Options mc) (b : Position)
    (first : GeneratedSquare mc b.val) (rest : List (GeneratedSquare mc b.val)) :
    List (GeneratedSquare mc b.val) :=
  if rest.isEmpty then [first] else opts.order b .deep

def run (mc : X86Config) (fc : FlipConfig) (opts : Options mc)
    (b : Position) (alpha : Int) (state : State) : Outcome :=
  match moves : legalSquares mc b.val with
  | [] =>
    if available : Board.hasLegalMoves mc (pass b).val = true then
      (run mc fc opts (pass b) (-(alpha + 1)) state).negate
    else ⟨some (Board.AwardedScore b.val), state⟩
  | first :: rest =>
    match EndgameCutoffs.beforeMoves fc b.val state.history opts.address .deep alpha
      ((first :: rest).map Subtype.val) with
    | some score => ⟨some score, state⟩
    | none =>
      let child := fun sq input => moveFrame sq (fun placed =>
        let next := (play mc fc b sq).val
        if placed.empties.count.toNat ≤ 6 then
          let outcome := EndgameSearch.run mc fc opts .shallow next (-(alpha + 1)) placed.history
          ⟨some outcome.score, { placed with history := outcome.history }⟩
        else run mc fc opts next (-(alpha + 1)) placed) input
      storeDeep b alpha (scanRaw child alpha (searchOrder opts b first rest) (-127) state)
termination_by Board.playRank mc b.val
decreasing_by
  · exact Board.legalCheckedStep_strict mc (.pass ⟨PvsBoardTree.stuck_of_nil moves, available⟩) b.property
  · exact (play mc fc b sq).property

private def ScoreValid (alpha value : Int) (state : State) (score : Option Int) : Prop :=
  match score with
  | none => state.polls.cache.latched = true
  | some score => SearchRange score ∧ NWS alpha value score

def Valid (mc : X86Config) (fc : FlipConfig) (input : State) (alpha value : Int)
    (out : Outcome) : Prop :=
  out.state.empties = input.empties ∧ soundHistory mc fc out.state.history ∧
    input.history <+: out.state.history ∧ ScoreValid alpha value out.state out.score

private theorem immediate (sound : soundHistory mc fc state.history)
    (score : ScoreValid alpha value state answer) :
    Valid mc fc state alpha value ⟨answer, state⟩ :=
  ⟨rfl, sound, ⟨[], by simp⟩, score⟩

private theorem negated (valid : Valid mc fc input (-(alpha + 1)) value out) :
    Valid mc fc input alpha (-value) out.negate := by
  obtain ⟨empties, sound, growth, bounded⟩ := valid
  refine ⟨empties, sound, growth, ?_⟩
  cases response : out.score with
  | none => simpa [Outcome.negate, ScoreValid, response] using bounded
  | some score =>
    simp only [response, ScoreValid] at bounded
    obtain ⟨⟨lo, hi⟩, nws⟩ := bounded
    simp only [Outcome.negate, response, Option.map_some, ScoreValid]
    exact ⟨⟨by omega, by omega⟩, Solve34.negate_nws nws⟩

private theorem moveFrame_valid (mc : X86Config) (fc : FlipConfig) (b : Position)
    (sq : GeneratedSquare mc b.val) (alpha : Int) (state : State)
    (aligned : state.empties.matches b) (sound : soundHistory mc fc state.history)
    (call : State → Outcome)
    (valid : ∀ input, input.empties.matches (play mc fc b sq).val →
      soundHistory mc fc input.history →
      Valid mc fc input (-(alpha + 1)) (discValue mc fc (play mc fc b sq).val.val) (call input)) :
    Valid mc fc state alpha (-discValue mc fc (play mc fc b sq).val.val) (moveFrame sq call state) := by
  unfold moveFrame
  split
  · rename_i cut
    exact ⟨rfl, sound, ⟨[], by simp [State.afterPoll]⟩, cut⟩
  · obtain ⟨_, _, childAligned, _, _, links, parity, count⟩ :=
      emptyStatePlayContract_proven mc fc b sq _ _ _ aligned
    let placed := { state.afterPoll with empties := state.empties.remove sq.val }
    let result := (call placed).negate
    have next : placed.empties.matches (play mc fc b sq).val := childAligned
    obtain ⟨restored, after, grew, bounds⟩ := negated (valid placed next sound)
    change result.state.empties = placed.empties at restored
    refine ⟨?_, after, grew, ?_⟩
    · change result.state.empties.restore sq.val = state.empties
      rw [restored]
      simp only [placed, EmptyState.remove, EmptyState.restore, links, parity, count]
    · change ScoreValid alpha (-discValue mc fc (play mc fc b sq).val.val)
        { result.state with empties := result.state.empties.restore sq.val } result.score
      change ScoreValid alpha (-discValue mc fc (play mc fc b sq).val.val) result.state result.score at bounds
      cases h : result.score <;> simpa [ScoreValid, h] using bounds

private def ScanValid (mc : X86Config) (fc : FlipConfig) (input : State)
    (alpha value : Int) (result : Int × State) : Prop :=
  result.2.empties = input.empties ∧ soundHistory mc fc result.2.history ∧
    input.history <+: result.2.history ∧
    (result.2.polls.occurred ≠ true → SearchRange result.1 ∧ NWS alpha value result.1)

private theorem scan_valid {Move : Type} (mc : X86Config) (fc : FlipConfig) (v : Move → Int)
    (call : Move → State → Outcome)
    (skip : ∀ sq input, input.polls.cache.latched = true → call sq input = ⟨none, input⟩)
    (alpha : Int) (moves : List Move) (seed best : Int)
    (state : State) (sound : soundHistory mc fc state.history)
    (calls : ∀ sq input, sq ∈ moves → input.empties = state.empties →
      soundHistory mc fc input.history → Valid mc fc input alpha (v sq) (call sq input))
    (seedSound : NWS alpha seed best) (bestRange : SearchRange best) :
    ScanValid mc fc state alpha (PvsMoveLoop.bestValue v moves seed) (scanRaw call alpha moves best state) := by
  induction moves generalizing seed best state with
  | nil => exact ⟨rfl, sound, ⟨[], by simp [scanRaw]⟩, fun _ => ⟨bestRange, seedSound⟩⟩
  | cons first rest ih =>
    let result := call first state
    obtain ⟨aligned, after, grew, bounded⟩ := calls first state (by simp) rfl sound
    change result.state.empties = state.empties at aligned
    change ScoreValid alpha (v first) result.state result.score at bounded
    rw [scanRaw]
    change ScanValid mc fc state alpha (PvsMoveLoop.bestValue v (first :: rest) seed)
      (if alpha + 1 ≤ result.score.getD 0 then (result.score.getD 0, result.state)
       else scanRaw call alpha rest (max best (result.score.getD 0)) result.state)
    cases response : result.score with
    | none =>
      have cut : result.state.polls.cache.latched = true := by simpa only [ScoreValid, response] using bounded
      have untouched : (if alpha + 1 ≤ 0 then (0, result.state)
          else scanRaw call alpha rest (max best 0) result.state).2 = result.state := by
        split
        · rfl
        · exact scanRaw_stopped call skip alpha rest _ result.state cut
      simp only [Option.getD_none, ScanValid, untouched]
      exact ⟨aligned, after, grew, fun pending => (pending (occurred_latched cut)).elim⟩
    | some score =>
      simp only [Option.getD_some]
      obtain ⟨range, bounds⟩ : SearchRange score ∧ NWS alpha (v first) score := by
        simpa only [response, ScoreValid] using bounded
      split
      · rename_i high
        have below := PvsMoveLoop.bestValue_ge_move v (first :: rest) seed (by simp : first ∈ first :: rest)
        exact ⟨aligned, after, grew, fun _ =>
          ⟨range, fun h => by omega, fun h => Int.le_trans (bounds.2 h) below⟩⟩
      · rw [PvsMoveLoop.bestValue]
        have merged := Solve34.max_nws bounds seedSound
        have mergedRange : SearchRange (max best score) := by
          unfold SearchRange at bestRange range ⊢
          omega
        obtain ⟨ctxEq, cache, growth, resultBounds⟩ := ih (max seed (v first)) (max best score)
          result.state after (fun sq input mem ctxEq => calls sq input (List.mem_cons_of_mem _ mem)
            (ctxEq.trans aligned)) merged mergedRange
        exact ⟨ctxEq.trans aligned, cache, grew.trans growth, resultBounds⟩

private theorem store_valid (b : Position)
    (valid : ScanValid mc fc input alpha (discValue mc fc b.val) result) :
    Valid mc fc input alpha (discValue mc fc b.val) (storeDeep b alpha result) := by
  obtain ⟨aligned, sound, grew, bounds⟩ := valid
  simp only [storeDeep]
  split
  · rename_i cut
    exact ⟨aligned, sound, grew, cut⟩
  · rename_i openGate
    obtain ⟨range, nws⟩ := bounds openGate
    exact ⟨aligned, EndgameTreeCache.append_sound mc fc _ sound _
      (EndgameCache.searchRange_store range) nws, grew.trans (List.prefix_append _ _), range, nws⟩

theorem run_valid (mc : X86Config) (fc : FlipConfig) (opts : Options mc)
    (correct : CorrectOptions mc fc opts) (b : Position) (alpha : Int) (state : State)
    (aligned : state.empties.matches b) (sound : soundHistory mc fc state.history)
    (window : SafeThreshold alpha) :
    Valid mc fc state alpha (discValue mc fc b.val) (run mc fc opts b alpha state) := by
  have childWindow : SafeThreshold (-(alpha + 1)) := ⟨by have := window.2; omega, by have := window.1; omega⟩
  rw [run.eq_def]
  split
  next moves =>
    split
    next available =>
      have forced : Board.ForcedPass mc b.val := ⟨PvsBoardTree.stuck_of_nil moves, available⟩
      have a := EndgameTreeCache.scaled_discValue mc fc b
      have c := EndgameTreeCache.scaled_discValue mc fc (pass b)
      rw [PvsBoardTree.minimax_pass mc fc b moves available] at a
      have eq : discValue mc fc b.val = -discValue mc fc (pass b).val := by omega
      rw [eq]
      exact negated (run_valid mc fc opts correct (pass b) (-(alpha + 1)) state
        (emptyState_pass b _ _ _ aligned) sound childWindow)
    next unavailable =>
      have no : Board.hasLegalMoves mc (pass b).val = false := Bool.eq_false_iff.mpr unavailable
      have scaled := EndgameTreeCache.scaled_discValue mc fc b
      rw [PvsBoardTree.terminal_value mc fc b moves no] at scaled
      have eq : discValue mc fc b.val = Board.AwardedScore b.val := by omega
      have ⟨lo, hi⟩ := EndgameTreeCache.discValue_range mc fc b.val
      exact immediate sound (show SearchRange (Board.AwardedScore b.val) ∧
        NWS alpha (discValue mc fc b.val) (Board.AwardedScore b.val)
        from ⟨⟨by omega, by omega⟩, eq ▸ Solve2.nws_exact _ _⟩)
  next first rest moves =>
    split
    next score hit =>
      obtain ⟨range, bounds⟩ := EndgameCutoffs.contract_proven mc fc b state.history opts.address .deep
        alpha score (first :: rest) sound (fun sq member => moves.symm ▸ member) hit
      exact immediate sound ⟨EndgameCache.cacheRange_search range, bounds⟩
    next =>
      apply store_valid b
      let v := fun sq => -discValue mc fc (play mc fc b sq).val.val
      have eq : discValue mc fc b.val = PvsMoveLoop.bestValue v (searchOrder opts b first rest) (-127) := by
        rw [EndgameTreeCache.discValue_moves mc fc b first rest moves]
        apply PvsMoveLoop.bestValue_eq_of_same_moves
        intro sq
        unfold searchOrder
        split
        · have nil := List.isEmpty_iff.mp ‹rest.isEmpty = true›
          rw [nil]
        · rw [← moves]
          exact (correct.1 b .deep sq).symm
      rw [eq]
      apply scan_valid mc fc v _ (fun sq input cut => by simp [moveFrame, occurred_latched cut, input.afterPoll_latched cut]) alpha _ (-127) (-127) state sound
      · intro sq input _ same inputSound
        exact moveFrame_valid mc fc b sq alpha input (same.symm ▸ aligned) inputSound _ (by
          intro placed childAligned ready
          split
          · have ⟨range, bound, sound, grew⟩ := contract_proven mc fc opts .shallow
              (play mc fc b sq).val (-(alpha + 1)) placed.history correct ready (fun _ _ => childWindow)
            exact ⟨rfl, sound, grew, range, bound⟩
          · exact run_valid mc fc opts correct (play mc fc b sq).val (-(alpha + 1)) placed
              childAligned ready childWindow)
      · exact Solve2.nws_exact _ _
      · exact ⟨by decide, by decide⟩
termination_by Board.playRank mc b.val
decreasing_by
  · exact Board.legalCheckedStep_strict mc (.pass forced) b.property
  · exact (play mc fc b sq).property

def publish (result : Outcome) (aborted : Bool) (score : Score) : Option Score :=
  if result.state.polls.occurred || aborted then none else some score

def Contract : Prop :=
  ∀ (mc : X86Config) (fc : FlipConfig) (opts : Options mc)
    (b : Position) (alpha : Int) (state : State),
    CorrectOptions mc fc opts → state.empties.matches b →
    soundHistory mc fc state.history → SafeThreshold alpha →
    Valid mc fc state alpha (discValue mc fc b.val) (run mc fc opts b alpha state)

theorem contract_proven : Contract := by
  intro mc fc opts b alpha state correct aligned sound window
  exact run_valid mc fc opts correct b alpha state aligned sound window

def BoundaryContract : Prop :=
  ∀ (target : TargetConfig) (bmi2 : Bool) (mc : X86Config) (fc : FlipConfig)
    (address : Address) (b : Position) (alpha beta : Score) (state : State) (checked aborted : Bool),
    state.empties.matches b → soundHistory mc fc state.history →
    beta.raw.toInt = alpha.raw.toInt + 1 →
    let result := run mc fc (orderedOptions target bmi2 mc fc address) b (toDiscDiff alpha).raw.toInt state
    result.state.empties = state.empties ∧ soundHistory mc fc result.state.history ∧
    state.history <+: result.state.history ∧
    ∃ raw, fromDiscDiff checked ⟨BitVec.ofInt 32 (result.score.getD 0)⟩ = .ok raw ∧
      (result.score = none → raw.raw.toInt = 0 ∧ publish result aborted raw = none) ∧
      (publish result aborted raw = some raw →
        Aspiration.SoundResult (minimax mc fc b) alpha.raw.toInt beta.raw.toInt raw.raw.toInt)

theorem boundaryContract_proven : BoundaryContract := by
  intro target bmi2 mc fc address b alpha beta state checked aborted aligned sound window
  let opts := orderedOptions target bmi2 mc fc address
  let threshold := (toDiscDiff alpha).raw.toInt
  let result := run mc fc opts b threshold state
  have floor := ScoreRefresh.disc_floor alpha
  have safe : SafeThreshold threshold := by
    change SafeThreshold ((toDiscDiff alpha).raw.toInt)
    rw [floor]
    have low := BitVec.le_toInt alpha.raw
    have high := BitVec.toInt_lt (x := alpha.raw)
    exact ⟨by simp at low; omega, by simp at high; omega⟩
  obtain ⟨restored, cache, grew, valid⟩ :=
    contract_proven mc fc opts b threshold state (orderedOptions_correct target bmi2 mc fc address)
      aligned sound safe
  change ScoreValid threshold (discValue mc fc b.val) result.state result.score at valid
  have range : SearchRange (result.score.getD 0) := by
    cases answer : result.score with
    | none => exact ⟨by decide, by decide⟩
    | some value => exact (show SearchRange value ∧ NWS threshold (discValue mc fc b.val) value
        from by simpa only [ScoreValid, answer] using valid).1
  let score : DiscScore := ⟨BitVec.ofInt 32 (result.score.getD 0)⟩
  have asInt : score.raw.toInt = result.score.getD 0 := by
    obtain ⟨lo, hi⟩ := range
    exact BitVec.toInt_ofInt_eq_self (by decide) (by omega) (by omega)
  have lo : -127 ≤ score.raw.toInt := asInt ▸ range.1
  have hi : score.raw.toInt ≤ 127 := asInt ▸ range.2
  obtain ⟨raw, cast, _, scaled, _⟩ := ScoreRefresh.bounded_roundtrip score lo hi
  have converted : fromDiscDiff checked score = .ok raw := by
    have within : ¬ (score.raw.toInt < -127 ∨ score.raw.toInt > 127) := by omega
    simpa [fromDiscDiff, within] using cast
  rw [asInt, Int.mul_comm] at scaled
  refine ⟨restored, cache, grew, raw, converted, ?_, ?_⟩
  · intro cut
    change result.score = none at cut
    change raw.raw.toInt = 0 ∧ publish result aborted raw = none
    have latched : result.state.polls.cache.latched = true := by
      simpa only [ScoreValid, cut] using valid
    exact ⟨by simpa [cut] using scaled, by simp [publish, occurred_latched latched]⟩
  · intro allowed
    change publish result aborted raw = some raw at allowed
    cases answer : result.score with
    | none =>
      have latched : result.state.polls.cache.latched = true := by simpa only [ScoreValid, answer] using valid
      simp [publish, occurred_latched latched] at allowed
    | some value =>
      have bound : SearchRange value ∧ NWS threshold (discValue mc fc b.val) value := by
        simpa only [ScoreValid, answer] using valid
      have exactRaw := EndgameTreeCache.rawWindow_sound mc fc b alpha.raw.toInt value bound.1
        (floor ▸ bound.2)
      rw [← window] at exactRaw
      simp only [answer, Option.getD_some] at scaled
      rw [scaled]
      exact exactRaw

end NeuralReversi.EndgameSearch.Cancellation
