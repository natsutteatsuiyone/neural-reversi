import NeuralReversi.Game.Transitions.Properties
import NeuralReversi.MidgameScoreRange.Domain

namespace NeuralReversi.GameAnalysis

open Board (Board)
open Game.Transitions (MovesConfig FlipConfig GameStateError isLegalMove nextBoardAfterPlay)
open MidgameAspiration (FiniteScore)
open ScaledScore (ScaledScore)

inductive Action where
  | play (square : Square.Square)
  | pass

inductive Step where
  | play (before : Board)
  | pass

def replayStep (mc : MovesConfig) (fc : FlipConfig) (board : Board) :
    Action → Except GameStateError (Step × Board)
  | .play sq =>
    if isLegalMove mc board sq then .ok (.play board, nextBoardAfterPlay fc board sq)
    else .error (.illegalMove sq)
  | .pass =>
    if Board.isGameOver mc board then .error .passAfterGameOver
    else if Board.hasLegalMoves mc board then .error .passWithLegalMoves
    else .ok (.pass, Board.switchPlayers board)

def replay (mc : MovesConfig) (fc : FlipConfig) (board : Board) :
    List Action → Except GameStateError (List Step × Board)
  | [] => .ok ([], board)
  | action :: rest =>
    match replayStep mc fc board action with
    | .error err => .error err
    | .ok (step, next) =>
      (replay mc fc next rest).map fun (steps, final) => (step :: steps, final)

inductive LegalHistory (mc : MovesConfig) (fc : FlipConfig) :
    Board → List Action → List Step → Board → Prop where
  | nil (board : Board) : LegalHistory mc fc board [] [] board
  | play {board rest steps final} (sq : Square.Square)
      (legal : Bitboard.Moves.LegalMove board.player board.opponent sq.val)
      (tail : LegalHistory mc fc (nextBoardAfterPlay fc board sq) rest steps final) :
      LegalHistory mc fc board (.play sq :: rest) (.play board :: steps) final
  | pass {board rest steps final}
      (forced : Board.ForcedPass mc board)
      (tail : LegalHistory mc fc (Board.switchPlayers board) rest steps final) :
      LegalHistory mc fc board (.pass :: rest) (.pass :: steps) final

private theorem switch_legal (mc : MovesConfig) (board : Board)
    (live : Board.isGameOver mc board = false) (stuck : Board.hasLegalMoves mc board = false) :
    Board.hasLegalMoves mc (Board.switchPlayers board) = true := by
  simpa [Board.isGameOver, stuck] using live

def ReplayContract : Prop :=
  (∀ mc fc board actions steps final,
    replay mc fc board actions = .ok (steps, final) ↔ LegalHistory mc fc board actions steps final) ∧
  (∀ mc fc board actions err,
    replayStep mc fc board actions = .error err →
      ∀ rest, replay mc fc board (actions :: rest) = .error err)

theorem replayContract_proven : ReplayContract := by
  constructor
  · intro mc fc board actions steps final
    constructor
    · induction actions generalizing board steps final with
      | nil =>
        intro result
        have same : ([], board) = (steps, final) := Except.ok.inj result
        cases same
        exact .nil board
      | cons action rest ih =>
        cases action with
        | play sq =>
          intro result
          by_cases legal : isLegalMove mc board sq = true
          · cases more : replay mc fc (nextBoardAfterPlay fc board sq) rest with
            | error err => simp [replay, replayStep, legal, more, Except.map] at result
            | ok pair =>
              obtain ⟨tail, last⟩ := pair
              have same : (.play board :: tail, last) = (steps, final) := by
                simpa [replay, replayStep, legal, more, Except.map] using result
              cases same
              exact .play sq ((Game.Transitions.isLegalMove_iff mc board sq).mp legal) (ih _ _ _ more)
          · simp [replay, replayStep, legal] at result
        | pass =>
          intro result
          by_cases terminal : Board.isGameOver mc board = true
          · simp [replay, replayStep, terminal] at result
          · by_cases mobile : Board.hasLegalMoves mc board = true
            · simp [replay, replayStep, terminal, mobile] at result
            · cases more : replay mc fc (Board.switchPlayers board) rest with
              | error err => simp [replay, replayStep, terminal, mobile, more, Except.map] at result
              | ok pair =>
                obtain ⟨tail, last⟩ := pair
                have same : (.pass :: tail, last) = (steps, final) := by
                  simpa [replay, replayStep, terminal, mobile, more, Except.map] using result
                cases same
                have stuck : Board.hasLegalMoves mc board = false := by simpa using mobile
                have live : Board.isGameOver mc board = false := by simpa using terminal
                exact .pass ⟨stuck, switch_legal mc board live stuck⟩ (ih _ _ _ more)
    · intro history
      induction history with
      | nil => rfl
      | @play board rest steps final sq legal tail ih =>
        have listed := (Game.Transitions.isLegalMove_iff mc board sq).mpr legal
        simp [replay, replayStep, listed, ih, Except.map]
      | @pass board rest steps final forced tail ih =>
        have live : Board.isGameOver mc board = false := by
          simp [Board.isGameOver, forced.1, forced.2]
        simp [replay, replayStep, live, forced.1, ih, Except.map]
  · intro mc fc board action err rejected rest
    simp [replay, rejected]

def checkedScore (score : ScaledScore) : Except Int ScaledScore :=
  if MidgameScoreRange.isFinite score.raw then .ok score else .error score.raw.toInt

def propagate (checked : Bool) (best previous : ScaledScore) :
    Except String (Int × Int × Int) :=
  (ScaledScore.negate checked previous).bind fun played =>
    (ScaledScore.subScore checked best played).map fun difference =>
      (played.raw.toInt, max difference.raw.toInt 0, max best.raw.toInt played.raw.toInt)

private theorem math_exact (checked : Bool) (value : Int) (message : String)
    (lower : -2147483648 ≤ value) (upper : value < 2147483648) :
    ScaledScore.checkedMathSpec checked value message = .ok value := by
  have safe : ¬ ScaledScore.outsideI32 value := by
    unfold ScaledScore.outsideI32
    omega
  have exactWord : value.bmod 4294967296 = value :=
    Int.bmod_eq_of_le (m := 4294967296) (by exact lower) (by exact upper)
  simp [ScaledScore.checkedMathSpec, safe, exactWord]

def PropagationContract : Prop :=
  (∀ score : ScaledScore,
    (checkedScore score = .ok score ↔ FiniteScore score.raw.toInt) ∧
    (checkedScore score = .error score.raw.toInt ↔ ¬ FiniteScore score.raw.toInt)) ∧
  (∀ checked (best previous : ScaledScore),
    FiniteScore best.raw.toInt → FiniteScore previous.raw.toInt →
    propagate checked best previous =
      .ok (-previous.raw.toInt, max (best.raw.toInt + previous.raw.toInt) 0,
        max best.raw.toInt (-previous.raw.toInt)) ∧
    FiniteScore (max best.raw.toInt (-previous.raw.toInt)) ∧
    0 ≤ max (best.raw.toInt + previous.raw.toInt) 0)

theorem propagationContract_proven : PropagationContract := by
  constructor
  · intro score
    have valid := MidgameScoreRange.gateContract_proven score.raw
    cases finite : MidgameScoreRange.isFinite score.raw <;> simp_all [checkedScore]
  · intro checked best previous bestFinite prevFinite
    have negated := ScaledScore.negate_spec checked previous
    have negLo : -2147483648 ≤ -previous.raw.toInt := by unfold FiniteScore at prevFinite; omega
    have negHi : -previous.raw.toInt < 2147483648 := by unfold FiniteScore at prevFinite; omega
    rw [math_exact checked _ _ negLo negHi] at negated
    cases playedResult : ScaledScore.negate checked previous with
    | error error =>
      simp [ScaledScore.observeScore, playedResult, Except.map] at negated
    | ok played =>
      have playedValue : played.raw.toInt = -previous.raw.toInt := by
        simpa [ScaledScore.observeScore, playedResult, Except.map, ScaledScore.value] using negated
      have difference := ScaledScore.subWords_spec checked best.raw played.raw
      rw [playedValue] at difference
      have lower : -2147483648 ≤ best.raw.toInt - -previous.raw.toInt := by
        unfold FiniteScore at bestFinite prevFinite
        omega
      have upper : best.raw.toInt - -previous.raw.toInt < 2147483648 := by
        unfold FiniteScore at bestFinite prevFinite
        omega
      rw [math_exact checked _ _ lower upper] at difference
      change ScaledScore.observeScore (ScaledScore.subScore checked best played) = _ at difference
      cases diffResult : ScaledScore.subScore checked best played with
      | error error =>
        simp [ScaledScore.observeScore, diffResult, Except.map] at difference
      | ok diff =>
        have diffValue : diff.raw.toInt = best.raw.toInt + previous.raw.toInt := by
          simp [ScaledScore.observeScore, diffResult, Except.map, ScaledScore.value] at difference
          omega
        exact ⟨by simp [propagate, playedResult, diffResult, Except.bind, Except.map, playedValue, diffValue],
          MidgameScoreRange.finite_max bestFinite (MidgameScoreRange.finite_neg prevFinite), by omega⟩

inductive Seed where
  | searchable
  | forcedPass
  | terminal
  deriving DecidableEq

def classifySeed (mc : MovesConfig) (board : Board) : Seed :=
  if Board.hasLegalMoves mc board then .searchable
  else if Board.isGameOver mc board then .terminal
  else .forcedPass

def SeedContract : Prop :=
  ∀ mc board,
    (classifySeed mc board = .searchable ↔ Board.hasLegalMoves mc board = true) ∧
    (classifySeed mc board = .forcedPass ↔ Board.ForcedPass mc board) ∧
    (classifySeed mc board = .terminal ↔
      Board.hasLegalMoves mc board = false ∧
        Board.hasLegalMoves mc (Board.switchPlayers board) = false)

theorem seedContract_proven : SeedContract := by
  intro mc board
  cases first : Board.hasLegalMoves mc board <;>
    cases second : Board.hasLegalMoves mc (Board.switchPlayers board) <;>
      simp [classifySeed, Board.isGameOver, Board.ForcedPass, first, second]

inductive Outcome where
  | complete
  | cancelled
  | searchError (message : String)
  | invalidScore (raw : Int)
  deriving DecidableEq

structure Progress where
  index : Nat
  best : Int
  played : Int
  loss : Int
  bestMove : Square.Square
  pv : List Square.Square
  deriving DecidableEq

structure Report where
  progress : List Progress
  outcome : Outcome
  deriving DecidableEq

structure SearchResult where
  score : ScaledScore
  bestMove : Square.Square
  pv : List Square.Square

abbrev Search := Nat → Board → Except String SearchResult

def backward (search : Search) (cancelled : Nat → Bool) (poll : Nat) (score : Int) :
    List (Nat × Step) → Report
  | [] => ⟨[], .complete⟩
  | (i, step) :: rest =>
    if cancelled poll then ⟨[], .cancelled⟩
    else match step with
    | .pass => backward search cancelled (poll + 1) (-score) rest
    | .play before =>
      match search i before with
      | .error message => ⟨[], .searchError message⟩
      | .ok value =>
        if cancelled (poll + 1) then ⟨[], .cancelled⟩
        else match checkedScore value.score with
        | .error raw => ⟨[], .invalidScore raw⟩
        | .ok valid =>
          let best := valid.raw.toInt
          let tail := backward search cancelled (poll + 2) (max best (-score)) rest
          {tail with progress :=
            ⟨i, best, -score, max (best + score) 0, value.bestMove, value.pv⟩ :: tail.progress}

def playIndices : List (Nat × Step) → List Nat
  | [] => []
  | (_, .pass) :: rest => playIndices rest
  | (i, .play _) :: rest => i :: playIndices rest

def ProgressContract : Prop :=
  ∀ search cancelled poll score steps,
    let report := backward search cancelled poll score steps
    (report.progress.map Progress.index).IsPrefix (playIndices steps) ∧
    (report.outcome = .complete → report.progress.map Progress.index = playIndices steps)

theorem progressContract_proven : ProgressContract := by
  intro search cancelled poll score steps report
  subst report
  induction steps generalizing poll score with
  | nil => simp [backward, playIndices, List.IsPrefix]
  | cons item rest ih =>
    obtain ⟨i, step⟩ := item
    cases step with
    | pass =>
      by_cases stopped : cancelled poll = true
      · simp [backward, playIndices, stopped, List.IsPrefix]
      · simpa [backward, playIndices, stopped] using ih (poll + 1) (-score)
    | play board =>
      by_cases stopped : cancelled poll = true
      · simp [backward, playIndices, stopped, List.IsPrefix]
      · cases found : search i board with
        | error message => simp [backward, playIndices, stopped, found, List.IsPrefix]
        | ok value =>
          by_cases late : cancelled (poll + 1) = true
          · simp [backward, playIndices, stopped, found, late, List.IsPrefix]
          · cases accepted : checkedScore value.score with
            | error raw => simp [backward, playIndices, stopped, found, late, accepted, List.IsPrefix]
            | ok valid =>
              obtain ⟨older, complete⟩ := ih (poll + 2) (max valid.raw.toInt (-score))
              simpa [backward, playIndices, stopped, found, late, accepted, older] using complete

def MetadataContract : Prop :=
  ∀ search cancelled poll score steps report,
    report ∈ (backward search cancelled poll score steps).progress →
    ∃ before result, (report.index, .play before) ∈ steps ∧
      search report.index before = .ok result ∧ report.bestMove = result.bestMove ∧ report.pv = result.pv

theorem metadataContract_proven : MetadataContract := by
  intro search cancelled poll score steps report
  induction steps generalizing poll score with
  | nil => simp [backward]
  | cons item rest ih =>
    obtain ⟨i, step⟩ := item
    intro emitted
    by_cases stopped : cancelled poll = true
    · simp [backward, stopped] at emitted
    · cases step with
      | pass =>
        obtain ⟨before, result, member, found, best, pv⟩ :=
          ih (poll + 1) (-score) (by simpa [backward, stopped] using emitted)
        exact ⟨before, result, List.mem_cons_of_mem _ member, found, best, pv⟩
      | play before =>
        cases found : search i before with
        | error message => simp [backward, stopped, found] at emitted
        | ok value =>
          by_cases late : cancelled (poll + 1) = true
          · simp [backward, stopped, found, late] at emitted
          · cases accepted : checkedScore value.score with
            | error raw => simp [backward, stopped, found, late, accepted] at emitted
            | ok valid =>
              simp only [backward, stopped, found, late, accepted] at emitted
              rcases List.mem_cons.mp emitted with rfl | tail
              · exact ⟨before, value, by simp, found, rfl, rfl⟩
              · obtain ⟨earlier, result, member, searched, best, pv⟩ :=
                  ih (poll + 2) (max valid.raw.toInt (-score)) tail
                exact ⟨earlier, result, List.mem_cons_of_mem _ member, searched, best, pv⟩

def searchSeed (search : Board → Except String SearchResult) (cancelled : Nat → Bool)
    (board : Board) : Except Outcome Int :=
  match search board with
  | .error message => .error (.searchError message)
  | .ok value =>
    if cancelled 1 then .error .cancelled
    else match checkedScore value.score with
    | .ok valid => .ok valid.raw.toInt
    | .error raw => .error (.invalidScore raw)

def reportGame (mc : MovesConfig) (fc : FlipConfig)
    (search : Option Nat → Board → Except String SearchResult) (cancelled : Nat → Bool)
    (board : Board) (actions : List Action) : Except GameStateError Report :=
  (replay mc fc board actions).map fun (steps, final) =>
    if cancelled 0 then ⟨[], .cancelled⟩
    else
      let reversed := steps.zipIdx.reverse.map fun (step, index) => (index, step)
      let analyzeSteps := backward (fun index => search (some index)) cancelled
      match classifySeed mc final with
      | .searchable =>
        match searchSeed (search none) cancelled final with
        | .ok score => analyzeSteps 2 score reversed
        | .error outcome => ⟨[], outcome⟩
      | .forcedPass =>
        match searchSeed (search none) cancelled (Board.switchPlayers final) with
        | .ok score => analyzeSteps 2 (-score) reversed
        | .error outcome => ⟨[], outcome⟩
      | .terminal => analyzeSteps 1 (Board.AwardedScore final * 256) reversed

end NeuralReversi.GameAnalysis
