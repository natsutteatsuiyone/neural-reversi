import NeuralReversi.Game.Transitions.Model
import NeuralReversi.Game.History.ReferenceProperties
import NeuralReversi.Game.History.Properties
import NeuralReversi.Board.MovesProperties
import NeuralReversi.Board.Move.Properties
import NeuralReversi.Board.InitializationProperties
import NeuralReversi.Board.SetPlacement

namespace NeuralReversi.Game.Transitions
open NeuralReversi.Game.History
open NeuralReversi.Board (ForcedPass)
open NeuralReversi.Bitboard.Moves (LegalMove)

theorem opposite_involutive (c : Disc) : c.opposite.opposite = c := by
  cases c <;> rfl

theorem isLegalMove_iff (movesConfig : MovesConfig) (b : Board) (sq : Square) :
    isLegalMove movesConfig b sq = true ↔ LegalMove b.player b.opponent sq.val :=
  NeuralReversi.Board.getMoves_exact movesConfig b sq.val

theorem isLegalMove_matches_board (movesConfig : MovesConfig)
    (debugAssertions overflowChecks : Bool) (b : Board) (sq : Square) :
    NeuralReversi.Board.isLegalMove movesConfig debugAssertions overflowChecks b sq =
      .ok (isLegalMove movesConfig b sq) :=
  (NeuralReversi.Board.x86IsLegalMoveContract_proven movesConfig debugAssertions overflowChecks b sq).1

theorem nextBoardAfterPlay_matches_board (flipConfig : FlipConfig) (overflowChecks : Bool)
    (b : Board) (sq : Square) :
    NeuralReversi.Board.makeMove flipConfig false overflowChecks b sq =
      .ok (nextBoardAfterPlay flipConfig b sq) := by
  rw [NeuralReversi.Board.makeMove_resolved]
  simp only [Bool.false_and, Bool.false_eq_true, ↓reduceIte]
  exact NeuralReversi.Board.placeAny_valid_eq false overflowChecks b _ sq

theorem applyAction_explicitPass_eq (flipConfig : FlipConfig) (state : ValueState) :
    applyAction flipConfig state .explicitPass = applyPass state .explicitPass := rfl

theorem applyAction_automaticPass_eq (flipConfig : FlipConfig) (state : ValueState) :
    applyAction flipConfig state .automaticPass = applyPass state .automaticPass := rfl

theorem makeMove_illegal (movesConfig : MovesConfig) (flipConfig : FlipConfig)
    (state : ValueState) (sq : Square)
    (illegal : ¬ LegalMove state.board.player state.board.opponent sq.val) :
    makeMove movesConfig flipConfig state sq = .error (.illegalMove sq) := by
  have hfalse : isLegalMove movesConfig state.board sq = false := by
    cases h : isLegalMove movesConfig state.board sq
    · rfl
    · exact (illegal ((isLegalMove_iff movesConfig state.board sq).mp h)).elim
  simp [makeMove, hfalse]

theorem makeMove_legal_no_autopass (movesConfig : MovesConfig) (flipConfig : FlipConfig)
    (state : ValueState) (sq : Square)
    (legal : LegalMove state.board.player state.board.opponent sq.val)
    (noPass : ¬ ForcedPass movesConfig (nextBoardAfterPlay flipConfig state.board sq)) :
    makeMove movesConfig flipConfig state sq =
      .ok { board := nextBoardAfterPlay flipConfig state.board sq
            side := state.side.opposite
            history := state.history ++ [⟨.play sq, state.board, state.side⟩] } := by
  have htrue : isLegalMove movesConfig state.board sq = true :=
    (isLegalMove_iff movesConfig state.board sq).mpr legal
  unfold makeMove
  simp only [htrue, Bool.not_true, Bool.false_eq_true, ↓reduceIte]
  have hcond : (!NeuralReversi.Board.hasLegalMoves movesConfig
      (nextBoardAfterPlay flipConfig state.board sq) &&
      NeuralReversi.Board.hasLegalMoves movesConfig
        (NeuralReversi.Board.switchPlayers (nextBoardAfterPlay flipConfig state.board sq))) = false := by
    cases h1 : NeuralReversi.Board.hasLegalMoves movesConfig
        (nextBoardAfterPlay flipConfig state.board sq) with
    | true => rfl
    | false =>
      cases h2 : NeuralReversi.Board.hasLegalMoves movesConfig
          (NeuralReversi.Board.switchPlayers (nextBoardAfterPlay flipConfig state.board sq)) with
      | false => rfl
      | true => exact (noPass ⟨h1, h2⟩).elim
  simp only [applyAction, hcond, Bool.false_eq_true, ↓reduceIte]

theorem makeMove_legal_autopass (movesConfig : MovesConfig) (flipConfig : FlipConfig)
    (state : ValueState) (sq : Square)
    (legal : LegalMove state.board.player state.board.opponent sq.val)
    (forced : ForcedPass movesConfig (nextBoardAfterPlay flipConfig state.board sq)) :
    let afterBoard := nextBoardAfterPlay flipConfig state.board sq
    makeMove movesConfig flipConfig state sq =
      .ok { board := NeuralReversi.Board.switchPlayers afterBoard
            side := state.side
            history := state.history ++
              [⟨.play sq, state.board, state.side⟩,
               ⟨.automaticPass, afterBoard, state.side.opposite⟩] } := by
  have htrue : isLegalMove movesConfig state.board sq = true :=
    (isLegalMove_iff movesConfig state.board sq).mpr legal
  unfold makeMove
  simp only [htrue, Bool.not_true, Bool.false_eq_true, ↓reduceIte]
  simp only [applyAction, forced.1, forced.2, Bool.not_false, Bool.and_self, ↓reduceIte,
    opposite_involutive, List.append_assoc, List.singleton_append]

theorem makeMove_legal_gameOver_no_autopass (movesConfig : MovesConfig) (flipConfig : FlipConfig)
    (state : ValueState) (sq : Square)
    (legal : LegalMove state.board.player state.board.opponent sq.val)
    (gameOver : NeuralReversi.Board.isGameOver movesConfig
      (nextBoardAfterPlay flipConfig state.board sq) = true) :
    makeMove movesConfig flipConfig state sq =
      .ok { board := nextBoardAfterPlay flipConfig state.board sq
            side := state.side.opposite
            history := state.history ++ [⟨.play sq, state.board, state.side⟩] } := by
  apply makeMove_legal_no_autopass movesConfig flipConfig state sq legal
  intro forced
  have hnot := NeuralReversi.Board.forcedPass_not_gameOver movesConfig _ forced
  rw [hnot] at gameOver
  cases gameOver

theorem makeMove_error_iff (movesConfig : MovesConfig) (flipConfig : FlipConfig)
    (state : ValueState) (sq : Square) (e : GameStateError) :
    makeMove movesConfig flipConfig state sq = .error e ↔
      ¬ LegalMove state.board.player state.board.opponent sq.val ∧ e = .illegalMove sq := by
  by_cases hlegal : LegalMove state.board.player state.board.opponent sq.val
  · constructor
    · intro herr
      by_cases hforced : ForcedPass movesConfig (nextBoardAfterPlay flipConfig state.board sq)
      · rw [makeMove_legal_autopass movesConfig flipConfig state sq hlegal hforced] at herr
        cases herr
      · rw [makeMove_legal_no_autopass movesConfig flipConfig state sq hlegal hforced] at herr
        cases herr
    · rintro ⟨hnot, _⟩
      exact (hnot hlegal).elim
  · rw [makeMove_illegal movesConfig flipConfig state sq hlegal]
    simp [hlegal, eq_comm]

theorem makeMove_undo_roundtrip (movesConfig : MovesConfig) (flipConfig : FlipConfig)
    (state next : ValueState) (sq : Square)
    (hok : makeMove movesConfig flipConfig state sq = .ok next) :
    Game.History.Reference.undo next = (true, state) := by
  by_cases hlegal : LegalMove state.board.player state.board.opponent sq.val
  · by_cases hforced : ForcedPass movesConfig (nextBoardAfterPlay flipConfig state.board sq)
    · rw [makeMove_legal_autopass movesConfig flipConfig state sq hlegal hforced] at hok
      injection hok with hnext
      subst hnext
      have hhist : state.history ++ [⟨.play sq, state.board, state.side⟩,
          ⟨.automaticPass, nextBoardAfterPlay flipConfig state.board sq, state.side.opposite⟩] =
          (state.history ++ [⟨.play sq, state.board, state.side⟩]) ++
            [⟨.automaticPass, nextBoardAfterPlay flipConfig state.board sq, state.side.opposite⟩] := by
        simp [List.append_assoc]
      unfold Game.History.Reference.undo
      dsimp only
      rw [hhist, Game.History.Reference.popLast_append]
      dsimp only
      rw [Game.History.Reference.popLast_append]
      rfl
    · rw [makeMove_legal_no_autopass movesConfig flipConfig state sq hlegal hforced] at hok
      injection hok with hnext
      subst hnext
      unfold Game.History.Reference.undo
      dsimp only
      rw [Game.History.Reference.popLast_append]
      rfl
  · rw [makeMove_illegal movesConfig flipConfig state sq hlegal] at hok
    cases hok

theorem makePass_gameOver (movesConfig : MovesConfig) (state : ValueState)
    (gameOver : NeuralReversi.Board.isGameOver movesConfig state.board = true) :
    makePass movesConfig state = .error .passAfterGameOver := by
  simp [makePass, gameOver]

theorem makePass_withLegalMoves (movesConfig : MovesConfig) (state : ValueState)
    (hasMoves : NeuralReversi.Board.hasLegalMoves movesConfig state.board = true) :
    makePass movesConfig state = .error .passWithLegalMoves := by
  have hnotOver : NeuralReversi.Board.isGameOver movesConfig state.board = false := by
    simp [NeuralReversi.Board.isGameOver, hasMoves]
  simp [makePass, hnotOver, hasMoves]

theorem makePass_forced (movesConfig : MovesConfig) (state : ValueState)
    (forced : ForcedPass movesConfig state.board) :
    makePass movesConfig state =
      .ok { board := NeuralReversi.Board.switchPlayers state.board
            side := state.side.opposite
            history := state.history ++ [⟨.explicitPass, state.board, state.side⟩] } := by
  have hnotOver := NeuralReversi.Board.forcedPass_not_gameOver movesConfig state.board forced
  simp [makePass, applyPass, hnotOver, forced.1]

theorem makePass_ok_iff (movesConfig : MovesConfig) (state next : ValueState) :
    makePass movesConfig state = .ok next ↔
      ForcedPass movesConfig state.board ∧ next = applyPass state .explicitPass := by
  cases hcur : NeuralReversi.Board.hasLegalMoves movesConfig state.board with
  | true =>
    have hnotForced : ¬ ForcedPass movesConfig state.board := by
      intro hf; rw [hf.1] at hcur; cases hcur
    rw [makePass_withLegalMoves movesConfig state hcur]
    simp [hnotForced]
  | false =>
    cases hopp : NeuralReversi.Board.hasLegalMoves movesConfig
        (NeuralReversi.Board.switchPlayers state.board) with
    | false =>
      have hover : NeuralReversi.Board.isGameOver movesConfig state.board = true := by
        simp [NeuralReversi.Board.isGameOver, hcur, hopp]
      have hnotForced : ¬ ForcedPass movesConfig state.board := by
        intro hf; rw [hf.2] at hopp; cases hopp
      rw [makePass_gameOver movesConfig state hover]
      simp [hnotForced]
    | true =>
      have hforced : ForcedPass movesConfig state.board := ⟨hcur, hopp⟩
      rw [makePass_forced movesConfig state hforced]
      simp [applyPass, hforced, eq_comm]

theorem makePass_undo_roundtrip (movesConfig : MovesConfig) (state next : ValueState)
    (hok : makePass movesConfig state = .ok next) :
    Game.History.Reference.undo next = (true, state) := by
  obtain ⟨_, rfl⟩ := (makePass_ok_iff movesConfig state next).mp hok
  unfold Game.History.Reference.undo applyPass
  dsimp only
  rw [Game.History.Reference.popLast_append]
  rfl

def ConfirmedPassRoundtripContract : Prop :=
  ∀ mc fc state sq (stored : State),
    makeMove mc fc state sq = .ok (VecView.projectState stored) →
    let afterPlay := applyAction fc state (.play sq)
    ForcedPass mc afterPlay.board →
    let confirmed := confirmAutomaticPass stored afterPlay.side
    let undone := Game.History.undo confirmed.2
    confirmed.1 = true ∧ undone.1 = true ∧ VecView.projectState undone.2 = afterPlay ∧
      makePass mc (VecView.projectState undone.2) = .ok (VecView.projectState confirmed.2)

theorem confirmedPass_roundtrip_contract : ConfirmedPassRoundtripContract := by
  intro mc fc state sq stored played afterPlay forced confirmedPass undonePass
  have auto : VecView.projectState stored = applyPass afterPlay .automaticPass := by
    unfold makeMove at played
    split at played
    · cases played
    · change (if !Board.hasLegalMoves mc afterPlay.board &&
          Board.hasLegalMoves mc (Board.switchPlayers afterPlay.board)
        then .ok (applyPass afterPlay .automaticPass) else .ok afterPlay) =
          Except.ok (VecView.projectState stored) at played
      simpa only [forced.1, forced.2, Bool.not_false, Bool.and_self, ↓reduceIte,
        Except.ok.injEq] using played.symm
  have confirmation : (confirmedPass.1, VecView.projectState confirmedPass.2) =
      Reference.confirm (VecView.projectState stored) afterPlay.side :=
    VecView.confirmState_value_refinement entryBytes stored afterPlay.side
  rw [auto] at confirmation
  have tag : Reference.confirm (applyPass afterPlay .automaticPass) afterPlay.side =
      (true, applyPass afterPlay .explicitPass) := by
    simp [Reference.confirm, applyPass, Reference.popLast_append]
  rw [tag] at confirmation
  obtain ⟨confirmed, confirmedState⟩ := Prod.mk.inj confirmation
  have pass := (makePass_ok_iff mc afterPlay (applyPass afterPlay .explicitPass)).mpr ⟨forced, rfl⟩
  have undo : (undonePass.1, VecView.projectState undonePass.2) =
      Reference.undo (VecView.projectState confirmedPass.2) :=
    VecView.undo_value_refinement entryBytes confirmedPass.2
  rw [confirmedState, makePass_undo_roundtrip mc _ _ pass] at undo
  obtain ⟨undone, restored⟩ := Prod.mk.inj undo
  refine ⟨confirmed, undone, restored, ?_⟩
  simpa only [restored, confirmedState] using pass

theorem fromMoves_nil (movesConfig : MovesConfig) (flipConfig : FlipConfig) :
    fromMoves movesConfig flipConfig [] = .ok newValueState := rfl

theorem replayMoves_append_ok (movesConfig : MovesConfig) (flipConfig : FlipConfig)
    (startIndex : Nat) (state mid : ValueState) (xs ys : List Square)
    (hxs : replayMoves movesConfig flipConfig startIndex state xs = .ok mid) :
    replayMoves movesConfig flipConfig startIndex state (xs ++ ys) =
      replayMoves movesConfig flipConfig (startIndex + xs.length) mid ys := by
  induction xs generalizing startIndex state with
  | nil =>
    simp only [replayMoves] at hxs
    injection hxs with hEq
    subst hEq
    simp
  | cons sq rest ih =>
    simp only [replayMoves] at hxs
    cases hm : makeMove movesConfig flipConfig state sq with
    | error e =>
      rw [hm] at hxs
      cases hxs
    | ok next =>
      rw [hm] at hxs
      simp only [List.cons_append, replayMoves, hm, List.length_cons]
      have hstep := ih (startIndex + 1) next hxs
      have harith : startIndex + 1 + rest.length = startIndex + (rest.length + 1) := by omega
      rw [harith] at hstep
      exact hstep

theorem replayMoves_append_error (movesConfig : MovesConfig) (flipConfig : FlipConfig)
    (startIndex : Nat) (state : ValueState) (xs ys : List Square) (e : GameStateError)
    (herr : replayMoves movesConfig flipConfig startIndex state xs = .error e) :
    replayMoves movesConfig flipConfig startIndex state (xs ++ ys) = .error e := by
  induction xs generalizing startIndex state with
  | nil =>
    simp only [replayMoves] at herr
    cases herr
  | cons sq rest ih =>
    simp only [replayMoves] at herr
    cases hm : makeMove movesConfig flipConfig state sq with
    | error errMove =>
      rw [hm] at herr
      simp only [List.cons_append, replayMoves, hm]
      exact herr
    | ok next =>
      rw [hm] at herr
      simp only [List.cons_append, replayMoves, hm]
      exact ih (startIndex + 1) next herr

theorem fromMoves_first_error (movesConfig : MovesConfig) (flipConfig : FlipConfig)
    (pre suf : List Square) (sq : Square) (mid : ValueState) (e : GameStateError)
    (hpre : replayMoves movesConfig flipConfig 0 newValueState pre = .ok mid)
    (hfail : makeMove movesConfig flipConfig mid sq = .error e) :
    fromMoves movesConfig flipConfig (pre ++ sq :: suf) =
      .error (.illegalMoveAt (pre.length + 1) sq) := by
  unfold fromMoves
  rw [replayMoves_append_ok movesConfig flipConfig 0 newValueState mid pre (sq :: suf) hpre]
  simp [replayMoves, hfail]

def playedSquares (history : List Entry) : List Square :=
  history.filterMap fun entry =>
    match entry.action with
    | .play sq => some sq
    | .explicitPass | .automaticPass => none

theorem makeMove_playedSquares (movesConfig : MovesConfig) (flipConfig : FlipConfig)
    (state next : ValueState) (sq : Square)
    (hok : makeMove movesConfig flipConfig state sq = .ok next) :
    playedSquares next.history = playedSquares state.history ++ [sq] := by
  by_cases hlegal : LegalMove state.board.player state.board.opponent sq.val
  · by_cases hforced : ForcedPass movesConfig (nextBoardAfterPlay flipConfig state.board sq)
    · rw [makeMove_legal_autopass movesConfig flipConfig state sq hlegal hforced] at hok
      injection hok with hnext
      subst hnext
      simp [playedSquares]
    · rw [makeMove_legal_no_autopass movesConfig flipConfig state sq hlegal hforced] at hok
      injection hok with hnext
      subst hnext
      simp [playedSquares]
  · rw [makeMove_illegal movesConfig flipConfig state sq hlegal] at hok
    cases hok

theorem replayMoves_playedSquares (movesConfig : MovesConfig) (flipConfig : FlipConfig)
    (startIndex : Nat) (state final : ValueState) (moves : List Square)
    (hok : replayMoves movesConfig flipConfig startIndex state moves = .ok final) :
    playedSquares final.history = playedSquares state.history ++ moves := by
  induction moves generalizing startIndex state with
  | nil =>
    simp only [replayMoves] at hok
    injection hok with hEq
    subst hEq
    simp
  | cons sq rest ih =>
    simp only [replayMoves] at hok
    cases hm : makeMove movesConfig flipConfig state sq with
    | error e =>
      rw [hm] at hok
      cases hok
    | ok next =>
      rw [hm] at hok
      have hrest := ih (startIndex + 1) next hok
      rw [hrest, makeMove_playedSquares movesConfig flipConfig state next sq hm]
      simp [List.append_assoc]

theorem fromMoves_playedSquares (movesConfig : MovesConfig) (flipConfig : FlipConfig)
    (moves : List Square) (final : ValueState)
    (hok : fromMoves movesConfig flipConfig moves = .ok final) :
    playedSquares final.history = moves := by
  have h := replayMoves_playedSquares movesConfig flipConfig 0 newValueState final moves hok
  simpa [newValueState, playedSquares] using h

private theorem makeMove_emptyCount (mc : MovesConfig) (fc : FlipConfig)
    (state next : ValueState) (sq : Square) (wf : Board.wellFormed state.board)
    (played : makeMove mc fc state sq = .ok next) :
    Board.wellFormed next.board ∧
      (Board.emptyCount next.board).toNat + 1 = (Board.emptyCount state.board).toNat := by
  by_cases legal : LegalMove state.board.player state.board.opponent sq.val
  · let b := state.board
    have subset : Flip.Dispatch.resultValue fc sq b.player b.opponent &&& ~~~b.opponent = 0#64 := by
      rw [Flip.Dispatch.value_disjoint_scalar fc sq b.player b.opponent wf]
      apply BitVec.eq_of_getLsbD_eq
      intro i hi
      cases bit : (Flip.value sq b.player b.opponent).getLsbD i with
      | false => simp only [BitVec.getLsbD_and, bit, Bool.false_and, BitVec.getLsbD_zero]
      | true =>
        have opponent := Flip.flipped_is_opponent _ _ _ _ bit
        simp only [BitVec.getLsbD_and, BitVec.getLsbD_not, opponent, Bool.not_true,
          Bool.and_false, BitVec.getLsbD_zero]
    have empty : Bitboard.squareMask sq &&& (b.player ||| b.opponent) = 0#64 := by
      rw [BitVec.and_comm, Bitboard.and_squareMask_zero, BitVec.getLsbD_or]
      exact Bool.or_eq_false_iff.mpr ⟨legal.2.1, legal.2.2.1⟩
    have placed := Board.place_wellFormed b (Flip.Dispatch.resultValue fc sq b.player b.opponent) sq wf
    have count := Board.place_emptyCount_step b _ sq wf subset empty
    by_cases forced : ForcedPass mc (nextBoardAfterPlay fc b sq)
    · rw [makeMove_legal_autopass mc fc state sq legal forced] at played
      cases played
      exact ⟨Board.switchPlayers_wellFormed _ placed, by rwa [Board.switchPlayers_emptyCount]⟩
    · rw [makeMove_legal_no_autopass mc fc state sq legal forced] at played
      cases played
      exact ⟨placed, count⟩
  · rw [makeMove_illegal mc fc state sq legal] at played
    cases played

private theorem replayMoves_emptyCount (mc : MovesConfig) (fc : FlipConfig) (moves : List Square)
    (startIndex : Nat) (state final : ValueState) (wf : Board.wellFormed state.board)
    (replayed : replayMoves mc fc startIndex state moves = .ok final) :
    (Board.emptyCount final.board).toNat + moves.length = (Board.emptyCount state.board).toNat := by
  induction moves generalizing startIndex state with
  | nil =>
    cases replayed
    simp
  | cons sq rest ih =>
    simp only [replayMoves] at replayed
    cases played : makeMove mc fc state sq with
    | error e => rw [played] at replayed; cases replayed
    | ok next =>
      rw [played] at replayed
      obtain ⟨nextWf, step⟩ := makeMove_emptyCount mc fc state next sq wf played
      have remaining := ih (startIndex + 1) next nextWf replayed
      rw [List.length_cons]
      omega

def FromMovesEmptyCountContract : Prop :=
  ∀ mc fc moves final, fromMoves mc fc moves = .ok final →
    (Board.emptyCount final.board).toNat + moves.length = 60

theorem fromMoves_emptyCount_contract : FromMovesEmptyCountContract := by
  intro mc fc moves final replayed
  have initial := Board.initialBoardContract_proven Board.newBoard (by simp)
  exact (replayMoves_emptyCount mc fc moves 0 newValueState final initial.1 replayed).trans
    initial.2.2.2.1

def undoMoves : Nat → State → Option State
  | 0, state => some state
  | n + 1, state => do
    let prior ← undoMoves n state
    let (ok, next) := Game.History.undo prior
    if ok then some next else none

private theorem replayMoves_undo (mc : MovesConfig) (fc : FlipConfig) (moves : List Square)
    (index : Nat) (start : ValueState) (stored : State)
    (played : replayMoves mc fc index start moves = .ok (VecView.projectState stored)) :
    ∃ restored, undoMoves moves.length stored = some restored ∧ VecView.projectState restored = start := by
  induction moves generalizing index start with
  | nil =>
    exact ⟨stored, rfl, (Except.ok.inj played).symm⟩
  | cons sq rest ih =>
    simp only [replayMoves] at played
    cases move : makeMove mc fc start sq with
    | error e => rw [move] at played; cases played
    | ok next =>
      rw [move] at played
      obtain ⟨prior, undone, projects⟩ := ih (index + 1) next played
      have step := VecView.undo_value_refinement entryBytes prior
      rw [projects, makeMove_undo_roundtrip mc fc start next sq move] at step
      obtain ⟨success, original⟩ := Prod.mk.inj step
      refine ⟨(Game.History.undo prior).2, ?_, original⟩
      simp [undoMoves, undone, Game.History.undo, success]

def FromMovesUndoPrefixContract : Prop :=
  ∀ mc fc pre suffix (stored : State),
    fromMoves mc fc (pre ++ suffix) = .ok (VecView.projectState stored) →
    ∃ restored, undoMoves suffix.length stored = some restored ∧
      fromMoves mc fc pre = .ok (VecView.projectState restored)

theorem fromMoves_undoPrefix_contract : FromMovesUndoPrefixContract := by
  intro mc fc pre suffix stored played
  unfold fromMoves at *
  cases prefixResult : replayMoves mc fc 0 newValueState pre with
  | error e =>
    rw [replayMoves_append_error mc fc 0 newValueState pre suffix e prefixResult] at played
    cases played
  | ok mid =>
    rw [replayMoves_append_ok mc fc 0 newValueState mid pre suffix prefixResult] at played
    obtain ⟨restored, undone, projects⟩ := replayMoves_undo mc fc suffix _ mid stored played
    exact ⟨restored, undone, congrArg Except.ok projects.symm⟩

private theorem makeMove_backend_eq (mc₁ mc₂ : MovesConfig) (fc₁ fc₂ : FlipConfig)
    (state : ValueState) (sq : Square) (wf : Board.wellFormed state.board) :
    makeMove mc₁ fc₁ state sq = makeMove mc₂ fc₂ state sq := by
  have masks (b : Board) : Board.getMoves mc₁ b = Board.getMoves mc₂ b := by
    apply BitVec.eq_of_getLsbD_eq
    intro i _
    exact Bool.eq_iff_iff.mpr ((Board.getMoves_exact mc₁ b i).trans (Board.getMoves_exact mc₂ b i).symm)
  have flips := (Flip.Dispatch.value_disjoint_scalar fc₁ sq _ _ wf).trans
    (Flip.Dispatch.value_disjoint_scalar fc₂ sq _ _ wf).symm
  simp only [makeMove, isLegalMove, Board.hasLegalMoves, applyAction, nextBoardAfterPlay, masks, flips]
  rfl

def FromMovesBackendContract : Prop :=
  ∀ mc₁ mc₂ fc₁ fc₂ moves, fromMoves mc₁ fc₁ moves = fromMoves mc₂ fc₂ moves

theorem fromMoves_backend_contract : FromMovesBackendContract := by
  intro mc₁ mc₂ fc₁ fc₂ moves
  suffices replay : ∀ index (state : ValueState), Board.wellFormed state.board →
      replayMoves mc₁ fc₁ index state moves = replayMoves mc₂ fc₂ index state moves by
    exact replay 0 newValueState (Board.initialBoardContract_proven Board.newBoard (by simp)).1
  induction moves with
  | nil => intros; rfl
  | cons sq rest ih =>
    intro index state wf
    simp only [replayMoves, makeMove_backend_eq mc₁ mc₂ fc₁ fc₂ state sq wf]
    cases played : makeMove mc₂ fc₂ state sq with
    | error e => rfl
    | ok next => exact ih (index + 1) next (makeMove_emptyCount mc₂ fc₂ state next sq wf played).1

private theorem makeMove_emptySquares (mc : MovesConfig) (fc : FlipConfig)
    (state next : ValueState) (sq : Square) (played : makeMove mc fc state sq = .ok next) :
    (Board.emptySquares state.board).getLsbD sq.val = true ∧
    ∀ other : Square, (Board.emptySquares next.board).getLsbD other.val = true →
      other ≠ sq ∧ (Board.emptySquares state.board).getLsbD other.val = true := by
  have legal : LegalMove state.board.player state.board.opponent sq.val := by
    apply Classical.byContradiction
    intro illegal
    rw [makeMove_illegal mc fc state sq illegal] at played
    cases played
  constructor
  · simp only [Board.emptySquares, BitVec.getLsbD_not, sq.isLt, decide_true, Bool.true_and,
      BitVec.getLsbD_or, legal.2.1, legal.2.2.1, Bool.or_self, Bool.not_false]
  · have after : Board.emptySquares next.board =
        Board.emptySquares (nextBoardAfterPlay fc state.board sq) := by
      by_cases forced : ForcedPass mc (nextBoardAfterPlay fc state.board sq)
      · rw [makeMove_legal_autopass mc fc state sq legal forced] at played
        cases played
        exact (Board.perspectiveExchange_proven _).2.2.2
      · rw [makeMove_legal_no_autopass mc fc state sq legal forced] at played
        cases played
        rfl
    intro other empty
    rw [after] at empty
    let flips := Flip.Dispatch.resultValue fc sq state.board.player state.board.opponent
    let placed := Board.SetPlacement.placeMask state.board flips (Bitboard.squareMask sq)
    change (~~~(placed.player ||| placed.opponent)).getLsbD other.val = true at empty
    rw [Board.SetPlacement.occupancy_mask_exact] at empty
    simp only [Board.emptySquares, BitVec.getLsbD_not, other.isLt, decide_true,
      Bool.true_and, Bool.not_eq_true', BitVec.getLsbD_or, Bool.or_eq_false_iff,
      Bitboard.squareMask_bit, decide_eq_false_iff_not] at empty ⊢
    exact ⟨fun same => empty.1.2 (congrArg Fin.val same), empty.1.1⟩

def FromMovesDistinctSquaresContract : Prop :=
  ∀ mc fc moves final, fromMoves mc fc moves = .ok final →
    moves.Nodup ∧ ∀ sq ∈ moves, (Board.emptySquares Board.newBoard).getLsbD sq.val = true

theorem fromMoves_distinctSquares_contract : FromMovesDistinctSquaresContract := by
  intro mc fc moves final
  suffices replay : ∀ index state, replayMoves mc fc index state moves = .ok final →
      moves.Nodup ∧ ∀ sq ∈ moves, (Board.emptySquares state.board).getLsbD sq.val = true by
    exact replay 0 newValueState
  induction moves with
  | nil => simp
  | cons sq rest ih =>
    intro index state replayed
    simp only [replayMoves] at replayed
    cases move : makeMove mc fc state sq with
    | error e => rw [move] at replayed; cases replayed
    | ok next =>
      rw [move] at replayed
      obtain ⟨distinct, remaining⟩ := ih (index + 1) next replayed
      obtain ⟨empty, earlier⟩ := makeMove_emptySquares mc fc state next sq move
      refine ⟨List.nodup_cons.mpr ⟨fun mem => (earlier sq (remaining sq mem)).1 rfl, distinct⟩, ?_⟩
      intro other mem
      rcases List.mem_cons.mp mem with rfl | later
      · exact empty
      · exact (earlier other (remaining other later)).2

private theorem makeMove_no_forcedPass (mc : MovesConfig) (fc : FlipConfig)
    (state next : ValueState) (sq : Square) (played : makeMove mc fc state sq = .ok next) :
    ¬ ForcedPass mc next.board := by
  by_cases legal : LegalMove state.board.player state.board.opponent sq.val
  · by_cases forced : ForcedPass mc (nextBoardAfterPlay fc state.board sq)
    · rw [makeMove_legal_autopass mc fc state sq legal forced] at played
      cases played
      exact Board.forcedPass_not_repeat mc _ forced
    · rw [makeMove_legal_no_autopass mc fc state sq legal forced] at played
      cases played
      exact forced
  · rw [makeMove_illegal mc fc state sq legal] at played
    cases played

private theorem replayMoves_no_forcedPass (mc : MovesConfig) (fc : FlipConfig)
    (index : Nat) (state final : ValueState) (moves : List Square)
    (ready : ¬ ForcedPass mc state.board)
    (replayed : replayMoves mc fc index state moves = .ok final) :
    ¬ ForcedPass mc final.board := by
  induction moves generalizing index state with
  | nil => cases replayed; exact ready
  | cons sq rest ih =>
    simp only [replayMoves] at replayed
    cases played : makeMove mc fc state sq with
    | error e => rw [played] at replayed; cases replayed
    | ok next =>
      rw [played] at replayed
      exact ih (index + 1) next (makeMove_no_forcedPass mc fc state next sq played) replayed

def FromMovesContinuationContract : Prop :=
  ∀ mc fc moves state, fromMoves mc fc moves = .ok state →
    (Board.isGameOver mc state.board = false ↔
      ∃ sq next, fromMoves mc fc (moves ++ [sq]) = .ok next)

theorem fromMoves_continuation_contract : FromMovesContinuationContract := by
  intro mc fc moves state replayed
  have openingMove := (Board.getMoves_exact ⟨false, false⟩ Board.newBoard 19).mp (by decide)
  have opening := (Board.hasLegalMoves_exact mc Board.newBoard).mpr ⟨19, openingMove⟩
  have ready := replayMoves_no_forcedPass mc fc 0 newValueState state moves
    (fun forced => Bool.noConfusion (opening.symm.trans forced.1)) replayed
  have append (sq : Square) := replayMoves_append_ok mc fc 0 newValueState state moves [sq] replayed
  simp only [fromMoves, append, replayMoves]
  constructor
  · intro ongoing
    have available : Board.hasLegalMoves mc state.board = true := by
      cases h : Board.hasLegalMoves mc state.board with
      | true => rfl
      | false =>
        simp [Board.isGameOver, h] at ongoing
        exact (ready ⟨h, ongoing⟩).elim
    obtain ⟨i, legal⟩ := (Board.hasLegalMoves_exact mc state.board).mp available
    let sq : Square := ⟨i, legal.1⟩
    cases played : makeMove mc fc state sq with
    | ok next => exact ⟨sq, next, by simp [played]⟩
    | error e => exact (((makeMove_error_iff mc fc state sq e).mp played).1 legal).elim
  · rintro ⟨sq, next, extended⟩
    by_cases legal : LegalMove state.board.player state.board.opponent sq.val
    · have available := (Board.hasLegalMoves_exact mc state.board).mpr ⟨sq.val, legal⟩
      simp [Board.isGameOver, available]
    · rw [makeMove_illegal mc fc state sq legal] at extended
      cases extended

def MakeMoveContract : Prop :=
  ∀ (movesConfig : MovesConfig) (flipConfig : FlipConfig) (state : ValueState) (sq : Square),
    (¬ LegalMove state.board.player state.board.opponent sq.val →
      makeMove movesConfig flipConfig state sq = .error (.illegalMove sq)) ∧
    (LegalMove state.board.player state.board.opponent sq.val →
      ¬ ForcedPass movesConfig (nextBoardAfterPlay flipConfig state.board sq) →
      makeMove movesConfig flipConfig state sq =
        .ok { board := nextBoardAfterPlay flipConfig state.board sq
              side := state.side.opposite
              history := state.history ++ [⟨.play sq, state.board, state.side⟩] }) ∧
    (LegalMove state.board.player state.board.opponent sq.val →
      ForcedPass movesConfig (nextBoardAfterPlay flipConfig state.board sq) →
      let afterBoard := nextBoardAfterPlay flipConfig state.board sq
      makeMove movesConfig flipConfig state sq =
        .ok { board := NeuralReversi.Board.switchPlayers afterBoard
              side := state.side
              history := state.history ++
                [⟨.play sq, state.board, state.side⟩,
                 ⟨.automaticPass, afterBoard, state.side.opposite⟩] })

theorem makeMove_contract : MakeMoveContract := by
  intro movesConfig flipConfig state sq
  exact ⟨makeMove_illegal movesConfig flipConfig state sq,
    makeMove_legal_no_autopass movesConfig flipConfig state sq,
    makeMove_legal_autopass movesConfig flipConfig state sq⟩

def MakePassContract : Prop :=
  ∀ (movesConfig : MovesConfig) (state : ValueState),
    (NeuralReversi.Board.isGameOver movesConfig state.board = true →
      makePass movesConfig state = .error .passAfterGameOver) ∧
    (NeuralReversi.Board.hasLegalMoves movesConfig state.board = true →
      makePass movesConfig state = .error .passWithLegalMoves) ∧
    (ForcedPass movesConfig state.board →
      makePass movesConfig state =
        .ok { board := NeuralReversi.Board.switchPlayers state.board
              side := state.side.opposite
              history := state.history ++ [⟨.explicitPass, state.board, state.side⟩] })

theorem makePass_contract : MakePassContract := by
  intro movesConfig state
  exact ⟨makePass_gameOver movesConfig state,
    makePass_withLegalMoves movesConfig state,
    makePass_forced movesConfig state⟩

def FromMovesContract : Prop :=
  ∀ (movesConfig : MovesConfig) (flipConfig : FlipConfig),
    fromMoves movesConfig flipConfig [] = .ok newValueState ∧
    (∀ moves final, fromMoves movesConfig flipConfig moves = .ok final →
      playedSquares final.history = moves) ∧
    (∀ pre suf sq mid e,
      replayMoves movesConfig flipConfig 0 newValueState pre = .ok mid →
      makeMove movesConfig flipConfig mid sq = .error e →
      fromMoves movesConfig flipConfig (pre ++ sq :: suf) =
        .error (.illegalMoveAt (pre.length + 1) sq))

theorem fromMoves_contract : FromMovesContract := by
  intro movesConfig flipConfig
  exact ⟨fromMoves_nil movesConfig flipConfig,
    fun moves final hok => fromMoves_playedSquares movesConfig flipConfig moves final hok,
    fun pre suf sq mid e hpre hfail =>
      fromMoves_first_error movesConfig flipConfig pre suf sq mid e hpre hfail⟩

private def portableMoves : MovesConfig := ⟨false, false⟩
private def portableFlip : FlipConfig := ⟨false, false, false⟩
private def sqD3 : Square := ⟨19, by decide⟩

example : fromMoves portableMoves portableFlip [] = .ok newValueState := rfl

example : makePass portableMoves newValueState = .error .passWithLegalMoves := rfl

example : makePass portableMoves { board := ⟨0, 0⟩, side := .black, history := [] } =
    .error .passAfterGameOver := rfl

example : makePass portableMoves { board := ⟨2, 1⟩, side := .white, history := [] } =
    .ok { board := ⟨1, 2⟩, side := .black, history := [⟨.explicitPass, ⟨2, 1⟩, .white⟩] } := rfl

example : fromMoves portableMoves portableFlip [sqD3, sqD3] =
    .error (.illegalMoveAt 2 sqD3) := rfl

end NeuralReversi.Game.Transitions
