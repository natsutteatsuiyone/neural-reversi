import NeuralReversi.Solve1.FlipCount
import NeuralReversi.Solve1.ScalarVerified
import NeuralReversi.Solve1.NeonVerified
import NeuralReversi.EndgameTreeCache

namespace NeuralReversi.Solve1

open Bitboard (count squareMask)
open Bitboard.Moves (X86Config)
open PvsBoardTree (Position GeneratedSquare legalSquares play pass)
open EndgameTreeCache (discValue)

private abbrev FlipConfig := Flip.Dispatch.X86Config

def Bmi2.Board.asBoard (b : Bmi2.Board) : Board.Board :=
  ⟨b.player, ~~~(b.player ||| squareMask b.square)⟩

private theorem wf (b : Bmi2.Board) : Board.wellFormed b.asBoard := by
  change b.player &&& ~~~(b.player ||| squareMask b.square) = 0#64
  rw [BitVec.not_or, ← BitVec.and_assoc]
  simp

private theorem one_empty (b : Bmi2.Board) : Board.emptySquares b.asBoard = squareMask b.square := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  by_cases target : i = b.square.val
  · subst i
    simp [Board.emptySquares, Bmi2.Board.asBoard, Bitboard.squareMask_bit, b.empty,
      b.square.isLt, -BitVec.getLsbD_eq_getElem]
  · simp [Board.emptySquares, Bmi2.Board.asBoard, Bitboard.squareMask_bit, target, hi,
      -BitVec.getLsbD_eq_getElem]

private theorem empty_eq {b : Board.Board} {sq i : Bitboard.Square}
    (one : Board.emptySquares b = squareMask sq) :
    (b.player.getLsbD i.val = false ∧ b.opponent.getLsbD i.val = false) ↔ i = sq := by
  have bit := congrArg (fun (m : Bitboard.Bits) => m.getLsbD i.val) one
  simp only [Board.emptySquares, BitVec.getLsbD_not, i.isLt, decide_true, Bool.true_and,
    BitVec.getLsbD_or, Bitboard.squareMask_bit] at bit
  cases hp : b.player.getLsbD i.val <;> cases ho : b.opponent.getLsbD i.val <;>
    simp_all [Fin.ext_iff, -BitVec.getLsbD_eq_getElem]

private def flips (b : Board.Board) (sq : Bitboard.Square) : Bitboard.Bits :=
  Flip.value sq b.player b.opponent

private theorem available_iff (mc : X86Config) (b : Position) (sq : Bitboard.Square)
    (one : Board.emptySquares b.val = squareMask sq) :
    Board.hasLegalMoves mc b.val = true ↔ flips b.val sq ≠ 0 := by
  rw [Board.hasLegalMoves_exact]
  constructor
  · rintro ⟨i, legal⟩
    let candidate : Bitboard.Square := ⟨i, legal.1⟩
    have eq := (empty_eq (i := candidate) one).mp ⟨legal.2.1, legal.2.2.1⟩
    have nonzero := ((Board.legal_iff_empty_flip b.val candidate b.property).mp legal).2
    exact eq ▸ nonzero
  · intro nonzero
    exact ⟨sq.val, (Board.legal_iff_empty_flip b.val sq b.property).mpr
      ⟨(empty_eq one).mpr rfl, nonzero⟩⟩

private theorem legal_empty (mc : X86Config) (b : Board.Board) (zero : Board.emptySquares b = 0) :
    Board.hasLegalMoves mc b = false := by
  cases available : Board.hasLegalMoves mc b
  · rfl
  · obtain ⟨i, within, p, o, _⟩ := (Board.hasLegalMoves_exact mc b).mp available
    have bit := congrArg (fun (m : Bitboard.Bits) => m.getLsbD i) zero
    simp [Board.emptySquares, within, p, o, -BitVec.getLsbD_eq_getElem] at bit

private theorem terminal_disc (mc : X86Config) (fc : FlipConfig) (b : Position)
    (stuck : Board.hasLegalMoves mc b.val = false) (other : Board.hasLegalMoves mc (pass b).val = false) :
    discValue mc fc b.val = Board.AwardedScore b.val := by
  have exactValue := EndgameTreeCache.scaled_discValue mc fc b
  rw [PvsBoardTree.terminal_value mc fc b (PvsBoardTree.nil_of_stuck stuck) other] at exactValue
  omega

private theorem full_disc (mc : X86Config) (fc : FlipConfig) (b : Position)
    (zero : (Board.emptyCount b.val).toNat = 0) :
    discValue mc fc b.val = 2 * (Board.playerCount b.val).toNat - 64 := by
  have full : Board.emptySquares b.val = 0 := by
    apply (Bitboard.populationCountContract_proven (Board.emptySquares b.val)).2.2.1.mp
    apply BitVec.eq_of_toNat_eq
    exact zero
  have fullOther : Board.emptySquares (pass b).val = 0 := by
    change ~~~(b.val.opponent ||| b.val.player) = 0#64
    rw [BitVec.or_comm]
    exact full
  rw [terminal_disc mc fc b (legal_empty mc b.val full) (legal_empty mc (pass b).val fullOther)]
  have total := Board.countConservation_proven b.val b.property
  unfold Board.AwardedScore
  rw [zero]
  split <;> (try split) <;> omega

private theorem unflipped_count (b : Board.Board) (sq : Bitboard.Square) :
    (count (b.opponent &&& ~~~flips b sq)).toNat + (count (flips b sq)).toNat =
      (Board.opponentCount b).toNat := by
  let f := flips b sq
  have separate : (b.opponent &&& ~~~f) &&& f = 0 := by
    rw [BitVec.and_assoc]
    simp
  have cover : (b.opponent &&& ~~~f) ||| f = b.opponent := by
    apply BitVec.eq_of_getLsbD_eq
    intro i within
    cases hit : f.getLsbD i
    · simp [hit, within, -BitVec.getLsbD_eq_getElem]
    · have enemy := Flip.flipped_is_opponent sq b.player b.opponent i hit
      simp [hit, enemy, -BitVec.getLsbD_eq_getElem]
  change (count (b.opponent &&& ~~~f)).toNat + (count f).toNat = (count b.opponent).toNat
  simpa only [cover] using (Bitboard.count_or_of_disjoint _ _ separate).symm

private theorem last_child (mc : X86Config) (fc : FlipConfig) (b : Position) (sq : Bitboard.Square)
    (one : Board.emptySquares b.val = squareMask sq) (move : GeneratedSquare mc b.val) :
    -discValue mc fc (play mc fc b move).val.val =
      64 - 2 * ((Board.opponentCount b.val).toNat - (count (flips b.val sq)).toNat : Int) := by
  let next := (play mc fc b move).val
  have before : (Board.emptyCount b.val).toNat = 1 := by
    change (count (Board.emptySquares b.val)).toNat = 1
    rw [one, Board.squareMask_count_one]
  have legal := (Board.getMoves_exact mc b.val move.val.val).mp move.property
  have unique := (empty_eq one).mp ⟨legal.2.1, legal.2.2.1⟩
  have accepted :=
    (Board.precomputedMoveContract_proven mc fc true true b.val move.val b.property move.property).1
  have after := (Board.checkedPlacementProgress_proven _ _ _ _ _ b.property accepted).2
  change (Board.emptyCount next.val).toNat + 1 = (Board.emptyCount b.val).toNat at after
  have left : next.val.player = b.val.opponent &&& ~~~flips b.val sq := by
    change b.val.opponent &&& ~~~(Flip.Dispatch.resultValue fc move.val b.val.player b.val.opponent) = _
    rw [Flip.Dispatch.value_disjoint_scalar _ _ _ _ b.property, unique]
    rfl
  change -discValue mc fc next.val = _
  rw [full_disc mc fc next (by omega)]
  have remains := unflipped_count b.val sq
  change (count (b.val.opponent &&& ~~~flips b.val sq)).toNat + _ = _ at remains
  simp only [Board.playerCount, left]
  omega

private theorem last_move (mc : X86Config) (fc : FlipConfig) (b : Position) (sq : Bitboard.Square)
    (one : Board.emptySquares b.val = squareMask sq) (available : Board.hasLegalMoves mc b.val = true) :
    discValue mc fc b.val =
      64 - 2 * ((Board.opponentCount b.val).toNat - (count (flips b.val sq)).toNat : Int) := by
  cases moves : legalSquares mc b.val with
  | nil =>
    have no := PvsBoardTree.stuck_of_nil moves
    rw [no] at available
    contradiction
  | cons first rest =>
    rw [EndgameTreeCache.discValue_moves mc fc b first rest moves]
    have head := last_child mc fc b sq one first
    have range := (EndgameTreeCache.discValue_range mc fc (play mc fc b first).val.val).2
    apply Int.le_antisymm
    · exact PvsMoveLoop.bestValue_le _ _ _ _ (by omega)
        (fun move _ => Int.le_of_eq (last_child mc fc b sq one move))
    · rw [← head]
      exact PvsMoveLoop.bestValue_ge_move _ _ _ (by simp)

private theorem opponent_flipCount (b : Bmi2.Board) :
    (count (flips (Board.switchPlayers b.asBoard) b.square)).toNat =
      Bmi2.reference (~~~b.player) b.square.val := by
  rw [← reference_reconstructedOpponent, reference_eq_flipCount]
  have restore : ~~~(b.asBoard.opponent ||| squareMask b.square) = b.player := by
    apply BitVec.eq_of_getLsbD_eq
    intro i hi
    by_cases target : i = b.square.val
    · subst i
      simp [Bmi2.Board.asBoard, Bitboard.squareMask_bit, b.empty, b.square.isLt,
        -BitVec.getLsbD_eq_getElem]
    · simp [Bmi2.Board.asBoard, Bitboard.squareMask_bit, target, hi, -BitVec.getLsbD_eq_getElem]
  change (count (Flip.value b.square b.asBoard.opponent b.player)).toNat =
    (count (Flip.value b.square b.asBoard.opponent (~~~(b.asBoard.opponent ||| squareMask b.square)))).toNat
  rw [restore]

theorem exactScore_eq_tree (mc : X86Config) (fc : FlipConfig) (b : Bmi2.Board) :
    Bmi2.exactScore b = discValue mc fc b.asBoard := by
  let pos : Position := ⟨b.asBoard, wf b⟩
  have one := one_empty b
  have oneOther : Board.emptySquares (pass pos).val = squareMask b.square := by
    change ~~~(b.asBoard.opponent ||| b.asBoard.player) = _
    rw [BitVec.or_comm]
    exact one
  have emptyCount : (Board.emptyCount b.asBoard).toNat = 1 := by
    change (count (Board.emptySquares b.asBoard)).toNat = 1
    rw [one, Board.squareMask_count_one]
  have total := Board.countConservation_proven b.asBoard (wf b)
  have player : (Board.playerCount b.asBoard).toNat = Bmi2.popCount b.player :=
    Bitboard.count_toNat b.player
  have remaining : Bmi2.popCount b.player + (Board.opponentCount pos.val).toNat = 63 := by
    change Bmi2.popCount b.player + (Board.opponentCount b.asBoard).toNat = 63
    omega
  have present : (count (flips pos.val b.square)).toNat = Bmi2.reference b.player b.square.val :=
    (reference_eq_flipCount b.player b.square).symm
  have reply : (count (flips (pass pos).val b.square)).toNat = Bmi2.reference (~~~b.player) b.square.val :=
    opponent_flipCount b
  have playable (p : Position) (empty : Board.emptySquares p.val = squareMask b.square) :
      Board.hasLegalMoves mc p.val = true ↔ 0 < (count (flips p.val b.square)).toNat := by
    rw [available_iff mc p b.square empty]
    have zero := (Bitboard.populationCountContract_proven (flips p.val b.square)).2.2.1
    have countZero : (count (flips p.val b.square)).toNat = 0 ↔ flips p.val b.square = 0#64 := by
      rw [← zero]
      exact ⟨fun h => BitVec.eq_of_toNat_eq h, fun h => by rw [h]; rfl⟩
    change (¬ flips p.val b.square = 0#64) ↔ _
    rw [← countZero]
    omega
  rw [Bmi2.exactScore_terminal]
  split
  next hasMove =>
    have value := last_move mc fc pos b.square one ((playable pos one).mpr (by omega))
    change discValue mc fc b.asBoard = _ at value
    omega
  next noMove =>
    have none : Board.hasLegalMoves mc pos.val = false := by
      cases avail : Board.hasLegalMoves mc pos.val
      · rfl
      · have positive := (playable pos one).mp avail
        omega
    split
    next otherMoves =>
      have available := (playable (pass pos) oneOther).mpr (by omega)
      have a := EndgameTreeCache.scaled_discValue mc fc pos
      have c := EndgameTreeCache.scaled_discValue mc fc (pass pos)
      rw [PvsBoardTree.minimax_pass mc fc pos (PvsBoardTree.nil_of_stuck none) available] at a
      have value := last_move mc fc (pass pos) b.square oneOther available
      symm
      change 256 * discValue mc fc b.asBoard = _ at a
      change (Board.opponentCount (pass pos).val).toNat = Bmi2.popCount b.player at player
      omega
    next otherNone =>
      have no : Board.hasLegalMoves mc (pass pos).val = false := by
        cases avail : Board.hasLegalMoves mc (pass pos).val
        · rfl
        · have positive := (playable (pass pos) oneOther).mp avail
          omega
      change _ = discValue mc fc pos.val
      rw [terminal_disc mc fc pos none no]
      change _ = Board.AwardedScore b.asBoard
      unfold Board.AwardedScore
      rw [emptyCount]
      split <;> (try split) <;> (try split) <;> omega

def TreeContract : Prop :=
  ∀ (mc : X86Config) (fc : FlipConfig) (b : Bmi2.Board) (alpha : ScoreWord),
    let exactValue := discValue mc fc b.asBoard
    NWSResult alpha.toInt exactValue (Bmi2.solve b alpha).toInt ∧
    NWSResult alpha.toInt exactValue (Scalar.solve b alpha).toInt ∧
    NWSResult alpha.toInt exactValue (Neon.solve b alpha).toInt

theorem treeContract_proven : TreeContract := by
  intro mc fc b alpha
  have bounds := exactScore_eq_tree mc fc b ▸ Bmi2.solve_nws b alpha
  exact ⟨bounds, Scalar.solve_eq_bmi2 b alpha ▸ bounds, Neon.solve_eq_bmi2 b alpha ▸ bounds⟩

end NeuralReversi.Solve1
