import { getNotation, opponentPlayer } from "@/domain/game/game-logic";
import { applyMove, checkGameOver, type Move } from "@/domain/game/store-helpers";
import type { Board, Player } from "@/domain/game/types";

/**
 * Pure solver position-timeline logic. A {@link SolverHistoryEntry} is one
 * stop on the navigation timeline; advancing/rooting are pure functions of
 * (board, player) so the Solver Session can stay a thin orchestrator.
 */

export interface SolverHistoryEntry {
  board: Board;
  player: Player;
  moveFrom: string | null;
}

export function createSolverRootEntry(board: Board, player: Player): SolverHistoryEntry {
  return { board, player, moveFrom: null };
}

export interface AdvanceSolverPositionResult {
  board: Board;
  player: Player;
  entry: SolverHistoryEntry;
  gameOver: boolean;
}

export function advanceSolverPosition(
  board: Board,
  player: Player,
  row: number,
  col: number,
): AdvanceSolverPositionResult {
  const move: Move = { row, col, isAI: false, score: 0 };
  const nextBoard = applyMove(board, move, player);
  let nextPlayer = opponentPlayer(player);

  // Collapse an implicit pass into this solver step so one undo unwinds it.
  const { gameOver, shouldPass } = checkGameOver(nextBoard, nextPlayer);
  if (shouldPass) {
    nextPlayer = opponentPlayer(nextPlayer);
  }

  return {
    board: nextBoard,
    player: nextPlayer,
    entry: {
      board: nextBoard,
      player: nextPlayer,
      moveFrom: getNotation(row, col),
    },
    gameOver,
  };
}
