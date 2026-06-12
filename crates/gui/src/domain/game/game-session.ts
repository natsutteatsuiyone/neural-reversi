import { getValidMoves, opponentPlayer as nextPlayer } from "@/domain/game/game-logic";
import { BOARD_SIZE } from "@/domain/game/constants";
import { MoveHistory } from "@/domain/game/move-history";
import {
  checkGameOver,
  cloneBoard,
  createPassMove,
  reconstructBoardFromMoves,
  type Move,
} from "@/domain/game/store-helpers";
import type { Board, GameStatus, MoveRecord, Player } from "@/domain/game/types";

export type HistoryNavigationDirection = "undo" | "redo";

export interface DerivedGamePosition {
  board: Board;
  currentPlayer: Player;
  validMoves: [number, number][];
  lastMove: Move | null;
}

export interface HistoryNavigationState {
  gameStatus: GameStatus;
  moveHistory: MoveHistory;
  historyStartBoard: Board;
  historyStartPlayer: Player;
  gameTimeLimitMs: number;
}

export interface GameHistoryPatch extends DerivedGamePosition {
  moveHistory: MoveHistory;
  isPass: false;
  gameOver: boolean;
  gameStatus: GameStatus;
  skipAnimation: true;
  aiRemainingTime: number;
}

export interface PassTurnState {
  board: Board;
  moveHistory: MoveHistory;
  currentPlayer: Player;
  aiRemainingTime: number;
}

export interface PassTurnPatch {
  board: Board;
  moveHistory: MoveHistory;
  currentPlayer: Player;
  validMoves: [number, number][];
  isPass: true;
}

export function resolveRemainingTime(history: MoveHistory, defaultMs: number): number {
  return history.length > 0 ? (history.lastMove?.remainingTime ?? defaultMs) : defaultMs;
}

export function derivePositionFromMoves(
  moves: readonly MoveRecord[],
  historyStartBoard: Board,
  historyStartPlayer: Player,
): DerivedGamePosition {
  const { board, currentPlayer } = reconstructBoardFromMoves(
    moves,
    historyStartBoard,
    historyStartPlayer,
  );

  return {
    board,
    currentPlayer,
    validMoves: getValidMoves(board, currentPlayer),
    lastMove: toLastMove(moves),
  };
}

export function createHistoryNavigationPatch(
  state: HistoryNavigationState,
  direction: HistoryNavigationDirection,
): GameHistoryPatch | null {
  const isUndo = direction === "undo";

  if (state.gameStatus === "waiting") return null;
  if (isUndo ? !state.moveHistory.canUndo : !state.moveHistory.canRedo) return null;

  const moveHistory = isUndo ? state.moveHistory.undo(1) : state.moveHistory.redo(1);
  const derived = derivePositionFromMoves(
    moveHistory.currentMoves,
    state.historyStartBoard,
    state.historyStartPlayer,
  );
  const gameOver = isUndo ? false : checkGameOver(derived.board, derived.currentPlayer).gameOver;

  return {
    ...derived,
    moveHistory,
    isPass: false,
    gameOver,
    gameStatus: gameOver ? "finished" : "playing",
    skipAnimation: true,
    aiRemainingTime: resolveRemainingTime(moveHistory, state.gameTimeLimitMs),
  };
}

export function createGoToMovePatch(
  state: HistoryNavigationState,
  position: number,
): GameHistoryPatch | null {
  const moveHistory = state.moveHistory.goTo(position);
  if (moveHistory.length === state.moveHistory.length) return null;

  const derived = derivePositionFromMoves(
    moveHistory.currentMoves,
    state.historyStartBoard,
    state.historyStartPlayer,
  );
  const isAtEnd = position >= state.moveHistory.totalLength;
  const gameOver = isAtEnd ? checkGameOver(derived.board, derived.currentPlayer).gameOver : false;
  const gameStatus = gameOver
    ? "finished"
    : state.gameStatus === "finished"
      ? "playing"
      : state.gameStatus;

  return {
    ...derived,
    moveHistory,
    isPass: false,
    gameOver,
    gameStatus,
    skipAnimation: true,
    aiRemainingTime: resolveRemainingTime(moveHistory, state.gameTimeLimitMs),
  };
}

export function createPassTurnPatch(
  state: PassTurnState,
  passingPlayer = state.currentPlayer,
): PassTurnPatch {
  const passMove = createPassMove(state.moveHistory.length, passingPlayer, state.aiRemainingTime);
  const currentPlayer = nextPlayer(passingPlayer);
  const board = cloneBoard(state.board);

  return {
    board,
    moveHistory: state.moveHistory.append(passMove),
    currentPlayer,
    validMoves: getValidMoves(board, currentPlayer),
    isPass: true,
  };
}

export function hasFlippedDiscs(oldBoard: Board, newBoard: Board): boolean {
  for (let row = 0; row < BOARD_SIZE; row++) {
    for (let col = 0; col < BOARD_SIZE; col++) {
      const oldCell = oldBoard[row][col];
      const newCell = newBoard[row][col];
      if (oldCell.color && newCell.color && oldCell.color !== newCell.color) {
        return true;
      }
    }
  }

  return false;
}

function toLastMove(moves: readonly MoveRecord[]): Move | null {
  const last = moves.length > 0 ? moves[moves.length - 1] : undefined;
  if (!last || last.row < 0 || last.col < 0) {
    return null;
  }

  return {
    row: last.row,
    col: last.col,
    isAI: Boolean(last.isAI),
    score: last.score,
  };
}
