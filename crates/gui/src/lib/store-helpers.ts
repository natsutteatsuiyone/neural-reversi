import {
  getFlippedDiscs,
  getNotation,
  getValidMoves,
  initializeBoard,
  opponentPlayer as nextPlayer,
} from "@/lib/game-logic";
import type { Board, MoveRecord, Player } from "@/types";
import { MoveHistory } from "@/lib/move-history";

export interface Move {
  row: number;
  col: number;
  isAI: boolean;
  score?: number;
}

export function cloneBoard(board: Board): Board {
  return board.map((row) => row.map((cell) => ({ ...cell })));
}

export function createMoveRecord(
  moveId: number,
  player: Player,
  move: Move,
  remainingTime?: number
): MoveRecord {
  return {
    id: moveId,
    player,
    row: move.row,
    col: move.col,
    notation: getNotation(move.row, move.col),
    score: move.score,
    isAI: move.isAI,
    remainingTime,
  };
}

export function applyMove(board: Board, move: Move, player: Player): Board {
  const newBoard = cloneBoard(board);
  const flipped = getFlippedDiscs(board, move.row, move.col, player);

  newBoard[move.row][move.col] = {
    color: player,
    isNew: true,
  };

  for (const [r, c] of flipped) {
    newBoard[r][c] = { color: player };
  }

  return newBoard;
}

export function createPassMove(moveId: number, player: Player, remainingTime?: number): MoveRecord {
  return {
    id: moveId,
    player,
    row: -1,
    col: -1,
    notation: "Pass",
    remainingTime,
  };
}

export function reconstructBoardFromMoves(
  moves: MoveRecord[],
  historyStartBoard: Board = initializeBoard(),
  historyStartPlayer: Player = "black"
): {
  board: Board;
  currentPlayer: Player;
} {
  const board = cloneBoard(historyStartBoard);

  for (const move of moves) {
    if (move.row >= 0 && move.col >= 0) {
      const flipped = getFlippedDiscs(board, move.row, move.col, move.player);

      board[move.row][move.col] = {
        color: move.player,
      };

      for (const [r, c] of flipped) {
        board[r][c] = { color: move.player };
      }
    }
  }

  const currentPlayer: Player =
    moves.length > 0 ? nextPlayer(moves[moves.length - 1].player) : historyStartPlayer;

  return { board, currentPlayer };
}

export function createGameStartState(
  board: Board,
  currentPlayer: Player,
  gameStatus: "waiting" | "playing",
  gameTimeLimitMs: number,
) {
  return {
    board,
    historyStartBoard: cloneBoard(board),
    historyStartPlayer: currentPlayer,
    moveHistory: MoveHistory.empty(),
    currentPlayer,
    gameStatus,
    gameOver: false,
    isPass: false,
    lastMove: null,
    lastAIMove: null,
    showPassNotification: null,
    analyzeResults: null,
    isAIThinking: false,
    isAnalyzing: false,
    aiMoveProgress: null,
    aiThinkingHistory: [],
    aiRemainingTime: gameTimeLimitMs,
    searchTimer: null,
    validMoves: gameStatus === "playing" ? getValidMoves(board, currentPlayer) : [],
    skipAnimation: true,
    paused: false,
    gameAnalysisResult: null,
  };
}

export function checkGameOver(board: Board, currentPlayer: Player): {
  gameOver: boolean;
  shouldPass: boolean;
} {
  const currentMoves = getValidMoves(board, currentPlayer);

  if (currentMoves.length > 0) {
    return { gameOver: false, shouldPass: false };
  }

  const opponentMoves = getValidMoves(board, nextPlayer(currentPlayer));

  if (opponentMoves.length === 0) {
    return { gameOver: true, shouldPass: false };
  }

  return { gameOver: false, shouldPass: true };
}
