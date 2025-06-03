import {
  getFlippedDiscs,
  getNotation,
  getValidMoves,
  initializeBoard,
  opponentPlayer as nextPlayer,
} from "@/lib/game-logic";
import type { Board, MoveRecord, Player } from "@/types";

export interface Move {
  row: number;
  col: number;
  isAI: boolean;
  score?: number;
}

export function createMoveRecord(
  moveId: number,
  player: Player,
  move: Move
): MoveRecord {
  return {
    id: moveId,
    player,
    row: move.row,
    col: move.col,
    notation: getNotation(move.row, move.col),
    score: move.score,
    isAI: move.isAI,
  };
}

export function applyMove(board: Board, move: Move, player: Player): Board {
  const newBoard = board.map((row) => row.map((cell) => ({ ...cell })));
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

export function createPassMove(moveId: number, player: Player): MoveRecord {
  return {
    id: moveId,
    player,
    row: -1,
    col: -1,
    notation: "Pass",
  };
}

export function reconstructBoardFromMoves(moves: MoveRecord[]): {
  board: Board;
  currentPlayer: Player;
} {
  const board = initializeBoard();

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
    moves.length > 0 ? nextPlayer(moves[moves.length - 1].player) : "black";

  return { board, currentPlayer };
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

export function getUndoMoves(
  moves: MoveRecord[],
  gameMode: "analyze" | "ai-black" | "ai-white"
): MoveRecord[] {
  if (moves.length === 0) {
    return [];
  }

  const newMoves = [...moves];

  if (gameMode === "ai-black" || gameMode === "ai-white") {
    const lastMove = newMoves[newMoves.length - 1];

    if (lastMove.isAI) {
      newMoves.pop();
      if (newMoves.length > 0 && !newMoves[newMoves.length - 1].isAI) {
        newMoves.pop();
      }
    } else {
      newMoves.pop();
    }
  } else {
    newMoves.pop();
  }

  return newMoves;
}
