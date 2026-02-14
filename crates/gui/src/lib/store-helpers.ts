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

// Helper function to determine whose turn it is from the actual move history.
// Unlike modulo-based alternation, this correctly handles pass moves
// where the same player may appear in consecutive move records.
function getCurrentPlayer(moves: MoveRecord[], historyStartPlayer: Player): Player {
  if (moves.length === 0) return historyStartPlayer;
  return nextPlayer(moves[moves.length - 1].player);
}

// Helper function to check if it's the player's turn
function isPlayerTurn(
  moves: MoveRecord[],
  gameMode: "ai-black" | "ai-white",
  historyStartPlayer: Player
): boolean {
  const currentTurn = getCurrentPlayer(moves, historyStartPlayer);
  const playerIsBlack = gameMode === "ai-white";
  return (playerIsBlack && currentTurn === "black") || (!playerIsBlack && currentTurn === "white");
}

export function getUndoMoves(
  moves: MoveRecord[],
  gameMode: "analyze" | "ai-black" | "ai-white",
  historyStartPlayer: Player = "black"
): MoveRecord[] {
  if (moves.length === 0) {
    return [];
  }

  const newMoves = [...moves];

  if (gameMode === "analyze") {
    // In analyze mode, just remove one move
    newMoves.pop();
    return newMoves;
  }

  // In AI mode, undo to the previous player's turn
  const currentIsPlayerTurn = isPlayerTurn(newMoves, gameMode, historyStartPlayer);

  if (currentIsPlayerTurn) {
    // Currently player's turn, go back to previous player's turn (remove 2 moves)
    if (newMoves.length >= 2) {
      newMoves.pop();
      newMoves.pop();
    }
  } else {
    // Currently AI's turn, go back to player's turn (remove 1 move)
    if (newMoves.length >= 1) {
      newMoves.pop();
    }
  }

  return newMoves;
}

export function getRedoMoves(
  currentMoves: MoveRecord[],
  allMoves: MoveRecord[],
  gameMode: "analyze" | "ai-black" | "ai-white",
  historyStartPlayer: Player = "black"
): MoveRecord[] {
  // No moves to redo
  if (currentMoves.length >= allMoves.length) {
    return currentMoves;
  }

  // In analyze mode, redo one move at a time
  if (gameMode === "analyze") {
    return [...currentMoves, allMoves[currentMoves.length]];
  }

  // In AI mode, redo until it's the player's turn
  const newMoves = [...currentMoves];
  let index = currentMoves.length;

  // Add moves until we reach a player's turn
  while (index < allMoves.length) {
    const move = allMoves[index];
    newMoves.push(move);
    index++;

    // Check if it's now the player's turn
    if (isPlayerTurn(newMoves, gameMode, historyStartPlayer)) {
      // Stop at player's turn
      break;
    }
  }

  return newMoves;
}
