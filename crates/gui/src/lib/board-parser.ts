import type { Board, Player } from "@/types";
import { BOARD_SIZE } from "./constants";
import {
  createEmptyBoard,
  initializeBoard,
  getFlippedDiscs,
  getValidMoves,
  opponentPlayer as opponentOf,
} from "./game-logic";

export type ParseResult =
  | { ok: true; board: Board; currentPlayer: Player }
  | { ok: false; error: string };

/**
 * Parses a move transcript string like "F5D6C3D3C4" into a board position.
 *
 * Each move is 2 characters: column letter (a-h) + row number (1-8), case-insensitive.
 * Replays moves from the initial position with legality checks.
 * Auto-handles passes when a player has no valid moves.
 */
export function parseTranscript(input: string): ParseResult {
  const transcript = input.trim();

  if (transcript.length === 0) {
    const board = initializeBoard();
    return { ok: true, board, currentPlayer: "black" };
  }

  if (transcript.length % 2 !== 0) {
    return { ok: false, error: "invalidLength" };
  }

  const board = initializeBoard();
  let currentPlayer: Player = "black";

  for (let i = 0; i < transcript.length; i += 2) {
    const colChar = transcript[i].toLowerCase();
    const rowChar = transcript[i + 1];

    const col = colChar.charCodeAt(0) - "a".charCodeAt(0);
    const row = parseInt(rowChar, 10) - 1;

    if (col < 0 || col >= BOARD_SIZE || row < 0 || row >= BOARD_SIZE || isNaN(row)) {
      return { ok: false, error: "invalidMove" };
    }

    // Auto-handle pass: if current player has no moves, switch to opponent
    const currentMoves = getValidMoves(board, currentPlayer);
    if (currentMoves.length === 0) {
      const opponentMoves = getValidMoves(board, opponentOf(currentPlayer));
      if (opponentMoves.length === 0) {
        return { ok: false, error: "gameAlreadyOver" };
      }
      currentPlayer = opponentOf(currentPlayer);
    }

    const flipped = getFlippedDiscs(board, row, col, currentPlayer);
    if (flipped.length === 0 || board[row][col].color !== null) {
      const notation = `${colChar.toUpperCase()}${rowChar}`;
      return { ok: false, error: `illegalMove:${notation}` };
    }

    // Apply the move
    board[row][col] = { color: currentPlayer };
    for (const [fr, fc] of flipped) {
      board[fr][fc] = { color: currentPlayer };
    }

    currentPlayer = opponentOf(currentPlayer);
  }

  // After all moves, if current player has no valid moves but opponent does, switch
  const currentMoves = getValidMoves(board, currentPlayer);
  if (currentMoves.length === 0) {
    const opponent = opponentOf(currentPlayer);
    const opponentMoves = getValidMoves(board, opponent);
    if (opponentMoves.length > 0) {
      currentPlayer = opponent;
    }
  }

  return { ok: true, board, currentPlayer };
}

/**
 * Parses a 64-character board string into a board position.
 *
 * Character mappings (case-insensitive):
 * - X, B, * = black
 * - O, W = white
 * - -, . = empty
 *
 * Strips whitespace/newlines before parsing.
 */
export function parseBoardString(input: string): ParseResult {
  const stripped = input.replace(/\s/g, "");

  if (stripped.length === 0) {
    const board = createEmptyBoard();
    return { ok: true, board, currentPlayer: "black" };
  }

  if (stripped.length !== 64) {
    return { ok: false, error: "invalidBoardLength" };
  }

  const board = createEmptyBoard();

  for (let i = 0; i < 64; i++) {
    const ch = stripped[i].toUpperCase();
    const row = Math.floor(i / BOARD_SIZE);
    const col = i % BOARD_SIZE;

    switch (ch) {
      case "X":
      case "B":
      case "*":
        board[row][col] = { color: "black" };
        break;
      case "O":
      case "W":
        board[row][col] = { color: "white" };
        break;
      case "-":
      case ".":
        board[row][col] = { color: null };
        break;
      default:
        return { ok: false, error: "invalidCharacter" };
    }
  }

  return { ok: true, board, currentPlayer: "black" };
}

/**
 * Validates that a board is playable.
 *
 * Returns null if valid, or an error string:
 * - "needBothColors": Must have at least one black and one white stone
 * - "noValidMoves": Neither player has a legal move
 * - "currentPlayerNoMoves": Current player has no legal moves (would pass immediately)
 */
export function validateBoard(board: Board, currentPlayer: Player): string | null {
  let hasBlack = false;
  let hasWhite = false;

  for (let row = 0; row < BOARD_SIZE; row++) {
    for (let col = 0; col < BOARD_SIZE; col++) {
      if (board[row][col].color === "black") hasBlack = true;
      if (board[row][col].color === "white") hasWhite = true;
      if (hasBlack && hasWhite) break;
    }
    if (hasBlack && hasWhite) break;
  }

  if (!hasBlack || !hasWhite) {
    return "needBothColors";
  }

  const currentMoves = getValidMoves(board, currentPlayer);
  const opponentMoves = getValidMoves(board, opponentOf(currentPlayer));

  if (currentMoves.length === 0 && opponentMoves.length === 0) {
    return "noValidMoves";
  }

  if (currentMoves.length === 0) {
    return "currentPlayerNoMoves";
  }

  return null;
}

/**
 * Converts a Board to a 64-character string.
 *
 * X = black, O = white, - = empty.
 * Characters are ordered row by row, left to right, top to bottom.
 */
export function boardToString(board: Board): string {
  let result = "";

  for (let row = 0; row < BOARD_SIZE; row++) {
    for (let col = 0; col < BOARD_SIZE; col++) {
      const color = board[row][col].color;
      if (color === "black") {
        result += "X";
      } else if (color === "white") {
        result += "O";
      } else {
        result += "-";
      }
    }
  }

  return result;
}
