import { describe, expect, it } from "vitest";
import { initializeBoard } from "@/domain/game/game-logic";
import {
  createGoToMovePatch,
  createHistoryNavigationPatch,
  createPassTurnPatch,
  derivePositionFromMoves,
  hasFlippedDiscs,
} from "@/domain/game/game-session";
import { MoveHistory } from "@/domain/game/move-history";
import { applyMove, createMoveRecord, type Move } from "@/domain/game/store-helpers";

const GAME_TIME_LIMIT_MS = 60_000;

function firstMove(): Move {
  return { row: 2, col: 3, isAI: false, score: 4 };
}

function firstMoveHistory(): MoveHistory {
  return MoveHistory.empty().append(createMoveRecord(0, "black", firstMove(), 42_000));
}

describe("derivePositionFromMoves", () => {
  it("reconstructs board, next player, valid moves, and last move", () => {
    const startBoard = initializeBoard();
    const history = firstMoveHistory();

    const derived = derivePositionFromMoves(history.currentMoves, startBoard, "black");

    expect(derived.board[2][3].color).toBe("black");
    expect(derived.board[3][3].color).toBe("black");
    expect(derived.currentPlayer).toBe("white");
    expect(derived.validMoves.length).toBeGreaterThan(0);
    expect(derived.lastMove).toEqual(firstMove());
  });
});

describe("createHistoryNavigationPatch", () => {
  it("creates an undo patch without re-applying pass or automation concerns", () => {
    const startBoard = initializeBoard();
    const history = firstMoveHistory();

    const patch = createHistoryNavigationPatch({
      gameStatus: "playing",
      moveHistory: history,
      historyStartBoard: startBoard,
      historyStartPlayer: "black",
      gameTimeLimitMs: GAME_TIME_LIMIT_MS,
    }, "undo");

    expect(patch?.moveHistory.length).toBe(0);
    expect(patch?.currentPlayer).toBe("black");
    expect(patch?.gameOver).toBe(false);
    expect(patch?.gameStatus).toBe("playing");
    expect(patch?.aiRemainingTime).toBe(GAME_TIME_LIMIT_MS);
  });

  it("creates a redo patch and restores remaining time from the move record", () => {
    const startBoard = initializeBoard();
    const history = firstMoveHistory().undo(1);

    const patch = createHistoryNavigationPatch({
      gameStatus: "playing",
      moveHistory: history,
      historyStartBoard: startBoard,
      historyStartPlayer: "black",
      gameTimeLimitMs: GAME_TIME_LIMIT_MS,
    }, "redo");

    expect(patch?.moveHistory.length).toBe(1);
    expect(patch?.currentPlayer).toBe("white");
    expect(patch?.lastMove).toEqual(firstMove());
    expect(patch?.aiRemainingTime).toBe(42_000);
  });
});

describe("createGoToMovePatch", () => {
  it("moves a finished game back to playing when navigating before the end", () => {
    const startBoard = initializeBoard();
    const history = firstMoveHistory();

    const patch = createGoToMovePatch({
      gameStatus: "finished",
      moveHistory: history,
      historyStartBoard: startBoard,
      historyStartPlayer: "black",
      gameTimeLimitMs: GAME_TIME_LIMIT_MS,
    }, 0);

    expect(patch?.moveHistory.length).toBe(0);
    expect(patch?.gameStatus).toBe("playing");
    expect(patch?.gameOver).toBe(false);
  });
});

describe("createPassTurnPatch", () => {
  it("appends a pass move and switches player without mutating the board", () => {
    const board = initializeBoard();
    const patch = createPassTurnPatch({
      board,
      moveHistory: MoveHistory.empty(),
      currentPlayer: "black",
      aiRemainingTime: 12_345,
    });

    expect(patch.board).not.toBe(board);
    expect(patch.board[3][3].color).toBe(board[3][3].color);
    expect(patch.currentPlayer).toBe("white");
    expect(patch.moveHistory.lastMove?.notation).toBe("Pass");
    expect(patch.moveHistory.lastMove?.remainingTime).toBe(12_345);
    expect(patch.isPass).toBe(true);
  });
});

describe("hasFlippedDiscs", () => {
  it("detects color flips between two board states", () => {
    const board = initializeBoard();
    const nextBoard = applyMove(board, firstMove(), "black");

    expect(hasFlippedDiscs(board, nextBoard)).toBe(true);
    expect(hasFlippedDiscs(board, board)).toBe(false);
  });
});
