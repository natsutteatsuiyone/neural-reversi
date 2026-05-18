import { beforeEach, describe, expect, it, vi } from "vitest";
import { initializeBoard } from "@/domain/game/game-logic";
import { applyMove } from "@/domain/game/store-helpers";
import type { Board, Player } from "@/domain/game/types";
import { advanceSolverPosition } from "../solver-navigation";

let getValidMovesStub:
  | ((board: Board, player: Player) => [number, number][])
  | null = null;

vi.mock("@/domain/game/game-logic", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/domain/game/game-logic")>();
  return {
    ...actual,
    getValidMoves: (board: Board, player: Player) =>
      (getValidMovesStub ?? actual.getValidMoves)(board, player),
  };
});

beforeEach(() => {
  getValidMovesStub = null;
});

describe("advanceSolverPosition", () => {
  it("applies a move and flips the current player", () => {
    const board = initializeBoard();

    const result = advanceSolverPosition(board, "black", 2, 3);

    expect(result.entry.moveFrom).toBe("d3");
    expect(result.player).toBe("white");
    expect(result.board).toEqual(applyMove(board, { row: 2, col: 3, isAI: false, score: 0 }, "black"));
    expect(result.gameOver).toBe(false);
  });

  it("collapses an implicit pass into the same history entry", () => {
    const board = initializeBoard();
    getValidMovesStub = (_board, player) => (player === "white" ? [] : [[2, 3]]);

    const result = advanceSolverPosition(board, "black", 2, 3);

    expect(result.player).toBe("black");
    expect(result.entry.player).toBe("black");
    expect(result.gameOver).toBe(false);
  });

  it("marks gameOver when neither player can move", () => {
    const board = initializeBoard();
    getValidMovesStub = () => [];

    const result = advanceSolverPosition(board, "black", 2, 3);

    expect(result.player).toBe("white");
    expect(result.gameOver).toBe(true);
  });
});
