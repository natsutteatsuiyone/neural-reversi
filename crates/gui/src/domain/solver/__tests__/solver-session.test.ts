import { beforeEach, describe, expect, it, vi } from "vitest";
import { initializeBoard } from "@/domain/game/game-logic";
import { applyMove } from "@/domain/game/store-helpers";
import type { Board, Player } from "@/domain/game/types";
import type { SolverCandidate, SolverProgressPayload } from "@/services/types";
import {
  advanceSolverPosition,
  applySolverProgress,
  cacheCompleteSolverResult,
  createSolverResultCache,
  getCachedSolverResult,
  isCompleteSolverResult,
} from "../solver-session";

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

function candidate(overrides?: Partial<SolverCandidate>): SolverCandidate {
  return {
    move: "d3",
    row: 2,
    col: 3,
    score: 4,
    depth: 14,
    targetDepth: 14,
    acc: 100,
    pvLine: "d3",
    isEndgame: false,
    isComplete: true,
    ...overrides,
  };
}

function progress(overrides?: Partial<SolverProgressPayload>): SolverProgressPayload {
  return {
    runId: 1,
    bestMove: "d3",
    row: 2,
    col: 3,
    score: 4,
    depth: 14,
    targetDepth: 14,
    acc: 100,
    nodes: 1_000,
    pvLine: "d3",
    isEndgame: false,
    ...overrides,
  };
}

describe("solver result cache", () => {
  it("stores only complete candidate maps and returns a fresh copy", () => {
    const cache = createSolverResultCache();
    const board = initializeBoard();
    const candidates = new Map([["2,3", candidate()]]);

    cacheCompleteSolverResult(cache, board, "black", 100, "multiPv", candidates);

    const cached = getCachedSolverResult(cache, board, "black", 100, "multiPv");
    expect(cached).toEqual(candidates);
    expect(cached).not.toBe(candidates);

    cached?.set("4,5", candidate({ move: "f5", row: 4, col: 5 }));
    expect(getCachedSolverResult(cache, board, "black", 100, "multiPv")?.size).toBe(1);
  });

  it("does not cache incomplete or empty candidate maps", () => {
    const cache = createSolverResultCache();
    const board = initializeBoard();

    cacheCompleteSolverResult(cache, board, "black", 100, "multiPv", new Map());
    cacheCompleteSolverResult(
      cache,
      board,
      "black",
      95,
      "multiPv",
      new Map([["2,3", candidate({ isComplete: false })]]),
    );

    expect(getCachedSolverResult(cache, board, "black", 100, "multiPv")).toBeNull();
    expect(getCachedSolverResult(cache, board, "black", 95, "multiPv")).toBeNull();
  });
});

describe("isCompleteSolverResult", () => {
  it("requires at least one candidate and all candidates complete", () => {
    expect(isCompleteSolverResult(new Map())).toBe(false);
    expect(isCompleteSolverResult(new Map([["2,3", candidate()]]))).toBe(true);
    expect(isCompleteSolverResult(new Map([["2,3", candidate({ isComplete: false })]]))).toBe(false);
  });
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

describe("applySolverProgress", () => {
  it("returns the same map when progress does not change rendering fields", () => {
    const existing = new Map([["2,3", candidate()]]);

    const next = applySolverProgress(existing, progress(), 100, "multiPv");

    expect(next).toBe(existing);
  });

  it("marks endgame candidates complete once target selectivity is reached", () => {
    const next = applySolverProgress(new Map(), progress({
      isEndgame: true,
      acc: 95,
      depth: 20,
      targetDepth: 20,
    }), 95, "multiPv");

    expect(next.get("2,3")?.isComplete).toBe(true);
  });

  it("keeps only the latest candidate in bestOnly mode", () => {
    const first = applySolverProgress(new Map(), progress(), 100, "bestOnly");
    const second = applySolverProgress(first, progress({
      bestMove: "f5",
      row: 4,
      col: 5,
      pvLine: "f5",
    }), 100, "bestOnly");

    expect(second.size).toBe(1);
    expect(second.get("4,5")?.move).toBe("f5");
    expect(second.get("2,3")).toBeUndefined();
  });
});
