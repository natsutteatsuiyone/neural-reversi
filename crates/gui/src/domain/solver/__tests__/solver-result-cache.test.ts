import { describe, expect, it } from "vitest";
import { initializeBoard } from "@/domain/game/game-logic";
import type { SolverCandidate } from "@/services/types";
import { SolverResultCache } from "../solver-result-cache";

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

describe("SolverResultCache", () => {
  it("stores only complete candidate maps and returns a fresh copy", () => {
    const cache = new SolverResultCache();
    const board = initializeBoard();
    const candidates = new Map([["2,3", candidate()]]);

    cache.storeIfComplete(board, "black", 100, "multiPv", candidates);

    const cached = cache.get(board, "black", 100, "multiPv");
    expect(cached).toEqual(candidates);
    expect(cached).not.toBe(candidates);

    // Mutating a returned copy must not leak back into the cache.
    cached?.set("4,5", candidate({ move: "f5", row: 4, col: 5 }));
    expect(cache.get(board, "black", 100, "multiPv")?.size).toBe(1);
  });

  it("does not cache incomplete or empty candidate maps", () => {
    const cache = new SolverResultCache();
    const board = initializeBoard();

    cache.storeIfComplete(board, "black", 100, "multiPv", new Map());
    cache.storeIfComplete(
      board,
      "black",
      95,
      "multiPv",
      new Map([["2,3", candidate({ isComplete: false })]]),
    );

    expect(cache.get(board, "black", 100, "multiPv")).toBeNull();
    expect(cache.get(board, "black", 95, "multiPv")).toBeNull();
  });

  it("keys on the full (board, player, selectivity, mode) tuple", () => {
    const cache = new SolverResultCache();
    const board = initializeBoard();
    cache.storeIfComplete(board, "black", 100, "multiPv", new Map([["2,3", candidate()]]));

    expect(cache.get(board, "white", 100, "multiPv")).toBeNull();
    expect(cache.get(board, "black", 95, "multiPv")).toBeNull();
    expect(cache.get(board, "black", 100, "bestOnly")).toBeNull();
    expect(cache.get(board, "black", 100, "multiPv")?.size).toBe(1);
  });
});
