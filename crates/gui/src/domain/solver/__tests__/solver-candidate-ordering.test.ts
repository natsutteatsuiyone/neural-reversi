import { describe, expect, it } from "vitest";
import type { SolverCandidate } from "@/services/types";
import { getBestRoundedScore, sortSolverCandidates } from "../solver-candidate-ordering";

function candidate(move: string, score: number): SolverCandidate {
  return {
    move,
    row: 0,
    col: 0,
    score,
    depth: 20,
    targetDepth: 20,
    acc: 100,
    pvLine: move,
    isEndgame: true,
    isComplete: true,
  };
}

describe("sortSolverCandidates", () => {
  it("orders candidates by descending score without mutating the source collection", () => {
    const source = [candidate("a1", -4), candidate("b2", 7.4), candidate("c3", 7.6)];

    const sorted = sortSolverCandidates(source);

    expect(sorted.map((c) => c.move)).toEqual(["c3", "b2", "a1"]);
    expect(source.map((c) => c.move)).toEqual(["a1", "b2", "c3"]);
  });
});

describe("getBestRoundedScore", () => {
  it("returns the rounded score of the first sorted candidate", () => {
    expect(getBestRoundedScore([candidate("d4", 7.6)])).toBe(8);
  });

  it("returns null when there are no candidates", () => {
    expect(getBestRoundedScore([])).toBeNull();
  });
});
