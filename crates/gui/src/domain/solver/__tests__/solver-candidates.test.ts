import { describe, expect, it } from "vitest";
import type { SolverCandidate, SolverProgressPayload } from "@/services/types";
import {
  applySolverProgress,
  isCompleteSolverResult,
  solverCandidatesToAnalysisResults,
} from "../solver-candidates";

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

describe("isCompleteSolverResult", () => {
  it("requires at least one candidate and all candidates complete", () => {
    expect(isCompleteSolverResult(new Map())).toBe(false);
    expect(isCompleteSolverResult(new Map([["2,3", candidate()]]))).toBe(true);
    expect(isCompleteSolverResult(new Map([["2,3", candidate({ isComplete: false })]]))).toBe(false);
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

describe("solverCandidatesToAnalysisResults", () => {
  it("projects a candidate into the analysis-overlay shape", () => {
    const results = solverCandidatesToAnalysisResults(
      new Map([["2,3", candidate({ move: "d3", score: 7, acc: 95, isEndgame: false })]]),
    );

    expect(results.get("2,3")).toEqual({
      bestMove: "d3",
      row: 2,
      col: 3,
      score: 7,
      depth: 14,
      targetDepth: 14,
      acc: 95,
      nodes: 0,
      pvLine: "d3",
      isEndgame: false,
    });
  });

  it("forces acc=100 for a completed endgame candidate", () => {
    const results = solverCandidatesToAnalysisResults(
      new Map([["2,3", candidate({ acc: 73, isEndgame: true, isComplete: true })]]),
    );

    expect(results.get("2,3")?.acc).toBe(100);
  });

  it("keeps the raw acc for an incomplete endgame candidate", () => {
    const results = solverCandidatesToAnalysisResults(
      new Map([["2,3", candidate({ acc: 73, isEndgame: true, isComplete: false })]]),
    );

    expect(results.get("2,3")?.acc).toBe(73);
  });
});
