import { describe, expect, it } from "vitest";
import type { MoveAnalysis } from "@/domain/game/game-analysis";
import type { MoveRecord } from "@/domain/game/types";
import {
  createEvaluationChartData,
  resolveCursorMoveNumber,
  resolveYAxisDomain,
  resolveYAxisTicks,
} from "@/components/ai/evaluation-chart-model";

function move(overrides: Partial<MoveRecord>): MoveRecord {
  return {
    id: 0,
    player: "black",
    row: 2,
    col: 3,
    notation: "d3",
    ...overrides,
  };
}

function analysis(overrides: Partial<MoveAnalysis>): MoveAnalysis {
  return {
    moveIndex: 0,
    player: "black",
    playedMove: "d3",
    playedScore: 4,
    bestMove: "d3",
    bestScore: 4,
    scoreLoss: 0,
    depth: 12,
    ...overrides,
  };
}

describe("createEvaluationChartData", () => {
  it("skips pass moves but keeps timeline indices for navigation", () => {
    const data = createEvaluationChartData([
      move({ id: 0, row: 4, col: 5, notation: "f5" }),
      move({ id: 1, player: "white", row: -1, col: -1, notation: "Pass" }),
      move({ id: 2, player: "white", row: 5, col: 3, notation: "d6" }),
    ], null);

    expect(data.map((item) => item.move)).toEqual([1, 2]);
    expect(data.map((item) => item.timelineIndex)).toEqual([0, 2]);
  });

  it("uses analysis scores before AI fallback scores", () => {
    const data = createEvaluationChartData([
      move({ isAI: true, score: 2 }),
    ], [
      analysis({ playedScore: 7 }),
    ]);

    expect(data[0].score).toBe(7);
    expect(data[0].scoreDisplay).toBe("+7.0");
  });

  it("normalizes white scores so positive always means black-favored", () => {
    const data = createEvaluationChartData([
      move({ id: 0, player: "white", isAI: true, score: 5 }),
      move({ id: 1, player: "white", row: 2, col: 4, notation: "e3" }),
    ], [
      analysis({ moveIndex: 1, player: "white", playedMove: "e3", playedScore: -3 }),
    ]);

    expect(data[0].score).toBe(-5);
    expect(data[0].scoreDisplay).toBe("-5.0");
    expect(data[1].score).toBe(3);
    expect(data[1].scoreDisplay).toBe("+3.0");
  });
});

describe("resolveCursorMoveNumber", () => {
  it("returns the last visible move before the history cursor", () => {
    const data = createEvaluationChartData([
      move({ id: 0, row: 4, col: 5, notation: "f5" }),
      move({ id: 1, player: "white", row: -1, col: -1, notation: "Pass" }),
      move({ id: 2, player: "white", row: 5, col: 3, notation: "d6" }),
    ], null);

    expect(resolveCursorMoveNumber(data, 2, 3)).toBe(1);
    expect(resolveCursorMoveNumber(data, 3, 3)).toBeNull();
  });
});

describe("resolveYAxisDomain", () => {
  it("uses the default domain when no scores exist", () => {
    expect(resolveYAxisDomain([])).toEqual([-8, 8]);
  });

  it("keeps enough opposite-side space for one-sided scores", () => {
    const data = createEvaluationChartData([
      move({ id: 0, isAI: true, score: 20 }),
      move({ id: 1, row: 3, col: 2, notation: "c4", isAI: true, score: 28 }),
    ], null);

    expect(resolveYAxisDomain(data)).toEqual([-12, 32]);
  });

  it("caps large ranges to the chart score bounds", () => {
    const data = createEvaluationChartData([
      move({ id: 0, isAI: true, score: -80 }),
      move({ id: 1, row: 3, col: 2, notation: "c4", isAI: true, score: 80 }),
    ], null);

    expect(resolveYAxisDomain(data)).toEqual([-64, 64]);
  });
});

describe("resolveYAxisTicks", () => {
  it("creates stable ticks across the domain", () => {
    expect(resolveYAxisTicks([-8, 8])).toEqual([-8, -4, 0, 4, 8]);
  });
});
