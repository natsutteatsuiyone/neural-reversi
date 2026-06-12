import { describe, expect, it } from "vitest";
import {
  AI_PROGRESS_TRAIL_SIZE,
  EMPTY_AI_PROGRESS_TRAIL,
  nextAIProgressTrail,
  type AIProgressTrailCell,
} from "../ai-progress-trail";

const cell = (row: number, col: number, timestamp = 0): AIProgressTrailCell => ({
  row,
  col,
  timestamp,
});

describe("nextAIProgressTrail", () => {
  it("keeps the trail unchanged (same ref) when there is no cell but it is still the AI's turn", () => {
    const prev = [cell(1, 1)];
    expect(nextAIProgressTrail(prev, null, true, 100)).toBe(prev);
  });

  it("clears to the shared empty trail when there is no cell and it is no longer the AI's turn", () => {
    const prev = [cell(1, 1)];
    expect(nextAIProgressTrail(prev, null, false, 100)).toBe(EMPTY_AI_PROGRESS_TRAIL);
  });

  it("ignores a cell that repeats the current head (same ref)", () => {
    const prev = [cell(3, 4, 10), cell(2, 2, 5)];
    expect(nextAIProgressTrail(prev, { row: 3, col: 4 }, true, 99)).toBe(prev);
  });

  it("prepends a new cell with the given timestamp", () => {
    const prev = [cell(2, 2, 5)];
    const next = nextAIProgressTrail(prev, { row: 3, col: 4 }, true, 99);
    expect(next).toEqual([cell(3, 4, 99), cell(2, 2, 5)]);
  });

  it("caps the trail at AI_PROGRESS_TRAIL_SIZE, dropping the oldest", () => {
    let trail: AIProgressTrailCell[] = EMPTY_AI_PROGRESS_TRAIL;
    for (let i = 0; i < AI_PROGRESS_TRAIL_SIZE + 2; i++) {
      trail = nextAIProgressTrail(trail, { row: i, col: i }, true, i);
    }
    expect(trail).toHaveLength(AI_PROGRESS_TRAIL_SIZE);
    // newest first
    expect(trail[0]).toEqual(
      cell(AI_PROGRESS_TRAIL_SIZE + 1, AI_PROGRESS_TRAIL_SIZE + 1, AI_PROGRESS_TRAIL_SIZE + 1),
    );
  });

  it("prepends a distinct cell even when only row or only col changed", () => {
    const prev = [cell(3, 4, 1)];
    expect(nextAIProgressTrail(prev, { row: 3, col: 5 }, true, 2)).toEqual([
      cell(3, 5, 2),
      cell(3, 4, 1),
    ]);
  });
});
