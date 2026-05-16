import { describe, expect, it } from "vitest";
import {
  appendGameAnalysisProgress,
  applyHintAnalysisProgress,
  createGameAnalysisMoveList,
} from "@/domain/game/game-analysis";
import type { AIMoveProgress, GameAnalysisProgress } from "@/services/types";
import type { MoveRecord } from "@/domain/game/types";

function hintProgress(overrides: Partial<AIMoveProgress> = {}): AIMoveProgress {
  return {
    bestMove: "d3",
    row: 2,
    col: 3,
    score: 10,
    depth: 8,
    targetDepth: 12,
    acc: 0,
    nodes: 100,
    pvLine: "d3 c3",
    isEndgame: false,
    ...overrides,
  };
}

function moveRecord(overrides: Partial<MoveRecord> = {}): MoveRecord {
  return {
    id: 0,
    player: "black",
    row: 2,
    col: 3,
    notation: "d3",
    ...overrides,
  };
}

function gameProgress(overrides: Partial<GameAnalysisProgress> = {}): GameAnalysisProgress {
  return {
    moveIndex: 0,
    bestMove: "d3",
    bestScore: 20,
    playedScore: 10,
    scoreLoss: 10,
    depth: 12,
    ...overrides,
  };
}

describe("applyHintAnalysisProgress", () => {
  it("adds progress keyed by board coordinate", () => {
    const results = applyHintAnalysisProgress(new Map(), hintProgress());

    expect(results?.get("2,3")?.bestMove).toBe("d3");
  });

  it("returns null for no-op progress re-emits", () => {
    const progress = hintProgress();
    const first = applyHintAnalysisProgress(new Map(), progress);
    expect(first).not.toBeNull();

    const second = applyHintAnalysisProgress(first!, progress);

    expect(second).toBeNull();
  });

  it("returns a fresh map when progress changes", () => {
    const first = applyHintAnalysisProgress(new Map(), hintProgress());
    const second = applyHintAnalysisProgress(first!, hintProgress({ depth: 9 }));

    expect(second).not.toBe(first);
    expect(second?.get("2,3")?.depth).toBe(9);
  });
});

describe("createGameAnalysisMoveList", () => {
  it("converts normal moves and passes to backend notation", () => {
    const moves = [
      moveRecord({ row: 4, col: 5, notation: "f5" }),
      moveRecord({ id: 1, player: "white", row: -1, col: -1, notation: "Pass" }),
    ];

    expect(createGameAnalysisMoveList(moves)).toEqual(["f5", "--"]);
  });
});

describe("appendGameAnalysisProgress", () => {
  it("appends display-ready analysis for the played move", () => {
    const results = appendGameAnalysisProgress(
      [],
      [moveRecord({ score: 1 })],
      gameProgress({ bestMove: "c4", scoreLoss: 7 }),
    );

    expect(results).toEqual([{
      moveIndex: 0,
      player: "black",
      playedMove: "d3",
      playedScore: 10,
      bestMove: "c4",
      bestScore: 20,
      scoreLoss: 7,
      depth: 12,
    }]);
  });
});
