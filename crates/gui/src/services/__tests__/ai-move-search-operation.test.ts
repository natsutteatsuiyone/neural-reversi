import { describe, expect, it, vi } from "vitest";
import { initializeBoard } from "@/domain/game/game-logic";
import { createMockAIService } from "@/services/mock-ai-service";
import { runAIMoveSearch } from "@/services/ai-move-search-operation";
import type { AIMoveProgress } from "@/services/types";

function createDeferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

function progress(overrides?: Partial<AIMoveProgress>): AIMoveProgress {
  return {
    bestMove: "d3",
    row: 2,
    col: 3,
    score: 4,
    depth: 10,
    targetDepth: 10,
    acc: 100,
    nodes: 1_000,
    pvLine: "d3",
    isEndgame: false,
    ...overrides,
  };
}

describe("runAIMoveSearch", () => {
  it("dedupes progress and computes nps", async () => {
    vi.useFakeTimers();
    try {
      vi.setSystemTime(0);
      const onProgress = vi.fn();
      const ai = createMockAIService({
        getAIMove: vi.fn().mockImplementation(async (_board, _player, _level, _time, _remaining, callback) => {
          vi.setSystemTime(1_000);
          callback(progress());
          callback(progress());
          vi.setSystemTime(1_500);
          callback(progress({ depth: 11, nodes: 2_000 }));
          return null;
        }),
      });

      await runAIMoveSearch({
        ai,
        board: initializeBoard(),
        player: "black",
        level: 1,
        mode: "level",
        timeLimitSeconds: 1,
        remainingTimeMs: 60_000,
        getRemainingTime: () => 60_000,
        onStart: vi.fn(),
        onTimerChange: vi.fn(),
        onRemainingTime: vi.fn(),
        onProgress,
        onFinish: vi.fn(),
      });

      expect(onProgress).toHaveBeenCalledTimes(2);
      expect(onProgress.mock.calls[0][0]).toMatchObject({
        progress: expect.objectContaining({ depth: 10 }),
        nps: 1_000,
      });
      expect(onProgress.mock.calls[1][0]).toMatchObject({
        progress: expect.objectContaining({ depth: 11 }),
        nps: expect.closeTo(1_333.333, 3),
      });
    } finally {
      vi.useRealTimers();
    }
  });

  it("ticks down game-time mode and applies final engine time", async () => {
    vi.useFakeTimers();
    try {
      let remainingTime = 60_000;
      let timer: ReturnType<typeof setInterval> | null = null;
      const moveDeferred = createDeferred<{
        row: number;
        col: number;
        score: number;
        depth: number;
        acc: number;
        timeTaken: number;
      }>();
      const ai = createMockAIService({
        getAIMove: vi.fn().mockReturnValue(moveDeferred.promise),
      });

      const pending = runAIMoveSearch({
        ai,
        board: initializeBoard(),
        player: "black",
        level: 1,
        mode: "game-time",
        timeLimitSeconds: 1,
        remainingTimeMs: remainingTime,
        getRemainingTime: () => remainingTime,
        onStart: vi.fn(),
        onTimerChange: (nextTimer) => {
          timer = nextTimer;
        },
        onRemainingTime: (nextRemainingTime) => {
          remainingTime = nextRemainingTime;
        },
        onProgress: vi.fn(),
        onFinish: vi.fn(),
      });

      expect(timer).not.toBeNull();
      vi.advanceTimersByTime(1_000);
      await Promise.resolve();
      expect(remainingTime).toBe(59_000);

      moveDeferred.resolve({
        row: 2,
        col: 3,
        score: 1,
        depth: 10,
        acc: 100,
        timeTaken: 1_090,
      });
      await pending;

      expect(remainingTime).toBe(58_910);
      expect(timer).toBeNull();
    } finally {
      vi.useRealTimers();
    }
  });

  it("stops managing remaining time after an external clock write", async () => {
    vi.useFakeTimers();
    try {
      let remainingTime = 60_000;
      const moveDeferred = createDeferred<null>();
      const ai = createMockAIService({
        getAIMove: vi.fn().mockReturnValue(moveDeferred.promise),
      });

      const pending = runAIMoveSearch({
        ai,
        board: initializeBoard(),
        player: "black",
        level: 1,
        mode: "game-time",
        timeLimitSeconds: 1,
        remainingTimeMs: remainingTime,
        getRemainingTime: () => remainingTime,
        onStart: vi.fn(),
        onTimerChange: vi.fn(),
        onRemainingTime: (nextRemainingTime) => {
          remainingTime = nextRemainingTime;
        },
        onProgress: vi.fn(),
        onFinish: vi.fn(),
      });

      remainingTime = 42_000;
      vi.advanceTimersByTime(100);
      await Promise.resolve();
      moveDeferred.resolve(null);
      await pending;

      expect(remainingTime).toBe(42_000);
    } finally {
      vi.useRealTimers();
    }
  });

  it("cleans up the timer and finishes when search rejects", async () => {
    vi.useFakeTimers();
    try {
      const onTimerChange = vi.fn();
      const onFinish = vi.fn();
      const ai = createMockAIService({
        getAIMove: vi.fn().mockRejectedValue(new Error("boom")),
      });

      await expect(runAIMoveSearch({
        ai,
        board: initializeBoard(),
        player: "black",
        level: 1,
        mode: "game-time",
        timeLimitSeconds: 1,
        remainingTimeMs: 60_000,
        getRemainingTime: () => 60_000,
        onStart: vi.fn(),
        onTimerChange,
        onRemainingTime: vi.fn(),
        onProgress: vi.fn(),
        onFinish,
      })).rejects.toThrow("boom");

      expect(onTimerChange).toHaveBeenLastCalledWith(null);
      expect(onFinish).toHaveBeenCalledTimes(1);
    } finally {
      vi.useRealTimers();
    }
  });
});
