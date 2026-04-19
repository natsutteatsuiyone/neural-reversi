import { afterAll, beforeEach, describe, expect, it, vi } from "vitest";
import { createMockAIService } from "@/services/mock-ai-service";
import { createTestStore, createDeferred } from "./test-helpers";

const consoleErrorSpy = vi.spyOn(console, "error").mockImplementation(() => {});

beforeEach(() => {
  vi.clearAllMocks();
});

afterAll(() => {
  consoleErrorSpy.mockRestore();
});

describe("makeAIMove", () => {
  it("resets isAIThinking to false when getAIMove throws", async () => {
    const { store } = createTestStore({
      ai: createMockAIService({
        getAIMove: vi.fn().mockRejectedValue(new Error("boom")),
      }),
    });
    await store.getState().startGame();

    await store.getState().makeAIMove();

    expect(store.getState().isAIThinking).toBe(false);
    expect(store.getState().aiMoveProgress).toBeNull();
    expect(store.getState().searchTimer).toBeNull();
  });

  it("does not overwrite aiRemainingTime written during the search", async () => {
    const moveDeferred = createDeferred<null>();
    const { store } = createTestStore({
      ai: createMockAIService({
        getAIMove: vi.fn().mockReturnValue(moveDeferred.promise),
      }),
    });
    await store.getState().startGame();
    store.setState({ aiMode: "game-time", aiRemainingTime: 60_000 });

    const pending = store.getState().makeAIMove();

    // Simulate an external writer (e.g. undo/redo) adjusting aiRemainingTime
    // while the search is still running.
    store.setState({ aiRemainingTime: 42_000 });

    moveDeferred.resolve(null);
    await pending;

    // With the previous buggy code this would be 60_000 - aiMove.timeTaken,
    // clobbering the 42_000 that undo wrote mid-search.
    expect(store.getState().aiRemainingTime).toBe(42_000);
  });

  it("applies the exact game-time spent before recording the AI move", async () => {
    vi.useFakeTimers();
    try {
      const moveDeferred = createDeferred<{
        row: number;
        col: number;
        score: number;
        depth: number;
        acc: number;
        timeTaken: number;
      }>();
      const { store } = createTestStore({
        ai: createMockAIService({
          getAIMove: vi.fn().mockReturnValue(moveDeferred.promise),
        }),
      });
      await store.getState().startGame();
      store.setState({
        gameMode: "pvp",
        aiMode: "game-time",
        aiRemainingTime: 60_000,
      });

      const pending = store.getState().makeAIMove();

      vi.advanceTimersByTime(1_000);
      await Promise.resolve();
      expect(store.getState().aiRemainingTime).toBe(59_000);

      moveDeferred.resolve({
        row: 2,
        col: 3,
        score: 12,
        depth: 10,
        acc: 100,
        timeTaken: 1_090,
      });
      await pending;

      expect(store.getState().aiRemainingTime).toBe(58_910);
      expect(store.getState().moveHistory.lastMove?.remainingTime).toBe(58_910);
    } finally {
      vi.useRealTimers();
    }
  });
});

describe("abortAIMove", () => {
  it("keeps isAIThinking true when abortSearch throws", async () => {
    const { store } = createTestStore({
      ai: createMockAIService({
        abortSearch: vi.fn().mockRejectedValue(new Error("abort failed")),
      }),
    });
    store.setState({ isAIThinking: true });

    await store.getState().abortAIMove();

    expect(store.getState().isAIThinking).toBe(true);
  });

  it("clears flags when abortSearch resolves", async () => {
    const { store, services } = createTestStore();
    store.setState({ isAIThinking: true, isAnalyzing: true });

    await store.getState().abortAIMove();

    expect(services.ai.abortSearch).toHaveBeenCalledTimes(1);
    expect(store.getState().isAIThinking).toBe(false);
    expect(store.getState().isAnalyzing).toBe(false);
    expect(store.getState().aiMoveProgress).toBeNull();
  });

  it("is a no-op when neither thinking nor analyzing", async () => {
    const { store, services } = createTestStore();

    await store.getState().abortAIMove();

    expect(services.ai.abortSearch).not.toHaveBeenCalled();
  });
});
