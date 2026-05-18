import { afterAll, beforeEach, describe, expect, it, vi } from "vitest";
import { createMockAIService } from "@/services/mock-ai-service";
import type { AIMoveResult } from "@/services/types";
import { createTestStore, createDeferred } from "./test-helpers";

const consoleErrorSpy = vi.spyOn(console, "error").mockImplementation(() => {});

beforeEach(() => {
  vi.clearAllMocks();
});

afterAll(() => {
  consoleErrorSpy.mockRestore();
});

describe("makeAIMove", () => {
  it("does not start a second search while another search is active", async () => {
    const { store, services } = createTestStore();
    // An Engine Search is already in flight (CONTEXT.md → Engine Activity);
    // isAIThinking is its projection.
    store.setState({
      isAIThinking: true,
      engineActivity: { kind: "ai-move", runId: 1 },
    });

    await store.getState().makeAIMove();

    expect(services.ai.getAIMove).not.toHaveBeenCalled();
  });

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
      // Let the EngineSearch run register and start the game-time interval.
      await Promise.resolve();

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

  it("ignores a move returned after the AI search was aborted", async () => {
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
    store.setState({ gameMode: "ai-black", currentPlayer: "black" });

    const pending = store.getState().makeAIMove();
    await Promise.resolve();
    expect(store.getState().isAIThinking).toBe(true);

    await store.getState().abortAIMove();
    expect(store.getState().paused).toBe(true);

    moveDeferred.resolve({
      row: 2,
      col: 3,
      score: 12,
      depth: 10,
      acc: 100,
      timeTaken: 50,
    });
    await pending;

    expect(store.getState().moveHistory.length).toBe(0);
    expect(store.getState().currentPlayer).toBe("black");
    expect(store.getState().lastAIMove).toBeNull();
    expect(store.getState().paused).toBe(true);
  });

  it("an AI move aborted mid-search does not apply its late result (replaces ignoredAIMoveRunIds)", async () => {
    const moveDeferred = createDeferred<AIMoveResult>();
    const { store, services } = createTestStore({
      ai: createMockAIService({
        getAIMove: vi.fn().mockReturnValue(moveDeferred.promise),
      }),
    });
    await store.getState().startGame();

    const aiPending = store.getState().makeAIMove();
    await Promise.resolve();
    expect(store.getState().isAIThinking).toBe(true);

    await store.getState().abortAIMove();
    expect(store.getState().isAIThinking).toBe(false);

    // The aborted search resolves late with a real move — it must NOT be played.
    moveDeferred.resolve({ row: 2, col: 3, score: 0, depth: 1, acc: 0, timeTaken: 0 });
    await aiPending;

    expect(store.getState().lastAIMove).toBeNull();
    expect(store.getState().isAIThinking).toBe(false);
    expect(services.ai.getAIMove).toHaveBeenCalledTimes(1);
  });
});

describe("abortAIMove", () => {
  it("clears thinking state even when abortSearch throws", async () => {
    const { store } = createTestStore({
      ai: createMockAIService({
        abortSearch: vi.fn().mockRejectedValue(new Error("abort failed")),
      }),
    });
    store.setState({ isAIThinking: true, isAnalyzing: true });

    await store.getState().abortAIMove();

    // EngineSearch.abort runs onSettled regardless of backend-abort success
    // (unified teardown contract; prevents a stuck "thinking" UI).
    expect(store.getState().isAIThinking).toBe(false);
    expect(store.getState().isAnalyzing).toBe(false);
    expect(store.getState().aiMoveProgress).toBeNull();
  });

  it("clears thinking state and drops a late move when abortSearch throws", async () => {
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
        abortSearch: vi.fn().mockRejectedValue(new Error("abort failed")),
      }),
    });
    await store.getState().startGame();
    store.setState({ gameMode: "ai-black", currentPlayer: "black" });

    const pending = store.getState().makeAIMove();
    // Let the EngineSearch run register before aborting it.
    await Promise.resolve();
    await store.getState().abortAIMove();
    // Abort-reject now clears immediately (unified teardown contract); the
    // superseded run's late move is dropped, not held in an ignored set.
    expect(store.getState().isAIThinking).toBe(false);

    moveDeferred.resolve({
      row: 2,
      col: 3,
      score: 12,
      depth: 10,
      acc: 100,
      timeTaken: 50,
    });
    await pending;

    expect(store.getState().isAIThinking).toBe(false);
    expect(store.getState().moveHistory.length).toBe(0);
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

  it("pauses the game when stopping an AI turn with legal moves", async () => {
    const { store } = createTestStore();
    await store.getState().startGame();
    store.setState({
      gameMode: "ai-black",
      currentPlayer: "black",
      isAIThinking: true,
      paused: false,
    });

    await store.getState().abortAIMove();

    expect(store.getState().paused).toBe(true);
  });

  it("does not pause when aborting hint analysis without an AI move search", async () => {
    const { store } = createTestStore();
    await store.getState().startGame();
    store.setState({
      gameMode: "ai-black",
      currentPlayer: "black",
      isAIThinking: false,
      isAnalyzing: true,
      paused: false,
    });

    await store.getState().abortAIMove();

    expect(store.getState().paused).toBe(false);
  });

  it("is a no-op when neither thinking nor analyzing", async () => {
    const { store, services } = createTestStore();

    await store.getState().abortAIMove();

    expect(services.ai.abortSearch).not.toHaveBeenCalled();
  });

  // Regression: a hint abort-then-restart stamps isAnalyzing=false
  // synchronously while its backend abort + restart are still in flight.
  // abortAIMove must still supersede that pending run (it routes the
  // EngineSearch generation forward), so it cannot early-return here.
  it("still aborts when only a hint abort is pending", async () => {
    const { store, services } = createTestStore();
    store.setState({
      isAIThinking: false,
      isAnalyzing: false,
      hintAnalysisAbortPending: true,
    });

    await store.getState().abortAIMove();

    expect(services.ai.abortSearch).toHaveBeenCalledTimes(1);
    expect(store.getState().paused).toBe(false);
  });
});
