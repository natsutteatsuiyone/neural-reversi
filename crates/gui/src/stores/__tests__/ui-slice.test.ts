import { beforeEach, describe, expect, it, vi } from "vitest";
import { createTestStore, createDeferred } from "./test-helpers";
import { createMockAIService } from "@/services/mock-ai-service";

beforeEach(() => {
  vi.clearAllMocks();
});

describe("setHintMode", () => {
  it("starts hint analysis when enabling hint mode", () => {
    const { store } = createTestStore();
    const analyzeBoardSpy = vi.spyOn(store.getState(), "analyzeBoard");

    store.getState().setHintMode(true);

    expect(store.getState().isHintMode).toBe(true);
    expect(analyzeBoardSpy).toHaveBeenCalled();
  });

  it("aborts only hint analysis when disabling during analysis", () => {
    const { store, services } = createTestStore();
    store.setState({
      isHintMode: true,
      isAnalyzing: true,
      isAIThinking: false,
      analyzeResults: new Map([["2,3", {} as never]]),
    });

    store.getState().setHintMode(false);

    expect(store.getState().isHintMode).toBe(false);
    expect(store.getState().isAnalyzing).toBe(true);
    expect(store.getState().hintAnalysisAbortPending).toBe(true);
    expect(store.getState().analyzeResults).toBeNull();
    expect(services.ai.abortSearch).toHaveBeenCalledTimes(1);
  });

  it("does not abort AI thinking when disabling hint mode", () => {
    const { store, services } = createTestStore();
    store.setState({
      isHintMode: true,
      isAIThinking: true,
      isAnalyzing: false,
    });

    store.getState().setHintMode(false);

    expect(store.getState().isHintMode).toBe(false);
    expect(store.getState().isAIThinking).toBe(true);
    expect(services.ai.abortSearch).not.toHaveBeenCalled();
  });

  it("waits for abortSearch to finish before clearing analysis state", async () => {
    const abortDeferred = createDeferred<void>();
    const { store } = createTestStore({
      ai: createMockAIService({
        abortSearch: vi.fn().mockReturnValue(abortDeferred.promise),
      }),
    });

    store.setState({
      isHintMode: true,
      isAnalyzing: true,
      isAIThinking: false,
    });

    store.getState().setHintMode(false);
    expect(store.getState().isAnalyzing).toBe(true);
    expect(store.getState().hintAnalysisAbortPending).toBe(true);

    abortDeferred.resolve();
    await abortDeferred.promise;
    await Promise.resolve();

    expect(store.getState().isAnalyzing).toBe(false);
    expect(store.getState().hintAnalysisAbortPending).toBe(false);
  });

  it("restarts hint analysis only after the pending abort completes", async () => {
    const abortDeferred = createDeferred<void>();
    const { store } = createTestStore({
      ai: createMockAIService({
        abortSearch: vi.fn().mockReturnValue(abortDeferred.promise),
      }),
    });
    const analyzeBoardSpy = vi.spyOn(store.getState(), "analyzeBoard");

    store.setState({
      isHintMode: true,
      isAnalyzing: true,
      isAIThinking: false,
    });

    store.getState().setHintMode(false);
    store.getState().setHintMode(true);

    expect(analyzeBoardSpy).not.toHaveBeenCalled();

    abortDeferred.resolve();
    await abortDeferred.promise;
    await Promise.resolve();

    expect(analyzeBoardSpy).toHaveBeenCalledTimes(1);
  });

  it("ignores stale hint progress after a cancelled analysis is re-enabled", async () => {
    const analyzeDeferred1 = createDeferred<void>();
    const analyzeDeferred2 = createDeferred<void>();
    const abortDeferred = createDeferred<void>();
    const analyzeCallbacks: Array<(progress: {
      row: number;
      col: number;
      depth: number;
      score: number;
      targetDepth: number;
      acc: number;
      nodes: number;
      pvLine: string;
      bestMove: string;
      isEndgame: boolean;
    }) => void> = [];

    const { store, services } = createTestStore({
      ai: createMockAIService({
        analyze: vi
          .fn()
          .mockImplementationOnce(async (_board, _player, _level, callback) => {
            analyzeCallbacks.push(callback);
            await analyzeDeferred1.promise;
          })
          .mockImplementationOnce(async (_board, _player, _level, callback) => {
            analyzeCallbacks.push(callback);
            await analyzeDeferred2.promise;
          }),
        abortSearch: vi.fn().mockReturnValue(abortDeferred.promise),
      }),
    });

    store.setState({
      gameStatus: "playing",
      isHintMode: false,
      isAIThinking: false,
      currentPlayer: "black",
      gameMode: "pvp",
    });

    store.getState().setHintMode(true);
    await Promise.resolve();
    expect(services.ai.analyze).toHaveBeenCalledTimes(1);

    store.getState().setHintMode(false);
    store.getState().setHintMode(true);

    analyzeCallbacks[0]({
      row: 2,
      col: 3,
      depth: 10,
      score: 1,
      targetDepth: 10,
      acc: 100,
      nodes: 10,
      pvLine: "d3",
      bestMove: "d3",
      isEndgame: false,
    });
    expect(store.getState().analyzeResults).toBeNull();

    abortDeferred.resolve();
    await abortDeferred.promise;
    await Promise.resolve();
    expect(services.ai.analyze).toHaveBeenCalledTimes(2);

    analyzeCallbacks[0]({
      row: 2,
      col: 3,
      depth: 11,
      score: 2,
      targetDepth: 11,
      acc: 100,
      nodes: 20,
      pvLine: "d3",
      bestMove: "d3",
      isEndgame: false,
    });
    expect(store.getState().analyzeResults).toBeNull();

    analyzeCallbacks[1]({
      row: 4,
      col: 5,
      depth: 12,
      score: 3,
      targetDepth: 12,
      acc: 100,
      nodes: 30,
      pvLine: "f5",
      bestMove: "f5",
      isEndgame: false,
    });
    expect(store.getState().analyzeResults?.get("4,5")?.bestMove).toBe("f5");

    analyzeDeferred1.resolve();
    analyzeDeferred2.resolve();
    await Promise.all([analyzeDeferred1.promise, analyzeDeferred2.promise]);
  });
});
