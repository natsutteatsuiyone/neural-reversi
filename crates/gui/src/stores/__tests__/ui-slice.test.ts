import { beforeEach, describe, expect, it, vi } from "vitest";
import { createTestStore, createDeferred } from "./test-helpers";
import { createMockAIService } from "@/services/mock-ai-service";
import type { AIMoveResult, GameAnalysisProgress } from "@/services/types";

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

  it("aborts only hint analysis when disabling during analysis", async () => {
    const { store, services } = createTestStore();
    store.setState({
      isHintMode: true,
      isAnalyzing: true,
      isAIThinking: false,
      analyzeResults: new Map([["2,3", {} as never]]),
    });

    store.getState().setHintMode(false);
    // EngineSearch.abort awaits supersede() before onAbort, so the abort-pending
    // snapshot is now observable one microtask later (behavior unchanged).
    await Promise.resolve();

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
    // EngineSearch.abort awaits supersede() before onAbort sets the pending flag.
    await Promise.resolve();
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

  it("ignores stale abort cleanup while a newer hint abort is pending", async () => {
    const abortDeferred1 = createDeferred<void>();
    const abortDeferred2 = createDeferred<void>();
    const { store, services } = createTestStore({
      ai: createMockAIService({
        abortSearch: vi
          .fn()
          .mockReturnValueOnce(abortDeferred1.promise)
          .mockReturnValueOnce(abortDeferred2.promise),
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
    store.getState().setHintMode(false);
    // EngineSearch.abort awaits supersede() before onAbort/abort, so both queued
    // abort chains land one microtask later (behavior unchanged).
    await Promise.resolve();

    expect(services.ai.abortSearch).toHaveBeenCalledTimes(2);
    expect(store.getState().hintAnalysisAbortPending).toBe(true);

    abortDeferred1.resolve();
    await abortDeferred1.promise;
    await Promise.resolve();

    expect(store.getState().hintAnalysisAbortPending).toBe(true);
    expect(store.getState().isAnalyzing).toBe(true);

    store.getState().setHintMode(true);
    expect(analyzeBoardSpy).not.toHaveBeenCalled();

    abortDeferred2.resolve();
    await abortDeferred2.promise;
    await Promise.resolve();

    expect(store.getState().hintAnalysisAbortPending).toBe(false);
    expect(store.getState().isAnalyzing).toBe(false);
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
    // EngineSearch.abort drains supersede() -> teardown -> onSettled ->
    // analyzeBoard -> start() -> supersede() before the re-analysis is issued,
    // so the restart now spans several microtasks (behavior unchanged).
    for (let i = 0; i < 10 && (services.ai.analyze as ReturnType<typeof vi.fn>).mock.calls.length < 2; i++) {
      await Promise.resolve();
    }
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

  it("does not strand the abort guard when another engine op supersedes an in-flight hint abort", async () => {
    const abortDeferred = createDeferred<void>();
    const { store } = createTestStore({
      ai: createMockAIService({
        abortSearch: vi.fn().mockReturnValue(abortDeferred.promise),
      }),
    });
    store.setState({ isHintMode: true, isAnalyzing: true, isAIThinking: false });

    // Hint abort claims the guard, then stalls on its slow backend abort.
    store.getState().setHintMode(false);
    await Promise.resolve();
    expect(store.getState().hintAnalysisAbortPending).toBe(true);

    // A different (non-hint-abort) engine op supersedes the in-flight hint
    // abort, so the hint abort's onSettled will be skipped.
    await store.getState().abortGameAnalysis();

    // The slow hint abort finally resolves: onSettled was skipped, but the
    // guaranteed-once onTeardown must still release the guard (no newer hint
    // abort claimed since), or hint analysis is dead for the session.
    abortDeferred.resolve();
    for (let i = 0; i < 10 && store.getState().hintAnalysisAbortPending; i++) await Promise.resolve();
    expect(store.getState().hintAnalysisAbortPending).toBe(false);
  });

  it("does not start a hint search queued behind a slow supersede after hint mode is turned off", async () => {
    const gameRunDeferred = createDeferred<void>();
    const gameAbortDeferred = createDeferred<void>();
    const analyzeSpy = vi.fn().mockResolvedValue(undefined);
    const analyzeGameMock = vi.fn().mockReturnValue(gameRunDeferred.promise);
    const { store } = createTestStore({
      ai: createMockAIService({
        analyzeGame: analyzeGameMock,
        abortGameAnalysis: vi.fn().mockReturnValue(gameAbortDeferred.promise),
        analyze: analyzeSpy,
      }),
    });
    await store.getState().startGame();
    store.setState({ gameMode: "pvp" }); // isAITurn() === false so analyzeBoard's guard passes
    await store.getState().makeMove({ row: 2, col: 3, isAI: false });

    // Game analysis installs a live EngineSearch run. isGameAnalyzing flips
    // synchronously (onClaim); wait for the backend run to actually start so
    // the hint claim below captures the game run as its prior.
    void store.getState().analyzeGame();
    for (let i = 0; i < 10 && analyzeGameMock.mock.calls.length === 0; i++) await Promise.resolve();
    expect(store.getState().isGameAnalyzing).toBe(true);

    // Enabling hint mode queues a hint analyzeBoard start; its claim() captures
    // the game-analysis run and stalls on the SLOW abortGameAnalysis, so the
    // hint start's onStart (which sets isAnalyzing) has NOT run yet.
    store.getState().setHintMode(true);
    await Promise.resolve();
    expect(store.getState().isAnalyzing).toBe(false);

    // User turns hint mode off during the wait. isAnalyzing is false here, so
    // setHintMode(false) does nothing — the queued hint start is still pending.
    store.getState().setHintMode(false);
    expect(store.getState().isHintMode).toBe(false);

    // Slow supersede resolves: the queued hint start now proceeds. It must
    // recheck hint mode and NOT launch a background search nor strand isAnalyzing.
    gameAbortDeferred.resolve();
    for (let i = 0; i < 12; i++) await Promise.resolve();

    expect(analyzeSpy).not.toHaveBeenCalled();
    expect(store.getState().isAnalyzing).toBe(false);

    gameRunDeferred.resolve();
  });

  it("turning hint mode off while an AI move is in flight does not abort the AI search", async () => {
    const moveDeferred = createDeferred<AIMoveResult>();
    const { store, services } = createTestStore({
      ai: createMockAIService({
        getAIMove: vi.fn().mockReturnValue(moveDeferred.promise),
      }),
    });
    await store.getState().startGame();
    store.setState({ isHintMode: true });

    const aiPending = store.getState().makeAIMove();
    await Promise.resolve();
    expect(store.getState().isAIThinking).toBe(true);

    store.getState().setHintMode(false);
    await Promise.resolve();

    expect(services.ai.abortSearch).not.toHaveBeenCalled();
    expect(store.getState().isAIThinking).toBe(true);

    moveDeferred.resolve(null);
    await aiPending;
  });
});

describe("analyzeBoard generation ownership", () => {
  it("a hint run queued behind a slow game-analysis supersede does not clear a newer hint run's isAnalyzing", async () => {
    const gameRunDeferred = createDeferred<void>();
    const gameAbortDeferred = createDeferred<void>();
    const hintRunDeferred = createDeferred<void>();
    const analyzeGameMock = vi.fn().mockReturnValue(gameRunDeferred.promise);    // game run stays live
    const analyzeMock = vi.fn().mockReturnValue(hintRunDeferred.promise);        // newer hint run
    const { store } = createTestStore({
      ai: createMockAIService({
        analyzeGame: analyzeGameMock,
        abortGameAnalysis: vi.fn().mockReturnValue(gameAbortDeferred.promise),   // SLOW supersede
        analyze: analyzeMock,
      }),
    });
    await store.getState().startGame();
    store.setState({ gameMode: "pvp" }); // isAITurn() === false
    await store.getState().makeMove({ row: 2, col: 3, isAI: false });
    vi.spyOn(store.getState(), "makeAIMove").mockResolvedValue(undefined);

    // Game analysis is live (an installed EngineSearch run).
    void store.getState().analyzeGame();
    for (let i = 0; i < 10 && analyzeGameMock.mock.calls.length === 0; i++) await Promise.resolve();
    expect(store.getState().isGameAnalyzing).toBe(true);

    // Enabling Hint queues analyzeBoard A: its claim supersedes the live game
    // run, so A blocks on the slow game-analysis backend abort WITHOUT
    // installing or setting isAnalyzing. (analyzeBoard does not guard
    // isGameAnalyzing, so the top guard lets it through.)
    store.getState().setHintMode(true);
    await Promise.resolve();
    expect(store.getState().isAnalyzing).toBe(false);

    // Changing the hint level while A is still queued starts analyzeBoard B.
    // isAnalyzing is still false (A never installed), so setHintLevel routes
    // to analyzeBoard, not restartHintAnalysisAfterAbort. B's claim sees no
    // live run and installs, taking ownership of isAnalyzing.
    store.getState().setHintLevel(store.getState().hintLevel + 1);
    for (let i = 0; i < 10 && analyzeMock.mock.calls.length === 0; i++) await Promise.resolve();
    expect(store.getState().isAnalyzing).toBe(true);

    // The slow game-analysis abort resolves: A is now superseded-before-install
    // and its guaranteed-once onTeardown fires. It MUST NOT clear isAnalyzing —
    // that flag is owned by the newer, still-running hint run B. Drain all
    // microtasks unconditionally: A's onTeardown runs one microtask AFTER the
    // game run's own teardown clears isGameAnalyzing, so a barrier that stops
    // on isGameAnalyzing===false would assert before the clobber and pass
    // spuriously.
    gameAbortDeferred.resolve();
    for (let i = 0; i < 30; i++) await Promise.resolve();
    expect(store.getState().isGameAnalyzing).toBe(false);
    expect(store.getState().isAnalyzing).toBe(true);

    hintRunDeferred.resolve();
    gameRunDeferred.resolve();
    await Promise.resolve();
  });
});

describe("analyzeGame", () => {
  it("marks game analysis pending synchronously so a queued analysis cannot run on a mutated history", async () => {
    const hintRunDeferred = createDeferred<void>();
    const abortDeferred = createDeferred<void>();
    const gameRunDeferred = createDeferred<void>();
    const { store } = createTestStore({
      ai: createMockAIService({
        analyze: vi.fn().mockReturnValue(hintRunDeferred.promise),   // hint run stays live
        abortSearch: vi.fn().mockReturnValue(abortDeferred.promise), // SLOW supersede
        analyzeGame: vi.fn().mockReturnValue(gameRunDeferred.promise),
      }),
    });
    await store.getState().startGame();
    store.setState({ gameMode: "pvp" }); // isAITurn() === false
    await store.getState().makeMove({ row: 2, col: 3, isAI: false });
    vi.spyOn(store.getState(), "makeAIMove").mockResolvedValue(undefined);
    const historyLen = store.getState().moveHistory.length;

    // Hint analysis in flight: a live EngineSearch run with a slow abort.
    store.setState({ isHintMode: true });
    void store.getState().analyzeBoard();
    for (let i = 0; i < 10 && !store.getState().isAnalyzing; i++) await Promise.resolve();
    expect(store.getState().isAnalyzing).toBe(true);

    // Analyze Game queues behind the slow hint-abort supersede. It must mark
    // itself pending SYNCHRONOUSLY so the board/history stay locked while the
    // backend will run against the move list captured at call time.
    const gamePending = store.getState().analyzeGame();
    expect(store.getState().isGameAnalyzing).toBe(true);

    // The user cannot mutate the history during the queued window.
    store.getState().undoMove();
    expect(store.getState().moveHistory.length).toBe(historyLen);

    abortDeferred.resolve();
    hintRunDeferred.resolve();
    gameRunDeferred.resolve();
    await gamePending;
    expect(store.getState().isGameAnalyzing).toBe(false);
  });

  it("clears the game-analysis pending flag when superseded before install", async () => {
    const hintRunDeferred = createDeferred<void>();
    const abortDeferred = createDeferred<void>();
    const gameRunDeferred = createDeferred<void>();
    const { store } = createTestStore({
      ai: createMockAIService({
        analyze: vi.fn().mockReturnValue(hintRunDeferred.promise),   // hint run stays live
        abortSearch: vi.fn().mockReturnValue(abortDeferred.promise), // SLOW supersede
        analyzeGame: vi.fn().mockReturnValue(gameRunDeferred.promise),
      }),
    });
    await store.getState().startGame();
    store.setState({ gameMode: "pvp" }); // isAITurn() === false
    await store.getState().makeMove({ row: 2, col: 3, isAI: false });
    vi.spyOn(store.getState(), "makeAIMove").mockResolvedValue(undefined);

    // Hint analysis in flight: a live EngineSearch run with a slow abort.
    store.setState({ isHintMode: true });
    void store.getState().analyzeBoard();
    for (let i = 0; i < 10 && !store.getState().isAnalyzing; i++) await Promise.resolve();
    expect(store.getState().isAnalyzing).toBe(true);

    // analyzeGame commits isGameAnalyzing synchronously (onClaim), then queues
    // behind the slow hint-abort supersede WITHOUT installing a run.
    const gamePending = store.getState().analyzeGame();
    expect(store.getState().isGameAnalyzing).toBe(true);

    // A newer engine op (abortAIMove) claims a higher generation while
    // analyzeGame is still queued — analyzeGame is now superseded BEFORE it
    // installs. abortAIMove never touches isGameAnalyzing, so only
    // analyzeGame's own guaranteed-once onTeardown can release the flag.
    const abortPending = store.getState().abortAIMove();

    abortDeferred.resolve();
    hintRunDeferred.resolve();
    gameRunDeferred.resolve();
    await Promise.all([gamePending, abortPending]);

    // The onClaim breadcrumb must be undone — the board/history must not be
    // left permanently locked by a stranded isGameAnalyzing.
    expect(store.getState().isGameAnalyzing).toBe(false);
  });

  it("resumes queued automation that fired during game analysis", async () => {
    vi.useFakeTimers();
    try {
      const analyzeDeferred = createDeferred<void>();
      const { store } = createTestStore({
        ai: createMockAIService({
          analyzeGame: vi.fn().mockImplementation(async () => {
            await analyzeDeferred.promise;
          }),
        }),
      });

      await store.getState().startGame();
      const makeAIMoveSpy = vi.spyOn(store.getState(), "makeAIMove").mockResolvedValue(undefined);

      await store.getState().makeMove({ row: 2, col: 3, isAI: false });
      expect(store.getState().currentPlayer).toBe("white");

      const analysis = store.getState().analyzeGame();
      // analyzeGame commits isGameAnalyzing synchronously via onClaim.
      for (let i = 0; i < 10 && !store.getState().isGameAnalyzing; i++) await Promise.resolve();
      expect(store.getState().isGameAnalyzing).toBe(true);

      vi.advanceTimersByTime(500);
      expect(makeAIMoveSpy).not.toHaveBeenCalled();

      analyzeDeferred.resolve();
      await analysis;

      expect(store.getState().isGameAnalyzing).toBe(false);
      expect(makeAIMoveSpy).toHaveBeenCalledTimes(1);
    } finally {
      vi.useRealTimers();
    }
  });

  it("keeps queued automation delayed when game analysis finishes before the timer fires", async () => {
    vi.useFakeTimers();
    try {
      const { store } = createTestStore();

      await store.getState().startGame();
      const makeAIMoveSpy = vi.spyOn(store.getState(), "makeAIMove").mockResolvedValue(undefined);

      await store.getState().makeMove({ row: 2, col: 3, isAI: false });
      const analysis = store.getState().analyzeGame();

      await analysis;

      expect(store.getState().isGameAnalyzing).toBe(false);
      expect(makeAIMoveSpy).not.toHaveBeenCalled();

      vi.advanceTimersByTime(400);

      expect(makeAIMoveSpy).toHaveBeenCalledTimes(1);
    } finally {
      vi.useRealTimers();
    }
  });

  it("resumes queued automation that fired before game analysis was aborted", async () => {
    vi.useFakeTimers();
    try {
      const analyzeDeferred = createDeferred<void>();
      const { store } = createTestStore({
        ai: createMockAIService({
          analyzeGame: vi.fn().mockImplementation(async () => {
            await analyzeDeferred.promise;
          }),
        }),
      });

      await store.getState().startGame();
      const makeAIMoveSpy = vi.spyOn(store.getState(), "makeAIMove").mockResolvedValue(undefined);

      await store.getState().makeMove({ row: 2, col: 3, isAI: false });
      const analysis = store.getState().analyzeGame();
      // analyzeGame commits isGameAnalyzing synchronously via onClaim.
      for (let i = 0; i < 10 && !store.getState().isGameAnalyzing; i++) await Promise.resolve();

      vi.advanceTimersByTime(500);
      expect(makeAIMoveSpy).not.toHaveBeenCalled();

      await store.getState().abortGameAnalysis();

      expect(store.getState().isGameAnalyzing).toBe(false);
      expect(makeAIMoveSpy).toHaveBeenCalledTimes(1);

      analyzeDeferred.resolve();
      await analysis;
    } finally {
      vi.useRealTimers();
    }
  });

  it("queues a resume request made during game analysis", async () => {
    const analyzeDeferred = createDeferred<void>();
    const { store } = createTestStore({
      ai: createMockAIService({
        analyzeGame: vi.fn().mockImplementation(async () => {
          await analyzeDeferred.promise;
        }),
      }),
    });

    await store.getState().startGame();
    const makeAIMoveSpy = vi.spyOn(store.getState(), "makeAIMove").mockResolvedValue(undefined);

    await store.getState().makeMove({ row: 2, col: 3, isAI: false });
    store.setState({ paused: true });

    const analysis = store.getState().analyzeGame();
    // analyzeGame commits isGameAnalyzing synchronously via onClaim.
    for (let i = 0; i < 10 && !store.getState().isGameAnalyzing; i++) await Promise.resolve();
    expect(store.getState().isGameAnalyzing).toBe(true);

    store.getState().resumeAI();

    expect(store.getState().paused).toBe(false);
    expect(store.getState().automationResumePending).toBe(true);
    expect(makeAIMoveSpy).not.toHaveBeenCalled();

    analyzeDeferred.resolve();
    await analysis;

    expect(store.getState().automationResumePending).toBe(false);
    expect(makeAIMoveSpy).toHaveBeenCalledTimes(1);
  });

  it("queues a resume request made before game analysis is aborted", async () => {
    const analyzeDeferred = createDeferred<void>();
    const { store } = createTestStore({
      ai: createMockAIService({
        analyzeGame: vi.fn().mockImplementation(async () => {
          await analyzeDeferred.promise;
        }),
      }),
    });

    await store.getState().startGame();
    const makeAIMoveSpy = vi.spyOn(store.getState(), "makeAIMove").mockResolvedValue(undefined);

    await store.getState().makeMove({ row: 2, col: 3, isAI: false });
    store.setState({ paused: true });

    const analysis = store.getState().analyzeGame();
    store.getState().resumeAI();

    await store.getState().abortGameAnalysis();

    expect(store.getState().isGameAnalyzing).toBe(false);
    expect(store.getState().automationResumePending).toBe(false);
    expect(makeAIMoveSpy).toHaveBeenCalledTimes(1);

    analyzeDeferred.resolve();
    await analysis;
  });

  it("ignores stale game-analysis progress and cleanup after abort plus restart", async () => {
    const analyzeDeferred1 = createDeferred<void>();
    const analyzeDeferred2 = createDeferred<void>();
    const analyzeCallbacks: Array<(progress: GameAnalysisProgress) => void> = [];

    const { store, services } = createTestStore({
      ai: createMockAIService({
        analyzeGame: vi
          .fn()
          .mockImplementationOnce(async (_board, _player, _moves, _level, callback) => {
            analyzeCallbacks.push(callback);
            await analyzeDeferred1.promise;
          })
          .mockImplementationOnce(async (_board, _player, _moves, _level, callback) => {
            analyzeCallbacks.push(callback);
            await analyzeDeferred2.promise;
          }),
      }),
    });

    await store.getState().startGame({
      gameMode: "pvp",
      aiLevel: 21,
      aiMode: "level",
      gameTimeLimit: 60,
    });
    await store.getState().makeMove({ row: 2, col: 3, isAI: false });

    const first = store.getState().analyzeGame();
    await Promise.resolve();
    expect(services.ai.analyzeGame).toHaveBeenCalledTimes(1);

    await store.getState().abortGameAnalysis();
    const second = store.getState().analyzeGame();
    await Promise.resolve();
    expect(services.ai.analyzeGame).toHaveBeenCalledTimes(2);

    analyzeCallbacks[0]({
      moveIndex: 0,
      bestMove: "c4",
      bestScore: 12,
      playedScore: 1,
      scoreLoss: 11,
      depth: 10,
    });
    expect(store.getState().gameAnalysisResult).toBeNull();

    analyzeCallbacks[1]({
      moveIndex: 0,
      bestMove: "d3",
      bestScore: 3,
      playedScore: 3,
      scoreLoss: 0,
      depth: 12,
    });
    expect(store.getState().gameAnalysisResult?.[0]).toMatchObject({
      bestMove: "d3",
      depth: 12,
    });

    analyzeDeferred1.resolve();
    await first;
    expect(store.getState().isGameAnalyzing).toBe(true);

    analyzeDeferred2.resolve();
    await second;
    expect(store.getState().isGameAnalyzing).toBe(false);
  });

  it("game analysis supersedes an in-flight hint analysis (no stuck isAnalyzing)", async () => {
    const hintDeferred = createDeferred<void>();
    const gameDeferred = createDeferred<void>();
    const { store } = createTestStore({
      ai: createMockAIService({
        analyze: vi.fn().mockReturnValue(hintDeferred.promise),
        analyzeGame: vi.fn().mockImplementation(async () => {
          await gameDeferred.promise;
        }),
      }),
    });
    await store.getState().startGame();
    vi.spyOn(store.getState(), "makeAIMove").mockResolvedValue(undefined);
    await store.getState().makeMove({ row: 2, col: 3, isAI: false });
    store.setState({ gameMode: "pvp", currentPlayer: "white", isHintMode: true });

    const hintPending = store.getState().analyzeBoard();
    await Promise.resolve();
    expect(store.getState().isAnalyzing).toBe(true);

    const gamePending = store.getState().analyzeGame();
    // analyzeGame marks isGameAnalyzing synchronously (onClaim); the hint run's
    // supersede (abort + teardown clearing isAnalyzing) still spans a few
    // microtasks — wait on that, the actual cross-feature handover signal.
    for (let i = 0; i < 10 && store.getState().isAnalyzing; i++) {
      await Promise.resolve();
    }

    // Shared engine: starting game analysis superseded the hint run; its
    // teardown ran exactly once. Pre-change (separate gameAnalysisSearch) this
    // assertion fails because isAnalyzing is still true.
    expect(store.getState().isAnalyzing).toBe(false);
    expect(store.getState().isGameAnalyzing).toBe(true);

    hintDeferred.resolve();
    await hintPending;
    expect(store.getState().isAnalyzing).toBe(false); // stays cleared

    gameDeferred.resolve();
    await gamePending;
  });
});
