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
    // A hint analysis is the current Engine Activity.
    store.setState({
      isHintMode: true,
      engineActivity: { kind: "hint", runId: 1 },
      analyzeResults: new Map([["2,3", {} as never]]),
    });

    store.getState().setHintMode(false);
    // EngineSearch.abort awaits supersede() before onAbort, so the abort-pending
    // snapshot is observable one microtask later.
    await Promise.resolve();

    expect(store.getState().isHintMode).toBe(false);
    expect(store.getState().engineActivity.kind).toBe("idle");
    expect(store.getState().hintAnalysisAbortPending).toBe(true);
    expect(store.getState().analyzeResults).toBeNull();
    expect(services.ai.abortSearch).toHaveBeenCalledTimes(1);
  });

  it("does not abort an AI move when disabling hint mode", () => {
    const { store, services } = createTestStore();
    store.setState({
      isHintMode: true,
      engineActivity: { kind: "ai-move", runId: 1 },
    });

    store.getState().setHintMode(false);

    expect(store.getState().isHintMode).toBe(false);
    expect(store.getState().engineActivity.kind).toBe("ai-move");
    expect(services.ai.abortSearch).not.toHaveBeenCalled();
  });

  it("waits for abortSearch to finish before clearing the hint abort guard", async () => {
    const abortDeferred = createDeferred<void>();
    const { store } = createTestStore({
      ai: createMockAIService({
        abortSearch: vi.fn().mockReturnValue(abortDeferred.promise),
      }),
    });

    store.setState({
      isHintMode: true,
      engineActivity: { kind: "hint", runId: 1 },
    });

    store.getState().setHintMode(false);
    await Promise.resolve();
    // Engine Activity is idle at abort request, but the dedupe guard persists
    // until the backend abort resolves.
    expect(store.getState().engineActivity.kind).toBe("idle");
    expect(store.getState().hintAnalysisAbortPending).toBe(true);

    abortDeferred.resolve();
    await abortDeferred.promise;
    await Promise.resolve();

    expect(store.getState().hintAnalysisAbortPending).toBe(false);
    expect(store.getState().engineActivity.kind).toBe("idle");
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
      engineActivity: { kind: "hint", runId: 1 },
    });

    store.getState().setHintMode(false);
    store.getState().setHintMode(true);

    expect(analyzeBoardSpy).not.toHaveBeenCalled();

    abortDeferred.resolve();
    await abortDeferred.promise;
    await Promise.resolve();

    expect(analyzeBoardSpy).toHaveBeenCalledTimes(1);
  });

  it("a stale superseded hint abort does not clear the guard the newer pending abort owns", async () => {
    // The `hintAnalysisAbortPending` dedupe guard is NOT owned by the Engine
    // Activity (it survives the busy-flag migration); its own generation
    // guard must still prevent a stale, superseded hint abort from dropping
    // the guard a newer pending hint abort still owns.
    const backendAbort = createDeferred<void>();
    const { store } = createTestStore({
      ai: createMockAIService({
        abortSearch: vi.fn().mockReturnValue(backendAbort.promise), // slow
      }),
    });
    store.setState({ isHintMode: true });

    // Two overlapping hint aborts: #2's claim bumps the generation, so #1 is
    // superseded and bails to its guaranteed-once onTeardown while #2 owns
    // the guard.
    store.getState().restartHintAnalysisAfterAbort();
    store.getState().restartHintAnalysisAfterAbort();
    expect(store.getState().hintAnalysisAbortPending).toBe(true);

    // Drain microtasks: #1's superseded onTeardown runs here. Its generation
    // is stale, so it must NOT clear the guard #2 still owns.
    for (let i = 0; i < 5; i++) await Promise.resolve();
    expect(store.getState().hintAnalysisAbortPending).toBe(true);

    // #2 settles normally: only now is the guard released.
    backendAbort.resolve();
    for (let i = 0; i < 10 && store.getState().hintAnalysisAbortPending; i++) {
      await Promise.resolve();
    }
    expect(store.getState().hintAnalysisAbortPending).toBe(false);
  });

  it("ignores stale hint progress after a cancelled analysis is re-enabled", async () => {
    const analyzeDeferred1 = createDeferred<void>();
    const analyzeDeferred2 = createDeferred<void>();
    const abortDeferred = createDeferred<void>();
    const analyzeCallbacks: Array<
      (progress: {
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
      }) => void
    > = [];

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
    for (
      let i = 0;
      i < 10 && (services.ai.analyze as ReturnType<typeof vi.fn>).mock.calls.length < 2;
      i++
    ) {
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
    store.setState({
      isHintMode: true,
      engineActivity: { kind: "hint", runId: 1 },
    });

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
    for (let i = 0; i < 10 && store.getState().hintAnalysisAbortPending; i++)
      await Promise.resolve();
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

    // Game analysis claims activity synchronously; wait until its backend run
    // starts so the hint claim captures it as the prior run.
    void store.getState().analyzeGame();
    for (let i = 0; i < 10 && analyzeGameMock.mock.calls.length === 0; i++) await Promise.resolve();
    expect(store.getState().engineActivity.kind).toBe("game-analysis");

    // Enabling hint mode queues a hint analyzeBoard start; its claim()
    // supersedes the live game run and stamps the hint activity synchronously
    // (CONTEXT.md → Engine Activity), then stalls on the SLOW abortGameAnalysis.
    store.getState().setHintMode(true);
    await Promise.resolve();
    expect(store.getState().engineActivity.kind).toBe("hint");

    // User turns hint mode off while the hint start is still queued. The hint
    // activity is current, so this supersedes the queued hint start via a
    // restart-abort.
    store.getState().setHintMode(false);
    expect(store.getState().isHintMode).toBe(false);

    // Slow supersede resolves: the queued hint start never launches a backend
    // search and its activity is not stranded.
    gameAbortDeferred.resolve();
    for (let i = 0; i < 12; i++) await Promise.resolve();

    expect(analyzeSpy).not.toHaveBeenCalled();
    expect(store.getState().engineActivity.kind).toBe("idle");

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
    expect(store.getState().engineActivity.kind).toBe("ai-move");

    store.getState().setHintMode(false);
    await Promise.resolve();

    expect(services.ai.abortSearch).not.toHaveBeenCalled();
    expect(store.getState().engineActivity.kind).toBe("ai-move");

    moveDeferred.resolve(null);
    await aiPending;
  });
});

describe("Engine Activity ownership", () => {
  it("a stale superseded hint run does not clear the newer game-analysis activity", async () => {
    // The invariant the deleted hint generation counter used to enforce, now
    // owned by the Engine Activity (CONTEXT.md → Engine Activity): a
    // superseded run's teardown returns the activity to idle ONLY while it is
    // still the current run, so it can never clobber a newer run's activity.
    const hintRunDeferred = createDeferred<void>();
    const gameRunDeferred = createDeferred<void>();
    const analyzeMock = vi.fn().mockReturnValue(hintRunDeferred.promise); // hint run stays live
    const analyzeGameMock = vi.fn().mockReturnValue(gameRunDeferred.promise); // newer run
    const { store } = createTestStore({
      ai: createMockAIService({
        analyze: analyzeMock,
        analyzeGame: analyzeGameMock,
      }),
    });
    await store.getState().startGame();
    store.setState({ gameMode: "pvp" }); // isAITurn() === false
    await store.getState().makeMove({ row: 2, col: 3, isAI: false });
    vi.spyOn(store.getState(), "makeAIMove").mockResolvedValue(undefined);

    // A hint analysis is the live Engine Activity.
    store.getState().setHintMode(true);
    for (let i = 0; i < 10 && analyzeMock.mock.calls.length === 0; i++) await Promise.resolve();
    expect(store.getState().engineActivity.kind).toBe("hint");

    // Game analysis supersedes the hint run and becomes the current activity.
    void store.getState().analyzeGame();
    for (let i = 0; i < 10 && analyzeGameMock.mock.calls.length === 0; i++) await Promise.resolve();
    expect(store.getState().engineActivity.kind).toBe("game-analysis");

    // The superseded hint run resolves late: its teardown must NOT return the
    // activity to idle — the newer game-analysis run owns it now.
    hintRunDeferred.resolve();
    for (let i = 0; i < 20; i++) await Promise.resolve();
    expect(store.getState().engineActivity.kind).toBe("game-analysis");

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
        analyze: vi.fn().mockReturnValue(hintRunDeferred.promise), // hint run stays live
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
    for (let i = 0; i < 10 && store.getState().engineActivity.kind !== "hint"; i++)
      await Promise.resolve();
    expect(store.getState().engineActivity.kind).toBe("hint");

    // Analyze Game queues behind the slow hint-abort supersede. It must mark
    // itself pending SYNCHRONOUSLY so the board/history stay locked while the
    // backend will run against the move list captured at call time.
    const gamePending = store.getState().analyzeGame();
    expect(store.getState().engineActivity.kind).toBe("game-analysis");

    // The user cannot mutate the history during the queued window.
    store.getState().undoMove();
    expect(store.getState().moveHistory.length).toBe(historyLen);

    abortDeferred.resolve();
    hintRunDeferred.resolve();
    gameRunDeferred.resolve();
    await gamePending;
    expect(store.getState().engineActivity.kind).toBe("idle");
  });

  it("clears the game-analysis pending flag when superseded before install", async () => {
    const hintRunDeferred = createDeferred<void>();
    const abortDeferred = createDeferred<void>();
    const gameRunDeferred = createDeferred<void>();
    const { store } = createTestStore({
      ai: createMockAIService({
        analyze: vi.fn().mockReturnValue(hintRunDeferred.promise), // hint run stays live
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
    for (let i = 0; i < 10 && store.getState().engineActivity.kind !== "hint"; i++)
      await Promise.resolve();
    expect(store.getState().engineActivity.kind).toBe("hint");

    // Game analysis claims activity synchronously, then queues behind the
    // slow hint-abort supersede without installing a run.
    const gamePending = store.getState().analyzeGame();
    expect(store.getState().engineActivity.kind).toBe("game-analysis");

    // A newer engine operation claims a higher generation before game analysis
    // installs; guaranteed teardown must release its activity.
    const abortPending = store.getState().abortAIMove();

    abortDeferred.resolve();
    hintRunDeferred.resolve();
    gameRunDeferred.resolve();
    await Promise.all([gamePending, abortPending]);

    expect(store.getState().engineActivity.kind).toBe("idle");
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
      for (let i = 0; i < 10 && store.getState().engineActivity.kind !== "game-analysis"; i++)
        await Promise.resolve();
      expect(store.getState().engineActivity.kind).toBe("game-analysis");

      vi.advanceTimersByTime(500);
      expect(makeAIMoveSpy).not.toHaveBeenCalled();

      analyzeDeferred.resolve();
      await analysis;

      expect(store.getState().engineActivity.kind).toBe("idle");
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

      expect(store.getState().engineActivity.kind).toBe("idle");
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
      for (let i = 0; i < 10 && store.getState().engineActivity.kind !== "game-analysis"; i++)
        await Promise.resolve();

      vi.advanceTimersByTime(500);
      expect(makeAIMoveSpy).not.toHaveBeenCalled();

      await store.getState().abortGameAnalysis();

      expect(store.getState().engineActivity.kind).toBe("idle");
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
    for (let i = 0; i < 10 && store.getState().engineActivity.kind !== "game-analysis"; i++)
      await Promise.resolve();
    expect(store.getState().engineActivity.kind).toBe("game-analysis");

    store.getState().resumeAI();

    expect(store.getState().paused).toBe(false);
    // The resume is deferred (queue flag private to Automation); the
    // observable proxy is that the AI does not move until analysis ends.
    expect(makeAIMoveSpy).not.toHaveBeenCalled();

    analyzeDeferred.resolve();
    await analysis;

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

    expect(store.getState().engineActivity.kind).toBe("idle");
    // Resume ran exactly once (queue flag is private to Automation).
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
      pvLine: "c4",
    });
    expect(store.getState().gameAnalysisResult).toBeNull();

    analyzeCallbacks[1]({
      moveIndex: 0,
      bestMove: "d3",
      bestScore: 3,
      playedScore: 3,
      scoreLoss: 0,
      depth: 12,
      pvLine: "d3",
    });
    expect(store.getState().gameAnalysisResult?.[0]).toMatchObject({
      bestMove: "d3",
      depth: 12,
    });

    analyzeDeferred1.resolve();
    await first;
    expect(store.getState().engineActivity.kind).toBe("game-analysis");

    analyzeDeferred2.resolve();
    await second;
    expect(store.getState().engineActivity.kind).toBe("idle");
  });

  it("game analysis supersedes an in-flight hint analysis", async () => {
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
    expect(store.getState().engineActivity.kind).toBe("hint");

    const gamePending = store.getState().analyzeGame();
    // Game analysis synchronously takes ownership from the hint run.
    expect(store.getState().engineActivity.kind).toBe("game-analysis");

    hintDeferred.resolve();
    await hintPending;
    expect(store.getState().engineActivity.kind).toBe("game-analysis");

    gameDeferred.resolve();
    await gamePending;
  });
});
