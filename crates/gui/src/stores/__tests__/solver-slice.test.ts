import { afterAll, beforeEach, describe, expect, it, vi } from "vitest";
import { createDeferred, createTestStore, type TestStore } from "./test-helpers";
import { createMockAIService } from "@/services/mock-ai-service";
import { createMockSolverService } from "@/services/mock-solver-service";
import { getNotation, initializeBoard } from "@/domain/game/game-logic";
import { applyMove } from "@/domain/game/store-helpers";
import type { Board, Player } from "@/domain/game/types";
import type { SolverCandidate, SolverProgressPayload } from "@/services/types";
import type { SolverHistoryEntry } from "@/stores/slices/types";

const consoleErrorSpy = vi.spyOn(console, "error").mockImplementation(() => {});

afterAll(() => {
  consoleErrorSpy.mockRestore();
});

// Allow individual tests to override `getValidMoves` (e.g. to simulate a
// position with no legal moves for the next player). Tests that don't set a
// stub fall through to the real implementation.
let getValidMovesStub: ((board: Board, player: Player) => [number, number][]) | null = null;

vi.mock("@/domain/game/game-logic", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/domain/game/game-logic")>();
  return {
    ...actual,
    getValidMoves: (board: Board, player: Player) =>
      (getValidMovesStub ?? actual.getValidMoves)(board, player),
  };
});

beforeEach(() => {
  vi.clearAllMocks();
  getValidMovesStub = null;
});

function buildHistoryEntry(
  board: Board,
  player: Player,
  moveFrom: string | null,
): SolverHistoryEntry {
  return { board, player, moveFrom };
}

function latestSolverRunId(services: ReturnType<typeof createTestStore>["services"]): number {
  const calls = vi.mocked(services.solver.startSearch).mock.calls;
  const runId = calls[calls.length - 1]?.[4];
  if (typeof runId !== "number") {
    throw new Error("No solver search run id was captured");
  }
  return runId;
}

function startSolverFromPosition(store: TestStore, board: Board, player: Player): Promise<boolean> {
  store.setState({
    setupTab: "manual",
    setupBoard: board,
    setupCurrentPlayer: player,
    setupError: null,
  });
  return store.getState().startSolverFromSetup();
}

describe("solver modal", () => {
  it("resets setup when opened and closes", () => {
    const { store } = createTestStore();
    const resetSetupSpy = vi.spyOn(store.getState(), "resetSetup");

    store.getState().openSolverModal();
    expect(resetSetupSpy).toHaveBeenCalledTimes(1);
    expect(store.getState().isSolverModalOpen).toBe(true);

    store.getState().closeSolverModal();
    expect(store.getState().isSolverModalOpen).toBe(false);
  });
});

describe("subscribeSolverProgress", () => {
  it("routes injected solver progress through the Solver Session", async () => {
    const progressCallbacks: Array<(payload: SolverProgressPayload) => void> = [];
    const unlisten = vi.fn();
    const { store, services } = createTestStore({
      solver: createMockSolverService({
        onProgress: vi.fn(async (callback) => {
          progressCallbacks.push(callback);
          return unlisten;
        }),
      }),
    });
    const board = initializeBoard();
    await startSolverFromPosition(store, board, "black");

    const unsubscribe = await store.getState().subscribeSolverProgress();
    expect(progressCallbacks).toHaveLength(1);
    progressCallbacks[0]({
      runId: latestSolverRunId(services),
      bestMove: "d3",
      row: 2,
      col: 3,
      score: 4,
      depth: 14,
      targetDepth: 14,
      acc: 100,
      nodes: 1000,
      pvLine: "d3",
      isEndgame: false,
    });
    unsubscribe();

    expect(store.getState().solverCandidates.get("2,3")?.move).toBe("d3");
    expect(unlisten).toHaveBeenCalledTimes(1);
  });
});

describe("startSolverFromSetup", () => {
  it("resolves the manual tab and starts the solver", async () => {
    const { store, services } = createTestStore();
    const board = initializeBoard();
    store.setState({
      setupTab: "manual",
      setupBoard: board,
      setupCurrentPlayer: "black",
      setupError: null,
    });

    const result = await store.getState().startSolverFromSetup();

    expect(result).toBe(true);
    const state = store.getState();
    expect(state.isSolverActive).toBe(true);
    expect(state.solverHistory).toHaveLength(1);
    expect(state.setupError).toBeNull();
    expect(services.solver.startSearch).toHaveBeenCalledTimes(1);
  });
});

describe("exitSolver", () => {
  it("aborts the search and clears solver state while preserving targetSelectivity", async () => {
    const { store, services } = createTestStore();
    const board = initializeBoard();
    const rootEntry = buildHistoryEntry(board, "black", null);
    const candidates = new Map<string, SolverCandidate>([
      [
        "2,3",
        {
          move: "d3",
          row: 2,
          col: 3,
          score: 4,
          depth: 14,
          targetDepth: 14,
          acc: 100,
          pvLine: "d3",
          isEndgame: true,
          isComplete: true,
        },
      ],
    ]);
    store.setState({
      isSolverActive: true,
      solverHistory: [rootEntry],
      solverCandidates: candidates,
      engineActivity: { kind: "solver", runId: 1 },
      targetSelectivity: 95,
    });

    await store.getState().exitSolver();

    expect(services.solver.abort).toHaveBeenCalledTimes(1);
    const state = store.getState();
    expect(state.isSolverActive).toBe(false);
    expect(state.solverHistory).toEqual([]);
    expect(state.solverCandidates.size).toBe(0);
    expect(state.engineActivity.kind).toBe("idle");
    expect(state.targetSelectivity).toBe(95);
  });
});

describe("advanceSolver", () => {
  it("applies the move, pushes history, and re-runs the search", async () => {
    const { store, services } = createTestStore();
    const board = initializeBoard();
    const rootEntry = buildHistoryEntry(board, "black", null);
    store.setState({
      isSolverActive: true,
      solverHistory: [rootEntry],
      solverCandidates: new Map([
        [
          "0,0",
          {
            move: "a1",
            row: 0,
            col: 0,
            score: 0,
            depth: 1,
            targetDepth: 1,
            acc: 100,
            pvLine: "a1",
            isEndgame: true,
            isComplete: true,
          },
        ],
      ]),
    });

    await store.getState().advanceSolver(2, 3);

    expect(services.solver.abort).toHaveBeenCalledTimes(1);

    const state = store.getState();
    expect(state.solverHistory).toHaveLength(2);
    expect(state.solverHistory[1].moveFrom).toBe(getNotation(2, 3));
    expect(state.solverHistory[state.solverHistory.length - 1]?.player).toBe("white");

    // The board should match applying the move on the original board.
    const expectedBoard = applyMove(board, { row: 2, col: 3, isAI: false, score: 0 }, "black");
    expect(state.solverHistory[state.solverHistory.length - 1]?.board).toEqual(expectedBoard);

    expect(state.solverCandidates.size).toBe(0);
    expect(state.engineActivity.kind).toBe("idle");

    expect(services.solver.startSearch).toHaveBeenCalledTimes(1);
    expect(services.solver.startSearch).toHaveBeenCalledWith(
      expectedBoard,
      "white",
      100,
      "multiPv",
      expect.any(Number),
    );
  });

  it("skips the search when the new position has no valid moves", async () => {
    const { store, services } = createTestStore();
    const board = initializeBoard();
    store.setState({
      isSolverActive: true,
      solverHistory: [buildHistoryEntry(board, "black", null)],
      solverCandidates: new Map(),
    });

    // Pretend the position after the move has zero legal moves for any player.
    getValidMovesStub = () => [];

    await store.getState().advanceSolver(2, 3);

    const state = store.getState();
    expect(state.engineActivity.kind).toBe("idle");
    expect(services.solver.startSearch).not.toHaveBeenCalled();
    // The advance itself still happened.
    expect(state.solverHistory).toHaveLength(2);
    expect(state.solverCandidates.size).toBe(0);
    // Both players empty ↁEgameOver, no auto-pass, turn stays flipped.
    expect(state.solverHistory[state.solverHistory.length - 1]?.player).toBe("white");
  });

  it("auto-passes when the next player has no moves but the current player still does", async () => {
    const { store, services } = createTestStore();
    const board = initializeBoard();
    store.setState({
      isSolverActive: true,
      solverHistory: [buildHistoryEntry(board, "black", null)],
      solverCandidates: new Map(),
    });

    // White (next player) has no moves, but black still does.
    getValidMovesStub = (_board, player) => (player === "white" ? [] : [[2, 3]]);

    await store.getState().advanceSolver(2, 3);

    const state = store.getState();
    expect(state.solverHistory).toHaveLength(2);
    // Auto-pass flipped the turn back to black.
    expect(state.solverHistory[state.solverHistory.length - 1]?.player).toBe("black");
    expect(state.engineActivity.kind).toBe("idle");

    const expectedBoard = applyMove(board, { row: 2, col: 3, isAI: false, score: 0 }, "black");
    expect(state.solverHistory[state.solverHistory.length - 1]?.board).toEqual(expectedBoard);
    expect(services.solver.startSearch).toHaveBeenCalledWith(
      expectedBoard,
      "black",
      100,
      "multiPv",
      expect.any(Number),
    );
  });
});

describe("undoSolver", () => {
  it("pops the last history entry and re-runs the search with the previous position", async () => {
    const { store, services } = createTestStore();
    const rootBoard = initializeBoard();
    const secondBoard = applyMove(rootBoard, { row: 2, col: 3, isAI: false, score: 0 }, "black");

    store.setState({
      isSolverActive: true,
      solverHistory: [
        buildHistoryEntry(rootBoard, "black", null),
        buildHistoryEntry(secondBoard, "white", "d3"),
      ],
      solverCandidates: new Map([
        [
          "2,2",
          {
            move: "c3",
            row: 2,
            col: 2,
            score: -2,
            depth: 12,
            targetDepth: 12,
            acc: 100,
            pvLine: "c3",
            isEndgame: true,
            isComplete: true,
          },
        ],
      ]),
    });

    await store.getState().undoSolver();

    expect(services.solver.abort).toHaveBeenCalledTimes(1);
    const state = store.getState();
    expect(state.solverHistory).toHaveLength(1);
    expect(state.solverHistory[state.solverHistory.length - 1]?.board).toBe(rootBoard);
    expect(state.solverHistory[state.solverHistory.length - 1]?.player).toBe("black");
    expect(state.solverCandidates.size).toBe(0);
    expect(state.engineActivity.kind).toBe("idle");
    expect(services.solver.startSearch).toHaveBeenCalledWith(
      rootBoard,
      "black",
      100,
      "multiPv",
      expect.any(Number),
    );
  });

  it("does not drop the second of two concurrent Back presses", async () => {
    const abortDeferred = createDeferred<void>();
    const abortMock = vi.fn().mockReturnValue(abortDeferred.promise);
    const { store } = createTestStore({
      solver: createMockSolverService({ abort: abortMock }),
    });

    const rootBoard = initializeBoard();
    const secondBoard = applyMove(rootBoard, { row: 2, col: 3, isAI: false, score: 0 }, "black");
    const thirdBoard = applyMove(secondBoard, { row: 2, col: 2, isAI: false, score: 0 }, "white");

    store.setState({
      isSolverActive: true,
      solverHistory: [
        buildHistoryEntry(rootBoard, "black", null),
        buildHistoryEntry(secondBoard, "white", "d3"),
        buildHistoryEntry(thirdBoard, "black", "c3"),
      ],
      solverCandidates: new Map(),
    });

    const firstUndo = store.getState().undoSolver();
    const secondUndo = store.getState().undoSolver();

    await Promise.resolve();
    abortDeferred.resolve();
    await firstUndo;
    await secondUndo;

    const state = store.getState();
    expect(state.solverHistory).toHaveLength(1);
    expect(state.solverHistory[state.solverHistory.length - 1]?.board).toBe(rootBoard);
    expect(state.solverHistory[state.solverHistory.length - 1]?.player).toBe("black");
  });

  it("is a no-op when history length is 1", async () => {
    const { store, services } = createTestStore();
    const board = initializeBoard();
    store.setState({
      isSolverActive: true,
      solverHistory: [buildHistoryEntry(board, "black", null)],
    });

    await store.getState().undoSolver();

    expect(services.solver.abort).not.toHaveBeenCalled();
    expect(services.solver.startSearch).not.toHaveBeenCalled();
    expect(store.getState().solverHistory).toHaveLength(1);
  });
});

describe("setTargetSelectivity", () => {
  it("updates state and restarts the search when solver is active and searching", async () => {
    const { store, services } = createTestStore();
    const board = initializeBoard();
    store.setState({
      isSolverActive: true,
      solverHistory: [buildHistoryEntry(board, "black", null)],
      solverCandidates: new Map([
        [
          "2,3",
          {
            move: "d3",
            row: 2,
            col: 3,
            score: 2,
            depth: 10,
            targetDepth: 10,
            acc: 99,
            pvLine: "d3",
            isEndgame: true,
            isComplete: false,
          },
        ],
      ]),
      engineActivity: { kind: "solver", runId: 1 },
    });

    await store.getState().setTargetSelectivity(95);

    const state = store.getState();
    expect(state.targetSelectivity).toBe(95);
    expect(services.solver.abort).toHaveBeenCalledTimes(1);
    expect(services.solver.startSearch).toHaveBeenCalledTimes(1);
    expect(services.solver.startSearch).toHaveBeenCalledWith(
      board,
      "black",
      95,
      "multiPv",
      expect.any(Number),
    );
    expect(state.solverCandidates.size).toBe(0);
    expect(state.engineActivity.kind).toBe("idle");
  });

  it("updates without searching when solver is inactive", async () => {
    const { store, services } = createTestStore();

    await store.getState().setTargetSelectivity(95);

    expect(store.getState().targetSelectivity).toBe(95);
    expect(services.solver.abort).not.toHaveBeenCalled();
    expect(services.solver.startSearch).not.toHaveBeenCalled();
  });

  it("restarts the search when solver is active even if not currently searching", async () => {
    const { store, services } = createTestStore();
    const board = initializeBoard();

    // The initial synchronous mock search has already completed.
    await startSolverFromPosition(store, board, "black");
    vi.mocked(services.solver.startSearch).mockClear();
    vi.mocked(services.solver.abort).mockClear();

    await store.getState().setTargetSelectivity(95);

    expect(store.getState().targetSelectivity).toBe(95);
    expect(services.solver.abort).toHaveBeenCalledTimes(1);
    expect(services.solver.startSearch).toHaveBeenCalledTimes(1);
    expect(services.solver.startSearch).toHaveBeenCalledWith(
      expect.anything(),
      "black",
      95,
      "multiPv",
      expect.any(Number),
    );
    expect(store.getState().engineActivity.kind).toBe("idle");
  });
});

describe("applySolverProgress", () => {
  it("drops payloads when solver mode is inactive", async () => {
    const { store, services } = createTestStore();

    const payload: SolverProgressPayload = {
      runId: 1,
      bestMove: "d3",
      row: 2,
      col: 3,
      score: 4,
      depth: 14,
      targetDepth: 14,
      acc: 100,
      nodes: 1000,
      pvLine: "d3",
      isEndgame: true,
    };

    // Solver inactive  Epayload dropped.
    store.getState().applySolverProgress(payload);
    expect(store.getState().solverCandidates.size).toBe(0);

    // Active, still searching  Eaccepted.
    await startSolverFromPosition(store, initializeBoard(), "black");
    store.getState().applySolverProgress({
      ...payload,
      runId: latestSolverRunId(services),
    });
    expect(store.getState().solverCandidates.size).toBe(1);
  });

  it("drops payloads from a superseded run", async () => {
    // Regression guard for the Codex review finding: late solver-progress
    // events from an aborted run must not leak into the state of the
    // newly-started run, even though `isSolverActive` stays true across
    // root-start/undo/reset/selectivity-change restarts.
    const { store, services } = createTestStore();
    await startSolverFromPosition(store, initializeBoard(), "black");
    const staleRunId = latestSolverRunId(services);
    await store.getState().setTargetSelectivity(95);
    const currentRunId = latestSolverRunId(services);

    const stalePayload: SolverProgressPayload = {
      runId: staleRunId, // emitted by the previous run, arrived late
      bestMove: "d3",
      row: 2,
      col: 3,
      score: 4,
      depth: 14,
      targetDepth: 14,
      acc: 100,
      nodes: 1000,
      pvLine: "d3",
      isEndgame: true,
    };

    store.getState().applySolverProgress(stalePayload);
    expect(store.getState().solverCandidates.size).toBe(0);

    // A payload from the current run still lands.
    store.getState().applySolverProgress({ ...stalePayload, runId: currentRunId });
    expect(store.getState().solverCandidates.size).toBe(1);
  });

  it("accepts trailing payloads after solver activity clears", async () => {
    // Solver progress can still be queued on the JS side after startSearch
    // resolves; those payloads must still reach the store.
    const { store, services } = createTestStore();
    await startSolverFromPosition(store, initializeBoard(), "black");

    const payload: SolverProgressPayload = {
      runId: latestSolverRunId(services),
      bestMove: "d3",
      row: 2,
      col: 3,
      score: 4,
      depth: 14,
      targetDepth: 14,
      acc: 100,
      nodes: 1000,
      pvLine: "d3",
      isEndgame: true,
    };

    store.getState().applySolverProgress(payload);
    expect(store.getState().solverCandidates.size).toBe(1);
    expect(store.getState().solverCandidates.get("2,3")?.isComplete).toBe(true);
  });
});

describe("runSolverSearch error handling", () => {
  it("returns activity to idle when startSearch fails", async () => {
    const { store, services } = createTestStore({
      solver: createMockSolverService({
        startSearch: vi.fn().mockRejectedValue(new Error("boom")),
      }),
    });

    const consoleErrorSpy = vi.spyOn(console, "error").mockImplementation(() => {});

    const board = initializeBoard();
    await startSolverFromPosition(store, board, "black");

    expect(store.getState().engineActivity.kind).toBe("idle");
    // Solver stays active; only search activity ends.
    expect(store.getState().isSolverActive).toBe(true);
    expect(consoleErrorSpy).toHaveBeenCalled();
    expect(services.solver.startSearch).toHaveBeenCalledTimes(1);

    consoleErrorSpy.mockRestore();
  });

  it("stale search errors do not clobber a newer search", async () => {
    const firstDeferred = createDeferred<void>();
    const secondDeferred = createDeferred<void>();
    const startSearchMock = vi
      .fn()
      .mockImplementationOnce(() => firstDeferred.promise)
      .mockImplementationOnce(() => secondDeferred.promise);

    const { store } = createTestStore({
      solver: createMockSolverService({
        startSearch: startSearchMock,
      }),
    });

    const consoleErrorSpy = vi.spyOn(console, "error").mockImplementation(() => {});

    // Flush pending microtasks until `predicate` is true, up to 50 ticks.
    const flushUntil = async (predicate: () => boolean) => {
      for (let i = 0; i < 50; i++) {
        if (predicate()) return;
        await Promise.resolve();
      }
      throw new Error("flushUntil: predicate never became true");
    };

    const board = initializeBoard();

    // Kick off the first search (its startSearch promise hangs).
    const firstPromise = startSolverFromPosition(store, board, "black");

    // Wait until the first search has actually reached startSearch.
    await flushUntil(() => startSearchMock.mock.calls.length === 1);
    expect(store.getState().engineActivity.kind).toBe("solver");
    const firstRunId = startSearchMock.mock.calls[0][4] as number;

    // Kick off a second search before the first settles. This simulates
    // the race: the user clicks again while a search is in flight. The
    // second call bumps the run id so the first's eventual rejection cannot
    // clear the newer activity.
    const secondPromise = startSolverFromPosition(store, board, "white");

    // Wait until the second search has reached startSearch too.
    await flushUntil(() => startSearchMock.mock.calls.length === 2);
    const secondRunId = startSearchMock.mock.calls[1][4] as number;
    expect(secondRunId).toBeGreaterThan(firstRunId);
    expect(store.getState().engineActivity.kind).toBe("solver");

    // Reject the first; its stale teardown must leave newer activity alone.
    firstDeferred.reject(new Error("aborted"));
    await firstPromise;

    // The newer run is still active (its promise hasn't settled yet).
    expect(store.getState().engineActivity.kind).toBe("solver");
    store.getState().applySolverProgress({
      runId: firstRunId,
      bestMove: "d3",
      row: 2,
      col: 3,
      score: 4,
      depth: 14,
      targetDepth: 14,
      acc: 100,
      nodes: 1000,
      pvLine: "d3",
      isEndgame: true,
    });
    expect(store.getState().solverCandidates.size).toBe(0);

    // Now let the second run resolve cleanly.
    secondDeferred.resolve();
    await secondPromise;

    expect(startSearchMock.mock.calls[1][4]).toBe(secondRunId);

    consoleErrorSpy.mockRestore();
  });
});

describe("solver result cache", () => {
  it("does not share cached candidates across store instances", async () => {
    const board = initializeBoard();
    let firstStore: ReturnType<typeof createTestStore>["store"] | null = null;

    const startSearchMock = vi.fn((_board, _player, _selectivity, _mode, runId) => {
      firstStore!.getState().applySolverProgress({
        runId,
        bestMove: "d3",
        row: 2,
        col: 3,
        score: 4,
        depth: 14,
        targetDepth: 14,
        acc: 100,
        nodes: 1000,
        pvLine: "d3",
        isEndgame: false,
      });
      return Promise.resolve();
    });

    const first = createTestStore({
      solver: createMockSolverService({ startSearch: startSearchMock }),
    });
    firstStore = first.store;

    await startSolverFromPosition(first.store, board, "black");
    expect(first.store.getState().solverCandidates.size).toBe(1);
    expect(startSearchMock).toHaveBeenCalledTimes(1);

    const secondStartSearchMock = vi.fn().mockResolvedValue(undefined);
    const second = createTestStore({
      solver: createMockSolverService({ startSearch: secondStartSearchMock }),
    });
    second.store.setState({
      isSolverActive: true,
      solverHistory: [buildHistoryEntry(board, "black", null)],
      targetSelectivity: 100,
      solverMode: "multiPv",
      isSolverStopped: true,
    });

    await second.store.getState().resumeSolverSearch();

    expect(secondStartSearchMock).toHaveBeenCalledTimes(1);
    expect(second.store.getState().solverCandidates.size).toBe(0);
  });
});

describe("shared EngineSearch cross-feature supersede", () => {
  it("starting an AI move supersedes an in-flight solver search and filters its stale progress", async () => {
    // The solver and AI share one EngineSearch instance (created once per
    // store). A solver search in flight is the live run; starting an AI
    // move supersedes it: the solver's registered abort runs, the
    // generation is bumped, so a late solver-progress stamped with the
    // superseded run's id is filtered by engineSearch.accepts and cannot
    // mutate solverCandidates.
    const startSearchDeferred = createDeferred<void>();
    const getAIMoveDeferred = createDeferred<null>();
    let solverRunId = -1;
    const startSearch = vi.fn((_b: Board, _p: Player, _s: number, _m: string, runId: number) => {
      solverRunId = runId;
      return startSearchDeferred.promise;
    });
    const { store, services } = createTestStore({
      solver: createMockSolverService({ startSearch }),
      ai: createMockAIService({
        getAIMove: vi.fn().mockReturnValue(getAIMoveDeferred.promise),
      }),
    });
    const board = initializeBoard();

    const solverPending = startSolverFromPosition(store, board, "black");
    for (let i = 0; i < 20 && solverRunId < 0; i++) await Promise.resolve();
    expect(store.getState().engineActivity.kind).toBe("solver");

    // Starting an AI move supersedes the in-flight solver run. Drain
    // microtasks until the AI search has taken over.
    const aiPending = store.getState().makeAIMove();
    for (let i = 0; i < 20 && store.getState().engineActivity.kind !== "ai-move"; i++) {
      await Promise.resolve();
    }
    expect(store.getState().engineActivity.kind).toBe("ai-move");
    // The solver run's registered abort fired during the supersede.
    expect(services.solver.abort).toHaveBeenCalled();

    // A late solver-progress stamped with the superseded run's id must NOT
    // mutate solverCandidates (filtered by engineSearch.accepts).
    store.getState().applySolverProgress({
      runId: solverRunId,
      bestMove: "d3",
      row: 2,
      col: 3,
      score: 4,
      depth: 14,
      targetDepth: 14,
      acc: 100,
      nodes: 1000,
      pvLine: "d3",
      isEndgame: true,
    });
    expect(store.getState().solverCandidates.size).toBe(0);

    getAIMoveDeferred.resolve(null);
    await aiPending;
    expect(store.getState().engineActivity.kind).toBe("idle");

    startSearchDeferred.resolve();
    await solverPending;
    expect(store.getState().engineActivity.kind).toBe("idle");
    expect(store.getState().solverCandidates.size).toBe(0);
  });

  it("a cache-hit navigation supersedes the prior run so its late progress cannot overwrite the cached candidates", async () => {
    // Note: a genuine RED-first (cache-hit as a bare commit) could not be
    // constructed through the public store API — the shared engine's
    // generation always advances past the in-flight run's id before the
    // late progress arrives, so the bare-commit variant also filters it
    // (verified empirically). This is therefore a GREEN behavioral guard
    // that the cache-hit path commits the cached candidates and a late
    // solver-progress stamped with the prior in-flight run's id does not
    // corrupt them; the underlying supersede correctness is independently
    // covered by the cross-feature test above and "drops payloads from a
    // superseded run".
    const hang = createDeferred<void>();
    let hangRunId = -1;
    let cacheRootMode = true;
    const startSearch = vi.fn((_b: Board, _p: Player, _s: number, _m: string, runId: number) => {
      if (cacheRootMode) {
        store.getState().applySolverProgress({
          runId,
          bestMove: "d3",
          row: 2,
          col: 3,
          score: 4,
          depth: 14,
          targetDepth: 14,
          acc: 100,
          nodes: 1000,
          pvLine: "d3",
          isEndgame: true,
        });
        return Promise.resolve();
      }
      hangRunId = runId;
      return hang.promise;
    });
    const { store } = createTestStore({
      solver: createMockSolverService({ startSearch }),
    });
    const board = initializeBoard();

    // Root search completes with a complete candidate -> root is cached.
    await startSolverFromPosition(store, board, "black");
    expect(store.getState().solverCandidates.size).toBe(1);
    // Advance d3 then undo back to the cached root.
    await store.getState().advanceSolver(2, 3);
    await store.getState().undoSolver();
    expect(store.getState().solverHistory).toHaveLength(1);

    // Advance to a different move whose search hangs (prior run in flight).
    cacheRootMode = false;
    const pending = store.getState().advanceSolver(2, 2);
    for (let i = 0; i < 30 && hangRunId < 0; i++) await Promise.resolve();
    expect(store.getState().engineActivity.kind).toBe("solver");

    // Undo back to root: root is cached -> cache-hit path supersedes the
    // in-flight run and commits the cached candidates.
    await store.getState().undoSolver();
    const cachedCandidates = store.getState().solverCandidates;
    expect(cachedCandidates.size).toBe(1);
    expect(cachedCandidates.get("2,3")?.move).toBe("d3");

    // Late progress stamped with the now-superseded in-flight run's id
    // must NOT overwrite the committed cached candidates.
    store.getState().applySolverProgress({
      runId: hangRunId,
      bestMove: "a1",
      row: 0,
      col: 0,
      score: 99,
      depth: 1,
      targetDepth: 1,
      acc: 100,
      nodes: 1,
      pvLine: "a1",
      isEndgame: true,
    });
    expect(store.getState().solverCandidates).toBe(cachedCandidates);
    expect(store.getState().solverCandidates.has("0,0")).toBe(false);
    expect(store.getState().solverCandidates.size).toBe(1);

    hang.resolve();
    await pending;
    expect(store.getState().solverCandidates.has("0,0")).toBe(false);
  });
});

describe("superseded solver teardown does not poison the prior position's cache (P1 #2)", () => {
  it("a superseded run's teardown must not cache the superseding position's candidates under its own (board,player) key", async () => {
    // Scenario reproducing the P1 #2 race:
    //  1. Root P0 search completes -> P0 cached with candidate "2,3"->d3.
    //  2. Advance to P1 (after move 2,3); P1's search hangs with empty
    //     candidates.
    //  3. Undo back to P0: P0 is cached -> cache-hit path. Its onClaim
    //     synchronously commits P0's cached candidates AND supersedes R_p1.
    //  4. R_p1's abort (registered as () => solver.abort()) is slow; while
    //     it is awaited inside supersede(), the new position is already P0
    //     with P0's candidates committed.
    //  5. Resolving R_p1's abort fires its solverTeardown in supersede()'s
    //     finally. With the BUG it unconditionally caches
    //     this.read().solverCandidates (== P0's complete candidates) under
    //     R_p1's captured (P1,white) key -> P1's cache is POISONED with
    //     P0's moves. With the fix it skips caching for a superseded run.
    //  6. Re-advancing to P1 must therefore issue a FRESH search (cache
    //     miss), not return P0's poisoned candidates as a cache hit.
    // abort() resolves immediately unless `hangAbort` is set, in which
    // case the FIRST such call returns a deferred we control. This lets us
    // freeze precisely R_p1's registered abort while it is awaited inside
    // the undo cache-hit's supersede(), so the new (P0) position and its
    // candidates are already committed when R_p1's teardown finally runs.
    interface VoidDeferred {
      promise: Promise<void>;
      resolve: (value: void | PromiseLike<void>) => void;
      reject: (reason?: unknown) => void;
    }
    const makeDeferred = (): VoidDeferred => createDeferred<void>();
    let hangAbort = false;
    const hungAbortBox: { current: VoidDeferred | null } = { current: null };
    const abortMock = vi.fn(() => {
      if (hangAbort && !hungAbortBox.current) {
        hungAbortBox.current = makeDeferred();
        return hungAbortBox.current.promise;
      }
      return Promise.resolve();
    });

    const p1HangBox: { current: VoidDeferred | null } = { current: null };
    let p1RunId = -1;
    let rootMode = true;
    const startSearch = vi.fn((_b: Board, _p: Player, _s: number, _m: string, runId: number) => {
      if (rootMode) {
        // Root P0 search: emit a complete candidate and resolve so
        // P0 is cached by its natural (ok) teardown.
        store.getState().applySolverProgress({
          runId,
          bestMove: "d3",
          row: 2,
          col: 3,
          score: 4,
          depth: 14,
          targetDepth: 14,
          acc: 100,
          nodes: 1000,
          pvLine: "d3",
          isEndgame: true,
        });
        return Promise.resolve();
      }
      // P1 search hangs so R_p1 is the live run when we undo.
      p1RunId = runId;
      p1HangBox.current = makeDeferred();
      return p1HangBox.current.promise;
    });

    const { store, services } = createTestStore({
      solver: createMockSolverService({ startSearch, abort: abortMock }),
    });
    const board = initializeBoard();

    // 1. Root P0 search completes -> P0 cached.
    await startSolverFromPosition(store, board, "black");
    expect(store.getState().solverCandidates.get("2,3")?.move).toBe("d3");

    // 2. Advance to P1; its search hangs (R_p1 live, in flight).
    rootMode = false;
    const p1Pending = store.getState().advanceSolver(2, 3);
    for (let i = 0; i < 30 && p1RunId < 0; i++) await Promise.resolve();
    expect(store.getState().engineActivity.kind).toBe("solver");
    const p1State = store.getState();
    const p1Position = p1State.solverHistory[p1State.solverHistory.length - 1]!;
    const { board: p1Board, player: p1Player } = p1Position;
    expect(store.getState().solverCandidates.size).toBe(0);

    // 3 + 4 + 5. Undo back to P0 (cache hit): commits P0's cached
    // candidates, supersedes R_p1. R_p1's registered abort hangs (the
    // first hung abort), so when we resolve it the new (P0) position and
    // its complete candidates are already committed before R_p1's
    // teardown runs in supersede()'s finally.
    hangAbort = true;
    const undoPending = store.getState().undoSolver();
    for (let i = 0; i < 30 && !hungAbortBox.current; i++) await Promise.resolve();
    const rootStateBeforeAbort = store.getState();
    expect(
      rootStateBeforeAbort.solverHistory[rootStateBeforeAbort.solverHistory.length - 1]?.board,
    ).toEqual(board);
    expect(store.getState().solverCandidates.get("2,3")?.move).toBe("d3");
    // Release R_p1's abort: supersede() proceeds and fires R_p1's
    // teardown while solverCandidates already holds P0's complete map.
    hangAbort = false;
    hungAbortBox.current?.resolve();
    for (let i = 0; i < 10; i++) await Promise.resolve();
    await undoPending;
    const rootStateAfterAbort = store.getState();
    expect(
      rootStateAfterAbort.solverHistory[rootStateAfterAbort.solverHistory.length - 1]?.board,
    ).toEqual(board);
    expect(store.getState().solverCandidates.get("2,3")?.move).toBe("d3");

    // Let the hung P1 search settle so its start() promise resolves.
    p1HangBox.current?.resolve();
    await p1Pending;

    // 6. Re-advance to P1. If R_p1's teardown poisoned P1's cache with
    // P0's candidates, this is a (wrong) cache hit returning d3 and NO
    // fresh search is issued. With the fix P1 is uncached -> a fresh
    // search runs for the P1 (board,player).
    startSearch.mockClear();
    p1RunId = -1;
    rootMode = false;
    const rePending = store.getState().advanceSolver(2, 3);
    for (let i = 0; i < 30 && startSearch.mock.calls.length === 0; i++) {
      await Promise.resolve();
    }
    const readvancedState = store.getState();
    const readvancedPosition =
      readvancedState.solverHistory[readvancedState.solverHistory.length - 1];
    expect(readvancedPosition?.board).toEqual(p1Board);
    expect(readvancedPosition?.player).toBe(p1Player);
    // The cache for P1 must NOT have been poisoned with P0's candidates.
    expect(services.solver.startSearch).toHaveBeenCalledWith(
      p1Board,
      p1Player,
      100,
      "multiPv",
      expect.any(Number),
    );
    expect(store.getState().solverCandidates.get("2,3")).toBeUndefined();

    p1HangBox.current?.resolve();
    await rePending;
  });
});
