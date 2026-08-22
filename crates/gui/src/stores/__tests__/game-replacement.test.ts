import { afterAll, beforeEach, describe, expect, it, vi } from "vitest";
import { abortInFlightGameSearches, runGameReplacement } from "@/stores/game-replacement";
import { createMockAIService } from "@/services/mock-ai-service";
import { createEmptyBoard } from "@/domain/game/game-logic";
import { createTestStore } from "./test-helpers";
const consoleErrorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
beforeEach(() => {
  consoleErrorSpy.mockClear();
});

afterAll(() => {
  consoleErrorSpy.mockRestore();
});

// Transaction behavior lives at the Game Replacement seam. Slice tests only
// pin that each public starter routes to the correct target.
describe("runGameReplacement", () => {
  it("returns false and re-initialises nothing when AI is not ready", async () => {
    const { store, services } = createTestStore({
      ai: createMockAIService({
        checkReady: vi.fn().mockRejectedValue(new Error("not ready")),
      }),
    });

    const ok = await runGameReplacement(services, store.getState, store.setState, {
      kind: "new-game",
    });

    expect(ok).toBe(false);
    expect(services.solver.abort).not.toHaveBeenCalled();
    expect(services.ai.initialize).not.toHaveBeenCalled();
    expect(consoleErrorSpy).toHaveBeenCalledWith("AI readiness check failed:", expect.any(Error));
  });

  it("returns true after re-initialising the backend", async () => {
    const { store, services } = createTestStore();

    const ok = await runGameReplacement(services, store.getState, store.setState, {
      kind: "new-game",
    });

    expect(ok).toBe(true);
    expect(services.ai.resizeTT).toHaveBeenCalledWith(store.getState().hashSize);
    const abortOrder = (services.solver.abort as ReturnType<typeof vi.fn>).mock
      .invocationCallOrder[0];
    const initOrder = (services.ai.initialize as ReturnType<typeof vi.fn>).mock
      .invocationCallOrder[0];
    expect(abortOrder).toBeLessThan(initOrder);
  });

  it("restores paused and re-triggers automation when init fails (no game analysis)", async () => {
    const { store, services } = createTestStore({
      ai: createMockAIService({
        initialize: vi.fn().mockRejectedValue(new Error("init failed")),
      }),
    });
    store.setState({ paused: true });
    const triggerSpy = vi.spyOn(store.getState(), "triggerAutomation");

    const ok = await runGameReplacement(services, store.getState, store.setState, {
      kind: "new-game",
    });

    expect(ok).toBe(false);
    expect(store.getState().paused).toBe(true);
    expect(triggerSpy).toHaveBeenCalled();
    expect(consoleErrorSpy).toHaveBeenCalledWith(
      "Failed to prepare AI for a new position:",
      expect.any(Error),
    );
  });

  it("resumes a superseded game analysis when init fails", async () => {
    const { store, services } = createTestStore({
      ai: createMockAIService({
        initialize: vi.fn().mockRejectedValue(new Error("init failed")),
      }),
    });
    store.setState({ engineActivity: { kind: "game-analysis", runId: 1 } });
    const analyzeGameSpy = vi.spyOn(store.getState(), "analyzeGame");
    const queueResumeSpy = vi.spyOn(store.getState(), "queueResumeAutomation");

    const ok = await runGameReplacement(services, store.getState, store.setState, {
      kind: "new-game",
    });

    expect(ok).toBe(false);
    expect(analyzeGameSpy).toHaveBeenCalled();
    expect(queueResumeSpy).toHaveBeenCalled();
    expect(consoleErrorSpy).toHaveBeenCalledWith(
      "Failed to prepare AI for a new position:",
      expect.any(Error),
    );
  });

  it("setup-game exits solver mode only after a successful replacement", async () => {
    const { store, services } = createTestStore();
    const solverBoard = store.getState().board;
    store.setState({
      isSolverActive: true,
      solverHistory: [{ board: solverBoard, player: "black", moveFrom: null }],
    });

    const ok = await runGameReplacement(services, store.getState, store.setState, {
      kind: "setup-game",
    });

    expect(ok).toBe(true);
    expect(services.solver.abort).toHaveBeenCalledTimes(2);
    expect(store.getState().isSolverActive).toBe(false);
    expect(store.getState().solverHistory).toEqual([]);
    expect(store.getState().setupError).toBeNull();
  });

  it("setup-game preserves solver state when backend init fails", async () => {
    const { store, services } = createTestStore({
      ai: createMockAIService({
        initialize: vi.fn().mockRejectedValue(new Error("init failed")),
      }),
    });
    const solverBoard = store.getState().board;
    store.setState({
      isSolverActive: true,
      solverHistory: [{ board: solverBoard, player: "black", moveFrom: null }],
    });

    const ok = await runGameReplacement(services, store.getState, store.setState, {
      kind: "setup-game",
    });

    expect(ok).toBe(false);
    expect(services.solver.abort).toHaveBeenCalledTimes(1);
    expect(store.getState().isSolverActive).toBe(true);
    const state = store.getState();
    expect(state.solverHistory[state.solverHistory.length - 1]?.board).toBe(solverBoard);
    expect(store.getState().setupError).toBe("aiInitFailed");
    expect(consoleErrorSpy).toHaveBeenCalledWith(
      "Failed to prepare AI for a new position:",
      expect.any(Error),
    );
  });

  it("rejects an invalid setup before re-initialising the backend", async () => {
    const { store, services } = createTestStore();
    store.setState({ setupBoard: createEmptyBoard() });

    const ok = await runGameReplacement(services, store.getState, store.setState, {
      kind: "setup-game",
    });

    expect(ok).toBe(false);
    expect(store.getState().setupError).toBe("needBothColors");
    expect(services.ai.initialize).not.toHaveBeenCalled();
  });

  it("installs solver config and starts the selected setup", async () => {
    const { store, services } = createTestStore();
    const board = store.getState().setupBoard;
    const startSolver = vi.fn().mockResolvedValue(undefined);
    store.setState({ isSolverModalOpen: true });

    const ok = await runGameReplacement(services, store.getState, store.setState, {
      kind: "setup-solver",
      config: { selectivity: 95, mode: "bestOnly" },
      startSolver,
    });

    expect(ok).toBe(true);
    expect(store.getState()).toMatchObject({
      gameStatus: "waiting",
      isSolverModalOpen: false,
      targetSelectivity: 95,
      solverMode: "bestOnly",
      setupError: null,
    });
    expect(startSolver).toHaveBeenCalledWith(board, "black");
    expect(services.settings.saveSetting).toHaveBeenCalledWith("solverTargetSelectivity", 95);
    expect(services.settings.saveSetting).toHaveBeenCalledWith("solverMode", "bestOnly");
  });

  // Launch auto-start: a new-game target with `pauseForAITurn` starts paused
  // when the AI moves first, so the AI does not play unprompted on launch. The
  // user resumes via the AI card's Resume button (Sidebar's `paused &&
  // isAITurn`); `triggerAutomation` no-ops while paused.
  it("starts paused without auto-playing when the AI moves first (pauseForAITurn)", async () => {
    const { store, services } = createTestStore();
    const makeAIMoveSpy = vi.spyOn(store.getState(), "makeAIMove");

    const ok = await runGameReplacement(services, store.getState, store.setState, {
      kind: "new-game",
      settings: { gameMode: "ai-black", aiLevel: 5, aiMode: "level", gameTimeLimit: 60 },
      pauseForAITurn: true,
    });

    expect(ok).toBe(true);
    expect(store.getState().gameStatus).toBe("playing");
    expect(store.getState().paused).toBe(true);
    expect(makeAIMoveSpy).not.toHaveBeenCalled();
  });

  it("does not pause when the human moves first (pauseForAITurn)", async () => {
    const { store, services } = createTestStore();

    const ok = await runGameReplacement(services, store.getState, store.setState, {
      kind: "new-game",
      settings: { gameMode: "ai-white", aiLevel: 5, aiMode: "level", gameTimeLimit: 60 },
      pauseForAITurn: true,
    });

    expect(ok).toBe(true);
    expect(store.getState().paused).toBe(false);
  });

  it("does not pause in pvp mode (pauseForAITurn)", async () => {
    const { store, services } = createTestStore();

    const ok = await runGameReplacement(services, store.getState, store.setState, {
      kind: "new-game",
      settings: { gameMode: "pvp", aiLevel: 5, aiMode: "level", gameTimeLimit: 60 },
      pauseForAITurn: true,
    });

    expect(ok).toBe(true);
    expect(store.getState().paused).toBe(false);
  });
});

describe("abortInFlightGameSearches", () => {
  it("aborts the AI-move search while one is in flight", async () => {
    const { store } = createTestStore();
    store.setState({ engineActivity: { kind: "ai-move", runId: 1 } });
    const abortSpy = vi.spyOn(store.getState(), "abortAIMove").mockResolvedValue(undefined);

    await abortInFlightGameSearches(store.getState);

    expect(abortSpy).toHaveBeenCalled();
  });

  // Regression: a hint abort-then-restart returns Engine Activity to idle
  // synchronously while its backend abort + restart are still in flight.
  // `hintAnalysisAbortPending` remains the breadcrumb for that window.
  it("aborts via abortAIMove while a hint abort is still pending", async () => {
    const { store } = createTestStore();
    store.setState({
      hintAnalysisAbortPending: true,
    });
    const abortSpy = vi.spyOn(store.getState(), "abortAIMove").mockResolvedValue(undefined);

    await abortInFlightGameSearches(store.getState);

    expect(abortSpy).toHaveBeenCalled();
  });

  it("aborts game analysis while it is in flight", async () => {
    const { store } = createTestStore();
    store.setState({ engineActivity: { kind: "game-analysis", runId: 1 } });
    const abortSpy = vi.spyOn(store.getState(), "abortGameAnalysis").mockResolvedValue(undefined);

    await abortInFlightGameSearches(store.getState);

    expect(abortSpy).toHaveBeenCalled();
  });

  it("is a no-op when nothing is in flight", async () => {
    const { store } = createTestStore();
    const abortAI = vi.spyOn(store.getState(), "abortAIMove");
    const abortGA = vi.spyOn(store.getState(), "abortGameAnalysis");

    await abortInFlightGameSearches(store.getState);

    expect(abortAI).not.toHaveBeenCalled();
    expect(abortGA).not.toHaveBeenCalled();
  });
});
