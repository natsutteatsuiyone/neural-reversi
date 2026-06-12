import { describe, expect, it, vi } from "vitest";
import { abortInFlightGameSearches, prepareGameReplacement } from "@/stores/game-replacement";
import { createMockAIService } from "@/services/mock-ai-service";
import { createTestStore } from "./test-helpers";

// The Game Replacement seam, tested through its own interface (CONTEXT.md →
// Game Replacement). The slice tests cover the same paths via startGame /
// startFromSetup / startSolver; these pin the module's contract in isolation.
describe("prepareGameReplacement", () => {
  it("returns false and re-initialises nothing when AI is not ready", async () => {
    const { store, services } = createTestStore({
      ai: createMockAIService({
        checkReady: vi.fn().mockRejectedValue(new Error("not ready")),
      }),
    });

    const ok = await prepareGameReplacement(services, store.getState, store.setState);

    expect(ok).toBe(false);
    expect(services.ai.initialize).not.toHaveBeenCalled();
  });

  it("returns true after re-initialising the backend", async () => {
    const { store, services } = createTestStore();

    const ok = await prepareGameReplacement(services, store.getState, store.setState);

    expect(ok).toBe(true);
    expect(services.ai.initialize).toHaveBeenCalled();
    expect(services.ai.resizeTT).toHaveBeenCalledWith(store.getState().hashSize);
  });

  // The solver is the fourth Engine Search and contends for the same backend
  // engine/mutex (CONTEXT.md → Engine Search). Re-initialising the backend
  // would deadlock on that mutex if a solver search still held it, so the
  // Game Replacement seam frees the solver search itself — every starter no
  // longer has to remember a pre-abort (and startFromSetup no longer omits
  // it).
  it("frees the solver search before re-initialising the backend", async () => {
    const { store, services } = createTestStore();

    const ok = await prepareGameReplacement(services, store.getState, store.setState);

    expect(ok).toBe(true);
    expect(services.solver.abort).toHaveBeenCalled();
    const abortOrder = (services.solver.abort as ReturnType<typeof vi.fn>).mock
      .invocationCallOrder[0];
    const initOrder = (services.ai.initialize as ReturnType<typeof vi.fn>).mock
      .invocationCallOrder[0];
    expect(abortOrder).toBeLessThan(initOrder);
  });

  it("does not free the solver search when AI is not ready", async () => {
    const { store, services } = createTestStore({
      ai: createMockAIService({
        checkReady: vi.fn().mockRejectedValue(new Error("not ready")),
      }),
    });

    const ok = await prepareGameReplacement(services, store.getState, store.setState);

    expect(ok).toBe(false);
    expect(services.solver.abort).not.toHaveBeenCalled();
    expect(services.ai.initialize).not.toHaveBeenCalled();
  });

  it("restores paused and re-triggers automation when init fails (no game analysis)", async () => {
    const { store, services } = createTestStore({
      ai: createMockAIService({
        initialize: vi.fn().mockRejectedValue(new Error("init failed")),
      }),
    });
    store.setState({ paused: true });
    const triggerSpy = vi.spyOn(store.getState(), "triggerAutomation");

    const ok = await prepareGameReplacement(services, store.getState, store.setState);

    expect(ok).toBe(false);
    expect(store.getState().paused).toBe(true);
    expect(triggerSpy).toHaveBeenCalled();
  });

  it("resumes a superseded game analysis when init fails", async () => {
    const { store, services } = createTestStore({
      ai: createMockAIService({
        initialize: vi.fn().mockRejectedValue(new Error("init failed")),
      }),
    });
    store.setState({ isGameAnalyzing: true });
    const analyzeGameSpy = vi.spyOn(store.getState(), "analyzeGame");
    const queueResumeSpy = vi.spyOn(store.getState(), "queueResumeAutomation");

    const ok = await prepareGameReplacement(services, store.getState, store.setState);

    expect(ok).toBe(false);
    expect(analyzeGameSpy).toHaveBeenCalled();
    expect(queueResumeSpy).toHaveBeenCalled();
  });
});

describe("abortInFlightGameSearches", () => {
  it("aborts the AI-move search while one is in flight", async () => {
    const { store } = createTestStore();
    store.setState({ isAIThinking: true });
    const abortSpy = vi.spyOn(store.getState(), "abortAIMove").mockResolvedValue(undefined);

    await abortInFlightGameSearches(store.getState);

    expect(abortSpy).toHaveBeenCalled();
  });

  // Regression: a hint abort-then-restart stamps Engine Activity back to idle
  // synchronously (isAnalyzing=false) while its backend abort + restart are
  // still in flight; `hintAnalysisAbortPending` is the breadcrumb for that
  // window. Replacement must still abort, or the pending hint teardown would
  // restart analysis on the just-reinitialised backend.
  it("aborts via abortAIMove while a hint abort is still pending", async () => {
    const { store } = createTestStore();
    store.setState({
      isAIThinking: false,
      isAnalyzing: false,
      hintAnalysisAbortPending: true,
    });
    const abortSpy = vi.spyOn(store.getState(), "abortAIMove").mockResolvedValue(undefined);

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
