import { describe, expect, it, vi, beforeEach } from "vitest";
import { createTestStore, createDeferred } from "./test-helpers";
import { createMockAIService } from "@/services/mock-ai-service";
import type { AppSettings } from "@/services/types";

beforeEach(() => {
  vi.clearAllMocks();
});

describe("initial state", () => {
  it("has correct default values", () => {
    const { store } = createTestStore();
    const s = store.getState();
    expect(s.gameMode).toBe("ai-white");
    expect(s.gameTimeLimit).toBe(60);
    expect(s.hintLevel).toBe(21);
    expect(s.aiAnalysisPanelOpen).toBe(false);
    expect(s.language).toBeNull();
    expect(s.targetSelectivity).toBe(100);
  });
});

describe("hydrateSettings", () => {
  it("hydrates loaded settings without persisting them again", () => {
    const { store, services } = createTestStore();
    const settings: AppSettings = {
      gameMode: "ai-black",
      aiLevel: 18,
      aiMode: "level",
      gameTimeLimit: 180,
      hintLevel: 12,
      gameAnalysisLevel: 16,
      hashSize: 1024,
      aiAnalysisPanelOpen: true,
      rightPanelSize: 30,
      bottomPanelSize: 35,
      language: "ja",
      solverTargetSelectivity: 95,
      solverMode: "bestOnly",
    };

    store.getState().hydrateSettings(settings);

    const s = store.getState();
    expect(s.gameMode).toBe("ai-black");
    expect(s.aiLevel).toBe(18);
    expect(s.aiMode).toBe("level");
    expect(s.gameTimeLimit).toBe(180);
    expect(s.hintLevel).toBe(12);
    expect(s.gameAnalysisLevel).toBe(16);
    expect(s.hashSize).toBe(1024);
    expect(s.aiAnalysisPanelOpen).toBe(true);
    expect(s.language).toBe("ja");
    expect(s.targetSelectivity).toBe(95);
    expect(s.solverMode).toBe("bestOnly");
    expect(services.settings.saveSetting).not.toHaveBeenCalled();
    expect(services.ai.resizeTT).toHaveBeenCalledWith(1024);
  });
});

describe("setHintLevel", () => {
  it("updates hintLevel state", () => {
    const { store } = createTestStore();
    store.getState().setHintLevel(10);
    expect(store.getState().hintLevel).toBe(10);
  });

  it("resets analyzeResults to null", () => {
    const { store } = createTestStore();
    store.setState({ analyzeResults: new Map([["0,0", {} as never]]) });
    store.getState().setHintLevel(10);
    expect(store.getState().analyzeResults).toBeNull();
  });

  it("calls saveSetting with hintLevel", () => {
    const { store, services } = createTestStore();
    store.getState().setHintLevel(10);
    expect(services.settings.saveSetting).toHaveBeenCalledWith("hintLevel", 10);
  });

  it("calls analyzeBoard when isHintMode is true", () => {
    const { store } = createTestStore();
    store.setState({ isHintMode: true });
    const analyzeBoardSpy = vi.spyOn(store.getState(), "analyzeBoard");
    store.getState().setHintLevel(10);
    expect(analyzeBoardSpy).toHaveBeenCalled();
  });

  it("does not call analyzeBoard when isHintMode is false", () => {
    const { store } = createTestStore();
    const analyzeBoardSpy = vi.spyOn(store.getState(), "analyzeBoard");
    store.getState().setHintLevel(10);
    expect(analyzeBoardSpy).not.toHaveBeenCalled();
  });

  it("aborts running analyze and restarts with the new level", async () => {
    const abortDeferred = createDeferred<void>();
    const { store, services } = createTestStore({
      ai: createMockAIService({
        abortSearch: vi.fn().mockReturnValue(abortDeferred.promise),
      }),
    });
    store.setState({
      isHintMode: true,
      engineActivity: { kind: "hint", runId: 1 },
    });
    const analyzeBoardSpy = vi.spyOn(store.getState(), "analyzeBoard");

    store.getState().setHintLevel(10);
    // EngineSearch.abort awaits supersede() before onAbort/abort, so the
    // abort-pending snapshot is observable one microtask later (unchanged).
    await Promise.resolve();

    expect(services.ai.abortSearch).toHaveBeenCalledTimes(1);
    expect(store.getState().hintAnalysisAbortPending).toBe(true);
    expect(analyzeBoardSpy).not.toHaveBeenCalled();

    abortDeferred.resolve();
    await abortDeferred.promise;
    for (let i = 0; i < 10 && !analyzeBoardSpy.mock.calls.length; i++) {
      await Promise.resolve();
    }

    expect(store.getState().hintAnalysisAbortPending).toBe(false);
    expect(store.getState().engineActivity.kind).toBe("idle");
    expect(analyzeBoardSpy).toHaveBeenCalled();
  });

  it("does not queue a second hint restart while abort is pending", async () => {
    const abortDeferred = createDeferred<void>();
    const { store, services } = createTestStore({
      ai: createMockAIService({
        abortSearch: vi.fn().mockReturnValue(abortDeferred.promise),
      }),
    });
    store.setState({
      isHintMode: true,
      engineActivity: { kind: "hint", runId: 1 },
    });
    const analyzeBoardSpy = vi.spyOn(store.getState(), "analyzeBoard");

    store.getState().setHintLevel(10);
    store.getState().setHintLevel(12);

    // Dedup behavior is preserved synchronously (the 2nd setHintLevel sees
    // hintAnalysisAbortPending=true and early-returns, so only ONE restart is
    // queued). EngineSearch.abort always awaits supersede() before issuing the
    // backend abort, so abortSearch() is observable one microtask later.
    await Promise.resolve();

    expect(store.getState().hintLevel).toBe(12);
    expect(services.ai.abortSearch).toHaveBeenCalledTimes(1);
    expect(store.getState().hintAnalysisAbortPending).toBe(true);
    expect(analyzeBoardSpy).not.toHaveBeenCalled();

    abortDeferred.resolve();
    await abortDeferred.promise;
    await Promise.resolve();

    expect(store.getState().hintAnalysisAbortPending).toBe(false);
    expect(store.getState().engineActivity.kind).toBe("idle");
    expect(analyzeBoardSpy).toHaveBeenCalledTimes(1);
  });

  it("dedupes a redundant backend abort while a hint abort is pending", async () => {
    // `hintAnalysisAbortPending` is the dedupe guard (not owned by the Engine
    // Activity): a same-tick second level change must not issue a redundant
    // backend abort while a hint abort is still in flight.
    const backendAbort = createDeferred<void>();
    const { store, services } = createTestStore({
      ai: createMockAIService({
        abortSearch: vi.fn().mockReturnValue(backendAbort.promise), // slow
      }),
    });
    const analyzeBoardSpy = vi.spyOn(store.getState(), "analyzeBoard");

    // A hint analysis is the current Engine Activity.
    store.setState({
      isHintMode: true,
      engineActivity: { kind: "hint", runId: 1 },
    });

    const base = store.getState().hintLevel;
    store.getState().setHintLevel(base + 1); // aborts + restarts hint once
    store.getState().setHintLevel(base + 2); // guarded out: no redundant abort
    for (let i = 0; i < 5; i++) await Promise.resolve();

    expect(services.ai.abortSearch).toHaveBeenCalledTimes(1);
    expect(store.getState().hintAnalysisAbortPending).toBe(true);
    expect(analyzeBoardSpy).not.toHaveBeenCalled();

    // The backend abort resolves: the guard releases and hint re-analyzes.
    backendAbort.resolve();
    for (let i = 0; i < 10 && !analyzeBoardSpy.mock.calls.length; i++) {
      await Promise.resolve();
    }
    expect(store.getState().hintAnalysisAbortPending).toBe(false);
    expect(store.getState().engineActivity.kind).toBe("idle");
    expect(analyzeBoardSpy).toHaveBeenCalledTimes(1);
  });
});

describe("setAIAnalysisPanelOpen", () => {
  it("updates aiAnalysisPanelOpen state", () => {
    const { store } = createTestStore();
    store.getState().setAIAnalysisPanelOpen(true);
    expect(store.getState().aiAnalysisPanelOpen).toBe(true);

    store.getState().setAIAnalysisPanelOpen(false);
    expect(store.getState().aiAnalysisPanelOpen).toBe(false);
  });

  it("calls saveSetting with aiAnalysisPanelOpen", () => {
    const { store, services } = createTestStore();
    store.getState().setAIAnalysisPanelOpen(true);
    expect(services.settings.saveSetting).toHaveBeenCalledWith("aiAnalysisPanelOpen", true);
  });
});

describe("setLanguagePreference", () => {
  it("updates language state", async () => {
    const { store } = createTestStore();
    await store.getState().setLanguagePreference("en");
    expect(store.getState().language).toBe("en");
  });

  it("calls saveSetting with language", async () => {
    const { store, services } = createTestStore();
    await store.getState().setLanguagePreference("ja");
    expect(services.settings.saveSetting).toHaveBeenCalledWith("language", "ja");
  });
});
