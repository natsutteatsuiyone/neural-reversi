import { describe, expect, it, vi, beforeEach } from "vitest";
import { createTestStore } from "./test-helpers";
import type { AppSettings } from "@/services/types";

beforeEach(() => {
  vi.clearAllMocks();
});

describe("initial state", () => {
  it("has correct default values", () => {
    const { store } = createTestStore();
    const s = store.getState();
    expect(s.gameMode).toBe("ai-white");
    expect(s.timeLimit).toBe(1);
    expect(s.gameTimeLimit).toBe(60);
    expect(s.hintLevel).toBe(21);
    expect(s.aiAnalysisPanelOpen).toBe(false);
    expect(s.language).toBeNull();
  });
});

describe("hydrateSettings", () => {
  it("hydrates loaded settings without persisting them again", () => {
    const { store, services } = createTestStore();
    const settings: AppSettings = {
      gameMode: "ai-black",
      aiLevel: 18,
      aiMode: "level",
      timeLimit: 3,
      gameTimeLimit: 180,
      hintLevel: 12,
      gameAnalysisLevel: 16,
      hashSize: 1024,
      aiAnalysisPanelOpen: true,
      language: "ja",
    };

    store.getState().hydrateSettings(settings);

    const s = store.getState();
    expect(s.gameMode).toBe("ai-black");
    expect(s.aiLevel).toBe(18);
    expect(s.aiMode).toBe("level");
    expect(s.timeLimit).toBe(3);
    expect(s.gameTimeLimit).toBe(180);
    expect(s.hintLevel).toBe(12);
    expect(s.gameAnalysisLevel).toBe(16);
    expect(s.hashSize).toBe(1024);
    expect(s.aiAnalysisPanelOpen).toBe(true);
    expect(s.language).toBe("ja");
    expect(services.settings.saveSetting).not.toHaveBeenCalled();
  });
});

describe("setGameMode", () => {
  it("updates gameMode state", () => {
    const { store } = createTestStore();
    store.getState().setGameMode("ai-black");
    expect(store.getState().gameMode).toBe("ai-black");
  });

  it("resets analyzeResults to null", () => {
    const { store } = createTestStore();
    store.setState({ analyzeResults: new Map([["0,0", {} as never]]) });
    store.getState().setGameMode("ai-black");
    expect(store.getState().analyzeResults).toBeNull();
  });

  it("calls saveSetting with gameMode", () => {
    const { store, services } = createTestStore();
    store.getState().setGameMode("ai-black");
    expect(services.settings.saveSetting).toHaveBeenCalledWith("gameMode", "ai-black");
  });
});

describe("setTimeLimit", () => {
  it("updates timeLimit state", () => {
    const { store } = createTestStore();
    store.getState().setTimeLimit(5);
    expect(store.getState().timeLimit).toBe(5);
  });

  it("calls saveSetting with timeLimit", () => {
    const { store, services } = createTestStore();
    store.getState().setTimeLimit(5);
    expect(services.settings.saveSetting).toHaveBeenCalledWith("timeLimit", 5);
  });
});

describe("setGameTimeLimit", () => {
  it("updates gameTimeLimit state", () => {
    const { store } = createTestStore();
    store.getState().setGameTimeLimit(120);
    expect(store.getState().gameTimeLimit).toBe(120);
  });

  it("calls saveSetting with gameTimeLimit", () => {
    const { store, services } = createTestStore();
    store.getState().setGameTimeLimit(120);
    expect(services.settings.saveSetting).toHaveBeenCalledWith("gameTimeLimit", 120);
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
