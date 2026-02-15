import { describe, expect, it, vi, beforeEach } from "vitest";
import { create } from "zustand";
import { createGameSlice } from "@/stores/slices/game-slice";
import { createAISlice } from "@/stores/slices/ai-slice";
import { createUISlice } from "@/stores/slices/ui-slice";
import { createSettingsSlice } from "@/stores/slices/settings-slice";
import { createSetupSlice } from "@/stores/slices/setup-slice";
import type { ReversiState } from "@/stores/slices/types";
import { saveSetting } from "@/lib/settings-store";

vi.mock("@/lib/ai", () => ({
  initializeAI: vi.fn().mockResolvedValue(undefined),
  abortAISearch: vi.fn().mockResolvedValue(undefined),
  getAIMove: vi.fn().mockResolvedValue(null),
  analyze: vi.fn().mockResolvedValue(undefined),
}));

vi.mock("@/lib/settings-store", () => ({
  saveSetting: vi.fn(),
  loadSettings: vi.fn().mockResolvedValue({}),
}));

function createTestStore() {
  return create<ReversiState>()((...a) => ({
    ...createGameSlice(...a),
    ...createAISlice(...a),
    ...createUISlice(...a),
    ...createSettingsSlice(...a),
    ...createSetupSlice(...a),
  }));
}

beforeEach(() => {
  vi.mocked(saveSetting).mockClear();
});

describe("initial state", () => {
  it("has correct default values", () => {
    const store = createTestStore();
    const s = store.getState();
    expect(s.gameMode).toBe("ai-white");
    expect(s.timeLimit).toBe(1);
    expect(s.gameTimeLimit).toBe(60);
    expect(s.hintLevel).toBe(21);
    expect(s.aiAnalysisPanelOpen).toBe(false);
  });
});

describe("setGameMode", () => {
  it("updates gameMode state", () => {
    const store = createTestStore();
    store.getState().setGameMode("ai-black");
    expect(store.getState().gameMode).toBe("ai-black");
  });

  it("resets analyzeResults to null", () => {
    const store = createTestStore();
    store.setState({ analyzeResults: new Map([["0,0", {} as never]]) });
    store.getState().setGameMode("ai-black");
    expect(store.getState().analyzeResults).toBeNull();
  });

  it("calls saveSetting with gameMode", () => {
    const store = createTestStore();
    store.getState().setGameMode("ai-black");
    expect(saveSetting).toHaveBeenCalledWith("gameMode", "ai-black");
  });
});

describe("setTimeLimit", () => {
  it("updates timeLimit state", () => {
    const store = createTestStore();
    store.getState().setTimeLimit(5);
    expect(store.getState().timeLimit).toBe(5);
  });

  it("calls saveSetting with timeLimit", () => {
    const store = createTestStore();
    store.getState().setTimeLimit(5);
    expect(saveSetting).toHaveBeenCalledWith("timeLimit", 5);
  });
});

describe("setGameTimeLimit", () => {
  it("updates gameTimeLimit state", () => {
    const store = createTestStore();
    store.getState().setGameTimeLimit(120);
    expect(store.getState().gameTimeLimit).toBe(120);
  });

  it("calls saveSetting with gameTimeLimit", () => {
    const store = createTestStore();
    store.getState().setGameTimeLimit(120);
    expect(saveSetting).toHaveBeenCalledWith("gameTimeLimit", 120);
  });
});

describe("setHintLevel", () => {
  it("updates hintLevel state", () => {
    const store = createTestStore();
    store.getState().setHintLevel(10);
    expect(store.getState().hintLevel).toBe(10);
  });

  it("resets analyzeResults to null", () => {
    const store = createTestStore();
    store.setState({ analyzeResults: new Map([["0,0", {} as never]]) });
    store.getState().setHintLevel(10);
    expect(store.getState().analyzeResults).toBeNull();
  });

  it("calls saveSetting with hintLevel", () => {
    const store = createTestStore();
    store.getState().setHintLevel(10);
    expect(saveSetting).toHaveBeenCalledWith("hintLevel", 10);
  });

  it("calls analyzeBoard when isHintMode is true", () => {
    const store = createTestStore();
    store.setState({ isHintMode: true });
    const analyzeBoardSpy = vi.spyOn(store.getState(), "analyzeBoard");
    store.getState().setHintLevel(10);
    expect(analyzeBoardSpy).toHaveBeenCalled();
  });

  it("does not call analyzeBoard when isHintMode is false", () => {
    const store = createTestStore();
    const analyzeBoardSpy = vi.spyOn(store.getState(), "analyzeBoard");
    store.getState().setHintLevel(10);
    expect(analyzeBoardSpy).not.toHaveBeenCalled();
  });
});

describe("setAIAnalysisPanelOpen", () => {
  it("updates aiAnalysisPanelOpen state", () => {
    const store = createTestStore();
    store.getState().setAIAnalysisPanelOpen(true);
    expect(store.getState().aiAnalysisPanelOpen).toBe(true);

    store.getState().setAIAnalysisPanelOpen(false);
    expect(store.getState().aiAnalysisPanelOpen).toBe(false);
  });

  it("calls saveSetting with aiAnalysisPanelOpen", () => {
    const store = createTestStore();
    store.getState().setAIAnalysisPanelOpen(true);
    expect(saveSetting).toHaveBeenCalledWith("aiAnalysisPanelOpen", true);
  });
});
