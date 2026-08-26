import { describe, expect, it, vi } from "vitest";
import { DEFAULT_SETTINGS } from "@/services/types";
import { createMockAIService } from "@/services/mock-ai-service";
import { hydrateReversiStore } from "@/stores/use-reversi-store";
import { createMemoryStorage, createTestStore } from "./test-helpers";

const STORAGE_KEY = "neural-reversi-settings";

describe("settings persistence", () => {
  it("writes a versioned settings payload to settings storage", async () => {
    const { store, storage } = createTestStore();
    await store.persist.rehydrate();

    store.getState().setAIAnalysisPanelOpen(true);
    await Promise.resolve();

    const raw = storage.getItem(STORAGE_KEY);
    expect(raw).toBeTypeOf("string");
    expect(JSON.parse(raw as string)).toMatchObject({
      version: 2,
      state: { aiAnalysisPanelOpen: true },
    });
  });

  it("replaces malformed persisted values with defaults", async () => {
    const storage = createMemoryStorage();
    storage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        version: 1,
        state: {
          gameMode: "online",
          aiLevel: -1,
          aiMode: "unsupported",
          gameTimeLimit: 1.5,
          hintLevel: 31,
          isHintMode: "yes",
          gameAnalysisLevel: null,
          hashSize: 0,
          aiAnalysisPanelOpen: "yes",
          rightPanelSize: Number.NaN,
          bottomPanelSize: "wide",
          language: "fr",
          targetSelectivity: 74,
          solverMode: "all",
        },
      }),
    );

    const { store } = createTestStore(undefined, storage);
    await store.persist.rehydrate();
    const state = store.getState();

    expect({
      gameMode: state.gameMode,
      aiLevel: state.aiLevel,
      aiMode: state.aiMode,
      gameTimeLimit: state.gameTimeLimit,
      hintLevel: state.hintLevel,
      isHintMode: state.isHintMode,
      gameAnalysisLevel: state.gameAnalysisLevel,
      hashSize: state.hashSize,
      aiAnalysisPanelOpen: state.aiAnalysisPanelOpen,
      rightPanelSize: state.rightPanelSize,
      bottomPanelSize: state.bottomPanelSize,
      language: state.language,
      solverTargetSelectivity: state.targetSelectivity,
      solverMode: state.solverMode,
    }).toEqual(DEFAULT_SETTINGS);
  });

  it("does not rewrite storage for volatile state changes", async () => {
    const storage = createMemoryStorage();
    const setItem = vi.spyOn(storage, "setItem");
    const { store } = createTestStore(undefined, storage);
    await store.persist.rehydrate();

    store.getState().setAIAnalysisPanelOpen(true);
    store.setState({ showGameOverNotification: true });
    await Promise.resolve();

    expect(setItem).toHaveBeenCalledTimes(1);
  });

  it("applies a persisted hash size to the backend at startup", async () => {
    const storage = createMemoryStorage();
    storage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        version: 1,
        state: { hashSize: 1024 },
      }),
    );
    const ai = createMockAIService();

    const { store } = createTestStore({ ai }, storage);
    await store.persist.rehydrate();

    expect(ai.resizeTT).toHaveBeenCalledWith(1024);
  });

  it("does not read storage before explicit startup hydration", async () => {
    const storage = createMemoryStorage();
    const getItem = vi.spyOn(storage, "getItem");
    const { store } = createTestStore(undefined, storage);

    expect(getItem).not.toHaveBeenCalled();

    await store.persist.rehydrate();

    expect(getItem).toHaveBeenCalledWith(STORAGE_KEY);
  });

  it("retries an unchanged payload after a storage write fails", async () => {
    let stored: string | null = null;
    const setItem = vi
      .fn()
      .mockRejectedValueOnce(new Error("disk full"))
      .mockImplementation(async (_name: string, value: string) => {
        stored = value;
      });
    const storage = {
      getItem: () => stored,
      setItem,
      removeItem: () => {
        stored = null;
      },
    };
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => {});
    const { store } = createTestStore(undefined, storage);
    await store.persist.rehydrate();

    try {
      await (store.setState({ aiAnalysisPanelOpen: true }) as unknown as Promise<void>);
      await (store.setState({ aiAnalysisPanelOpen: true }) as unknown as Promise<void>);
    } finally {
      consoleError.mockRestore();
    }

    expect(setItem).toHaveBeenCalledTimes(2);
  });

  it("coalesces an unchanged payload while an async write is pending", async () => {
    let finishWrite!: () => void;
    const pendingWrite = new Promise<void>((resolve) => {
      finishWrite = resolve;
    });
    const setItem = vi.fn(() => pendingWrite);
    const storage = {
      getItem: () => null,
      setItem,
      removeItem: () => {},
    };
    const { store } = createTestStore(undefined, storage);
    await store.persist.rehydrate();

    store.getState().setAIAnalysisPanelOpen(true);
    store.setState({ showGameOverNotification: true });
    await Promise.resolve();

    expect(setItem).toHaveBeenCalledTimes(1);

    finishWrite();
    await pendingWrite;
  });

  it("rejects startup and keeps persistence disabled after hydration fails", async () => {
    const setItem = vi.fn();
    const storage = {
      getItem: async () => {
        throw new Error("read failed");
      },
      setItem,
      removeItem: () => {},
    };
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => {});
    const { store } = createTestStore(undefined, storage);

    try {
      await expect(hydrateReversiStore(store)).rejects.toThrow("Failed to hydrate settings");
      store.getState().setAIAnalysisPanelOpen(true);
      await Promise.resolve();
    } finally {
      consoleError.mockRestore();
    }

    expect(store.persist.hasHydrated()).toBe(false);
    expect(setItem).not.toHaveBeenCalled();
  });

  it("persists the hint mode preference", async () => {
    const { store, storage } = createTestStore();
    await store.persist.rehydrate();

    store.getState().setHintMode(true);
    await Promise.resolve();

    const raw = storage.getItem(STORAGE_KEY);
    expect(JSON.parse(raw as string)).toMatchObject({
      state: { isHintMode: true },
    });
  });

  it("restores the enabled hint mode preference", async () => {
    const storage = createMemoryStorage();
    storage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        version: 2,
        state: { isHintMode: true },
      }),
    );
    const { store } = createTestStore(undefined, storage);

    await store.persist.rehydrate();

    expect(store.getState().isHintMode).toBe(true);
  });
});
