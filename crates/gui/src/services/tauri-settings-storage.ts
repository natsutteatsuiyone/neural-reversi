import { load, type Store } from "@tauri-apps/plugin-store";
import type { StateStorage } from "zustand/middleware";

const LEGACY_KEYS = [
  "gameMode",
  "aiLevel",
  "aiMode",
  "gameTimeLimit",
  "hintLevel",
  "gameAnalysisLevel",
  "hashSize",
  "aiAnalysisPanelOpen",
  "rightPanelSize",
  "bottomPanelSize",
  "language",
  "solverTargetSelectivity",
  "solverMode",
] as const;

async function readLegacySettings(store: Store): Promise<string | null> {
  const values = await Promise.all(LEGACY_KEYS.map((key) => store.get(key)));
  const state: Record<string, unknown> = {};
  for (const [index, value] of values.entries()) {
    if (value !== undefined) {
      const key = LEGACY_KEYS[index];
      state[key === "solverTargetSelectivity" ? "targetSelectivity" : key] = value;
    }
  }
  return Object.keys(state).length === 0 ? null : JSON.stringify({ version: 0, state });
}

export function createTauriSettingsStorage(): StateStorage {
  let storePromise: Promise<Store> | undefined;
  let removeLegacyAfterWrite = false;
  const getStore = () => {
    storePromise ??= load("settings.json", { autoSave: true, defaults: {} }).catch((error) => {
      storePromise = undefined;
      throw error;
    });
    return storePromise;
  };

  return {
    getItem: async (name) => {
      const store = await getStore();
      const current = await store.get<string>(name);
      if (current !== undefined) return current;
      const legacy = await readLegacySettings(store);
      removeLegacyAfterWrite = legacy !== null;
      return legacy;
    },
    setItem: async (name, value) => {
      const store = await getStore();
      await store.set(name, value);
      if (removeLegacyAfterWrite) {
        await Promise.all(LEGACY_KEYS.map((key) => store.delete(key)));
        removeLegacyAfterWrite = false;
      }
    },
    removeItem: async (name) => {
      await (await getStore()).delete(name);
    },
  };
}
