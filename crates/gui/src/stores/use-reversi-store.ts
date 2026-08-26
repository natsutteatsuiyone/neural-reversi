import { create, type StoreApi } from "zustand";
import {
  createJSONStorage,
  persist,
  type PersistStorage,
  type StateStorage,
  type StorageValue,
} from "zustand/middleware";
import type { ReversiState } from "./slices/types";
import {
  DEFAULT_SETTINGS,
  SOLVER_MODES,
  SOLVER_SELECTIVITIES,
  type Services,
} from "@/services/types";
import { createGameSlice } from "./slices/game-slice";
import { createAISlice } from "./slices/ai-slice";
import { createUISlice } from "./slices/ui-slice";
import { createSettingsSlice } from "./slices/settings-slice";
import { createSetupSlice } from "./slices/setup-slice";
import { createSolverSlice } from "./slices/solver-slice";
import { createEngineSearch } from "@/domain/engine/engine-search";
import {
  HintAnalysisSession,
  type HintAnalysisSessionCommit,
} from "@/domain/game/hint-analysis-session";
import {
  GameAnalysisSession,
  type GameAnalysisSessionCommit,
} from "@/domain/game/game-analysis-session";
import { IDLE_ENGINE_ACTIVITY } from "./engine-activity";
import { defaultServices } from "@/services/default-services";
import { createTauriSettingsStorage } from "@/services/tauri-settings-storage";

const SETTINGS_STORAGE_KEY = "neural-reversi-settings";
const SETTINGS_STORAGE_VERSION = 2;

type PersistedSettings = Pick<
  ReversiState,
  | "gameMode"
  | "aiLevel"
  | "aiMode"
  | "gameTimeLimit"
  | "hintLevel"
  | "isHintMode"
  | "gameAnalysisLevel"
  | "hashSize"
  | "aiAnalysisPanelOpen"
  | "rightPanelSize"
  | "bottomPanelSize"
  | "language"
  | "targetSelectivity"
  | "solverMode"
>;
const MAX_ENGINE_LEVEL = 30;
const MAX_HASH_SIZE = 16_384;
const GAME_MODES = ["ai-black", "ai-white", "pvp"] as const;
const AI_MODES = ["level", "game-time"] as const;
const LANGUAGES = ["en", "ja"] as const;

function isIntegerInRange(value: unknown, min: number, max: number): value is number {
  return typeof value === "number" && Number.isInteger(value) && value >= min && value <= max;
}

function isPositiveInteger(value: unknown): value is number {
  return typeof value === "number" && Number.isInteger(value) && value > 0;
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function isOneOf<T>(value: unknown, choices: readonly T[]): value is T {
  return choices.includes(value as T);
}

function sanitizePersistedSettings(value: unknown): PersistedSettings {
  const stored =
    typeof value === "object" && value !== null ? (value as Record<string, unknown>) : {};
  return {
    gameMode: isOneOf(stored.gameMode, GAME_MODES) ? stored.gameMode : DEFAULT_SETTINGS.gameMode,
    aiLevel: isIntegerInRange(stored.aiLevel, 0, MAX_ENGINE_LEVEL)
      ? stored.aiLevel
      : DEFAULT_SETTINGS.aiLevel,
    aiMode: isOneOf(stored.aiMode, AI_MODES) ? stored.aiMode : DEFAULT_SETTINGS.aiMode,
    gameTimeLimit: isPositiveInteger(stored.gameTimeLimit)
      ? stored.gameTimeLimit
      : DEFAULT_SETTINGS.gameTimeLimit,
    hintLevel: isIntegerInRange(stored.hintLevel, 0, MAX_ENGINE_LEVEL)
      ? stored.hintLevel
      : DEFAULT_SETTINGS.hintLevel,
    isHintMode:
      typeof stored.isHintMode === "boolean" ? stored.isHintMode : DEFAULT_SETTINGS.isHintMode,
    gameAnalysisLevel: isIntegerInRange(stored.gameAnalysisLevel, 0, MAX_ENGINE_LEVEL)
      ? stored.gameAnalysisLevel
      : DEFAULT_SETTINGS.gameAnalysisLevel,
    hashSize: isIntegerInRange(stored.hashSize, 1, MAX_HASH_SIZE)
      ? stored.hashSize
      : DEFAULT_SETTINGS.hashSize,
    aiAnalysisPanelOpen:
      typeof stored.aiAnalysisPanelOpen === "boolean"
        ? stored.aiAnalysisPanelOpen
        : DEFAULT_SETTINGS.aiAnalysisPanelOpen,
    rightPanelSize: isFiniteNumber(stored.rightPanelSize)
      ? stored.rightPanelSize
      : DEFAULT_SETTINGS.rightPanelSize,
    bottomPanelSize: isFiniteNumber(stored.bottomPanelSize)
      ? stored.bottomPanelSize
      : DEFAULT_SETTINGS.bottomPanelSize,
    language:
      stored.language === null || isOneOf(stored.language, LANGUAGES)
        ? stored.language
        : DEFAULT_SETTINGS.language,
    targetSelectivity: isOneOf(stored.targetSelectivity, SOLVER_SELECTIVITIES)
      ? stored.targetSelectivity
      : DEFAULT_SETTINGS.solverTargetSelectivity,
    solverMode: isOneOf(stored.solverMode, SOLVER_MODES)
      ? stored.solverMode
      : DEFAULT_SETTINGS.solverMode,
  };
}

function selectPersistedSettings(state: ReversiState): PersistedSettings {
  return {
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
    targetSelectivity: state.targetSelectivity,
    solverMode: state.solverMode,
  };
}

function deduplicateStorage(
  storage: PersistStorage<PersistedSettings>,
): PersistStorage<PersistedSettings> {
  let hasRead = false;
  let lastValue: string | null | undefined;
  let scheduledValue: string | null | undefined;
  let generation = 0;
  let writeQueue = Promise.resolve();

  const remember = (value: StorageValue<PersistedSettings> | null) => {
    hasRead = true;
    lastValue = value === null ? null : JSON.stringify(value);
    scheduledValue = lastValue;
    return value;
  };
  const schedule = (
    operation: () => unknown,
    value: string | undefined,
    failureMessage: string,
  ) => {
    const writeGeneration = ++generation;
    scheduledValue = value;
    writeQueue = writeQueue.then(async () => {
      try {
        await operation();
        lastValue = value;
      } catch (error) {
        console.error(failureMessage, error);
        if (generation === writeGeneration) {
          scheduledValue = lastValue;
        }
      }
    });
    return writeQueue;
  };

  return {
    getItem: (name) => {
      const value = storage.getItem(name);
      return value instanceof Promise ? value.then(remember) : remember(value);
    },
    setItem: (name, value) => {
      if (!hasRead) return writeQueue;
      const serialized = JSON.stringify(value);
      if (serialized === scheduledValue) return writeQueue;
      return schedule(() => storage.setItem(name, value), serialized, "Failed to save settings:");
    },
    removeItem: (name) =>
      schedule(() => storage.removeItem(name), undefined, "Failed to clear settings:"),
  };
}

export function createReversiStore(services: Services, stateStorage?: StateStorage) {
  // Captured after the store exists; `onActivityChange` only fires during an
  // async start/abort, long after this assignment.
  let setState: StoreApi<ReversiState>["setState"] | null = null;
  const engineSearch = createEngineSearch({
    onActivityChange: (engineActivity) => setState?.({ engineActivity }),
  });
  const jsonStorage = stateStorage
    ? createJSONStorage<PersistedSettings>(() => stateStorage)
    : undefined;
  const storage = jsonStorage ? deduplicateStorage(jsonStorage) : undefined;
  const store = create<ReversiState>()(
    persist<ReversiState, [], [], PersistedSettings>(
      (set, get, api) => {
        // One Hint Analysis Session, shared by the UI slice (toggle/analyze) and
        // the settings slice (level change), mirroring how `engineSearch` is
        // created once and injected into multiple slices.
        const hintCommit: HintAnalysisSessionCommit = (partial) =>
          set(partial as Parameters<typeof set>[0]);
        const hintSession = new HintAnalysisSession({
          ai: services.ai,
          read: get,
          commit: hintCommit,
          engineSearch,
        });
        // One Game Analysis Session, same shape as hintSession (CONTEXT.md →
        // Engine Activity): the UI slice delegates analyze/abort to it.
        const gameAnalysisCommit: GameAnalysisSessionCommit = (partial) =>
          set(partial as Parameters<typeof set>[0]);
        const gameAnalysisSession = new GameAnalysisSession({
          ai: services.ai,
          read: get,
          commit: gameAnalysisCommit,
          engineSearch,
        });
        return {
          engineActivity: IDLE_ENGINE_ACTIVITY,
          ...createGameSlice(services)(set, get, api),
          ...createAISlice(services, engineSearch)(set, get, api),
          ...createUISlice(hintSession, gameAnalysisSession)(set, get, api),
          ...createSettingsSlice(services, hintSession)(set, get, api),
          ...createSetupSlice(services)(set, get, api),
          ...createSolverSlice(services, engineSearch)(set, get, api),
        };
      },
      {
        name: SETTINGS_STORAGE_KEY,
        version: SETTINGS_STORAGE_VERSION,
        storage,
        partialize: selectPersistedSettings,
        migrate: sanitizePersistedSettings,
        merge: (persisted, current) => ({
          ...current,
          ...sanitizePersistedSettings(persisted),
        }),
        skipHydration: true,
        onRehydrateStorage: () => (state, error) => {
          if (error) {
            console.error("Failed to load settings:", error);
          } else if (state && state.hashSize !== DEFAULT_SETTINGS.hashSize) {
            void services.ai.resizeTT(state.hashSize);
          }
        },
      },
    ),
  );
  setState = store.setState;
  return store;
}

export interface HydratableReversiStore {
  persist: {
    rehydrate: () => Promise<void> | void;
    hasHydrated: () => boolean;
  };
}

export async function hydrateReversiStore(store: HydratableReversiStore): Promise<void> {
  await store.persist.rehydrate();
  if (!store.persist.hasHydrated()) {
    throw new Error("Failed to hydrate settings");
  }
}

export const useReversiStore = createReversiStore(defaultServices, createTauriSettingsStorage());
