import { load, type Store } from "@tauri-apps/plugin-store";
import type { AIMode, GameMode } from "@/domain/game/types";
import type { Language } from "@/i18n";
import {
  DEFAULT_SETTINGS,
  SOLVER_MODES,
  SOLVER_SELECTIVITIES,
  type AppSettings,
  type SettingsService,
  type SolverMode,
  type SolverSelectivity,
} from "./types";

// Mirrors `validate_level` in `src-tauri/src/lib.rs` and
// `reversi_core::level::MAX_LEVEL` (31 entries indexed 0..=30).
const MAX_ENGINE_LEVEL = 30;
const MAX_HASH_SIZE = 16384;
const GAME_MODES = ["ai-black", "ai-white", "pvp"] as const satisfies readonly GameMode[];
const AI_MODES = ["level", "time", "game-time"] as const satisfies readonly AIMode[];

function isIntegerInRange(value: unknown, min: number, max: number): value is number {
  return typeof value === "number" && Number.isInteger(value) && value >= min && value <= max;
}

function isPositiveFiniteInteger(value: unknown): value is number {
  return (
    typeof value === "number" && Number.isFinite(value) && Number.isInteger(value) && value > 0
  );
}

function isValidGameMode(value: unknown): value is GameMode {
  return typeof value === "string" && (GAME_MODES as readonly string[]).includes(value);
}

function isValidAiMode(value: unknown): value is AIMode {
  return typeof value === "string" && (AI_MODES as readonly string[]).includes(value);
}

function isValidSolverSelectivity(value: unknown): value is SolverSelectivity {
  return typeof value === "number" && (SOLVER_SELECTIVITIES as readonly number[]).includes(value);
}

function isValidSolverMode(value: unknown): value is SolverMode {
  return typeof value === "string" && (SOLVER_MODES as readonly string[]).includes(value);
}

export class TauriSettingsService implements SettingsService {
  private storePromise: Promise<Store> | null = null;

  private getStore(): Promise<Store> {
    if (!this.storePromise) {
      this.storePromise = load("settings.json", { autoSave: true, defaults: {} }).catch((error) => {
        this.storePromise = null;
        throw error;
      });
    }
    return this.storePromise;
  }

  async loadSettings(): Promise<AppSettings> {
    try {
      const s = await this.getStore();
      const [
        gameMode,
        aiLevel,
        aiMode,
        timeLimit,
        gameTimeLimit,
        hintLevel,
        gameAnalysisLevel,
        hashSize,
        aiAnalysisPanelOpen,
        rightPanelSize,
        bottomPanelSize,
        language,
        solverTargetSelectivity,
        solverMode,
      ] = await Promise.all([
        s.get<GameMode>("gameMode"),
        s.get<number>("aiLevel"),
        s.get<AIMode>("aiMode"),
        s.get<number>("timeLimit"),
        s.get<number>("gameTimeLimit"),
        s.get<number>("hintLevel"),
        s.get<number>("gameAnalysisLevel"),
        s.get<number>("hashSize"),
        s.get<boolean>("aiAnalysisPanelOpen"),
        s.get<number>("rightPanelSize"),
        s.get<number>("bottomPanelSize"),
        s.get<Language | null>("language"),
        s.get<number>("solverTargetSelectivity"),
        s.get<string>("solverMode"),
      ]);

      return {
        gameMode: isValidGameMode(gameMode) ? gameMode : DEFAULT_SETTINGS.gameMode,
        aiLevel: isIntegerInRange(aiLevel, 0, MAX_ENGINE_LEVEL)
          ? aiLevel
          : DEFAULT_SETTINGS.aiLevel,
        aiMode: isValidAiMode(aiMode) ? aiMode : DEFAULT_SETTINGS.aiMode,
        timeLimit: isPositiveFiniteInteger(timeLimit) ? timeLimit : DEFAULT_SETTINGS.timeLimit,
        gameTimeLimit: isPositiveFiniteInteger(gameTimeLimit)
          ? gameTimeLimit
          : DEFAULT_SETTINGS.gameTimeLimit,
        hintLevel: isIntegerInRange(hintLevel, 0, MAX_ENGINE_LEVEL)
          ? hintLevel
          : DEFAULT_SETTINGS.hintLevel,
        gameAnalysisLevel: isIntegerInRange(gameAnalysisLevel, 0, MAX_ENGINE_LEVEL)
          ? gameAnalysisLevel
          : DEFAULT_SETTINGS.gameAnalysisLevel,
        hashSize: isIntegerInRange(hashSize, 1, MAX_HASH_SIZE)
          ? hashSize
          : DEFAULT_SETTINGS.hashSize,
        aiAnalysisPanelOpen: aiAnalysisPanelOpen ?? DEFAULT_SETTINGS.aiAnalysisPanelOpen,
        rightPanelSize: rightPanelSize ?? DEFAULT_SETTINGS.rightPanelSize,
        bottomPanelSize: bottomPanelSize ?? DEFAULT_SETTINGS.bottomPanelSize,
        language: language ?? DEFAULT_SETTINGS.language,
        solverTargetSelectivity: isValidSolverSelectivity(solverTargetSelectivity)
          ? solverTargetSelectivity
          : DEFAULT_SETTINGS.solverTargetSelectivity,
        solverMode: isValidSolverMode(solverMode) ? solverMode : DEFAULT_SETTINGS.solverMode,
      };
    } catch (error) {
      console.error("Failed to load settings:", error);
      return DEFAULT_SETTINGS;
    }
  }

  async saveSetting<K extends keyof AppSettings>(key: K, value: AppSettings[K]): Promise<boolean> {
    try {
      const s = await this.getStore();
      await s.set(key, value);
      return true;
    } catch (error) {
      console.error(`Failed to save setting ${key}:`, error);
      return false;
    }
  }
}
