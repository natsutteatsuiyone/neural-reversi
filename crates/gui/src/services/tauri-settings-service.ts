import { load, type Store } from "@tauri-apps/plugin-store";
import type { AIMode, GameMode } from "@/types";
import type { Language } from "@/i18n";
import {
  DEFAULT_SETTINGS,
  SOLVER_SELECTIVITIES,
  type AppSettings,
  type SettingsService,
  type SolverSelectivity,
} from "./types";

function isValidSolverSelectivity(value: unknown): value is SolverSelectivity {
  return (
    typeof value === "number" &&
    (SOLVER_SELECTIVITIES as readonly number[]).includes(value)
  );
}

export class TauriSettingsService implements SettingsService {
  private storePromise: Promise<Store> | null = null;

  private getStore(): Promise<Store> {
    if (!this.storePromise) {
      this.storePromise = load("settings.json", { autoSave: true, defaults: {} })
        .catch((error) => {
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
        gameMode, aiLevel, aiMode, timeLimit, gameTimeLimit,
        hintLevel, gameAnalysisLevel, hashSize, aiAnalysisPanelOpen,
        rightPanelSize, bottomPanelSize, language, solverTargetSelectivity,
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
      ]);

      return {
        gameMode: gameMode ?? DEFAULT_SETTINGS.gameMode,
        aiLevel: aiLevel ?? DEFAULT_SETTINGS.aiLevel,
        aiMode: aiMode ?? DEFAULT_SETTINGS.aiMode,
        timeLimit: timeLimit ?? DEFAULT_SETTINGS.timeLimit,
        gameTimeLimit: gameTimeLimit ?? DEFAULT_SETTINGS.gameTimeLimit,
        hintLevel: hintLevel ?? DEFAULT_SETTINGS.hintLevel,
        gameAnalysisLevel: gameAnalysisLevel ?? DEFAULT_SETTINGS.gameAnalysisLevel,
        hashSize: hashSize ?? DEFAULT_SETTINGS.hashSize,
        aiAnalysisPanelOpen: aiAnalysisPanelOpen ?? DEFAULT_SETTINGS.aiAnalysisPanelOpen,
        rightPanelSize: rightPanelSize ?? DEFAULT_SETTINGS.rightPanelSize,
        bottomPanelSize: bottomPanelSize ?? DEFAULT_SETTINGS.bottomPanelSize,
        language: language ?? DEFAULT_SETTINGS.language,
        solverTargetSelectivity: isValidSolverSelectivity(solverTargetSelectivity)
          ? solverTargetSelectivity
          : DEFAULT_SETTINGS.solverTargetSelectivity,
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
