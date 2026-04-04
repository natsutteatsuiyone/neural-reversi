import { load, type Store } from "@tauri-apps/plugin-store";
import type { AIMode, GameMode } from "@/types";
import type { Language } from "@/i18n";
import { DEFAULT_SETTINGS, type AppSettings, type SettingsService } from "./types";

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
        hintLevel, gameAnalysisLevel, hashSize, aiAnalysisPanelOpen, language,
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
        s.get<Language | null>("language"),
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
        language: language ?? DEFAULT_SETTINGS.language,
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
