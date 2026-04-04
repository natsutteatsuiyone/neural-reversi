export type { AIService, SettingsService, Services, AIMoveResult, AIMoveProgress, GameAnalysisProgress, AppSettings } from "./types";
export { DEFAULT_SETTINGS } from "./types";

import type { Services } from "./types";
import { TauriAIService } from "./tauri-ai-service";
import { TauriSettingsService } from "./tauri-settings-service";

export const defaultServices: Services = {
  ai: new TauriAIService(),
  settings: new TauriSettingsService(),
};
