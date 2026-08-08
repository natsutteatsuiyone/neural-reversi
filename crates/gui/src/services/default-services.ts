import type { Services } from "./types";
import { TauriAIService } from "./tauri-ai-service";
import { TauriSettingsService } from "./tauri-settings-service";
import { TauriSolverService } from "./tauri-solver-service";

export const defaultServices: Services = {
  ai: new TauriAIService(),
  settings: new TauriSettingsService(),
  solver: new TauriSolverService(),
};
