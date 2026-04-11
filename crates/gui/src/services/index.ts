export type { AIService, SettingsService, Services, AIMoveResult, AIMoveProgress, GameAnalysisProgress, AppSettings, SolverService, SolverProgressPayload, SolverCandidate, SolverSelectivity } from "./types";
export { DEFAULT_SETTINGS, SOLVER_SELECTIVITIES, SOLVER_SELECTIVITY_TO_U8 } from "./types";

import type { Services } from "./types";
import { TauriAIService } from "./tauri-ai-service";
import { TauriSettingsService } from "./tauri-settings-service";
import { TauriSolverService } from "./tauri-solver-service";

export const defaultServices: Services = {
  ai: new TauriAIService(),
  settings: new TauriSettingsService(),
  solver: new TauriSolverService(),
};
