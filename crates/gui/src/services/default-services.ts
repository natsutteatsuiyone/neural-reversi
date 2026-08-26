import type { Services } from "./types";
import { TauriAIService } from "./tauri-ai-service";
import { TauriSolverService } from "./tauri-solver-service";

export const defaultServices: Services = {
  ai: new TauriAIService(),
  solver: new TauriSolverService(),
};
