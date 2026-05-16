import { create } from "zustand";
import type { ReversiState } from "./slices/types";
import type { Services } from "@/services/types";
import { createGameSlice } from "./slices/game-slice";
import { createAISlice } from "./slices/ai-slice";
import { createUISlice } from "./slices/ui-slice";
import { createSettingsSlice } from "./slices/settings-slice";
import { createSetupSlice } from "./slices/setup-slice";
import { createSolverSlice } from "./slices/solver-slice";
import { createEngineSearch } from "@/domain/engine/engine-search";
import { defaultServices } from "@/services";

export function createReversiStore(services: Services) {
  const engineSearch = createEngineSearch();
  return create<ReversiState>()((...a) => ({
    ...createGameSlice(services)(...a),
    ...createAISlice(services, engineSearch)(...a),
    ...createUISlice(services, engineSearch)(...a),
    ...createSettingsSlice(services)(...a),
    ...createSetupSlice(services)(...a),
    ...createSolverSlice(services, engineSearch)(...a),
  }));
}

export const useReversiStore = createReversiStore(defaultServices);
