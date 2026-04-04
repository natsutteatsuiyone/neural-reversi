import { create } from "zustand";
import type { ReversiState } from "./slices/types";
import type { Services } from "@/services/types";
import { createGameSlice } from "./slices/game-slice";
import { createAISlice } from "./slices/ai-slice";
import { createUISlice } from "./slices/ui-slice";
import { createSettingsSlice } from "./slices/settings-slice";
import { createSetupSlice } from "./slices/setup-slice";
import { defaultServices } from "@/services";

export function createReversiStore(services: Services) {
  return create<ReversiState>()((...a) => ({
    ...createGameSlice(services)(...a),
    ...createAISlice(services)(...a),
    ...createUISlice(services)(...a),
    ...createSettingsSlice(services)(...a),
    ...createSetupSlice(services)(...a),
  }));
}

export const useReversiStore = createReversiStore(defaultServices);
