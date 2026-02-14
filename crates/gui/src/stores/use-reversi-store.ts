import { create } from "zustand";
import type { ReversiState } from "./slices/types";
import { createGameSlice } from "./slices/game-slice";
import { createAISlice } from "./slices/ai-slice";
import { createUISlice } from "./slices/ui-slice";
import { createSettingsSlice } from "./slices/settings-slice";
import { createSetupSlice } from "./slices/setup-slice";

export const useReversiStore = create<ReversiState>()((...a) => ({
  ...createGameSlice(...a),
  ...createAISlice(...a),
  ...createUISlice(...a),
  ...createSettingsSlice(...a),
  ...createSetupSlice(...a),
}));
