import { create, type StoreApi } from "zustand";
import type { ReversiState } from "./slices/types";
import type { Services } from "@/services/types";
import { createGameSlice } from "./slices/game-slice";
import { createAISlice } from "./slices/ai-slice";
import { createUISlice } from "./slices/ui-slice";
import { createSettingsSlice } from "./slices/settings-slice";
import { createSetupSlice } from "./slices/setup-slice";
import { createSolverSlice } from "./slices/solver-slice";
import { createEngineSearch } from "@/domain/engine/engine-search";
import {
  HintAnalysisSession,
  type HintAnalysisSessionCommit,
} from "@/domain/game/hint-analysis-session";
import {
  GameAnalysisSession,
  type GameAnalysisSessionCommit,
} from "@/domain/game/game-analysis-session";
import { engineActivityPatch, IDLE_ENGINE_ACTIVITY } from "./engine-activity";
import { defaultServices } from "@/services";

export function createReversiStore(services: Services) {
  // Captured after the store exists; `onActivityChange` only fires during an
  // async start/abort, long after this assignment.
  let setState: StoreApi<ReversiState>["setState"] | null = null;
  const engineSearch = createEngineSearch({
    onActivityChange: (activity) =>
      setState?.(engineActivityPatch(activity)),
  });
  const store = create<ReversiState>()((set, get, api) => {
    // One Hint Analysis Session, shared by the UI slice (toggle/analyze) and
    // the settings slice (level change), mirroring how `engineSearch` is
    // created once and injected into multiple slices.
    const hintCommit: HintAnalysisSessionCommit = (partial) =>
      set(partial as Parameters<typeof set>[0]);
    const hintSession = new HintAnalysisSession({
      ai: services.ai,
      read: get,
      commit: hintCommit,
      engineSearch,
    });
    // One Game Analysis Session, same shape as hintSession (CONTEXT.md →
    // Engine Activity): the UI slice delegates analyze/abort to it.
    const gameAnalysisCommit: GameAnalysisSessionCommit = (partial) =>
      set(partial as Parameters<typeof set>[0]);
    const gameAnalysisSession = new GameAnalysisSession({
      ai: services.ai,
      read: get,
      commit: gameAnalysisCommit,
      engineSearch,
    });
    return {
      engineActivity: IDLE_ENGINE_ACTIVITY,
      ...createGameSlice(services)(set, get, api),
      ...createAISlice(services, engineSearch)(set, get, api),
      ...createUISlice(hintSession, gameAnalysisSession)(set, get, api),
      ...createSettingsSlice(services, hintSession)(set, get, api),
      ...createSetupSlice(services)(set, get, api),
      ...createSolverSlice(services, engineSearch)(set, get, api),
    };
  });
  setState = store.setState;
  return store;
}

export const useReversiStore = createReversiStore(defaultServices);
