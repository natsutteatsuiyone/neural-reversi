import { StateCreator } from "zustand";
import type { HintAnalysisSession } from "@/domain/game/hint-analysis-session";
import type { ReversiState, SettingsSlice } from "./types";
import { DEFAULT_SETTINGS, type Services } from "@/services/types";

export function createSettingsSlice(
  services: Services,
  hintSession: HintAnalysisSession,
): StateCreator<ReversiState, [], [], SettingsSlice> {
  return (set, get) => ({
    // Production hydrates before rendering. Non-hydrated test stores keep the
    // human moving first so automation does not start implicitly.
    gameMode: "ai-white",
    gameTimeLimit: DEFAULT_SETTINGS.gameTimeLimit,
    hintLevel: DEFAULT_SETTINGS.hintLevel,
    gameAnalysisLevel: DEFAULT_SETTINGS.gameAnalysisLevel,
    hashSize: DEFAULT_SETTINGS.hashSize,
    aiAnalysisPanelOpen: DEFAULT_SETTINGS.aiAnalysisPanelOpen,
    rightPanelSize: DEFAULT_SETTINGS.rightPanelSize,
    bottomPanelSize: DEFAULT_SETTINGS.bottomPanelSize,
    language: DEFAULT_SETTINGS.language,

    setHintLevel: (level) => {
      if (level === get().hintLevel) return;
      set({ hintLevel: level, analyzeResults: null });

      // The hint level-change coordination (dedupe guard + restart-vs-
      // analyze decision) belongs to the Hint Analysis feature; settings
      // only persists the level. (CONTEXT.md → Engine Activity.)
      hintSession.onLevelChanged();
    },

    setGameAnalysisLevel: (level) => {
      set({ gameAnalysisLevel: level });
    },

    setHashSize: (size) => {
      if (size === get().hashSize) return;
      set({ hashSize: size });
      void services.ai.resizeTT(size);
    },

    setAIAnalysisPanelOpen: (open) => {
      set({ aiAnalysisPanelOpen: open });
    },

    setRightPanelSize: (size) => {
      if (size === get().rightPanelSize) return;
      set({ rightPanelSize: size });
    },

    setBottomPanelSize: (size) => {
      if (size === get().bottomPanelSize) return;
      set({ bottomPanelSize: size });
    },

    setLanguagePreference: (language) => set({ language }),
  });
}
