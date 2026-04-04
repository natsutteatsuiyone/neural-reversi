import { StateCreator } from "zustand";
import type { ReversiState, SettingsSlice } from "./types";
import { DEFAULT_SETTINGS, type Services } from "@/services/types";

export function createSettingsSlice(services: Services): StateCreator<
    ReversiState,
    [],
    [],
    SettingsSlice
> {
  return (set, get) => ({
    gameMode: DEFAULT_SETTINGS.gameMode,
    timeLimit: DEFAULT_SETTINGS.timeLimit,
    gameTimeLimit: DEFAULT_SETTINGS.gameTimeLimit,
    hintLevel: DEFAULT_SETTINGS.hintLevel,
    gameAnalysisLevel: DEFAULT_SETTINGS.gameAnalysisLevel,
    hashSize: DEFAULT_SETTINGS.hashSize,
    aiAnalysisPanelOpen: DEFAULT_SETTINGS.aiAnalysisPanelOpen,

    setGameMode: (mode) => {
        set({
            gameMode: mode,
            analyzeResults: null
        });
        void services.settings.saveSetting("gameMode", mode);
    },

    setTimeLimit: (limit) => {
        set({ timeLimit: limit });
        void services.settings.saveSetting("timeLimit", limit);
    },

    setGameTimeLimit: (limit) => {
        set({ gameTimeLimit: limit });
        void services.settings.saveSetting("gameTimeLimit", limit);
    },

    setHintLevel: (level) => {
        set({ hintLevel: level, analyzeResults: null });
        void services.settings.saveSetting("hintLevel", level);
        if (get().isHintMode) {
            get().analyzeBoard();
        }
    },

    setGameAnalysisLevel: (level) => {
        set({ gameAnalysisLevel: level });
        void services.settings.saveSetting("gameAnalysisLevel", level);
    },

    setHashSize: (size) => {
        set({ hashSize: size });
        void services.settings.saveSetting("hashSize", size);
        void services.ai.resizeTT(size);
    },

    setAIAnalysisPanelOpen: (open) => {
        set({ aiAnalysisPanelOpen: open });
        void services.settings.saveSetting("aiAnalysisPanelOpen", open);
    },
  });
}
