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
    rightPanelSize: DEFAULT_SETTINGS.rightPanelSize,
    bottomPanelSize: DEFAULT_SETTINGS.bottomPanelSize,
    language: DEFAULT_SETTINGS.language,

    hydrateSettings: (settings) => {
        const shouldResizeTT = get().hashSize !== settings.hashSize;
        set({
            gameMode: settings.gameMode,
            timeLimit: settings.timeLimit,
            gameTimeLimit: settings.gameTimeLimit,
            hintLevel: settings.hintLevel,
            gameAnalysisLevel: settings.gameAnalysisLevel,
            hashSize: settings.hashSize,
            aiAnalysisPanelOpen: settings.aiAnalysisPanelOpen,
            rightPanelSize: settings.rightPanelSize,
            bottomPanelSize: settings.bottomPanelSize,
            aiLevel: settings.aiLevel,
            aiMode: settings.aiMode,
            language: settings.language,
            targetSelectivity: settings.solverTargetSelectivity,
            solverMode: settings.solverMode,
            analyzeResults: null,
        });
        if (shouldResizeTT) {
            void services.ai.resizeTT(settings.hashSize);
        }
    },

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
        if (level === get().hintLevel) return;
        set({ hintLevel: level, analyzeResults: null });
        void services.settings.saveSetting("hintLevel", level);

        const state = get();
        if (!state.isHintMode || state.hintAnalysisAbortPending) return;

        if (state.isAnalyzing && !state.isAIThinking) {
            state.restartHintAnalysisAfterAbort();
        } else {
            void state.analyzeBoard();
        }
    },

    setGameAnalysisLevel: (level) => {
        set({ gameAnalysisLevel: level });
        void services.settings.saveSetting("gameAnalysisLevel", level);
    },

    setHashSize: (size) => {
        if (size === get().hashSize) return;
        set({ hashSize: size });
        void services.settings.saveSetting("hashSize", size);
        void services.ai.resizeTT(size);
    },

    setAIAnalysisPanelOpen: (open) => {
        set({ aiAnalysisPanelOpen: open });
        void services.settings.saveSetting("aiAnalysisPanelOpen", open);
    },

    setRightPanelSize: (size) => {
        if (size === get().rightPanelSize) return;
        set({ rightPanelSize: size });
        void services.settings.saveSetting("rightPanelSize", size);
    },

    setBottomPanelSize: (size) => {
        if (size === get().bottomPanelSize) return;
        set({ bottomPanelSize: size });
        void services.settings.saveSetting("bottomPanelSize", size);
    },

    setLanguagePreference: async (language) => {
        set({ language });
        return services.settings.saveSetting("language", language);
    },
  });
}
