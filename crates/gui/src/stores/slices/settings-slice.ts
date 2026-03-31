import { StateCreator } from "zustand";
import type { ReversiState, SettingsSlice } from "./types";
import { saveSetting } from "@/lib/settings-store";
import { resizeTT } from "@/lib/ai";

export const createSettingsSlice: StateCreator<
    ReversiState,
    [],
    [],
    SettingsSlice
> = (set, get) => ({
    gameMode: "ai-white",
    timeLimit: 1,
    gameTimeLimit: 60, // 1 minute
    hintLevel: 21,
    hashSize: 512,
    aiAnalysisPanelOpen: false,

    setGameMode: (mode) => {
        set({
            gameMode: mode,
            analyzeResults: null
        });
        void saveSetting("gameMode", mode);
    },

    setTimeLimit: (limit) => {
        set({ timeLimit: limit });
        void saveSetting("timeLimit", limit);
    },

    setGameTimeLimit: (limit) => {
        set({ gameTimeLimit: limit });
        void saveSetting("gameTimeLimit", limit);
    },

    setHintLevel: (level) => {
        set({ hintLevel: level, analyzeResults: null });
        void saveSetting("hintLevel", level);
        if (get().isHintMode) {
            get().analyzeBoard();
        }
    },

    setHashSize: (size) => {
        set({ hashSize: size });
        void saveSetting("hashSize", size);
        void resizeTT(size);
    },

    setAIAnalysisPanelOpen: (open) => {
        set({ aiAnalysisPanelOpen: open });
        void saveSetting("aiAnalysisPanelOpen", open);
    },
});
