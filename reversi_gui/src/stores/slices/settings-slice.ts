import { StateCreator } from "zustand";
import type { ReversiState, SettingsSlice } from "./types";

export const createSettingsSlice: StateCreator<
    ReversiState,
    [],
    [],
    SettingsSlice
> = (set, get) => ({
    gameMode: "ai-white",
    timeLimit: 1,
    gameTimeLimit: 60, // 1 minute

    setGameMode: (mode) => {
        set({
            gameMode: mode,
            analyzeResults: null
        });

        if (mode === "analyze" && get().gameStatus === "playing") {
            void get().analyzeBoard();
        }
    },

    setTimeLimit: (limit) => set({ timeLimit: limit }),

    setGameTimeLimit: (limit) => set({ gameTimeLimit: limit }),
});
