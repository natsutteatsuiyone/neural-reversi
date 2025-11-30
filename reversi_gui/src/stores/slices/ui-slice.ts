import { StateCreator } from "zustand";
import { analyze, type AIMoveProgress } from "@/lib/ai";
import { abortAISearch } from "@/lib/ai";
import type { ReversiState, UISlice } from "./types";
import { triggerAutomation } from "./game-slice";

export const createUISlice: StateCreator<
    ReversiState,
    [],
    [],
    UISlice
> = (set, get) => ({
    showPassNotification: false,
    isAnalyzing: false,
    analyzeResults: null,

    hidePassNotification: () => {
        set({ showPassNotification: false });
        const { makePass } = get();
        makePass();
        triggerAutomation(get());
    },

    analyzeBoard: async () => {
        if (get().gameMode !== "analyze" || get().gameStatus !== "playing") {
            return;
        }

        await abortAISearch();

        const board = get().board;
        const player = get().currentPlayer;
        const results = new Map<string, AIMoveProgress>();

        set({ analyzeResults: null, isAnalyzing: true });

        try {
            await analyze(board, player, get().aiLevel, (ev) => {
                if (ev.payload.row !== undefined && ev.payload.col !== undefined) {
                    const key = `${ev.payload.row},${ev.payload.col}`;
                    results.set(key, ev.payload);
                    set({ analyzeResults: new Map(results) });
                }
            });
        } finally {
            set({ isAnalyzing: false });
        }
    },
});
