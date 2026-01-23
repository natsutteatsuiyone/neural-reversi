import { StateCreator } from "zustand";
import { analyze, type AIMoveProgress } from "@/lib/ai";
import type { ReversiState, UISlice } from "./types";
import { triggerAutomation } from "./game-slice";

export const createUISlice: StateCreator<
    ReversiState,
    [],
    [],
    UISlice
> = (set, get) => ({
    showPassNotification: null,
    isAnalyzing: false,
    analyzeResults: null,
    isNewGameModalOpen: false,
    isHintMode: false,

    setNewGameModalOpen: (open) => set({ isNewGameModalOpen: open }),

    setHintMode: (enabled) => {
        set({ isHintMode: enabled, analyzeResults: null });
        if (enabled) {
            get().analyzeBoard();
        } else {
            get().abortAIMove();
        }
    },

    hidePassNotification: () => {
        set({ showPassNotification: null });
        const { makePass } = get();
        makePass();
        triggerAutomation(get);
    },

    analyzeBoard: async () => {
        const { isHintMode, gameStatus, isAIThinking, isAITurn, isAnalyzing } = get();

        // Analyze only if Hint Mode is ON, game is playing, not AI thinking, not AI's turn, and not already analyzing
        if (!isHintMode || gameStatus !== "playing" || isAIThinking || isAITurn() || isAnalyzing) {
            return;
        }

        const board = get().board;
        const player = get().currentPlayer;
        const results = new Map<string, AIMoveProgress>();

        set({ analyzeResults: null, isAnalyzing: true });

        try {
            await analyze(board, player, get().hintLevel, (ev) => {
                // Check if hint mode was disabled or analysis was cancelled
                if (!get().isHintMode || !get().isAnalyzing) return;

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
