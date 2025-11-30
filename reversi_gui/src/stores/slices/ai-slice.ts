import { StateCreator } from "zustand";
import { abortAISearch, getAIMove } from "@/lib/ai";
import type { AISlice, ReversiState } from "./types";

export const createAISlice: StateCreator<
    ReversiState,
    [],
    [],
    AISlice
> = (set, get) => ({
    aiLevel: 21,
    aiMoveProgress: null,
    isAIThinking: false,
    lastAIMove: null,
    aiMode: "game-time",
    aiRemainingTime: 600000,
    searchTimer: null,

    makeAIMove: async () => {
        set({ isAIThinking: true });
        const player = get().currentPlayer;
        const board = get().board;
        const { aiLevel, aiMode, timeLimit, aiRemainingTime } = get();

        // Check if time is up for game-time mode
        if (aiMode === "game-time" && aiRemainingTime <= 0) {
            // Time is up! Make a random/quick move or just pass 0 time to let backend handle it (if it supports 0 time)
            // For now, we'll just proceed with 0 time, but ideally we should handle timeout.
        }

        // Start timer for game-time mode
        if (aiMode === "game-time") {
            const startTime = Date.now();
            const initialRemaining = aiRemainingTime;
            const timer = setInterval(() => {
                const elapsed = Date.now() - startTime;
                set({ aiRemainingTime: Math.max(0, initialRemaining - elapsed) });
            }, 100);
            set({ searchTimer: timer });
        }

        try {
            const aiMove = await getAIMove(
                board,
                player,
                aiLevel,
                aiMode === "time" ? timeLimit * 1000 : undefined,
                aiMode === "game-time" ? aiRemainingTime : undefined,
                (ev) => {
                    set({ aiMoveProgress: ev.payload });
                }
            );

            // Clear timer
            const { searchTimer } = get();
            if (searchTimer) {
                clearInterval(searchTimer);
                set({ searchTimer: null });
            }

            if (aiMove && aiMode === "game-time") {
                set({
                    aiRemainingTime: Math.max(0, aiRemainingTime - aiMove.timeTaken)
                });
            }
            set({ aiMoveProgress: null, isAIThinking: false });
            if (aiMove) {
                const move = {
                    row: aiMove.row,
                    col: aiMove.col,
                    score: aiMove.score,
                    isAI: true,
                };
                get().makeMove(move);
                set({
                    lastAIMove: aiMove,
                });
            }
        } catch (error) {
            // Ensure timer is cleared on error
            const { searchTimer } = get();
            if (searchTimer) {
                clearInterval(searchTimer);
                set({ searchTimer: null });
            }
            set({ isAIThinking: false, aiMoveProgress: null });
            console.error("AI Move failed:", error);
        }
    },

    abortAIMove: async () => {
        if (get().isAIThinking || get().isAnalyzing) {
            await abortAISearch();

            const { searchTimer } = get();
            if (searchTimer) {
                clearInterval(searchTimer);
            }

            set({
                isAIThinking: false,
                isAnalyzing: false,
                aiMoveProgress: null,
                searchTimer: null
            });
        }
    },

    setAILevelChange: (level) => set({ aiLevel: level }),

    setAIMode: (mode) => set({ aiMode: mode }),
});
