import { StateCreator } from "zustand";
import type { AISlice, ReversiState } from "./types";
import { DEFAULT_SETTINGS, type Services } from "@/services/types";

function clearSearchTimer(
    get: () => ReversiState,
    set: (partial: Partial<ReversiState>) => void,
): void {
    const { searchTimer } = get();
    if (searchTimer) {
        clearInterval(searchTimer);
        set({ searchTimer: null });
    }
}

export function createAISlice(services: Services): StateCreator<
    ReversiState,
    [],
    [],
    AISlice
> {
  return (set, get) => ({
    aiLevel: DEFAULT_SETTINGS.aiLevel,
    aiMoveProgress: null,
    aiThinkingHistory: [],
    aiSearchStartTime: null,
    isAIThinking: false,
    lastAIMove: null,
    aiMode: DEFAULT_SETTINGS.aiMode,
    aiRemainingTime: 600000,
    searchTimer: null,

    makeAIMove: async () => {
        set({ isAIThinking: true, aiThinkingHistory: [], aiSearchStartTime: Date.now() });
        const { currentPlayer: player, board, aiLevel, aiMode, timeLimit, aiRemainingTime } = get();

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
            const aiMove = await services.ai.getAIMove(
                board,
                player,
                aiLevel,
                aiMode === "time" ? timeLimit * 1000 : undefined,
                aiMode === "game-time" ? aiRemainingTime : undefined,
                (progress) => {
                    set((state) => {
                        // Skip duplicate entries
                        const last = state.aiThinkingHistory[state.aiThinkingHistory.length - 1];
                        if (last &&
                            last.depth === progress.depth &&
                            last.score === progress.score &&
                            last.nodes === progress.nodes &&
                            last.pvLine === progress.pvLine) {
                            return { aiMoveProgress: progress };
                        }

                        // Calculate NPS at this moment
                        const elapsedMs = state.aiSearchStartTime
                            ? Date.now() - state.aiSearchStartTime
                            : 0;
                        const nps = elapsedMs > 0 ? (progress.nodes / elapsedMs) * 1000 : 0;

                        return {
                            aiMoveProgress: progress,
                            aiThinkingHistory: [...state.aiThinkingHistory, { ...progress, nps }]
                        };
                    });
                }
            );

            clearSearchTimer(get, set);

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
                await get().makeMove(move);
                set({
                    lastAIMove: aiMove,
                });
            }
        } catch (error) {
            clearSearchTimer(get, set);
            set({ isAIThinking: false, aiMoveProgress: null });
            console.error("AI Move failed:", error);
        }
    },

    abortAIMove: async () => {
        const { isAIThinking, isAnalyzing } = get();
        if (isAIThinking || isAnalyzing) {
            await services.ai.abortSearch();

            clearSearchTimer(get, set);

            set({
                isAIThinking: false,
                isAnalyzing: false,
                aiMoveProgress: null,
            });
        }
    },

    setAILevelChange: (level) => {
        set({ aiLevel: level });
        void services.settings.saveSetting("aiLevel", level);
    },

    setAIMode: (mode) => {
        set({ aiMode: mode });
        void services.settings.saveSetting("aiMode", mode);
    },

    clearAiThinkingHistory: () => set({ aiThinkingHistory: [] }),
  });
}
