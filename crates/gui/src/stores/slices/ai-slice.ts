import { StateCreator } from "zustand";
import type { AISlice, ReversiState } from "./types";
import { DEFAULT_SETTINGS, type Services } from "@/services/types";
import { runAIMoveSearch } from "@/services/ai-move-search-operation";

export function clearSearchTimer(
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

    checkAIReady: async () => {
      try {
        await services.ai.checkReady();
        return true;
      } catch (error) {
        console.error("AI readiness check failed:", error);
        return false;
      }
    },

    makeAIMove: async () => {
        const { currentPlayer: player, board, aiLevel, aiMode, timeLimit, aiRemainingTime } = get();

        try {
            const aiMove = await runAIMoveSearch({
                ai: services.ai,
                board,
                player,
                level: aiLevel,
                mode: aiMode,
                timeLimitSeconds: timeLimit,
                remainingTimeMs: aiRemainingTime,
                getRemainingTime: () => get().aiRemainingTime,
                onStart: (aiSearchStartTime) => {
                    set({ isAIThinking: true, aiThinkingHistory: [], aiSearchStartTime });
                },
                onTimerChange: (searchTimer) => {
                    set({ searchTimer });
                },
                onRemainingTime: (remainingTime) => {
                    set({ aiRemainingTime: remainingTime });
                },
                onProgress: ({ progress, nps }) => {
                    set((state) => ({
                        aiMoveProgress: progress,
                        aiThinkingHistory: [...state.aiThinkingHistory, { ...progress, nps }],
                    }));
                },
                onFinish: () => {
                    set({ aiMoveProgress: null, isAIThinking: false });
                },
            });
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
            console.error("AI Move failed:", error);
        }
    },

    abortAIMove: async () => {
        const { isAIThinking, isAnalyzing } = get();
        if (!isAIThinking && !isAnalyzing) return;
        try {
            await services.ai.abortSearch();
            clearSearchTimer(get, set);
            set({
                isAIThinking: false,
                isAnalyzing: false,
                aiMoveProgress: null,
            });
        } catch (error) {
            console.error("AI abort failed:", error);
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
