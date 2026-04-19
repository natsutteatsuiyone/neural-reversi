import { StateCreator } from "zustand";
import type { AISlice, ReversiState } from "./types";
import { DEFAULT_SETTINGS, type Services } from "@/services/types";

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
        const aiSearchStartTime = Date.now();
        set({ isAIThinking: true, aiThinkingHistory: [], aiSearchStartTime });
        const { currentPlayer: player, board, aiLevel, aiMode, timeLimit, aiRemainingTime } = get();
        const initialRemainingTime = aiRemainingTime;
        const isGameTime = aiMode === "game-time";
        // When another actor (undo/redo) writes `aiRemainingTime` during the
        // search, stop our own updates so we don't clobber theirs.
        let managing = isGameTime;
        let lastWritten = initialRemainingTime;

        const applyManagedRemainingTime = (remainingTime: number) => {
            if (!managing) return;
            if (remainingTime === lastWritten) return;
            if (get().aiRemainingTime !== lastWritten) {
                managing = false;
                clearSearchTimer(get, set);
                return;
            }
            lastWritten = remainingTime;
            set({ aiRemainingTime: remainingTime });
        };

        if (isGameTime) {
            const timer = setInterval(() => {
                const elapsed = Date.now() - aiSearchStartTime;
                const remaining = Math.max(0, initialRemainingTime - elapsed);
                applyManagedRemainingTime(remaining);
                if (remaining === 0) clearSearchTimer(get, set);
            }, 100);
            set({ searchTimer: timer });
        }

        try {
            const aiMove = await services.ai.getAIMove(
                board,
                player,
                aiLevel,
                aiMode === "time" ? timeLimit * 1000 : undefined,
                isGameTime ? aiRemainingTime : undefined,
                (progress) => {
                    set((state) => {
                        const last = state.aiThinkingHistory[state.aiThinkingHistory.length - 1];
                        if (last &&
                            last.depth === progress.depth &&
                            last.score === progress.score &&
                            last.nodes === progress.nodes &&
                            last.pvLine === progress.pvLine) {
                            // Same reference short-circuits zustand's listener
                            // loop (Object.is), avoiding re-renders on no-op
                            // engine re-emissions.
                            return state;
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
            if (aiMove && isGameTime) {
                applyManagedRemainingTime(Math.max(0, initialRemainingTime - aiMove.timeTaken));
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
