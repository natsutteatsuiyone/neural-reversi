import { StateCreator } from "zustand";
import type { AISlice, ReversiState } from "./types";
import { DEFAULT_SETTINGS, type AIMoveProgress, type AIMoveResult, type Services } from "@/services/types";
import { runAIMoveSearch } from "@/services/ai-move-search-operation";
import type { EngineSearch } from "@/domain/engine/engine-search";

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

export function createAISlice(services: Services, engineSearch: EngineSearch): StateCreator<
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
      const state = get();
      if (state.isAIThinking || state.isAnalyzing || state.isGameAnalyzing) return;
      const { currentPlayer: player, board, aiLevel, aiMode, timeLimit, aiRemainingTime } = state;

      let aiMove: AIMoveResult = null;
      await engineSearch.start<{ progress: AIMoveProgress; nps: number }, AIMoveResult>({
        onStart: () => {},
        run: (accept, run) =>
          runAIMoveSearch({
            ai: services.ai, board, player, level: aiLevel, mode: aiMode,
            timeLimitSeconds: timeLimit, remainingTimeMs: aiRemainingTime,
            getRemainingTime: () => get().aiRemainingTime,
            onStart: (aiSearchStartTime) => {
              if (run.isCurrent())
                set({ isAIThinking: true, aiThinkingHistory: [], aiSearchStartTime });
            },
            onTimerChange: (searchTimer) => {
              if (run.isCurrent()) set({ searchTimer });
            },
            onRemainingTime: (remainingTime) => {
              if (run.isCurrent()) set({ aiRemainingTime: remainingTime });
            },
            onProgress: accept,
            onFinish: () => {},
          }),
        abort: () => services.ai.abortSearch(),
        onProgress: ({ progress, nps }) =>
          set((s) => ({
            aiMoveProgress: progress,
            aiThinkingHistory: [...s.aiThinkingHistory, { ...progress, nps }],
          })),
        onResult: (move) => { aiMove = move; },
        onError: (error) => console.error("AI Move failed:", error),
        onTeardown: () =>
          set({ aiMoveProgress: null, isAIThinking: false, aiSearchStartTime: null }),
      });

      // `aiMove` is only ever written from the `onResult` callback above, so
      // TS narrows the post-await read to `never`; reassert the declared type.
      const result = aiMove as AIMoveResult;
      if (result) {
        const move = { row: result.row, col: result.col, score: result.score, isAI: true };
        await get().makeMove(move);
        set({ lastAIMove: result });
      }
    },

    abortAIMove: async () => {
      const { isAIThinking, isAnalyzing } = get();
      if (!isAIThinking && !isAnalyzing) return;
      const shouldPauseAI = isAIThinking && get().isAITurn() && get().validMoves.length > 0;
      await engineSearch.abort({
        abort: () => services.ai.abortSearch(),
        onError: (error) => console.error("AI abort failed:", error),
        onSettled: () => {
          clearSearchTimer(get, set);
          set({
            isAIThinking: false, isAnalyzing: false, aiMoveProgress: null,
            aiSearchStartTime: null, paused: shouldPauseAI,
          });
        },
      });
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
