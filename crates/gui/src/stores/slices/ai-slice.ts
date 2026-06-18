import { StateCreator } from "zustand";
import type { AISlice, ReversiState } from "./types";
import {
  DEFAULT_SETTINGS,
  type AIMoveProgress,
  type AIMoveResult,
  type Services,
} from "@/services/types";
import { runAIMoveSearch } from "@/services/ai-move-search-operation";
import type { EngineSearch } from "@/domain/engine/engine-search";
import { isGameSearchActive } from "@/stores/engine-activity";

export function createAISlice(
  services: Services,
  engineSearch: EngineSearch,
): StateCreator<ReversiState, [], [], AISlice> {
  return (set, get) => {
    // The game-time countdown interval is an implementation detail of the
    // in-flight AI-move search, owned privately here (like Automation's timer
    // or HintAnalysisSession's generation counter) rather than mirrored into
    // public store state.
    let activeSearchTimer: ReturnType<typeof setInterval> | null = null;
    const cancelSearchTimer = (): void => {
      if (activeSearchTimer) {
        clearInterval(activeSearchTimer);
        activeSearchTimer = null;
      }
    };

    return {
      aiLevel: DEFAULT_SETTINGS.aiLevel,
      aiMoveProgress: null,
      aiThinkingHistory: [],
      isAIThinking: false,
      lastAIMove: null,
      aiMode: DEFAULT_SETTINGS.aiMode,
      aiRemainingTime: 600000,

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
        if (isGameSearchActive(state.engineActivity)) return;
        const { currentPlayer: player, board, aiLevel, aiMode, timeLimit, aiRemainingTime } = state;

        let aiMove: AIMoveResult = null;
        await engineSearch.start<{ progress: AIMoveProgress; nps: number }, AIMoveResult>({
          kind: "ai-move",
          onStart: () => {},
          run: (accept, run) =>
            runAIMoveSearch({
              ai: services.ai,
              board,
              player,
              level: aiLevel,
              mode: aiMode,
              timeLimitSeconds: timeLimit,
              remainingTimeMs: aiRemainingTime,
              getRemainingTime: () => get().aiRemainingTime,
              onStart: () => {
                // `isAIThinking` is the Engine Activity (stamped at claim by its
                // owner); only the AI-move payload is committed here, still
                // gated to the current run.
                if (run.isCurrent()) set({ aiThinkingHistory: [] });
              },
              onTimerChange: (timer) => {
                if (run.isCurrent()) activeSearchTimer = timer;
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
          onResult: (move) => {
            aiMove = move;
          },
          onError: (error) => console.error("AI Move failed:", error),
          // `isAIThinking` returns to idle via the Engine Activity owner; only
          // the AI-move payload is cleared here.
          onTeardown: () => set({ aiMoveProgress: null }),
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

      stopAIMove: async () => {
        cancelSearchTimer();
        if (!get().isAIThinking) return;

        try {
          // User-facing Stop keeps the current EngineSearch run alive. The
          // backend abort makes `getAIMove` return its best-so-far result,
          // which `makeAIMove` then commits through the normal result path.
          await services.ai.abortSearch();
        } catch (error) {
          console.error("AI stop failed:", error);
        }
      },

      abortAIMove: async () => {
        // Stop the countdown interval SYNCHRONOUSLY at teardown entry, exactly
        // as Game Replacement's `abortInFlightGameSearches` relies on: external
        // teardowns reach the AI-move feature only through this action, so the
        // timer never out-lives the search it belongs to.
        cancelSearchTimer();
        const { isAIThinking, isAnalyzing, hintAnalysisAbortPending } = get();
        // `hintAnalysisAbortPending` covers the window where a hint
        // abort-then-restart has already stamped Engine Activity back to idle
        // (so `isAnalyzing` is false) but its backend abort + restart are still
        // in flight. Aborting here supersedes that pending run via EngineSearch,
        // dropping its stale restart (CONTEXT.md → Engine Search).
        if (!isAIThinking && !isAnalyzing && !hintAnalysisAbortPending) return;
        const shouldPauseAI = isAIThinking && get().isAITurn() && get().validMoves.length > 0;
        await engineSearch.abort({
          abort: () => services.ai.abortSearch(),
          onError: (error) => console.error("AI abort failed:", error),
          onSettled: () => {
            // `isAIThinking`/`isAnalyzing` return to idle via the Engine
            // Activity owner (abort stamps idle at claim); only payload here.
            set({
              aiMoveProgress: null,
              paused: shouldPauseAI,
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
    };
  };
}
