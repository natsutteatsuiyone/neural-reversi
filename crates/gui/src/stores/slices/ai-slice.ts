import { StateCreator } from "zustand";
import type { AIThinkingEntry, AISlice, ReversiState } from "./types";
import {
  DEFAULT_SETTINGS,
  type AIMoveProgress,
  type AIMoveResult,
  type Services,
} from "@/services/types";
import { runAIMoveSearch } from "@/services/ai-move-search-operation";
import type { EngineSearch } from "@/domain/engine/engine-search";
import { isGameSearchActive } from "@/stores/engine-activity";

function isSameThinkingLogRow(a: AIThinkingEntry | undefined, b: AIThinkingEntry): boolean {
  return (
    a !== undefined &&
    a.depth === b.depth &&
    a.acc === b.acc &&
    a.score === b.score &&
    a.pvLine === b.pvLine
  );
}

function upsertThinkingHistoryEntry(
  history: readonly AIThinkingEntry[],
  entry: AIThinkingEntry,
): AIThinkingEntry[] {
  if (!isSameThinkingLogRow(history[history.length - 1], entry)) {
    return [...history, entry];
  }

  return [...history.slice(0, -1), entry];
}

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
        const { currentPlayer: player, board, aiLevel, aiMode, aiRemainingTime } = state;

        let aiMove: AIMoveResult = null;
        await engineSearch.start<{ progress: AIMoveProgress; nps: number }, AIMoveResult>({
          kind: "ai-move",
          run: (accept, run) =>
            runAIMoveSearch({
              ai: services.ai,
              board,
              player,
              level: aiLevel,
              mode: aiMode,
              remainingTimeMs: aiRemainingTime,
              getRemainingTime: () => get().aiRemainingTime,
              onStart: () => {
                // Only the AI-move payload is committed here, gated to the
                // current run.
                if (run.isCurrent()) set({ aiThinkingHistory: [] });
              },
              onTimerChange: (timer) => {
                if (run.isCurrent()) activeSearchTimer = timer;
              },
              onRemainingTime: (remainingTime) => {
                if (run.isCurrent()) set({ aiRemainingTime: remainingTime });
              },
              onProgress: accept,
            }),
          abort: () => services.ai.abortSearch(),
          onProgress: ({ progress, nps }) =>
            set((s) => ({
              aiMoveProgress: progress,
              aiThinkingHistory: upsertThinkingHistoryEntry(s.aiThinkingHistory, {
                ...progress,
                nps,
              }),
            })),
          onResult: (move) => {
            aiMove = move;
          },
          onError: (error) => console.error("AI Move failed:", error),
          // Only the AI-move payload is cleared here; EngineSearch owns activity.
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
        if (get().engineActivity.kind !== "ai-move") return;

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
        const { engineActivity, hintAnalysisAbortPending } = get();
        // `hintAnalysisAbortPending` covers the window where a hint
        // abort-then-restart has already returned Engine Activity to idle but
        // its backend abort + restart are still in flight.
        if (
          engineActivity.kind !== "ai-move" &&
          engineActivity.kind !== "hint" &&
          !hintAnalysisAbortPending
        ) {
          return;
        }
        const shouldPauseAI =
          engineActivity.kind === "ai-move" && get().isAITurn() && get().validMoves.length > 0;
        await engineSearch.abort({
          abort: () => services.ai.abortSearch(),
          onError: (error) => console.error("AI abort failed:", error),
          onSettled: () => {
            // EngineSearch returns activity to idle; clear only feature payload.
            set({
              aiMoveProgress: null,
              paused: shouldPauseAI,
            });
          },
        });
      },
    };
  };
}
