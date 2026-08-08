import { StateCreator } from "zustand";
import {
  calculateScores,
  getValidMoves,
  initializeBoard,
  opponentPlayer as nextPlayer,
} from "@/domain/game/game-logic";
import { type Move, applyMove, checkGameOver, createMoveRecord } from "@/domain/game/store-helpers";
import { createPassTurnPatch, hasFlippedDiscs } from "@/domain/game/game-session";
import { MoveHistory } from "@/domain/game/move-history";
import type { GameSlice, ReversiState } from "./types";
import type { Services } from "@/services/types";
import { createAutomation } from "@/stores/automation";
import { navigateHistory, goToHistoryMove } from "@/stores/history-navigation";
import { runGameReplacement } from "@/stores/game-replacement";

/**
 * A freshly played move/pass diverges from any analyzed line, so the stale
 * hint result and whole-game analysis result are invalidated. The single
 * expression of that rule for makeMove, including its automatic-pass branch.
 *
 * History navigation does NOT use this: it deliberately keeps
 * `gameAnalysisResult` so you can review the analyzed game while stepping
 * through it (see `withClears` in history-navigation.ts).
 */
function clearedStaleAnalysis(): {
  analyzeResults: null;
  gameAnalysisResult: null;
} {
  return { analyzeResults: null, gameAnalysisResult: null };
}

export function createGameSlice(services: Services): StateCreator<ReversiState, [], [], GameSlice> {
  return (set, get) => {
    // Automation owns the schedule timer / deferred flag in this closure;
    // they are not part of the public store state (CONTEXT.md → Automation).
    const automation = createAutomation(get);
    let initialGameStartPromise: Promise<boolean> | null = null;

    return {
      board: initializeBoard(),
      historyStartBoard: initializeBoard(),
      historyStartPlayer: "black",
      moveHistory: MoveHistory.empty(),
      currentPlayer: "black",
      gameOver: false,
      gameStatus: "waiting",
      lastMove: null,
      validMoves: [],
      skipAnimation: false,
      paused: false,

      triggerAutomation: () => automation.trigger(),
      resumeQueuedAutomation: () => automation.resumeIfQueued(),
      cancelAutomation: () => automation.cancel(),
      queueResumeAutomation: () => automation.queueResume(),

      getScores: () => {
        return calculateScores(get().board);
      },

      isAITurn: () => {
        const { gameMode, gameOver, currentPlayer } = get();
        if (gameOver || gameMode === "pvp") return false;
        return (
          (gameMode === "ai-black" && currentPlayer === "black") ||
          (gameMode === "ai-white" && currentPlayer === "white")
        );
      },

      isValidMove: (row, col) => {
        const { validMoves, gameStatus } = get();
        if (gameStatus !== "playing") {
          return false;
        }
        return validMoves.some((move) => move[0] === row && move[1] === col);
      },

      makeMove: async (move: Move) => {
        if (get().engineActivity.kind === "game-analysis") return;
        automation.cancel();

        // A user move makes any in-flight hint analysis stale. Abort it
        // through the canonical hint path so the Engine Search is properly
        // superseded and re-analysis targets the new position.
        if (!move.isAI && get().engineActivity.kind === "hint") {
          get().restartHintAnalysisAfterAbort();
        }

        const oldBoard = get().board;

        set((state) => {
          const currentPlayer = state.currentPlayer;
          const newBoard = applyMove(state.board, move, currentPlayer);
          const newMoveRecord = createMoveRecord(
            state.moveHistory.length,
            currentPlayer,
            move,
            state.aiRemainingTime,
          );
          const nextPlayerTurn = nextPlayer(currentPlayer);

          return {
            board: newBoard,
            moveHistory: state.moveHistory.append(newMoveRecord),
            currentPlayer: nextPlayerTurn,
            lastMove: move,
            validMoves: getValidMoves(newBoard, nextPlayerTurn),
            ...clearedStaleAnalysis(),
            skipAnimation: false,
          };
        });

        const updatedState = get();
        const { gameOver, shouldPass } = checkGameOver(
          updatedState.board,
          updatedState.currentPlayer,
        );

        if (gameOver) {
          set({ gameOver: true, gameStatus: "finished", showGameOverNotification: true });
          return;
        }

        if (shouldPass) {
          set((state) => ({
            ...createPassTurnPatch(state, updatedState.currentPlayer),
            ...clearedStaleAnalysis(),
            showPassNotification: updatedState.currentPlayer,
          }));

          automation.afterMove({ passed: true, flipped: false });
          return;
        }

        automation.afterMove({
          passed: false,
          flipped: !move.isAI && hasFlippedDiscs(oldBoard, updatedState.board),
        });
      },

      undoMove: () => navigateHistory(get, set, "undo"),

      redoMove: () => navigateHistory(get, set, "redo"),

      resumeAI: () => {
        if (get().engineActivity.kind === "game-analysis") {
          set({ paused: false });
          automation.queueResume();
          return;
        }
        set({ paused: false });
        automation.cancel();
        automation.trigger();
      },

      goToMove: (position: number) => goToHistoryMove(get, set, position),

      startGame: async (settings) =>
        runGameReplacement(services, get, set, { kind: "new-game", settings }),

      startInitialGame: () => {
        // React StrictMode replays mount effects in development. Keep launch
        // auto-start idempotent at the store boundary so concurrent callers
        // share one Game Replacement transaction. Best-effort: unlike the
        // modal starters (guarded by runGuardedStart), the launch caller has
        // no failure UI — resolve false so App still leaves the loading
        // screen and the user can start a game manually.
        initialGameStartPromise ??= runGameReplacement(services, get, set, {
          kind: "new-game",
          pauseForAITurn: true,
        }).catch((error) => {
          console.error("Failed to auto-start the initial game:", error);
          return false;
        });
        return initialGameStartPromise;
      },
    };
  };
}
