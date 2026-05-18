import { StateCreator } from "zustand";
import {
    calculateScores,
    getValidMoves,
    initializeBoard,
    opponentPlayer as nextPlayer,
} from "@/domain/game/game-logic";
import {
    type Move,
    applyMove,
    checkGameOver,
    createGameStartState,
    createMoveRecord,
} from "@/domain/game/store-helpers";
import {
    createPassTurnPatch,
    hasFlippedDiscs,
} from "@/domain/game/game-session";
import { MoveHistory } from "@/domain/game/move-history";
import type { GameSlice, ReversiState } from "./types";
import type { Services } from "@/services/types";
import { idleEngineActivityPatch } from "@/stores/engine-activity";
import { createAutomation } from "@/stores/automation";
import { navigateHistory, goToHistoryMove } from "@/stores/history-navigation";
import {
    abortInFlightGameSearches,
    prepareGameReplacement,
} from "@/stores/game-replacement";
import {
    createNewGamePatch,
    persistNewGameSettings,
    resolveNewGameSettings,
} from "@/stores/new-game";

/**
 * A freshly played move/pass diverges from any analyzed line, so the stale
 * hint result and whole-game analysis result are invalidated. The single
 * expression of that rule for the play paths (makeMove / makePass).
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

export function createGameSlice(services: Services): StateCreator<
    ReversiState,
    [],
    [],
    GameSlice
> {
  return (set, get) => {
    // Automation owns the schedule timer / deferred flag in this closure;
    // they are not part of the public store state (CONTEXT.md → Automation).
    const automation = createAutomation(get);

    return {
    board: initializeBoard(),
    historyStartBoard: initializeBoard(),
    historyStartPlayer: "black",
    moveHistory: MoveHistory.empty(),
    currentPlayer: "black",
    gameOver: false,
    gameStatus: "waiting",
    isPass: false,
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
        if (get().isGameAnalyzing) return;
        automation.cancel();

        // A user move makes any in-flight hint analysis stale. Abort it
        // through the canonical hint path so the Engine Search is properly
        // superseded (run id bumped, stale progress dropped, teardown exactly
        // once) and re-analysis targets the new position — instead of poking
        // the `isAnalyzing` projection and the backend directly, which left
        // the run un-superseded (CONTEXT.md → Engine Activity).
        if (!move.isAI && get().isAnalyzing) {
            get().restartHintAnalysisAfterAbort();
        }

        const oldBoard = get().board;

        set((state) => {
            const currentPlayer = state.currentPlayer;
            const newBoard = applyMove(state.board, move, currentPlayer);
            const newMoveRecord = createMoveRecord(state.moveHistory.length, currentPlayer, move, state.aiRemainingTime);
            const nextPlayerTurn = nextPlayer(currentPlayer);

            return {
                board: newBoard,
                moveHistory: state.moveHistory.append(newMoveRecord),
                currentPlayer: nextPlayerTurn,
                isPass: false,
                lastMove: move,
                validMoves: getValidMoves(newBoard, nextPlayerTurn),
                ...clearedStaleAnalysis(),
                skipAnimation: false,
            };
        });

        const updatedState = get();
        const { gameOver, shouldPass } = checkGameOver(updatedState.board, updatedState.currentPlayer);

        if (gameOver) {
            set({ gameOver: true, gameStatus: "finished" });
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

    makePass: () => {
        if (get().isGameAnalyzing) return;
        automation.cancel();
        set((state) => ({
            ...createPassTurnPatch(state),
            ...clearedStaleAnalysis(),
        }));
    },

    undoMove: () => navigateHistory(get, set, "undo"),

    redoMove: () => navigateHistory(get, set, "redo"),

    resumeAI: () => {
        if (get().isGameAnalyzing) {
            set({ paused: false });
            automation.queueResume();
            return;
        }
        set({ paused: false });
        automation.cancel();
        automation.trigger();
    },

    goToMove: (position: number) => goToHistoryMove(get, set, position),

    resetGame: async () => {
        automation.cancel();
        await abortInFlightGameSearches(get);

        const board = initializeBoard();
        set({
            ...createGameStartState(board, "black", "waiting", get().gameTimeLimit * 1000),
            ...idleEngineActivityPatch(),
        });
    },

    startGame: async (settings) => {
        const nextSettings = resolveNewGameSettings(get(), settings);

        // prepareGameReplacement frees the shared engine of every Engine
        // Search — including any in-flight solver search — before re-init
        // (CONTEXT.md → Game Replacement). What stays caller-specific is the
        // solver *session state* teardown: defer `exitSolver` to AFTER a
        // successful replacement so a failed init leaves the user's solver
        // session intact, matching how startGame preserves the current game
        // state on errors.
        const wasSolverActive = get().isSolverActive;

        if (!(await prepareGameReplacement(services, get, set))) {
            return false;
        }

        if (wasSolverActive) {
            await get().exitSolver();
        }

        set(createNewGamePatch(nextSettings, { board: initializeBoard(), currentPlayer: "black" }));
        persistNewGameSettings(services, nextSettings);

        automation.trigger();
        return true;
    },

    setGameStatus: (status) => set({ gameStatus: status }),
  };
  };
}
