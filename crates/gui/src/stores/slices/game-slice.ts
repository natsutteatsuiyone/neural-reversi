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
    createGoToMovePatch,
    createHistoryNavigationPatch,
    createPassTurnPatch,
    hasFlippedDiscs,
    type GameHistoryPatch,
    type HistoryNavigationState,
} from "@/domain/game/game-session";
import { MoveHistory } from "@/domain/game/move-history";
import type { GameSlice, NewGameSettings, ReversiState } from "./types";
import type { Services } from "@/services/types";
import { clearSearchTimer } from "./ai-slice";
import { FLIP_DURATION_S } from "@/components/board/board3d-utils";

const FLIP_DURATION_MS = FLIP_DURATION_S * 1000;
export const PASS_NOTIFICATION_DURATION_MS = 1500;

function createHistoryNavigationState(state: ReversiState): HistoryNavigationState {
    return {
        gameStatus: state.gameStatus,
        moveHistory: state.moveHistory,
        historyStartBoard: state.historyStartBoard,
        historyStartPlayer: state.historyStartPlayer,
        gameTimeLimitMs: state.gameTimeLimit * 1000,
    };
}

function withNavigationClears(patch: GameHistoryPatch): Partial<ReversiState> {
    return {
        ...patch,
        analyzeResults: null,
        showPassNotification: null,
    };
}

function finalizeNavigation(
    set: (partial: Partial<ReversiState> | ((state: ReversiState) => Partial<ReversiState>)) => void,
    get: () => ReversiState,
): void {
    const state = get();
    if (state.gameStatus !== "playing") return;

    const canResumeAI = state.isAITurn() && state.validMoves.length > 0;
    set({ paused: canResumeAI });
    if (state.isHintMode && state.validMoves.length > 0) {
        void state.analyzeBoard();
    }
}

export function clearAutomationTimer(
    get: () => ReversiState,
    set: (partial: Partial<ReversiState>) => void,
): void {
    const { automationTimer, automationResumePending } = get();
    if (automationTimer) {
        clearTimeout(automationTimer);
    }
    if (automationTimer || automationResumePending) {
        set({ automationTimer: null, automationResumePending: false });
    }
}

export function resumeQueuedAutomation(
    get: () => ReversiState,
    set: (partial: Partial<ReversiState>) => void,
): void {
    const state = get();
    if (!state.automationResumePending || state.isGameAnalyzing) return;
    set({ automationResumePending: false });
    triggerAutomation(get);
}

export async function cleanupActiveOperations(
    get: () => ReversiState,
    set: (partial: Partial<ReversiState>) => void,
): Promise<void> {
    const state = get();
    clearSearchTimer(get, set);
    const aborts: Promise<void>[] = [];
    if (state.isAIThinking || state.isAnalyzing) {
        aborts.push(state.abortAIMove());
    }
    if (state.isGameAnalyzing) {
        aborts.push(state.abortGameAnalysis());
    }
    await Promise.all(aborts);
}

function resolveNewGameSettings(
    state: ReversiState,
    overrides?: NewGameSettings,
): NewGameSettings {
    return {
        gameMode: overrides?.gameMode ?? state.gameMode,
        aiLevel: overrides?.aiLevel ?? state.aiLevel,
        aiMode: overrides?.aiMode ?? state.aiMode,
        gameTimeLimit: overrides?.gameTimeLimit ?? state.gameTimeLimit,
    };
}

export async function prepareToReplaceGame(
    services: Services,
    get: () => ReversiState,
    set: (partial: Partial<ReversiState>) => void,
): Promise<boolean> {
    if (!(await get().checkAIReady())) {
        return false;
    }

    const shouldResumeGameAnalysis = get().isGameAnalyzing;
    const wasPaused = get().paused;

    clearAutomationTimer(get, set);
    await cleanupActiveOperations(get, set);

    try {
        await services.ai.initialize();
        await services.ai.resizeTT(get().hashSize);
        return true;
    } catch (error) {
        console.error("Failed to prepare AI for a new position:", error);
        if (shouldResumeGameAnalysis) {
            void get().analyzeGame();
            set({ paused: wasPaused, automationResumePending: true });
        } else {
            set({ paused: wasPaused });
            triggerAutomation(get);
        }
        return false;
    }
}

function scheduleAutomation(
    get: () => ReversiState,
    set: (partial: Partial<ReversiState>) => void,
    delayMs: number,
): void {
    clearAutomationTimer(get, set);
    const timer = setTimeout(() => {
        if (get().isGameAnalyzing) {
            set({ automationTimer: null, automationResumePending: true });
            return;
        }
        set({ automationTimer: null, automationResumePending: false });
        triggerAutomation(get);
    }, delayMs);
    set({ automationTimer: timer, automationResumePending: false });
}

export function triggerAutomation(getState: () => ReversiState): void {
    const state = getState();

    if (state.gameStatus !== "playing") {
        return;
    }

    if (state.paused) {
        return;
    }

    if (state.isAIThinking || state.isAnalyzing || state.isGameAnalyzing) {
        return;
    }

    if (state.isAITurn()) {
        void state.makeAIMove();
        return;
    }

    if (state.isHintMode) {
        void state.analyzeBoard();
    }
}

export function createGameSlice(services: Services): StateCreator<
    ReversiState,
    [],
    [],
    GameSlice
> {
  return (set, get) => ({
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
    automationTimer: null,
    automationResumePending: false,

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
        clearAutomationTimer(get, set);

        // Abort analysis in background if it's a user move (don't await to avoid blocking)
        if (!move.isAI && get().isAnalyzing) {
            set({ isAnalyzing: false });
            void services.ai.abortSearch();
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
                analyzeResults: null,
                gameAnalysisResult: null,
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
                analyzeResults: null,
                gameAnalysisResult: null,
                showPassNotification: updatedState.currentPlayer,
            }));

            scheduleAutomation(get, set, PASS_NOTIFICATION_DURATION_MS);
            return;
        }

        if (!move.isAI && hasFlippedDiscs(oldBoard, updatedState.board)) {
            scheduleAutomation(get, set, FLIP_DURATION_MS);
        } else {
            triggerAutomation(get);
        }
    },

    makePass: () => {
        if (get().isGameAnalyzing) return;
        clearAutomationTimer(get, set);
        set((state) => ({
            ...createPassTurnPatch(state),
            analyzeResults: null,
            gameAnalysisResult: null,
        }));
    },

    undoMove: () => {
        if (get().isGameAnalyzing) return;
        clearAutomationTimer(get, set);
        set((state) => {
            const patch = createHistoryNavigationPatch(createHistoryNavigationState(state), "undo");
            return patch ? withNavigationClears(patch) : state;
        });
        finalizeNavigation(set, get);
    },

    redoMove: () => {
        if (get().isGameAnalyzing) return;
        clearAutomationTimer(get, set);
        set((state) => {
            const patch = createHistoryNavigationPatch(createHistoryNavigationState(state), "redo");
            return patch ? withNavigationClears(patch) : state;
        });
        finalizeNavigation(set, get);
    },

    resumeAI: () => {
        if (get().isGameAnalyzing) {
            set({ paused: false, automationResumePending: true });
            return;
        }
        set({ paused: false, automationResumePending: false });
        triggerAutomation(get);
    },

    goToMove: (position: number) => {
        const state = get();
        if (state.isAIThinking || state.isAnalyzing || state.isGameAnalyzing) return;
        clearAutomationTimer(get, set);

        const patch = createGoToMovePatch(createHistoryNavigationState(state), position);
        if (!patch) return;

        set(withNavigationClears(patch));

        if (!patch.gameOver && patch.gameStatus === "playing") {
            finalizeNavigation(set, get);
        }
    },

    resetGame: async () => {
        clearAutomationTimer(get, set);
        await cleanupActiveOperations(get, set);

        const board = initializeBoard();
        set(createGameStartState(board, "black", "waiting", get().gameTimeLimit * 1000));
    },

    startGame: async (settings) => {
        const nextSettings = resolveNewGameSettings(get(), settings);

        // If solver mode is active, abort its search first (without tearing
        // down solver state) so prepareToReplaceGame doesn't deadlock on the
        // shared backend mutex. Solver state cleanup is deferred to AFTER
        // setup succeeds so that a failed init leaves the user's solver
        // session intact, matching how startGame preserves the current game
        // state on errors.
        const wasSolverActive = get().isSolverActive;
        if (wasSolverActive) {
            await services.solver.abort();
        }

        if (!(await prepareToReplaceGame(services, get, set))) {
            return false;
        }

        if (wasSolverActive) {
            await get().exitSolver();
        }

        const board = initializeBoard();
        set({
            ...createGameStartState(board, "black", "playing", nextSettings.gameTimeLimit * 1000),
            gameMode: nextSettings.gameMode,
            aiLevel: nextSettings.aiLevel,
            aiMode: nextSettings.aiMode,
            gameTimeLimit: nextSettings.gameTimeLimit,
        });

        triggerAutomation(get);
        return true;
    },

    setGameStatus: (status) => set({ gameStatus: status }),
  });
}
