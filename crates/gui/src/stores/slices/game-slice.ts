import { StateCreator } from "zustand";
import {
    calculateScores,
    getValidMoves,
    initializeBoard,
    opponentPlayer as nextPlayer,
} from "@/lib/game-logic";
import {
    type Move,
    applyMove,
    checkGameOver,
    cloneBoard,
    createGameStartState,
    createMoveRecord,
    createPassMove,
    reconstructBoardFromMoves,
} from "@/lib/store-helpers";
import { MoveHistory } from "@/lib/move-history";
import type { Board, MoveRecord, Player } from "@/types";
import type { GameSlice, NewGameSettings, ReversiState } from "./types";
import type { Services } from "@/services/types";
import { clearSearchTimer } from "./ai-slice";
import { FLIP_DURATION_S } from "@/components/board/board3d-utils";

const FLIP_DURATION_MS = FLIP_DURATION_S * 1000;
export const PASS_NOTIFICATION_DURATION_MS = 1500;

function resolveAIRemainingTime(history: MoveHistory, gameTimeLimitMs: number): number {
    return history.length > 0
        ? (history.lastMove?.remainingTime ?? gameTimeLimitMs)
        : gameTimeLimitMs;
}

function toLastMove(moves: readonly MoveRecord[]): Move | null {
    const last = moves.length > 0 ? moves[moves.length - 1] : undefined;
    if (!last || last.row < 0 || last.col < 0) {
        return null;
    }

    return {
        row: last.row,
        col: last.col,
        isAI: Boolean(last.isAI),
        score: last.score,
    };
}

function deriveStateFromMoves(
    moves: readonly MoveRecord[],
    historyStartBoard: Board,
    historyStartPlayer: Player,
): {
    board: Board;
    currentPlayer: "black" | "white";
    validMoves: [number, number][];
    lastMove: Move | null;
} {
    const { board, currentPlayer } = reconstructBoardFromMoves(
        moves,
        historyStartBoard,
        historyStartPlayer,
    );

    return {
        board,
        currentPlayer,
        validMoves: getValidMoves(board, currentPlayer),
        lastMove: toLastMove(moves),
    };
}

function applyHistoryNavigation(
    state: ReversiState,
    direction: "undo" | "redo",
): Partial<ReversiState> | null {
    const isUndo = direction === "undo";

    if (state.gameStatus === "waiting") return null;
    if (isUndo ? !state.moveHistory.canUndo : !state.moveHistory.canRedo) return null;

    const newHistory = isUndo ? state.moveHistory.undo(1) : state.moveHistory.redo(1);
    const derived = deriveStateFromMoves(
        newHistory.currentMoves,
        state.historyStartBoard,
        state.historyStartPlayer,
    );

    const gameOver = isUndo ? false : checkGameOver(derived.board, derived.currentPlayer).gameOver;

    return {
        ...derived,
        moveHistory: newHistory,
        isPass: false,
        analyzeResults: null,
        showPassNotification: null,
        gameOver,
        gameStatus: gameOver ? "finished" : "playing",
        skipAnimation: true,
        aiRemainingTime: resolveAIRemainingTime(newHistory, state.gameTimeLimit * 1000),
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
    const { automationTimer } = get();
    if (automationTimer) {
        clearTimeout(automationTimer);
        set({ automationTimer: null });
    }
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
        }
        triggerAutomation(get);
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
        set({ automationTimer: null });
        triggerAutomation(get);
    }, delayMs);
    set({ automationTimer: timer });
}

function hasFlippedDiscs(oldBoard: Board, newBoard: Board): boolean {
    for (let r = 0; r < 8; r++) {
        for (let c = 0; c < 8; c++) {
            const oldCell = oldBoard[r][c];
            const newCell = newBoard[r][c];
            if (oldCell.color && newCell.color && oldCell.color !== newCell.color) {
                return true;
            }
        }
    }

    return false;
}

function applyPassState(state: ReversiState, passingPlayer: Player): Partial<ReversiState> {
    const passMove = createPassMove(state.moveHistory.length, passingPlayer, state.aiRemainingTime);
    const nextPlayerTurn = nextPlayer(passingPlayer);
    const boardClone = cloneBoard(state.board);

    return {
        board: boardClone,
        moveHistory: state.moveHistory.append(passMove),
        currentPlayer: nextPlayerTurn,
        validMoves: getValidMoves(boardClone, nextPlayerTurn),
        isPass: true,
        analyzeResults: null,
    };
}

export function triggerAutomation(getState: () => ReversiState): void {
    const state = getState();

    if (state.gameStatus !== "playing") {
        return;
    }

    if (state.paused) {
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
                ...applyPassState(state, updatedState.currentPlayer),
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
        clearAutomationTimer(get, set);
        set((state) => applyPassState(state, state.currentPlayer));
    },

    undoMove: () => {
        clearAutomationTimer(get, set);
        set((state) => applyHistoryNavigation(state, "undo") ?? state);
        finalizeNavigation(set, get);
    },

    redoMove: () => {
        clearAutomationTimer(get, set);
        set((state) => applyHistoryNavigation(state, "redo") ?? state);
        finalizeNavigation(set, get);
    },

    resumeAI: () => {
        set({ paused: false });
        triggerAutomation(get);
    },

    goToMove: (position: number) => {
        clearAutomationTimer(get, set);
        const state = get();
        if (state.isAIThinking || state.isAnalyzing) return;

        const newHistory = state.moveHistory.goTo(position);
        if (newHistory.length === state.moveHistory.length) return;

        const derived = deriveStateFromMoves(
            newHistory.currentMoves,
            state.historyStartBoard,
            state.historyStartPlayer,
        );

        const isAtEnd = position >= state.moveHistory.totalLength;
        const gameOver = isAtEnd
            ? checkGameOver(derived.board, derived.currentPlayer).gameOver
            : false;

        const newGameStatus = gameOver ? "finished" : (state.gameStatus === "finished" ? "playing" : state.gameStatus);

        set({
            ...derived,
            moveHistory: newHistory,
            isPass: false,
            analyzeResults: null,
            showPassNotification: null,
            gameOver,
            gameStatus: newGameStatus,
            skipAnimation: true,
            aiRemainingTime: resolveAIRemainingTime(newHistory, state.gameTimeLimit * 1000),
        });

        if (!gameOver && newGameStatus === "playing") {
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
