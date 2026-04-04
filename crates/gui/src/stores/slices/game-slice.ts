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
    getUndoCount,
    getRedoCount,
    reconstructBoardFromMoves,
} from "@/lib/store-helpers";
import { MoveHistory } from "@/lib/move-history";
import { abortAISearch } from "@/lib/ai";
import type { Board, MoveRecord, Player } from "@/types";
import type { GameSlice, ReversiState } from "./types";
import { initializeAI } from "@/lib/ai";
import { FLIP_DURATION_S } from "@/components/board/board3d-utils";

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
        moves as MoveRecord[],
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

    if (state.gameStatus !== "playing") return null;
    if (isUndo ? !state.moveHistory.canUndo : !state.moveHistory.canRedo) return null;

    const count = isUndo
        ? getUndoCount(state.moveHistory.currentMoves, state.gameMode, state.historyStartPlayer)
        : getRedoCount(
              state.moveHistory.currentMoves,
              [...state.moveHistory.currentMoves, ...state.moveHistory.redoMoves],
              state.gameMode,
              state.historyStartPlayer,
          );
    if (count === 0) return null;

    const newHistory = isUndo ? state.moveHistory.undo(count) : state.moveHistory.redo(count);
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
        gameOver,
        skipAnimation: true,
        aiRemainingTime: newHistory.length > 0
            ? (newHistory.lastMove!.remainingTime ?? state.gameTimeLimit * 1000)
            : state.gameTimeLimit * 1000,
    };
}

export function triggerAutomation(getState: () => ReversiState): void {
    const state = getState();

    if (state.gameStatus !== "playing") {
        return;
    }

    if (state.isAITurn()) {
        void state.makeAIMove();
        return;
    }

    // Re-fetch state to ensure we have the latest isHintMode value
    if (getState().isHintMode) {
        void state.analyzeBoard();
    }
}

export const createGameSlice: StateCreator<
    ReversiState,
    [],
    [],
    GameSlice
> = (set, get) => ({
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

    getScores: () => {
        return calculateScores(get().board);
    },

    isAITurn: () => {
        const { gameMode, gameOver, currentPlayer } = get();
        return (
            !gameOver &&
            ((gameMode === "ai-black" && currentPlayer === "black") ||
             (gameMode === "ai-white" && currentPlayer === "white"))
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
        // Abort analysis in background if it's a user move (don't await to avoid blocking)
        if (!move.isAI && get().isAnalyzing) {
            set({ isAnalyzing: false });
            void abortAISearch();
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
            set({ showPassNotification: updatedState.currentPlayer });
            return;
        }

        // Wait for flip animation to complete before triggering AI
        const FLIP_DURATION_MS = FLIP_DURATION_S * 1000;
        if (!move.isAI) {
            const newBoard = updatedState.board;
            let hasFlipped = false;
            for (let r = 0; r < 8 && !hasFlipped; r++) {
                for (let c = 0; c < 8 && !hasFlipped; c++) {
                    const oldCell = oldBoard[r][c];
                    const newCell = newBoard[r][c];
                    if (oldCell.color && newCell.color && oldCell.color !== newCell.color) {
                        hasFlipped = true;
                    }
                }
            }
            if (hasFlipped) {
                setTimeout(() => triggerAutomation(get), FLIP_DURATION_MS);
                return;
            }
        }

        triggerAutomation(get);
    },

    makePass: () => {
        set((state) => {
            const currentPlayer = state.currentPlayer;
            const passMove = createPassMove(state.moveHistory.length, currentPlayer, state.aiRemainingTime);
            const nextPlayerTurn = nextPlayer(currentPlayer);
            const boardClone = cloneBoard(state.board);

            return {
                board: boardClone,
                moveHistory: state.moveHistory.append(passMove),
                currentPlayer: nextPlayerTurn,
                validMoves: getValidMoves(boardClone, nextPlayerTurn),
                isPass: true,
                analyzeResults: null,
            };
        });
    },

    undoMove: () => {
        set((state) => applyHistoryNavigation(state, "undo") ?? state);

        const state = get();
        if (state.isHintMode && state.gameStatus === "playing") {
            void state.analyzeBoard();
        }
    },

    redoMove: () => {
        set((state) => applyHistoryNavigation(state, "redo") ?? state);

        const state = get();
        if (state.isHintMode && state.gameStatus === "playing") {
            void state.analyzeBoard();
        }
    },

    goToMove: (position: number) => {
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

        set({
            ...derived,
            moveHistory: newHistory,
            isPass: false,
            analyzeResults: null,
            gameOver,
            gameStatus: gameOver ? "finished" : state.gameStatus,
            skipAnimation: true,
            aiRemainingTime: newHistory.length > 0
                ? (newHistory.lastMove!.remainingTime ?? state.gameTimeLimit * 1000)
                : state.gameTimeLimit * 1000,
        });

        if (!gameOver && state.gameStatus === "playing") {
            const updated = get();
            if (updated.validMoves.length === 0) {
                const result = checkGameOver(updated.board, updated.currentPlayer);
                if (result.shouldPass) {
                    set({ showPassNotification: updated.currentPlayer });
                    return;
                }
            }
            triggerAutomation(get);
        }
    },

    resetGame: async () => {
        if (get().isAIThinking || get().isAnalyzing) {
            await get().abortAIMove();
        }
        if (get().isGameAnalyzing) {
            await get().abortGameAnalysis();
        }

        const { searchTimer } = get();
        if (searchTimer) {
            clearInterval(searchTimer);
        }

        const board = initializeBoard();
        set(createGameStartState(board, "black", "waiting", get().gameTimeLimit * 1000));
    },

    startGame: async () => {
        try {
            await initializeAI();
        } catch {
            return;
        }

        const board = initializeBoard();
        set(createGameStartState(board, "black", "playing", get().gameTimeLimit * 1000));

        triggerAutomation(get);
    },

    setGameStatus: (status) => set({ gameStatus: status }),
});
