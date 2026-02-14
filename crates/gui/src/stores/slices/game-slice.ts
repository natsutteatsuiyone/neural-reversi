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
    createMoveRecord,
    createPassMove,
    getUndoMoves,
    getRedoMoves,
    reconstructBoardFromMoves,
} from "@/lib/store-helpers";
import { abortAISearch } from "@/lib/ai";
import type { Board, MoveRecord, Player } from "@/types";
import type { GameSlice, ReversiState } from "./types";
import { initializeAI } from "@/lib/ai";

function toLastMove(moves: MoveRecord[]): Move | null {
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
    moves: MoveRecord[],
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
    moves: [],
    allMoves: [],
    currentPlayer: "black",
    gameOver: false,
    gameStatus: "waiting",
    isPass: false,
    lastMove: null,
    validMoves: [],

    getScores: () => {
        return calculateScores(get().board);
    },

    isAITurn: () => {
        const { gameMode, gameOver, currentPlayer } = get();
        return (
            (!gameOver && gameMode === "ai-black" && currentPlayer === "black") ||
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
        // Abort analysis in background if it's a user move (don't await to avoid blocking)
        if (!move.isAI && get().isAnalyzing) {
            set({ isAnalyzing: false });
            void abortAISearch();
        }

        set((state) => {
            const currentPlayer = state.currentPlayer;
            const newBoard = applyMove(state.board, move, currentPlayer);
            const newMoveRecord = createMoveRecord(state.moves.length, currentPlayer, move, state.aiRemainingTime);
            const nextPlayerTurn = nextPlayer(currentPlayer);

            return {
                board: newBoard,
                moves: [...state.moves, newMoveRecord],
                allMoves: [...state.moves, newMoveRecord], // Update allMoves when making a new move
                currentPlayer: nextPlayerTurn,
                isPass: false,
                lastMove: move,
                validMoves: getValidMoves(newBoard, nextPlayerTurn),
                analyzeResults: null,
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

        triggerAutomation(get);
    },

    makePass: () => {
        set((state) => {
            const currentPlayer = state.currentPlayer;
            const passMove = createPassMove(state.moves.length, currentPlayer, state.aiRemainingTime);
            const nextPlayerTurn = nextPlayer(currentPlayer);
            const boardClone = cloneBoard(state.board);

            return {
                board: boardClone,
                moves: [...state.moves, passMove],
                allMoves: [...state.moves, passMove], // Update allMoves when passing
                currentPlayer: nextPlayerTurn,
                validMoves: getValidMoves(boardClone, nextPlayerTurn),
                isPass: true,
                analyzeResults: null,
            };
        });
    },

    undoMove: () => {
        set((state) => {
            if (state.gameStatus !== "playing" || state.moves.length === 0) {
                return state;
            }

            const newMoves = getUndoMoves(state.moves, state.gameMode, state.historyStartPlayer);
            const derived = deriveStateFromMoves(
                newMoves,
                state.historyStartBoard,
                state.historyStartPlayer,
            );

            return {
                ...derived,
                moves: newMoves,
                isPass: false,
                analyzeResults: null,
                gameOver: false,
                aiRemainingTime: newMoves.length > 0
                    ? (newMoves[newMoves.length - 1].remainingTime ?? state.gameTimeLimit * 1000)
                    : state.gameTimeLimit * 1000,
            };
        });

        const state = get();
        if (state.isHintMode && state.gameStatus === "playing") {
            void state.analyzeBoard();
        }
    },

    redoMove: () => {
        set((state) => {
            if (state.gameStatus !== "playing" || state.moves.length >= state.allMoves.length) {
                return state;
            }

            const newMoves = getRedoMoves(
                state.moves,
                state.allMoves,
                state.gameMode,
                state.historyStartPlayer,
            );
            const derived = deriveStateFromMoves(
                newMoves,
                state.historyStartBoard,
                state.historyStartPlayer,
            );
            const { gameOver } = checkGameOver(derived.board, derived.currentPlayer);

            return {
                ...derived,
                moves: newMoves,
                isPass: false,
                analyzeResults: null,
                gameOver,
                aiRemainingTime: newMoves.length > 0
                    ? (newMoves[newMoves.length - 1].remainingTime ?? state.gameTimeLimit * 1000)
                    : state.gameTimeLimit * 1000,
            };
        });

        const state = get();
        if (state.isHintMode && state.gameStatus === "playing") {
            void state.analyzeBoard();
        }
    },

    resetGame: async () => {
        if (get().isAIThinking || get().isAnalyzing) {
            await get().abortAIMove();
        }

        const { searchTimer } = get();
        if (searchTimer) {
            clearInterval(searchTimer);
        }

        const board = initializeBoard();

        set({
            board,
            historyStartBoard: cloneBoard(board),
            historyStartPlayer: "black",
            moves: [],
            allMoves: [],
            currentPlayer: "black",
            gameOver: false,
            gameStatus: "waiting",
            isPass: false,
            lastMove: null,
            lastAIMove: null,
            showPassNotification: null,
            isAIThinking: false,
            isAnalyzing: false,
            analyzeResults: null,
            validMoves: [],
            aiMoveProgress: null,
            aiRemainingTime: get().gameTimeLimit * 1000,
            searchTimer: null,
        });
    },

    startGame: async () => {
        try {
            await initializeAI();
        } catch {
            return;
        }

        set(() => {
            const board = initializeBoard();
            const currentPlayer = "black";
            return {
                board,
                historyStartBoard: cloneBoard(board),
                historyStartPlayer: currentPlayer,
                moves: [],
                allMoves: [],
                currentPlayer,
                gameStatus: "playing",
                gameOver: false,
                isPass: false,
                lastMove: null,
                lastAIMove: null,
                validMoves: getValidMoves(board, currentPlayer),
                showPassNotification: null,
                analyzeResults: null,
                aiMoveProgress: null,
                aiThinkingHistory: [],
                aiRemainingTime: get().gameTimeLimit * 1000,
            };
        });

        triggerAutomation(get);
    },

    setGameStatus: (status) => set({ gameStatus: status }),
});
