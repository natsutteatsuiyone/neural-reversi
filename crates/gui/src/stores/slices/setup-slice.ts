import { StateCreator } from "zustand";
import { createEmptyBoard, initializeBoard, getValidMoves } from "@/lib/game-logic";
import { cloneBoard } from "@/lib/store-helpers";
import { MoveHistory } from "@/lib/move-history";
import { parseTranscript, parseBoardString, validateBoard } from "@/lib/board-parser";
import { initializeAI } from "@/lib/ai";
import type { Board, Player } from "@/types";
import type { ReversiState, SetupSlice, SetupTab } from "./types";
import { triggerAutomation } from "./game-slice";

type ResolvedSetupPosition =
    | { ok: true; board: Board; currentPlayer: Player }
    | { ok: false; error: string };

function resolveSetupPositionForTab(
    setupTab: SetupTab,
    setupBoard: Board,
    setupCurrentPlayer: Player,
    transcriptInput: string,
    boardStringInput: string,
): ResolvedSetupPosition {
    if (setupTab === "transcript") {
        const result = parseTranscript(transcriptInput);
        return result.ok
            ? { ok: true, board: result.board, currentPlayer: result.currentPlayer }
            : { ok: false, error: result.error };
    }

    if (setupTab === "boardString") {
        const result = parseBoardString(boardStringInput);
        return result.ok
            ? { ok: true, board: result.board, currentPlayer: setupCurrentPlayer }
            : { ok: false, error: result.error };
    }

    return { ok: true, board: setupBoard, currentPlayer: setupCurrentPlayer };
}

export const createSetupSlice: StateCreator<
    ReversiState,
    [],
    [],
    SetupSlice
> = (set, get) => ({
    setupBoard: initializeBoard(),
    setupCurrentPlayer: "black" as Player,
    setupTab: "manual" as const,
    transcriptInput: "",
    boardStringInput: "",
    setupError: null,

    resetSetup: () => {
        set({
            setupBoard: initializeBoard(),
            setupCurrentPlayer: "black",
            setupTab: "manual",
            transcriptInput: "",
            boardStringInput: "",
            setupError: null,
        });
    },

    setSetupTab: (tab) => set((state) => {
        const resolved = resolveSetupPositionForTab(
            tab,
            state.setupBoard,
            state.setupCurrentPlayer,
            state.transcriptInput,
            state.boardStringInput,
        );

        if (resolved.ok) {
            return {
                setupTab: tab,
                setupBoard: resolved.board,
                setupCurrentPlayer: resolved.currentPlayer,
                setupError: null,
            };
        }
        return {
            setupTab: tab,
            setupError: resolved.error,
        };
    }),

    setSetupCurrentPlayer: (player) => set({ setupCurrentPlayer: player, setupError: null }),

    setSetupBoard: (board) => set({ setupBoard: board }),

    setSetupCellColor: (row, col) => {
        set((state) => {
            const newBoard = cloneBoard(state.setupBoard);
            const current = newBoard[row][col].color;
            // Cycle: null -> black -> white -> null
            if (current === null) {
                newBoard[row][col] = { color: "black" };
            } else if (current === "black") {
                newBoard[row][col] = { color: "white" };
            } else {
                newBoard[row][col] = { color: null };
            }
            return { setupBoard: newBoard, setupError: null };
        });
    },

    setTranscriptInput: (input) => {
        const result = parseTranscript(input);
        if (result.ok) {
            set({
                transcriptInput: input,
                setupBoard: result.board,
                setupCurrentPlayer: result.currentPlayer,
                setupError: null,
            });
        } else {
            set({
                transcriptInput: input,
                setupError: result.error,
            });
        }
    },

    setBoardStringInput: (input) => {
        const result = parseBoardString(input);
        if (result.ok) {
            set({
                boardStringInput: input,
                setupBoard: result.board,
                setupError: null,
            });
        } else {
            set({
                boardStringInput: input,
                setupError: result.error,
            });
        }
    },

    clearSetupBoard: () => {
        set({ setupBoard: createEmptyBoard(), setupError: null });
    },

    resetSetupToInitial: () => {
        set({
            setupBoard: initializeBoard(),
            setupCurrentPlayer: "black",
            setupError: null,
        });
    },

    startFromSetup: async () => {
        const state = get();
        const resolved = resolveSetupPositionForTab(
            state.setupTab,
            state.setupBoard,
            state.setupCurrentPlayer,
            state.transcriptInput,
            state.boardStringInput,
        );
        if (!resolved.ok) {
            set({ setupError: resolved.error });
            return;
        }
        const { board: resolvedBoard, currentPlayer: resolvedCurrentPlayer } = resolved;

        const error = validateBoard(resolvedBoard, resolvedCurrentPlayer);
        if (error) {
            set({ setupError: error });
            return;
        }

        if (get().isAIThinking || get().isAnalyzing) {
            await get().abortAIMove();
        }

        const { searchTimer } = get();
        if (searchTimer) {
            clearInterval(searchTimer);
        }

        try {
            await initializeAI();
        } catch {
            set({ setupError: "aiInitFailed" });
            return;
        }

        const board = cloneBoard(resolvedBoard);
        const currentMoves = getValidMoves(board, resolvedCurrentPlayer);

        set({
            board,
            historyStartBoard: cloneBoard(board),
            historyStartPlayer: resolvedCurrentPlayer,
            moveHistory: MoveHistory.empty(),
            currentPlayer: resolvedCurrentPlayer,
            gameStatus: "playing",
            gameOver: false,
            isPass: false,
            lastMove: null,
            lastAIMove: null,
            validMoves: currentMoves,
            showPassNotification: null,
            setupError: null,
            analyzeResults: null,
            isAIThinking: false,
            isAnalyzing: false,
            aiMoveProgress: null,
            aiThinkingHistory: [],
            aiRemainingTime: get().gameTimeLimit * 1000,
            searchTimer: null,
        });

        if (currentMoves.length === 0) {
            set({ showPassNotification: resolvedCurrentPlayer });
            return;
        }

        triggerAutomation(get);
    },
});
