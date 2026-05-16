import { StateCreator } from "zustand";
import { createEmptyBoard, initializeBoard } from "@/domain/game/game-logic";
import { cloneBoard, createGameStartState } from "@/domain/game/store-helpers";
import { parseBoardString, parseTranscript } from "@/domain/game/board-parser";
import {
    resolveSetupPosition,
    resolveValidSetupPosition,
} from "@/domain/game/setup-position";
import type { Services } from "@/services/types";
import type { NewGameSettings, ReversiState, SetupSlice } from "./types";
import { prepareToReplaceGame, triggerAutomation } from "./game-slice";

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

export function createSetupSlice(services: Services): StateCreator<
    ReversiState,
    [],
    [],
    SetupSlice
> {
  return (set, get) => ({
    setupBoard: initializeBoard(),
    setupCurrentPlayer: "black",
    setupTab: "manual",
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
        const resolved = resolveSetupPosition({
            source: tab,
            board: state.setupBoard,
            currentPlayer: state.setupCurrentPlayer,
            transcriptInput: state.transcriptInput,
            boardStringInput: state.boardStringInput,
        });

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

    startFromSetup: async (settings) => {
        const state = get();
        const nextSettings = resolveNewGameSettings(state, settings);
        const resolved = resolveValidSetupPosition({
            source: state.setupTab,
            board: state.setupBoard,
            currentPlayer: state.setupCurrentPlayer,
            transcriptInput: state.transcriptInput,
            boardStringInput: state.boardStringInput,
        });
        if (!resolved.ok) {
            set({ setupError: resolved.error });
            return false;
        }
        const { board: resolvedBoard, currentPlayer: resolvedCurrentPlayer } = resolved;

        if (!(await prepareToReplaceGame(services, get, set))) {
            set({ setupError: "aiInitFailed" });
            return false;
        }

        const board = cloneBoard(resolvedBoard);
        const startState = createGameStartState(
            board,
            resolvedCurrentPlayer,
            "playing",
            nextSettings.gameTimeLimit * 1000,
        );
        set({
            ...startState,
            gameMode: nextSettings.gameMode,
            aiLevel: nextSettings.aiLevel,
            aiMode: nextSettings.aiMode,
            gameTimeLimit: nextSettings.gameTimeLimit,
            setupError: null,
        });

        triggerAutomation(get);
        return true;
    },
  });
}
