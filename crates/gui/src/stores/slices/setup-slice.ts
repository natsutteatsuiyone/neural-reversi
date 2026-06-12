import { StateCreator } from "zustand";
import { createEmptyBoard, initializeBoard } from "@/domain/game/game-logic";
import { cloneBoard } from "@/domain/game/store-helpers";
import { parseBoardString, parseTranscript } from "@/domain/game/board-parser";
import {
  resolveSetupPosition,
  resolveValidSetupPosition,
  type SetupPositionInput,
} from "@/domain/game/setup-position";
import type { Services } from "@/services/types";
import type { ReversiState, SetupSlice, SetupTab } from "./types";
import { prepareGameReplacement } from "@/stores/game-replacement";
import {
  createNewGamePatch,
  persistNewGameSettings,
  resolveNewGameSettings,
} from "@/stores/new-game";

/**
 * The single place the setup slice's raw fields are projected into a
 * {@link SetupPositionInput}. `source` defaults to the committed tab;
 * `setSetupTab` overrides it with the tab being switched to. Keeping this
 * mapping in one function is what stops the five-field shape leaking into
 * every starter.
 */
function readSetupPositionInput(
  state: ReversiState,
  source: SetupTab = state.setupTab,
): SetupPositionInput {
  return {
    source,
    board: state.setupBoard,
    currentPlayer: state.setupCurrentPlayer,
    transcriptInput: state.transcriptInput,
    boardStringInput: state.boardStringInput,
  };
}

export function createSetupSlice(
  services: Services,
): StateCreator<ReversiState, [], [], SetupSlice> {
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

    setSetupTab: (tab) =>
      set((state) => {
        const resolved = resolveSetupPosition(readSetupPositionInput(state, tab));

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

    resolveValidSetup: () => resolveValidSetupPosition(readSetupPositionInput(get())),

    startFromSetup: async (settings) => {
      const state = get();
      const nextSettings = resolveNewGameSettings(state, settings);
      const resolved = get().resolveValidSetup();
      if (!resolved.ok) {
        set({ setupError: resolved.error });
        return false;
      }
      const { board: resolvedBoard, currentPlayer: resolvedCurrentPlayer } = resolved;

      if (!(await prepareGameReplacement(services, get, set))) {
        set({ setupError: "aiInitFailed" });
        return false;
      }

      set({
        ...createNewGamePatch(nextSettings, {
          board: cloneBoard(resolvedBoard),
          currentPlayer: resolvedCurrentPlayer,
        }),
        setupError: null,
      });
      persistNewGameSettings(services, nextSettings);

      get().triggerAutomation();
      return true;
    },
  });
}
