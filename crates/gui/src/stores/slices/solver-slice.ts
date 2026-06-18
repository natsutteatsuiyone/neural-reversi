import { StateCreator } from "zustand";
import { SolverSession, type SolverSessionCommit } from "@/domain/solver/solver-session";
import type { EngineSearch } from "@/domain/engine/engine-search";
import type { Services, SolverMode, SolverSelectivity } from "@/services/types";
import { DEFAULT_SETTINGS } from "@/services/types";
import type { ReversiState, SetState, SolverSlice } from "./types";
import { runGameReplacement } from "@/stores/game-replacement";

function createSolverSessionCommit(set: SetState): SolverSessionCommit {
  return (partial) => {
    set(partial as Parameters<SetState>[0]);
  };
}

function saveTargetSelectivity(services: Services, selectivity: SolverSelectivity): void {
  void services.settings.saveSetting("solverTargetSelectivity", selectivity);
}

function saveSolverMode(services: Services, mode: SolverMode): void {
  void services.settings.saveSetting("solverMode", mode);
}

export function createSolverSlice(
  services: Services,
  engineSearch: EngineSearch,
): StateCreator<ReversiState, [], [], SolverSlice> {
  return (set, get) => {
    const solverSession = new SolverSession({
      solver: services.solver,
      read: get,
      commit: createSolverSessionCommit(set),
      engineSearch,
    });

    return {
      isSolverActive: false,
      isSolverModalOpen: false,
      solverRootBoard: null,
      solverRootPlayer: null,
      solverHistory: [],
      solverCurrentBoard: null,
      solverCurrentPlayer: null,
      targetSelectivity: DEFAULT_SETTINGS.solverTargetSelectivity,
      solverMode: DEFAULT_SETTINGS.solverMode,
      solverCandidates: new Map(),
      isSolverSearching: false,
      isSolverStopped: false,

      openSolverModal: () => {
        get().resetSetup();
        set({ isSolverModalOpen: true });
      },

      closeSolverModal: () => set({ isSolverModalOpen: false }),

      subscribeSolverProgress: () => solverSession.subscribeProgress(),

      startSolver: async (board, player, config) =>
        runGameReplacement(services, get, set, {
          kind: "solver-position",
          board,
          player,
          config,
          startSolver: (nextBoard, nextPlayer) => solverSession.start(nextBoard, nextPlayer),
        }),

      startSolverFromSetup: async (config) =>
        runGameReplacement(services, get, set, {
          kind: "setup-solver",
          config,
          startSolver: (board, player) => solverSession.start(board, player),
        }),

      exitSolver: async () => {
        await solverSession.exit();
      },

      advanceSolver: async (row, col) => {
        await solverSession.advance(row, col);
      },

      undoSolver: async () => {
        await solverSession.undo();
      },

      setTargetSelectivity: async (sel) => {
        set({ targetSelectivity: sel });
        saveTargetSelectivity(services, sel);
        await solverSession.repointCurrent();
      },

      setSolverMode: async (mode) => {
        if (get().solverMode === mode) return;
        set({ solverMode: mode });
        saveSolverMode(services, mode);
        await solverSession.repointCurrent();
      },

      stopSolverSearch: async () => {
        await solverSession.stop();
      },

      resumeSolverSearch: async () => {
        await solverSession.resume();
      },

      applySolverProgress: (payload) => {
        solverSession.applyProgress(payload);
      },
    };
  };
}
