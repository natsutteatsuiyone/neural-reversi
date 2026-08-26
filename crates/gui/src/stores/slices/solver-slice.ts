import { StateCreator } from "zustand";
import { SolverSession, type SolverSessionCommit } from "@/domain/solver/solver-session";
import type { EngineSearch } from "@/domain/engine/engine-search";
import type { Services } from "@/services/types";
import { DEFAULT_SETTINGS } from "@/services/types";
import type { ReversiState, SetState, SolverSlice } from "./types";
import { runGameReplacement } from "@/stores/game-replacement";

function createSolverSessionCommit(set: SetState): SolverSessionCommit {
  return (partial) => {
    set(partial as Parameters<SetState>[0]);
  };
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
      solverHistory: [],
      targetSelectivity: DEFAULT_SETTINGS.solverTargetSelectivity,
      solverMode: DEFAULT_SETTINGS.solverMode,
      solverCandidates: new Map(),
      isSolverStopped: false,

      openSolverModal: () => {
        get().resetSetup();
        set({ isSolverModalOpen: true });
      },

      closeSolverModal: () => set({ isSolverModalOpen: false }),

      subscribeSolverProgress: () => solverSession.subscribeProgress(),

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
        await solverSession.repointCurrent();
      },

      setSolverMode: async (mode) => {
        if (get().solverMode === mode) return;
        set({ solverMode: mode });
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
