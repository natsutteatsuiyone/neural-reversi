import { StateCreator } from "zustand";
import { resolveValidSetupPosition } from "@/domain/game/setup-position";
import { SolverSession, type SolverSessionCommit } from "@/domain/solver/solver-session";
import type { EngineSearch } from "@/domain/engine/engine-search";
import type { Services, SolverMode, SolverSelectivity } from "@/services/types";
import { DEFAULT_SETTINGS } from "@/services/types";
import type {
    ReversiState,
    SolverConfig,
    SolverSlice,
} from "./types";
import { prepareToReplaceGame } from "./game-slice";

type SetState = (
    partial:
        | Partial<ReversiState>
        | ((state: ReversiState) => Partial<ReversiState>),
) => void;

function createSolverSessionCommit(set: SetState): SolverSessionCommit {
    return (partial) => {
        set(partial as Parameters<SetState>[0]);
    };
}

function commitSolverConfig(
    services: Services,
    set: SetState,
    config: SolverConfig,
): void {
    set({
        targetSelectivity: config.selectivity,
        solverMode: config.mode,
    });
    void services.settings.saveSetting("solverTargetSelectivity", config.selectivity);
    void services.settings.saveSetting("solverMode", config.mode);
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

        return ({
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

        startSolver: async (board, player, config) => {
            // Abort any in-flight solver search first. Without this, a second
            // startSolver call would block inside prepareToReplaceGame on the
            // shared search mutex until the previous solve finishes, making
            // the new request appear hung.
            await services.solver.abort();

            if (!(await prepareToReplaceGame(services, get, set))) {
                return false;
            }

            await get().resetGame();

            if (config) {
                commitSolverConfig(services, set, config);
            }

            set({ isSolverModalOpen: false });
            await solverSession.start(board, player);
            return true;
        },

        startSolverFromSetup: async (config) => {
            const {
                setupTab,
                setupBoard,
                setupCurrentPlayer,
                transcriptInput,
                boardStringInput,
            } = get();

            const resolved = resolveValidSetupPosition({
                source: setupTab,
                board: setupBoard,
                currentPlayer: setupCurrentPlayer,
                transcriptInput,
                boardStringInput,
            });
            if (!resolved.ok) {
                set({ setupError: resolved.error });
                return false;
            }

            set({ setupError: null });
            const started = await get().startSolver(
                resolved.board,
                resolved.currentPlayer,
                config,
            );
            if (!started) {
                set({ setupError: "aiInitFailed" });
                return false;
            }
            return true;
        },

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
        });
    };
}
