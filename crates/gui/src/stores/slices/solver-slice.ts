import { StateCreator } from "zustand";
import { applyMove, checkGameOver, type Move } from "@/lib/store-helpers";
import { getNotation, opponentPlayer } from "@/lib/game-logic";
import { boardToString, validateBoard } from "@/lib/board-parser";
import type { Board, Player } from "@/types";
import type {
    Services,
    SolverCandidate,
    SolverMode,
    SolverProgressPayload,
    SolverSelectivity,
} from "@/services/types";
import { DEFAULT_SETTINGS } from "@/services/types";
import type {
    ReversiState,
    SolverHistoryEntry,
    SolverSlice,
} from "./types";
import { prepareToReplaceGame } from "./game-slice";
import { resolveSetupPositionForTab } from "./setup-slice";

type SetState = (
    partial:
        | Partial<ReversiState>
        | ((state: ReversiState) => Partial<ReversiState>),
) => void;

type GetState = () => ReversiState;

function createRootEntry(board: Board, player: Player): SolverHistoryEntry {
    return { board, player, moveFrom: null };
}

// Outside Zustand: no UI subscribers, survives exitSolver re-entry.
const SOLVER_CACHE_MAX_ENTRIES = 64;
const solverResultCache = new Map<string, Map<string, SolverCandidate>>();

function solverCacheKey(
    board: Board,
    player: Player,
    selectivity: SolverSelectivity,
    mode: SolverMode,
): string {
    return `${boardToString(board)}:${player}:${selectivity}:${mode}`;
}

function solverCacheGet(
    board: Board,
    player: Player,
    selectivity: SolverSelectivity,
    mode: SolverMode,
): Map<string, SolverCandidate> | null {
    const key = solverCacheKey(board, player, selectivity, mode);
    const entry = solverResultCache.get(key);
    if (!entry) return null;
    solverResultCache.delete(key);
    solverResultCache.set(key, entry);
    // Fresh copy so store-side mutations can't leak back into the cache.
    return new Map(entry);
}

function solverCachePut(
    board: Board,
    player: Player,
    selectivity: SolverSelectivity,
    mode: SolverMode,
    candidates: Map<string, SolverCandidate>,
): void {
    const key = solverCacheKey(board, player, selectivity, mode);
    solverResultCache.delete(key);
    // applySolverProgress always spreads into a fresh Map, so this ref is frozen.
    solverResultCache.set(key, candidates);
    if (solverResultCache.size > SOLVER_CACHE_MAX_ENTRIES) {
        const oldest = solverResultCache.keys().next().value;
        if (oldest !== undefined) solverResultCache.delete(oldest);
    }
}

/**
 * Wraps `services.solver.startSearch` and clears the searching flag when
 * the current run is still `runId`.
 */
async function runSolverSearch(
    services: Services,
    get: GetState,
    set: SetState,
    board: Board,
    player: Player,
    runId: number,
): Promise<void> {
    const selectivity = get().targetSelectivity;
    const mode = get().solverMode;
    try {
        await services.solver.startSearch(board, player, selectivity, mode, runId);
    } catch (error) {
        console.error("Failed to start solver search:", error);
    } finally {
        if (get().solverSearchRunId === runId) {
            // Skip partial maps: would restore a frozen "searching" view with no search actually running.
            const candidates = get().solverCandidates;
            let allComplete = candidates.size > 0;
            for (const c of candidates.values()) {
                if (!c.isComplete) {
                    allComplete = false;
                    break;
                }
            }
            if (allComplete) {
                solverCachePut(board, player, selectivity, mode, candidates);
            }
            set({ isSolverSearching: false });
        }
    }
}

/**
 * Navigates the solver to `(board, player)` under `selectivity`, merging any
 * caller-supplied `extra` state (history / current position) into the same
 * commit. Sync-mutates before `abort()` so a racing navigation reads the new
 * state instead of a stale pre-abort snapshot; post-abort run-id check keeps
 * the trailing `runSolverSearch` off a superseded run.
 */
async function repointSolver(
    services: Services,
    get: GetState,
    set: SetState,
    board: Board,
    player: Player,
    selectivity: SolverSelectivity,
    mode: SolverMode,
    extra: Partial<ReversiState> = {},
): Promise<void> {
    const cached = solverCacheGet(board, player, selectivity, mode);
    const nextRunId = get().solverSearchRunId + 1;

    set({
        ...extra,
        solverCandidates: cached ?? new Map<string, SolverCandidate>(),
        isSolverSearching: !cached,
        isSolverStopped: false,
        solverSearchRunId: nextRunId,
    });

    await services.solver.abort();

    if (cached || get().solverSearchRunId !== nextRunId) {
        return;
    }

    await runSolverSearch(services, get, set, board, player, nextRunId);
}

export function createSolverSlice(
    services: Services,
): StateCreator<ReversiState, [], [], SolverSlice> {
    return (set, get) => ({
        isSolverActive: false,
        isSolverModalOpen: false,
        solverRootBoard: null,
        solverRootPlayer: null,
        solverHistory: [],
        solverCurrentBoard: null,
        solverCurrentPlayer: null,
        targetSelectivity: DEFAULT_SETTINGS.solverTargetSelectivity,
        solverMode: DEFAULT_SETTINGS.solverMode,
        solverCandidates: new Map<string, SolverCandidate>(),
        isSolverSearching: false,
        isSolverStopped: false,
        solverSearchRunId: 0,

        openSolverModal: () => {
            get().resetSetup();
            set({ isSolverModalOpen: true });
        },

        closeSolverModal: () => set({ isSolverModalOpen: false }),

        startSolver: async (board, player) => {
            // Abort any in-flight solver search first. Without this, a second
            // startSolver call would block inside prepareToReplaceGame on the
            // shared search mutex until the previous solve finishes, making
            // the new request appear hung.
            await services.solver.abort();

            if (!(await prepareToReplaceGame(services, get, set))) {
                return false;
            }

            await get().resetGame();

            const rootEntry = createRootEntry(board, player);
            const nextRunId = get().solverSearchRunId + 1;
            set({
                isSolverActive: true,
                isSolverModalOpen: false,
                solverRootBoard: board,
                solverRootPlayer: player,
                solverCurrentBoard: board,
                solverCurrentPlayer: player,
                solverHistory: [rootEntry],
                solverCandidates: new Map<string, SolverCandidate>(),
                isSolverSearching: true,
                isSolverStopped: false,
                solverSearchRunId: nextRunId,
            });

            await runSolverSearch(services, get, set, board, player, nextRunId);
            return true;
        },

        startSolverFromSetup: async () => {
            const {
                setupTab,
                setupBoard,
                setupCurrentPlayer,
                transcriptInput,
                boardStringInput,
            } = get();

            const resolved = resolveSetupPositionForTab(
                setupTab,
                setupBoard,
                setupCurrentPlayer,
                transcriptInput,
                boardStringInput,
            );
            if (!resolved.ok) {
                set({ setupError: resolved.error });
                return false;
            }

            const validationError = validateBoard(resolved.board, resolved.currentPlayer);
            if (validationError) {
                set({ setupError: validationError });
                return false;
            }

            set({ setupError: null });
            const started = await get().startSolver(resolved.board, resolved.currentPlayer);
            if (!started) {
                set({ setupError: "aiInitFailed" });
                return false;
            }
            return true;
        },

        exitSolver: async () => {
            await services.solver.abort();
            set((state) => ({
                isSolverActive: false,
                solverRootBoard: null,
                solverRootPlayer: null,
                solverHistory: [],
                solverCurrentBoard: null,
                solverCurrentPlayer: null,
                solverCandidates: new Map<string, SolverCandidate>(),
                isSolverSearching: false,
                isSolverStopped: false,
                // Bump the run id so any in-flight search's catch branch sees
                // a stale id and does NOT clobber the post-exit state.
                solverSearchRunId: state.solverSearchRunId + 1,
            }));
        },

        advanceSolver: async (row, col) => {
            const initial = get();
            if (!initial.solverCurrentBoard || !initial.solverCurrentPlayer) {
                return;
            }

            // Claim this navigation's run id SYNCHRONOUSLY before awaiting
            // abort. Snapshotting state only after the await lets two quick
            // clicks compute from the same pre-abort position and append
            // root-based entries onto a history the racing call has
            // already extended, corrupting the breadcrumb trail. Bumping
            // up front also filters stale progress events from the
            // parent's search — including in the game-over branch below,
            // which starts no new search of its own.
            const nextRunId = initial.solverSearchRunId + 1;
            set({
                isSolverSearching: true,
                isSolverStopped: false,
                solverSearchRunId: nextRunId,
            });

            await services.solver.abort();

            // A racing advance/undo/reset may have superseded us while we
            // awaited abort. Bail so we don't clobber its history write.
            if (get().solverSearchRunId !== nextRunId) {
                return;
            }

            const currentBoard = get().solverCurrentBoard;
            const currentPlayer = get().solverCurrentPlayer;
            if (!currentBoard || !currentPlayer) {
                return;
            }

            const move: Move = { row, col, isAI: false, score: 0 };
            const newBoard = applyMove(currentBoard, move, currentPlayer);
            let nextPlayer = opponentPlayer(currentPlayer);

            // If the opponent has no legal replies but the original mover
            // still does, collapse the implicit pass into this entry so the
            // user can keep exploring without clicking through a dead end.
            // Single history entry → single undoSolver() unwinds everything.
            const { gameOver, shouldPass } = checkGameOver(newBoard, nextPlayer);
            if (shouldPass) {
                nextPlayer = opponentPlayer(nextPlayer);
            }

            const newEntry: SolverHistoryEntry = {
                board: newBoard,
                player: nextPlayer,
                moveFrom: getNotation(row, col),
            };

            if (gameOver) {
                set((s) => ({
                    solverHistory: [...s.solverHistory, newEntry],
                    solverCurrentBoard: newBoard,
                    solverCurrentPlayer: nextPlayer,
                    solverCandidates: new Map<string, SolverCandidate>(),
                    isSolverSearching: false,
                }));
                return;
            }

            const cached = solverCacheGet(
                newBoard,
                nextPlayer,
                get().targetSelectivity,
                get().solverMode,
            );
            if (cached) {
                set((s) => ({
                    solverHistory: [...s.solverHistory, newEntry],
                    solverCurrentBoard: newBoard,
                    solverCurrentPlayer: nextPlayer,
                    solverCandidates: cached,
                    isSolverSearching: false,
                    isSolverStopped: false,
                }));
                return;
            }

            set((s) => ({
                solverHistory: [...s.solverHistory, newEntry],
                solverCurrentBoard: newBoard,
                solverCurrentPlayer: nextPlayer,
                solverCandidates: new Map<string, SolverCandidate>(),
            }));

            await runSolverSearch(services, get, set, newBoard, nextPlayer, nextRunId);
        },

        undoSolver: async () => {
            const initial = get();
            if (initial.solverHistory.length <= 1) {
                return;
            }

            const newHistory = initial.solverHistory.slice(0, -1);
            const prevEntry = newHistory[newHistory.length - 1];

            await repointSolver(
                services,
                get,
                set,
                prevEntry.board,
                prevEntry.player,
                initial.targetSelectivity,
                initial.solverMode,
                {
                    solverHistory: newHistory,
                    solverCurrentBoard: prevEntry.board,
                    solverCurrentPlayer: prevEntry.player,
                },
            );
        },

        setTargetSelectivity: async (sel) => {
            set({ targetSelectivity: sel });
            void services.settings.saveSetting("solverTargetSelectivity", sel);

            const initial = get();
            if (
                !initial.isSolverActive ||
                !initial.solverCurrentBoard ||
                !initial.solverCurrentPlayer
            ) {
                return;
            }

            await repointSolver(
                services,
                get,
                set,
                initial.solverCurrentBoard,
                initial.solverCurrentPlayer,
                sel,
                initial.solverMode,
            );
        },

        setSolverMode: async (mode) => {
            if (get().solverMode === mode) return;
            set({ solverMode: mode });
            void services.settings.saveSetting("solverMode", mode);

            const initial = get();
            if (
                !initial.isSolverActive ||
                !initial.solverCurrentBoard ||
                !initial.solverCurrentPlayer
            ) {
                return;
            }

            await repointSolver(
                services,
                get,
                set,
                initial.solverCurrentBoard,
                initial.solverCurrentPlayer,
                initial.targetSelectivity,
                mode,
            );
        },

        stopSolverSearch: async () => {
            const state = get();
            if (!state.isSolverActive || !state.isSolverSearching) {
                return;
            }

            await services.solver.abort();

            // Leave candidates / history / current position intact — this
            // is a pause, not an exit. Bumping the run id drops trailing
            // `solver-progress` events from the aborted run.
            set((s) => ({
                isSolverSearching: false,
                isSolverStopped: true,
                solverSearchRunId: s.solverSearchRunId + 1,
            }));
        },

        resumeSolverSearch: async () => {
            const state = get();
            if (!state.isSolverActive || state.isSolverSearching || !state.isSolverStopped) {
                return;
            }
            const { solverCurrentBoard, solverCurrentPlayer } = state;
            if (!solverCurrentBoard || !solverCurrentPlayer) {
                return;
            }

            const cached = solverCacheGet(
                solverCurrentBoard,
                solverCurrentPlayer,
                state.targetSelectivity,
                state.solverMode,
            );
            const nextRunId = state.solverSearchRunId + 1;

            if (cached) {
                set({
                    solverCandidates: cached,
                    isSolverSearching: false,
                    isSolverStopped: false,
                    solverSearchRunId: nextRunId,
                });
                return;
            }

            // Keep existing candidates visible; new progress events will
            // overwrite them move-by-move as the re-run catches up.
            set({
                isSolverSearching: true,
                isSolverStopped: false,
                solverSearchRunId: nextRunId,
            });

            await runSolverSearch(
                services,
                get,
                set,
                solverCurrentBoard,
                solverCurrentPlayer,
                nextRunId,
            );
        },

        applySolverProgress: (payload: SolverProgressPayload) => {
            // Do NOT gate on `isSolverSearching`: `runSolverSearch` clears that
            // flag as soon as `startSearch` resolves, but trailing progress
            // events can still arrive and carry the final depth/accuracy.
            const state = get();
            if (!state.isSolverActive) {
                return;
            }
            // Drop events from a superseded run. After abort + restart (via
            // startSolver / undo / reset / setTargetSelectivity), late
            // `solver-progress` events from the previous run can still be
            // queued on the JS side. Without the run-id check, their stale
            // candidates would leak onto the new position.
            if (payload.runId !== state.solverSearchRunId) {
                return;
            }

            const isComplete = payload.isEndgame
                ? payload.acc >= state.targetSelectivity
                : payload.depth >= payload.targetDepth;
            const key = `${payload.row},${payload.col}`;
            const existing = state.solverCandidates.get(key);

            // Fast path: skip the Map clone when nothing that affects rendering
            // has changed. Relies on Zustand's selector equality — returning
            // the same Map ref means `SolverPanel` and `Board` subscribers bail
            // out of re-rendering.
            if (
                existing &&
                existing.score === payload.score &&
                existing.depth === payload.depth &&
                existing.targetDepth === payload.targetDepth &&
                existing.acc === payload.acc &&
                existing.isEndgame === payload.isEndgame &&
                existing.isComplete === isComplete &&
                existing.pvLine === payload.pvLine
            ) {
                return;
            }

            const candidate: SolverCandidate = {
                move: payload.bestMove,
                row: payload.row,
                col: payload.col,
                score: payload.score,
                depth: payload.depth,
                targetDepth: payload.targetDepth,
                acc: payload.acc,
                pvLine: payload.pvLine,
                isEndgame: payload.isEndgame,
                isComplete,
            };

            set((s) => {
                const next = new Map(s.solverCandidates);
                next.set(key, candidate);
                return { solverCandidates: next };
            });
        },
    });
}
