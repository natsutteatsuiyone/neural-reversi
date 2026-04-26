import { afterAll, beforeEach, describe, expect, it, vi } from "vitest";
import { createDeferred, createTestStore } from "./test-helpers";
import { createMockAIService } from "@/services/mock-ai-service";
import { createMockSolverService } from "@/services/mock-solver-service";
import {
    createEmptyBoard,
    getNotation,
    getValidMoves as realGetValidMoves,
    initializeBoard,
    opponentPlayer,
} from "@/lib/game-logic";
import { applyMove } from "@/lib/store-helpers";
import type { Board, Player } from "@/types";
import type {
    SolverCandidate,
    SolverProgressPayload,
} from "@/services/types";
import type { SolverHistoryEntry } from "@/stores/slices/types";

const consoleErrorSpy = vi.spyOn(console, "error").mockImplementation(() => {});

afterAll(() => {
    consoleErrorSpy.mockRestore();
});

// Allow individual tests to override `getValidMoves` (e.g. to simulate a
// position with no legal moves for the next player). Tests that don't set a
// stub fall through to the real implementation.
let getValidMovesStub:
    | ((board: Board, player: Player) => [number, number][])
    | null = null;

vi.mock("@/lib/game-logic", async (importOriginal) => {
    const actual = await importOriginal<typeof import("@/lib/game-logic")>();
    return {
        ...actual,
        getValidMoves: (board: Board, player: Player) =>
            (getValidMovesStub ?? actual.getValidMoves)(board, player),
    };
});

beforeEach(() => {
    vi.clearAllMocks();
    getValidMovesStub = null;
});

function buildHistoryEntry(
    board: Board,
    player: Player,
    moveFrom: string | null,
): SolverHistoryEntry {
    return { board, player, moveFrom };
}

describe("solver-slice initial state", () => {
    it("starts inactive with empty history and default selectivity", () => {
        const { store } = createTestStore();
        const state = store.getState();

        expect(state.isSolverActive).toBe(false);
        expect(state.isSolverModalOpen).toBe(false);
        expect(state.solverHistory).toEqual([]);
        expect(state.solverRootBoard).toBeNull();
        expect(state.solverRootPlayer).toBeNull();
        expect(state.solverCurrentBoard).toBeNull();
        expect(state.solverCurrentPlayer).toBeNull();
        expect(state.targetSelectivity).toBe(100);
        expect(state.solverCandidates.size).toBe(0);
        expect(state.isSolverSearching).toBe(false);
    });
});

describe("openSolverModal / closeSolverModal", () => {
    it("opens the modal and resets setup state", () => {
        const { store } = createTestStore();
        const resetSetupSpy = vi.spyOn(store.getState(), "resetSetup");

        store.getState().openSolverModal();

        expect(resetSetupSpy).toHaveBeenCalledTimes(1);
        expect(store.getState().isSolverModalOpen).toBe(true);
    });

    it("closes the modal", () => {
        const { store } = createTestStore();
        store.setState({ isSolverModalOpen: true });

        store.getState().closeSolverModal();

        expect(store.getState().isSolverModalOpen).toBe(false);
    });
});

describe("startSolver", () => {
    it("enters solver mode and kicks off a search", async () => {
        const { store, services } = createTestStore();
        const board = initializeBoard();

        const result = await store.getState().startSolver(board, "black");

        expect(result).toBe(true);
        const state = store.getState();
        expect(state.isSolverActive).toBe(true);
        expect(state.isSolverModalOpen).toBe(false);
        expect(state.solverRootBoard).toBe(board);
        expect(state.solverRootPlayer).toBe("black");
        expect(state.solverCurrentBoard).toBe(board);
        expect(state.solverCurrentPlayer).toBe("black");
        expect(state.solverHistory).toHaveLength(1);
        expect(state.solverHistory[0].moveFrom).toBeNull();
        // The mock startSearch resolves synchronously, so runSolverSearch's
        // finally block has already cleared the searching flag by the time
        // startSolver returns. Real Tauri runs hang here until completion.
        expect(state.isSolverSearching).toBe(false);
        expect(services.solver.startSearch).toHaveBeenCalledTimes(1);
        expect(services.solver.startSearch).toHaveBeenCalledWith(
            board,
            "black",
            100,
            "multiPv",
            expect.any(Number),
        );
        expect(services.ai.initialize).toHaveBeenCalled();
    });

    it("bails out when prepareToReplaceGame fails", async () => {
        const { store, services } = createTestStore({
            ai: createMockAIService({
                checkReady: vi.fn().mockRejectedValue(new Error("not ready")),
            }),
        });
        const board = initializeBoard();

        const result = await store.getState().startSolver(board, "black");

        expect(result).toBe(false);
        expect(store.getState().isSolverActive).toBe(false);
        expect(services.solver.startSearch).not.toHaveBeenCalled();
    });
});

describe("startSolverFromSetup", () => {
    it("resolves the manual tab and starts the solver", async () => {
        const { store, services } = createTestStore();
        const board = initializeBoard();
        store.setState({
            setupTab: "manual",
            setupBoard: board,
            setupCurrentPlayer: "black",
            setupError: null,
        });

        const result = await store.getState().startSolverFromSetup();

        expect(result).toBe(true);
        const state = store.getState();
        expect(state.isSolverActive).toBe(true);
        expect(state.solverHistory).toHaveLength(1);
        expect(state.setupError).toBeNull();
        expect(services.solver.startSearch).toHaveBeenCalledTimes(1);
    });

    it("sets setupError and does not start when board string is invalid", async () => {
        const { store, services } = createTestStore();
        store.setState({
            setupTab: "boardString",
            boardStringInput: "garbage",
        });

        const result = await store.getState().startSolverFromSetup();

        expect(result).toBe(false);
        const state = store.getState();
        expect(state.isSolverActive).toBe(false);
        expect(state.setupError).not.toBeNull();
        expect(services.solver.startSearch).not.toHaveBeenCalled();
    });

    it("sets setupError and does not start when transcript is invalid", async () => {
        const { store, services } = createTestStore();
        store.setState({
            setupTab: "transcript",
            transcriptInput: "zzz",
        });

        const result = await store.getState().startSolverFromSetup();

        expect(result).toBe(false);
        const state = store.getState();
        expect(state.isSolverActive).toBe(false);
        expect(state.setupError).not.toBeNull();
        expect(services.solver.startSearch).not.toHaveBeenCalled();
    });

    it("startSolverFromSetup returns false and sets aiInitFailed when startSolver bails", async () => {
        const { store, services } = createTestStore({
            ai: createMockAIService({
                checkReady: vi.fn().mockRejectedValue(new Error("not ready")),
            }),
        });

        store.setState({
            setupTab: "manual",
            setupBoard: initializeBoard(),
            setupCurrentPlayer: "black",
        });

        const result = await store.getState().startSolverFromSetup();

        expect(result).toBe(false);
        expect(store.getState().setupError).toBe("aiInitFailed");
        expect(store.getState().isSolverActive).toBe(false);
        expect(services.solver.startSearch).not.toHaveBeenCalled();
    });

    it("rejects a board that fails validateBoard (e.g. too few discs)", async () => {
        const { store, services } = createTestStore();
        // A nearly-empty board trips validateBoard's "tooFewDiscs" check.
        const sparseBoard = createEmptyBoard();
        sparseBoard[3][3] = { color: "black" };
        sparseBoard[4][4] = { color: "white" };

        store.setState({
            setupTab: "manual",
            setupBoard: sparseBoard,
            setupCurrentPlayer: "black",
        });

        const result = await store.getState().startSolverFromSetup();

        expect(result).toBe(false);
        expect(store.getState().setupError).not.toBeNull();
        expect(store.getState().isSolverActive).toBe(false);
        expect(services.solver.startSearch).not.toHaveBeenCalled();
    });
});

describe("exitSolver", () => {
    it("aborts the search and clears solver state while preserving targetSelectivity", async () => {
        const { store, services } = createTestStore();
        const board = initializeBoard();
        const rootEntry = buildHistoryEntry(board, "black", null);
        const candidates = new Map<string, SolverCandidate>([
            [
                "2,3",
                {
                    move: "d3",
                    row: 2,
                    col: 3,
                    score: 4,
                    depth: 14,
                    targetDepth: 14,
                    acc: 100,
                    pvLine: "d3",
                    isEndgame: true,
                    isComplete: true,
                },
            ],
        ]);
        store.setState({
            isSolverActive: true,
            solverRootBoard: board,
            solverRootPlayer: "black",
            solverHistory: [rootEntry],
            solverCurrentBoard: board,
            solverCurrentPlayer: "black",
            solverCandidates: candidates,
            isSolverSearching: true,
            targetSelectivity: 95,
        });

        await store.getState().exitSolver();

        expect(services.solver.abort).toHaveBeenCalledTimes(1);
        const state = store.getState();
        expect(state.isSolverActive).toBe(false);
        expect(state.solverRootBoard).toBeNull();
        expect(state.solverRootPlayer).toBeNull();
        expect(state.solverHistory).toEqual([]);
        expect(state.solverCurrentBoard).toBeNull();
        expect(state.solverCurrentPlayer).toBeNull();
        expect(state.solverCandidates.size).toBe(0);
        expect(state.isSolverSearching).toBe(false);
        expect(state.targetSelectivity).toBe(95);
    });
});

describe("advanceSolver", () => {
    it("applies the move, pushes history, and re-runs the search", async () => {
        const { store, services } = createTestStore();
        const board = initializeBoard();
        const rootEntry = buildHistoryEntry(board, "black", null);
        store.setState({
            isSolverActive: true,
            solverRootBoard: board,
            solverRootPlayer: "black",
            solverHistory: [rootEntry],
            solverCurrentBoard: board,
            solverCurrentPlayer: "black",
            solverCandidates: new Map([
                [
                    "0,0",
                    {
                        move: "a1",
                        row: 0,
                        col: 0,
                        score: 0,
                        depth: 1,
                        targetDepth: 1,
                        acc: 100,
                        pvLine: "a1",
                        isEndgame: true,
                        isComplete: true,
                    },
                ],
            ]),
            isSolverSearching: false,
        });

        await store.getState().advanceSolver(2, 3);

        expect(services.solver.abort).toHaveBeenCalledTimes(1);

        const state = store.getState();
        expect(state.solverHistory).toHaveLength(2);
        expect(state.solverHistory[1].moveFrom).toBe(getNotation(2, 3));
        expect(state.solverCurrentPlayer).toBe("white");

        // The board should match applying the move on the original board.
        const expectedBoard = applyMove(board, { row: 2, col: 3, isAI: false, score: 0 }, "black");
        expect(state.solverCurrentBoard).toEqual(expectedBoard);

        expect(state.solverCandidates.size).toBe(0);
        // Mock resolves synchronously → finally clears the flag.
        expect(state.isSolverSearching).toBe(false);

        expect(services.solver.startSearch).toHaveBeenCalledTimes(1);
        expect(services.solver.startSearch).toHaveBeenCalledWith(
            expectedBoard,
            "white",
            100,
            "multiPv",
            expect.any(Number),
        );
    });

    it("skips the search when the new position has no valid moves", async () => {
        const { store, services } = createTestStore();
        const board = initializeBoard();
        store.setState({
            isSolverActive: true,
            solverRootBoard: board,
            solverRootPlayer: "black",
            solverHistory: [buildHistoryEntry(board, "black", null)],
            solverCurrentBoard: board,
            solverCurrentPlayer: "black",
            solverCandidates: new Map(),
            isSolverSearching: false,
        });

        // Pretend the position after the move has zero legal moves for any player.
        getValidMovesStub = () => [];

        await store.getState().advanceSolver(2, 3);

        const state = store.getState();
        expect(state.isSolverSearching).toBe(false);
        expect(services.solver.startSearch).not.toHaveBeenCalled();
        // The advance itself still happened.
        expect(state.solverHistory).toHaveLength(2);
        expect(state.solverCandidates.size).toBe(0);
        // Both players empty → gameOver, no auto-pass, turn stays flipped.
        expect(state.solverCurrentPlayer).toBe("white");
    });

    it("auto-passes when the next player has no moves but the current player still does", async () => {
        const { store, services } = createTestStore();
        const board = initializeBoard();
        store.setState({
            isSolverActive: true,
            solverRootBoard: board,
            solverRootPlayer: "black",
            solverHistory: [buildHistoryEntry(board, "black", null)],
            solverCurrentBoard: board,
            solverCurrentPlayer: "black",
            solverCandidates: new Map(),
            isSolverSearching: false,
        });

        // White (next player) has no moves, but black still does.
        getValidMovesStub = (_board, player) =>
            player === "white" ? [] : [[2, 3]];

        await store.getState().advanceSolver(2, 3);

        const state = store.getState();
        expect(state.solverHistory).toHaveLength(2);
        // Auto-pass flipped the turn back to black.
        expect(state.solverCurrentPlayer).toBe("black");
        // Mock resolves synchronously → finally clears the flag.
        expect(state.isSolverSearching).toBe(false);

        const expectedBoard = applyMove(
            board,
            { row: 2, col: 3, isAI: false, score: 0 },
            "black",
        );
        expect(state.solverCurrentBoard).toEqual(expectedBoard);
        expect(services.solver.startSearch).toHaveBeenCalledWith(
            expectedBoard,
            "black",
            100,
            "multiPv",
            expect.any(Number),
        );
    });
});

describe("undoSolver", () => {
    it("pops the last history entry and re-runs the search with the previous position", async () => {
        const { store, services } = createTestStore();
        const rootBoard = initializeBoard();
        const secondBoard = applyMove(
            rootBoard,
            { row: 2, col: 3, isAI: false, score: 0 },
            "black",
        );

        store.setState({
            isSolverActive: true,
            solverRootBoard: rootBoard,
            solverRootPlayer: "black",
            solverHistory: [
                buildHistoryEntry(rootBoard, "black", null),
                buildHistoryEntry(secondBoard, "white", "d3"),
            ],
            solverCurrentBoard: secondBoard,
            solverCurrentPlayer: "white",
            solverCandidates: new Map([
                [
                    "2,2",
                    {
                        move: "c3",
                        row: 2,
                        col: 2,
                        score: -2,
                        depth: 12,
                        targetDepth: 12,
                        acc: 100,
                        pvLine: "c3",
                        isEndgame: true,
                        isComplete: true,
                    },
                ],
            ]),
            isSolverSearching: false,
        });

        await store.getState().undoSolver();

        expect(services.solver.abort).toHaveBeenCalledTimes(1);
        const state = store.getState();
        expect(state.solverHistory).toHaveLength(1);
        expect(state.solverCurrentBoard).toBe(rootBoard);
        expect(state.solverCurrentPlayer).toBe("black");
        expect(state.solverCandidates.size).toBe(0);
        // Mock resolves synchronously → finally clears the flag.
        expect(state.isSolverSearching).toBe(false);
        expect(services.solver.startSearch).toHaveBeenCalledWith(
            rootBoard,
            "black",
            100,
            "multiPv",
            expect.any(Number),
        );
    });

    it("does not drop the second of two concurrent Back presses", async () => {
        const abortDeferred = createDeferred<void>();
        const abortMock = vi.fn().mockReturnValue(abortDeferred.promise);
        const { store } = createTestStore({
            solver: createMockSolverService({ abort: abortMock }),
        });

        const rootBoard = initializeBoard();
        const secondBoard = applyMove(
            rootBoard,
            { row: 2, col: 3, isAI: false, score: 0 },
            "black",
        );
        const thirdBoard = applyMove(
            secondBoard,
            { row: 2, col: 2, isAI: false, score: 0 },
            "white",
        );

        store.setState({
            isSolverActive: true,
            solverRootBoard: rootBoard,
            solverRootPlayer: "black",
            solverHistory: [
                buildHistoryEntry(rootBoard, "black", null),
                buildHistoryEntry(secondBoard, "white", "d3"),
                buildHistoryEntry(thirdBoard, "black", "c3"),
            ],
            solverCurrentBoard: thirdBoard,
            solverCurrentPlayer: "black",
            solverCandidates: new Map(),
            isSolverSearching: false,
        });

        const firstUndo = store.getState().undoSolver();
        const secondUndo = store.getState().undoSolver();

        await Promise.resolve();
        abortDeferred.resolve();
        await firstUndo;
        await secondUndo;

        const state = store.getState();
        expect(state.solverHistory).toHaveLength(1);
        expect(state.solverCurrentBoard).toBe(rootBoard);
        expect(state.solverCurrentPlayer).toBe("black");
    });

    it("is a no-op when history length is 1", async () => {
        const { store, services } = createTestStore();
        const board = initializeBoard();
        store.setState({
            isSolverActive: true,
            solverRootBoard: board,
            solverRootPlayer: "black",
            solverHistory: [buildHistoryEntry(board, "black", null)],
            solverCurrentBoard: board,
            solverCurrentPlayer: "black",
            isSolverSearching: false,
        });

        await store.getState().undoSolver();

        expect(services.solver.abort).not.toHaveBeenCalled();
        expect(services.solver.startSearch).not.toHaveBeenCalled();
        expect(store.getState().solverHistory).toHaveLength(1);
    });
});

describe("setTargetSelectivity", () => {
    it("updates state and restarts the search when solver is active and searching", async () => {
        const { store, services } = createTestStore();
        const board = initializeBoard();
        store.setState({
            isSolverActive: true,
            solverRootBoard: board,
            solverRootPlayer: "black",
            solverHistory: [buildHistoryEntry(board, "black", null)],
            solverCurrentBoard: board,
            solverCurrentPlayer: "black",
            solverCandidates: new Map([
                [
                    "2,3",
                    {
                        move: "d3",
                        row: 2,
                        col: 3,
                        score: 2,
                        depth: 10,
                        targetDepth: 10,
                        acc: 99,
                        pvLine: "d3",
                        isEndgame: true,
                        isComplete: false,
                    },
                ],
            ]),
            isSolverSearching: true,
        });

        await store.getState().setTargetSelectivity(95);

        const state = store.getState();
        expect(state.targetSelectivity).toBe(95);
        expect(services.solver.abort).toHaveBeenCalledTimes(1);
        expect(services.solver.startSearch).toHaveBeenCalledTimes(1);
        expect(services.solver.startSearch).toHaveBeenCalledWith(
            board,
            "black",
            95,
            "multiPv",
            expect.any(Number),
        );
        expect(state.solverCandidates.size).toBe(0);
        // Mock resolves synchronously → finally clears the flag.
        expect(state.isSolverSearching).toBe(false);
    });

    it("only updates state when solver is inactive", async () => {
        const { store, services } = createTestStore();

        await store.getState().setTargetSelectivity(99);

        expect(store.getState().targetSelectivity).toBe(99);
        expect(services.solver.abort).not.toHaveBeenCalled();
        expect(services.solver.startSearch).not.toHaveBeenCalled();
    });

    it("persists the selectivity via the settings service", async () => {
        const { store, services } = createTestStore();
        await store.getState().setTargetSelectivity(95);
        expect(services.settings.saveSetting).toHaveBeenCalledWith(
            "solverTargetSelectivity",
            95,
        );
    });

    it("restarts the search when solver is active even if not currently searching", async () => {
        const { store, services } = createTestStore();
        const board = initializeBoard();

        // Start solver, then manually mark as "done searching".
        await store.getState().startSolver(board, "black");
        store.setState({ isSolverSearching: false });
        vi.mocked(services.solver.startSearch).mockClear();
        vi.mocked(services.solver.abort).mockClear();

        await store.getState().setTargetSelectivity(95);

        expect(store.getState().targetSelectivity).toBe(95);
        expect(services.solver.abort).toHaveBeenCalledTimes(1);
        expect(services.solver.startSearch).toHaveBeenCalledTimes(1);
        expect(services.solver.startSearch).toHaveBeenCalledWith(
            expect.anything(),
            "black",
            95,
            "multiPv",
            expect.any(Number),
        );
        // Mock resolves synchronously → finally clears the flag.
        expect(store.getState().isSolverSearching).toBe(false);
    });
});

describe("applySolverProgress", () => {
    it("upserts a candidate keyed by row,col and parses the PV line", () => {
        const { store } = createTestStore();
        // applySolverProgress is gated on isSolverActive + runId match.
        store.setState({ isSolverActive: true });

        const payload: SolverProgressPayload = {
            runId: store.getState().solverSearchRunId,
            bestMove: "d3",
            row: 2,
            col: 3,
            score: 4,
            depth: 14,
            targetDepth: 14,
            acc: 100,
            nodes: 123,
            pvLine: "d3 c5 f6",
            isEndgame: false,
        };

        store.getState().applySolverProgress(payload);

        const candidate = store.getState().solverCandidates.get("2,3");
        expect(candidate).toBeDefined();
        expect(candidate?.move).toBe("d3");
        expect(candidate?.row).toBe(2);
        expect(candidate?.col).toBe(3);
        expect(candidate?.score).toBe(4);
        expect(candidate?.depth).toBe(14);
        expect(candidate?.acc).toBe(100);
        expect(candidate?.pvLine).toBe("d3 c5 f6");
        expect(candidate?.isComplete).toBe(true);
    });

    it("marks a midgame candidate incomplete until depth reaches targetDepth", () => {
        const { store } = createTestStore();
        store.setState({ isSolverActive: true });

        const payload: SolverProgressPayload = {
            runId: store.getState().solverSearchRunId,
            bestMove: "e6",
            row: 5,
            col: 4,
            score: 0,
            depth: 8,
            targetDepth: 12,
            acc: 100,
            nodes: 10,
            pvLine: "e6",
            isEndgame: false,
        };

        store.getState().applySolverProgress(payload);

        const candidate = store.getState().solverCandidates.get("5,4");
        expect(candidate).toBeDefined();
        expect(candidate?.isComplete).toBe(false);
        expect(candidate?.isEndgame).toBe(false);
        expect(candidate?.targetDepth).toBe(12);
    });

    it("marks an endgame candidate complete once acc reaches the target selectivity (< 100)", () => {
        const { store } = createTestStore();
        store.setState({ isSolverActive: true, targetSelectivity: 73 });

        const payload: SolverProgressPayload = {
            runId: store.getState().solverSearchRunId,
            bestMove: "f6",
            row: 5,
            col: 5,
            score: 6,
            depth: 20,
            targetDepth: 20,
            acc: 73,
            nodes: 1000,
            pvLine: "f6",
            isEndgame: true,
        };

        store.getState().applySolverProgress(payload);

        const candidate = store.getState().solverCandidates.get("5,5");
        expect(candidate?.isComplete).toBe(true);
        expect(candidate?.isEndgame).toBe(true);
        expect(candidate?.acc).toBe(73);
    });

    it("drops payloads when solver mode is inactive", () => {
        const { store } = createTestStore();

        const payload: SolverProgressPayload = {
            runId: store.getState().solverSearchRunId,
            bestMove: "d3",
            row: 2,
            col: 3,
            score: 4,
            depth: 14,
            targetDepth: 14,
            acc: 100,
            nodes: 1000,
            pvLine: "d3",
            isEndgame: true,
        };

        // Solver inactive — payload dropped.
        store.getState().applySolverProgress(payload);
        expect(store.getState().solverCandidates.size).toBe(0);

        // Active, still searching — accepted.
        store.setState({ isSolverActive: true, isSolverSearching: true });
        store.getState().applySolverProgress(payload);
        expect(store.getState().solverCandidates.size).toBe(1);
    });

    it("drops payloads from a superseded run", () => {
        // Regression guard for the Codex review finding: late solver-progress
        // events from an aborted run must not leak into the state of the
        // newly-started run, even though `isSolverActive` stays true across
        // startSolver/undo/reset/setTargetSelectivity restarts.
        const { store } = createTestStore();
        store.setState({
            isSolverActive: true,
            isSolverSearching: true,
            solverSearchRunId: 7,
        });

        const stalePayload: SolverProgressPayload = {
            runId: 6, // emitted by the previous run, arrived late
            bestMove: "d3",
            row: 2,
            col: 3,
            score: 4,
            depth: 14,
            targetDepth: 14,
            acc: 100,
            nodes: 1000,
            pvLine: "d3",
            isEndgame: true,
        };

        store.getState().applySolverProgress(stalePayload);
        expect(store.getState().solverCandidates.size).toBe(0);

        // A payload from the current run still lands.
        store
            .getState()
            .applySolverProgress({ ...stalePayload, runId: 7 });
        expect(store.getState().solverCandidates.size).toBe(1);
    });

    it("accepts trailing payloads after isSolverSearching clears", () => {
        // Regression guard: runSolverSearch's finally block clears
        // isSolverSearching as soon as startSearch resolves, but trailing
        // solver-progress events can still be queued on the JS side.
        // Those payloads must still reach the store so the final
        // depth/accuracy update isn't lost.
        const { store } = createTestStore();
        store.setState({ isSolverActive: true, isSolverSearching: false });

        const payload: SolverProgressPayload = {
            runId: store.getState().solverSearchRunId,
            bestMove: "d3",
            row: 2,
            col: 3,
            score: 4,
            depth: 14,
            targetDepth: 14,
            acc: 100,
            nodes: 1000,
            pvLine: "d3",
            isEndgame: true,
        };

        store.getState().applySolverProgress(payload);
        expect(store.getState().solverCandidates.size).toBe(1);
        expect(store.getState().solverCandidates.get("2,3")?.isComplete).toBe(true);
    });
});

describe("runSolverSearch error handling", () => {
    it("clears isSolverSearching when startSearch fails", async () => {
        const { store, services } = createTestStore({
            solver: createMockSolverService({
                startSearch: vi.fn().mockRejectedValue(new Error("boom")),
            }),
        });

        const consoleErrorSpy = vi
            .spyOn(console, "error")
            .mockImplementation(() => {});

        const board = initializeBoard();
        await store.getState().startSolver(board, "black");

        expect(store.getState().isSolverSearching).toBe(false);
        // Solver stays active — only the searching flag is cleared.
        expect(store.getState().isSolverActive).toBe(true);
        expect(consoleErrorSpy).toHaveBeenCalled();
        expect(services.solver.startSearch).toHaveBeenCalledTimes(1);

        consoleErrorSpy.mockRestore();
    });

    it("stale search errors do not clobber a newer search", async () => {
        const firstDeferred = createDeferred<void>();
        const secondDeferred = createDeferred<void>();
        const startSearchMock = vi
            .fn()
            .mockImplementationOnce(() => firstDeferred.promise)
            .mockImplementationOnce(() => secondDeferred.promise);

        const { store } = createTestStore({
            solver: createMockSolverService({
                startSearch: startSearchMock,
            }),
        });

        const consoleErrorSpy = vi
            .spyOn(console, "error")
            .mockImplementation(() => {});

        // Flush pending microtasks until `predicate` is true, up to 50 ticks.
        const flushUntil = async (predicate: () => boolean) => {
            for (let i = 0; i < 50; i++) {
                if (predicate()) return;
                await Promise.resolve();
            }
            throw new Error("flushUntil: predicate never became true");
        };

        const board = initializeBoard();

        // Kick off the first search (its startSearch promise hangs).
        const firstPromise = store.getState().startSolver(board, "black");

        // Wait until the first search has actually reached startSearch.
        await flushUntil(() => startSearchMock.mock.calls.length === 1);
        expect(store.getState().isSolverSearching).toBe(true);
        const firstRunId = store.getState().solverSearchRunId;

        // Kick off a second search before the first settles. This simulates
        // the race: the user clicks again while a search is in flight. The
        // second call bumps the run id so the first's eventual rejection
        // should notice it is stale and leave isSolverSearching alone.
        const secondPromise = store.getState().startSolver(board, "white");

        // Wait until the second search has reached startSearch too.
        await flushUntil(() => startSearchMock.mock.calls.length === 2);
        const secondRunId = store.getState().solverSearchRunId;
        expect(secondRunId).toBeGreaterThan(firstRunId);
        expect(store.getState().isSolverSearching).toBe(true);

        // Reject the first — its catch branch must detect it is stale and
        // leave the newer run's `isSolverSearching` flag alone.
        firstDeferred.reject(new Error("aborted"));
        await firstPromise;

        // The newer run is still active (its promise hasn't settled yet).
        expect(store.getState().isSolverSearching).toBe(true);
        expect(store.getState().solverSearchRunId).toBe(secondRunId);

        // Now let the second run resolve cleanly.
        secondDeferred.resolve();
        await secondPromise;

        // Run id should NOT have been decremented by the stale error path.
        expect(store.getState().solverSearchRunId).toBe(secondRunId);

        consoleErrorSpy.mockRestore();
    });
});

// Sanity check that our mock respects the real module shape.
describe("game-logic mock sanity", () => {
    it("falls through to the real getValidMoves when no stub is set", () => {
        expect(getValidMovesStub).toBeNull();
        const board = initializeBoard();
        const moves = realGetValidMoves(board, "black");
        expect(moves.length).toBeGreaterThan(0);
        // Also exercise opponentPlayer so its import isn't flagged unused.
        expect(opponentPlayer("black")).toBe("white");
    });
});
