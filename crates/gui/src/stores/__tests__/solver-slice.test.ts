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
} from "@/domain/game/game-logic";
import { applyMove } from "@/domain/game/store-helpers";
import type { Board, Player } from "@/domain/game/types";
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

vi.mock("@/domain/game/game-logic", async (importOriginal) => {
    const actual = await importOriginal<typeof import("@/domain/game/game-logic")>();
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

function latestSolverRunId(services: ReturnType<typeof createTestStore>["services"]): number {
    const calls = vi.mocked(services.solver.startSearch).mock.calls;
    const runId = calls[calls.length - 1]?.[4];
    if (typeof runId !== "number") {
        throw new Error("No solver search run id was captured");
    }
    return runId;
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

describe("subscribeSolverProgress", () => {
    it("routes injected solver progress through the Solver Session", async () => {
        const progressCallbacks: Array<(payload: SolverProgressPayload) => void> = [];
        const unlisten = vi.fn();
        const { store, services } = createTestStore({
            solver: createMockSolverService({
                onProgress: vi.fn(async (callback) => {
                    progressCallbacks.push(callback);
                    return unlisten;
                }),
            }),
        });
        const board = initializeBoard();
        await store.getState().startSolver(board, "black");

        const unsubscribe = await store.getState().subscribeSolverProgress();
        expect(progressCallbacks).toHaveLength(1);
        progressCallbacks[0]({
            runId: latestSolverRunId(services),
            bestMove: "d3",
            row: 2,
            col: 3,
            score: 4,
            depth: 14,
            targetDepth: 14,
            acc: 100,
            nodes: 1000,
            pvLine: "d3",
            isEndgame: false,
        });
        unsubscribe();

        expect(store.getState().solverCandidates.get("2,3")?.move).toBe("d3");
        expect(unlisten).toHaveBeenCalledTimes(1);
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

    it("bails out when prepareGameReplacement fails", async () => {
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

    it("commits supplied solver config only after setup validation succeeds", async () => {
        const { store, services } = createTestStore();
        const board = initializeBoard();
        store.setState({
            setupTab: "manual",
            setupBoard: board,
            setupCurrentPlayer: "black",
            targetSelectivity: 100,
            solverMode: "multiPv",
        });

        const result = await store.getState().startSolverFromSetup({
            selectivity: 95,
            mode: "bestOnly",
        });

        expect(result).toBe(true);
        expect(store.getState().targetSelectivity).toBe(95);
        expect(store.getState().solverMode).toBe("bestOnly");
        expect(services.settings.saveSetting).toHaveBeenCalledWith(
            "solverTargetSelectivity",
            95,
        );
        expect(services.settings.saveSetting).toHaveBeenCalledWith(
            "solverMode",
            "bestOnly",
        );
        expect(services.solver.startSearch).toHaveBeenCalledWith(
            board,
            "black",
            95,
            "bestOnly",
            expect.any(Number),
        );
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

    it("does not commit supplied solver config when setup validation fails", async () => {
        const { store, services } = createTestStore();
        const sparseBoard = createEmptyBoard();
        sparseBoard[3][3] = { color: "black" };
        sparseBoard[4][4] = { color: "white" };

        store.setState({
            isSolverActive: true,
            setupTab: "manual",
            setupBoard: sparseBoard,
            setupCurrentPlayer: "black",
            targetSelectivity: 100,
            solverMode: "multiPv",
        });

        const result = await store.getState().startSolverFromSetup({
            selectivity: 95,
            mode: "bestOnly",
        });

        expect(result).toBe(false);
        expect(store.getState().targetSelectivity).toBe(100);
        expect(store.getState().solverMode).toBe("multiPv");
        expect(store.getState().setupError).not.toBeNull();
        expect(services.settings.saveSetting).not.toHaveBeenCalled();
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
        // Mock resolves synchronously ↁEfinally clears the flag.
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
        // Both players empty ↁEgameOver, no auto-pass, turn stays flipped.
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
        // Mock resolves synchronously ↁEfinally clears the flag.
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
        // Mock resolves synchronously ↁEfinally clears the flag.
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
        // Mock resolves synchronously ↁEfinally clears the flag.
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
        // Mock resolves synchronously ↁEfinally clears the flag.
        expect(store.getState().isSolverSearching).toBe(false);
    });
});

describe("applySolverProgress", () => {
    it("upserts a candidate keyed by row,col and parses the PV line", async () => {
        const { store, services } = createTestStore();
        const board = initializeBoard();
        await store.getState().startSolver(board, "black");

        const payload: SolverProgressPayload = {
            runId: latestSolverRunId(services),
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

    it("marks a midgame candidate incomplete until depth reaches targetDepth", async () => {
        const { store, services } = createTestStore();
        await store.getState().startSolver(initializeBoard(), "black");

        const payload: SolverProgressPayload = {
            runId: latestSolverRunId(services),
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

    it("marks an endgame candidate complete once acc reaches the target selectivity (< 100)", async () => {
        const { store, services } = createTestStore();
        await store.getState().startSolver(initializeBoard(), "black");
        store.setState({ targetSelectivity: 73 });

        const payload: SolverProgressPayload = {
            runId: latestSolverRunId(services),
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

    it("drops payloads when solver mode is inactive", async () => {
        const { store, services } = createTestStore();

        const payload: SolverProgressPayload = {
            runId: 1,
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

        // Solver inactive  Epayload dropped.
        store.getState().applySolverProgress(payload);
        expect(store.getState().solverCandidates.size).toBe(0);

        // Active, still searching  Eaccepted.
        await store.getState().startSolver(initializeBoard(), "black");
        store.getState().applySolverProgress({
            ...payload,
            runId: latestSolverRunId(services),
        });
        expect(store.getState().solverCandidates.size).toBe(1);
    });

    it("drops payloads from a superseded run", async () => {
        // Regression guard for the Codex review finding: late solver-progress
        // events from an aborted run must not leak into the state of the
        // newly-started run, even though `isSolverActive` stays true across
        // startSolver/undo/reset/setTargetSelectivity restarts.
        const { store, services } = createTestStore();
        await store.getState().startSolver(initializeBoard(), "black");
        const staleRunId = latestSolverRunId(services);
        await store.getState().setTargetSelectivity(95);
        const currentRunId = latestSolverRunId(services);

        const stalePayload: SolverProgressPayload = {
            runId: staleRunId, // emitted by the previous run, arrived late
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
            .applySolverProgress({ ...stalePayload, runId: currentRunId });
        expect(store.getState().solverCandidates.size).toBe(1);
    });

    it("in bestOnly mode keeps only the latest best move (drops earlier-selectivity picks)", async () => {
        const { store, services } = createTestStore();
        await store.getState().startSolver(initializeBoard(), "black");
        store.setState({
            solverMode: "bestOnly",
            targetSelectivity: 100,
        });

        const lowSelPayload: SolverProgressPayload = {
            runId: latestSolverRunId(services),
            bestMove: "d3",
            row: 2,
            col: 3,
            score: 4,
            depth: 20,
            targetDepth: 20,
            acc: 73,
            nodes: 100,
            pvLine: "d3",
            isEndgame: true,
        };
        const highSelPayload: SolverProgressPayload = {
            ...lowSelPayload,
            bestMove: "f5",
            row: 4,
            col: 5,
            score: 6,
            acc: 95,
            pvLine: "f5",
        };

        store.getState().applySolverProgress(lowSelPayload);
        store.getState().applySolverProgress(highSelPayload);

        const candidates = store.getState().solverCandidates;
        expect(candidates.size).toBe(1);
        expect(candidates.get("4,5")?.move).toBe("f5");
        expect(candidates.get("2,3")).toBeUndefined();
    });

    it("in multiPv mode preserves earlier candidates as new ones arrive", async () => {
        const { store, services } = createTestStore();
        await store.getState().startSolver(initializeBoard(), "black");
        store.setState({
            solverMode: "multiPv",
            targetSelectivity: 100,
        });

        const a: SolverProgressPayload = {
            runId: latestSolverRunId(services),
            bestMove: "d3",
            row: 2,
            col: 3,
            score: 4,
            depth: 20,
            targetDepth: 20,
            acc: 100,
            nodes: 100,
            pvLine: "d3",
            isEndgame: true,
        };
        const b: SolverProgressPayload = {
            ...a,
            bestMove: "f5",
            row: 4,
            col: 5,
            score: 2,
            pvLine: "f5",
        };

        store.getState().applySolverProgress(a);
        store.getState().applySolverProgress(b);

        expect(store.getState().solverCandidates.size).toBe(2);
    });

    it("accepts trailing payloads after isSolverSearching clears", async () => {
        // Regression guard: runSolverSearch's finally block clears
        // isSolverSearching as soon as startSearch resolves, but trailing
        // solver-progress events can still be queued on the JS side.
        // Those payloads must still reach the store so the final
        // depth/accuracy update isn't lost.
        const { store, services } = createTestStore();
        await store.getState().startSolver(initializeBoard(), "black");
        store.setState({ isSolverSearching: false });

        const payload: SolverProgressPayload = {
            runId: latestSolverRunId(services),
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
        // Solver stays active  Eonly the searching flag is cleared.
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
        const firstRunId = startSearchMock.mock.calls[0][4] as number;

        // Kick off a second search before the first settles. This simulates
        // the race: the user clicks again while a search is in flight. The
        // second call bumps the run id so the first's eventual rejection
        // should notice it is stale and leave isSolverSearching alone.
        const secondPromise = store.getState().startSolver(board, "white");

        // Wait until the second search has reached startSearch too.
        await flushUntil(() => startSearchMock.mock.calls.length === 2);
        const secondRunId = startSearchMock.mock.calls[1][4] as number;
        expect(secondRunId).toBeGreaterThan(firstRunId);
        expect(store.getState().isSolverSearching).toBe(true);

        // Reject the first  Eits catch branch must detect it is stale and
        // leave the newer run's `isSolverSearching` flag alone.
        firstDeferred.reject(new Error("aborted"));
        await firstPromise;

        // The newer run is still active (its promise hasn't settled yet).
        expect(store.getState().isSolverSearching).toBe(true);
        store.getState().applySolverProgress({
            runId: firstRunId,
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
        });
        expect(store.getState().solverCandidates.size).toBe(0);

        // Now let the second run resolve cleanly.
        secondDeferred.resolve();
        await secondPromise;

        expect(startSearchMock.mock.calls[1][4]).toBe(secondRunId);

        consoleErrorSpy.mockRestore();
    });
});

describe("solver result cache", () => {
    it("does not share cached candidates across store instances", async () => {
        const board = initializeBoard();
        let firstStore: ReturnType<typeof createTestStore>["store"] | null = null;

        const startSearchMock = vi.fn((_board, _player, _selectivity, _mode, runId) => {
            firstStore!.getState().applySolverProgress({
                runId,
                bestMove: "d3",
                row: 2,
                col: 3,
                score: 4,
                depth: 14,
                targetDepth: 14,
                acc: 100,
                nodes: 1000,
                pvLine: "d3",
                isEndgame: false,
            });
            return Promise.resolve();
        });

        const first = createTestStore({
            solver: createMockSolverService({ startSearch: startSearchMock }),
        });
        firstStore = first.store;

        await first.store.getState().startSolver(board, "black");
        expect(first.store.getState().solverCandidates.size).toBe(1);
        expect(startSearchMock).toHaveBeenCalledTimes(1);

        const secondStartSearchMock = vi.fn().mockResolvedValue(undefined);
        const second = createTestStore({
            solver: createMockSolverService({ startSearch: secondStartSearchMock }),
        });
        second.store.setState({
            isSolverActive: true,
            solverCurrentBoard: board,
            solverCurrentPlayer: "black",
            targetSelectivity: 100,
            solverMode: "multiPv",
            isSolverSearching: false,
            isSolverStopped: true,
        });

        await second.store.getState().resumeSolverSearch();

        expect(secondStartSearchMock).toHaveBeenCalledTimes(1);
        expect(second.store.getState().solverCandidates.size).toBe(0);
    });
});

describe("shared EngineSearch cross-feature supersede", () => {
    it("starting an AI move supersedes an in-flight solver search and filters its stale progress", async () => {
        // The solver and AI share one EngineSearch instance (created once per
        // store). A solver search in flight is the live run; starting an AI
        // move supersedes it: the solver's registered abort runs, the
        // generation is bumped, so a late solver-progress stamped with the
        // superseded run's id is filtered by engineSearch.accepts and cannot
        // mutate solverCandidates. isSolverSearching must not stick true.
        const startSearchDeferred = createDeferred<void>();
        const getAIMoveDeferred = createDeferred<null>();
        let solverRunId = -1;
        const startSearch = vi.fn(
            (_b: Board, _p: Player, _s: number, _m: string, runId: number) => {
                solverRunId = runId;
                return startSearchDeferred.promise;
            },
        );
        const { store, services } = createTestStore({
            solver: createMockSolverService({ startSearch }),
            ai: createMockAIService({
                getAIMove: vi.fn().mockReturnValue(getAIMoveDeferred.promise),
            }),
        });
        const board = initializeBoard();

        const solverPending = store.getState().startSolver(board, "black");
        for (let i = 0; i < 20 && solverRunId < 0; i++) await Promise.resolve();
        expect(store.getState().isSolverSearching).toBe(true);

        // Starting an AI move supersedes the in-flight solver run. Drain
        // microtasks until the AI search has taken over (it commits
        // isAIThinking after EngineSearch's await supersede()).
        const aiPending = store.getState().makeAIMove();
        for (let i = 0; i < 20 && !store.getState().isAIThinking; i++) {
            await Promise.resolve();
        }
        expect(store.getState().isSolverSearching).toBe(false);
        expect(store.getState().isAIThinking).toBe(true);
        // The solver run's registered abort fired during the supersede.
        expect(services.solver.abort).toHaveBeenCalled();

        // A late solver-progress stamped with the superseded run's id must NOT
        // mutate solverCandidates (filtered by engineSearch.accepts).
        store.getState().applySolverProgress({
            runId: solverRunId,
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
        });
        expect(store.getState().solverCandidates.size).toBe(0);

        getAIMoveDeferred.resolve(null);
        await aiPending;
        expect(store.getState().isSolverSearching).toBe(false);

        startSearchDeferred.resolve();
        await solverPending;
        expect(store.getState().isSolverSearching).toBe(false);
        expect(store.getState().solverCandidates.size).toBe(0);
    });

    it("a cache-hit navigation supersedes the prior run so its late progress cannot overwrite the cached candidates", async () => {
        // Note: a genuine RED-first (cache-hit as a bare commit) could not be
        // constructed through the public store API — the shared engine's
        // generation always advances past the in-flight run's id before the
        // late progress arrives, so the bare-commit variant also filters it
        // (verified empirically). This is therefore a GREEN behavioral guard
        // that the cache-hit path commits the cached candidates and a late
        // solver-progress stamped with the prior in-flight run's id does not
        // corrupt them; the underlying supersede correctness is independently
        // covered by the cross-feature test above and "drops payloads from a
        // superseded run".
        const hang = createDeferred<void>();
        let hangRunId = -1;
        let cacheRootMode = true;
        const startSearch = vi.fn(
            (_b: Board, _p: Player, _s: number, _m: string, runId: number) => {
                if (cacheRootMode) {
                    store.getState().applySolverProgress({
                        runId,
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
                    });
                    return Promise.resolve();
                }
                hangRunId = runId;
                return hang.promise;
            },
        );
        const { store } = createTestStore({
            solver: createMockSolverService({ startSearch }),
        });
        const board = initializeBoard();

        // Root search completes with a complete candidate -> root is cached.
        await store.getState().startSolver(board, "black");
        expect(store.getState().solverCandidates.size).toBe(1);
        // Advance d3 then undo back to the cached root.
        await store.getState().advanceSolver(2, 3);
        await store.getState().undoSolver();
        expect(store.getState().solverHistory).toHaveLength(1);

        // Advance to a different move whose search hangs (prior run in flight).
        cacheRootMode = false;
        const pending = store.getState().advanceSolver(2, 2);
        for (let i = 0; i < 30 && hangRunId < 0; i++) await Promise.resolve();
        expect(store.getState().isSolverSearching).toBe(true);

        // Undo back to root: root is cached -> cache-hit path supersedes the
        // in-flight run and commits the cached candidates.
        await store.getState().undoSolver();
        const cachedCandidates = store.getState().solverCandidates;
        expect(cachedCandidates.size).toBe(1);
        expect(cachedCandidates.get("2,3")?.move).toBe("d3");

        // Late progress stamped with the now-superseded in-flight run's id
        // must NOT overwrite the committed cached candidates.
        store.getState().applySolverProgress({
            runId: hangRunId,
            bestMove: "a1",
            row: 0,
            col: 0,
            score: 99,
            depth: 1,
            targetDepth: 1,
            acc: 100,
            nodes: 1,
            pvLine: "a1",
            isEndgame: true,
        });
        expect(store.getState().solverCandidates).toBe(cachedCandidates);
        expect(store.getState().solverCandidates.has("0,0")).toBe(false);
        expect(store.getState().solverCandidates.size).toBe(1);

        hang.resolve();
        await pending;
        expect(store.getState().solverCandidates.has("0,0")).toBe(false);
    });
});

describe("superseded solver teardown does not poison the prior position's cache (P1 #2)", () => {
    it("a superseded run's teardown must not cache the superseding position's candidates under its own (board,player) key", async () => {
        // Scenario reproducing the P1 #2 race:
        //  1. Root P0 search completes -> P0 cached with candidate "2,3"->d3.
        //  2. Advance to P1 (after move 2,3); P1's search HANGS -> run R_p1 is
        //     the live run, isSolverSearching=true, solverCandidates committed
        //     empty.
        //  3. Undo back to P0: P0 is cached -> cache-hit path. Its onClaim
        //     synchronously commits P0's cached candidates AND supersedes R_p1.
        //  4. R_p1's abort (registered as () => solver.abort()) is slow; while
        //     it is awaited inside supersede(), the new position is already P0
        //     with P0's candidates committed.
        //  5. Resolving R_p1's abort fires its solverTeardown in supersede()'s
        //     finally. With the BUG it unconditionally caches
        //     this.read().solverCandidates (== P0's complete candidates) under
        //     R_p1's captured (P1,white) key -> P1's cache is POISONED with
        //     P0's moves. With the fix it skips caching for a superseded run.
        //  6. Re-advancing to P1 must therefore issue a FRESH search (cache
        //     miss), not return P0's poisoned candidates as a cache hit.
        // abort() resolves immediately unless `hangAbort` is set, in which
        // case the FIRST such call returns a deferred we control. This lets us
        // freeze precisely R_p1's registered abort while it is awaited inside
        // the undo cache-hit's supersede(), so the new (P0) position and its
        // candidates are already committed when R_p1's teardown finally runs.
        interface VoidDeferred {
            promise: Promise<void>;
            resolve: (value: void | PromiseLike<void>) => void;
            reject: (reason?: unknown) => void;
        }
        const makeDeferred = (): VoidDeferred => createDeferred<void>();
        let hangAbort = false;
        const hungAbortBox: { current: VoidDeferred | null } = { current: null };
        const abortMock = vi.fn(() => {
            if (hangAbort && !hungAbortBox.current) {
                hungAbortBox.current = makeDeferred();
                return hungAbortBox.current.promise;
            }
            return Promise.resolve();
        });

        const p1HangBox: { current: VoidDeferred | null } = { current: null };
        let p1RunId = -1;
        let rootMode = true;
        const startSearch = vi.fn(
            (_b: Board, _p: Player, _s: number, _m: string, runId: number) => {
                if (rootMode) {
                    // Root P0 search: emit a complete candidate and resolve so
                    // P0 is cached by its natural (ok) teardown.
                    store.getState().applySolverProgress({
                        runId,
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
                    });
                    return Promise.resolve();
                }
                // P1 search hangs so R_p1 is the live run when we undo.
                p1RunId = runId;
                p1HangBox.current = makeDeferred();
                return p1HangBox.current.promise;
            },
        );

        const { store, services } = createTestStore({
            solver: createMockSolverService({ startSearch, abort: abortMock }),
        });
        const board = initializeBoard();

        // 1. Root P0 search completes -> P0 cached.
        await store.getState().startSolver(board, "black");
        expect(store.getState().solverCandidates.get("2,3")?.move).toBe("d3");

        // 2. Advance to P1; its search hangs (R_p1 live, in flight).
        rootMode = false;
        const p1Pending = store.getState().advanceSolver(2, 3);
        for (let i = 0; i < 30 && p1RunId < 0; i++) await Promise.resolve();
        expect(store.getState().isSolverSearching).toBe(true);
        const p1Board = store.getState().solverCurrentBoard;
        const p1Player = store.getState().solverCurrentPlayer;
        expect(store.getState().solverCandidates.size).toBe(0);

        // 3 + 4 + 5. Undo back to P0 (cache hit): commits P0's cached
        // candidates, supersedes R_p1. R_p1's registered abort hangs (the
        // first hung abort), so when we resolve it the new (P0) position and
        // its complete candidates are already committed before R_p1's
        // teardown runs in supersede()'s finally.
        hangAbort = true;
        const undoPending = store.getState().undoSolver();
        for (let i = 0; i < 30 && !hungAbortBox.current; i++) await Promise.resolve();
        expect(store.getState().solverCurrentBoard).toEqual(board);
        expect(store.getState().solverCandidates.get("2,3")?.move).toBe("d3");
        // Release R_p1's abort: supersede() proceeds and fires R_p1's
        // teardown while solverCandidates already holds P0's complete map.
        hangAbort = false;
        hungAbortBox.current?.resolve();
        for (let i = 0; i < 10; i++) await Promise.resolve();
        await undoPending;
        expect(store.getState().solverCurrentBoard).toEqual(board);
        expect(store.getState().solverCandidates.get("2,3")?.move).toBe("d3");

        // Let the hung P1 search settle so its start() promise resolves.
        p1HangBox.current?.resolve();
        await p1Pending;

        // 6. Re-advance to P1. If R_p1's teardown poisoned P1's cache with
        // P0's candidates, this is a (wrong) cache hit returning d3 and NO
        // fresh search is issued. With the fix P1 is uncached -> a fresh
        // search runs for the P1 (board,player).
        startSearch.mockClear();
        p1RunId = -1;
        rootMode = false;
        const rePending = store.getState().advanceSolver(2, 3);
        for (let i = 0; i < 30 && startSearch.mock.calls.length === 0; i++) {
            await Promise.resolve();
        }
        expect(store.getState().solverCurrentBoard).toEqual(p1Board);
        expect(store.getState().solverCurrentPlayer).toBe(p1Player);
        // The cache for P1 must NOT have been poisoned with P0's candidates.
        expect(services.solver.startSearch).toHaveBeenCalledWith(
            p1Board,
            p1Player,
            100,
            "multiPv",
            expect.any(Number),
        );
        expect(store.getState().solverCandidates.get("2,3")).toBeUndefined();

        p1HangBox.current?.resolve();
        await rePending;
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
