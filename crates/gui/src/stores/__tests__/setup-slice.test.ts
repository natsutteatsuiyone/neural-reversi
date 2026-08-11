import { afterAll, beforeEach, describe, expect, it, vi } from "vitest";
import { createMockAIService } from "@/services/mock-ai-service";
import { createEmptyBoard } from "@/domain/game/game-logic";
import { createTestStore, type TestStore } from "./test-helpers";

const consoleErrorSpy = vi.spyOn(console, "error").mockImplementation(() => {});

afterAll(() => {
  consoleErrorSpy.mockRestore();
});

describe("setup editing", () => {
  it("resets all setup state", () => {
    const { store } = createTestStore();
    store.setState({
      setupTab: "transcript",
      setupCurrentPlayer: "white",
      transcriptInput: "F5",
      boardStringInput: "X".repeat(64),
      setupError: "someError",
    });

    store.getState().resetSetup();

    expect(store.getState()).toMatchObject({
      setupCurrentPlayer: "black",
      setupTab: "manual",
      transcriptInput: "",
      boardStringInput: "",
      setupError: null,
    });
    expect(store.getState().setupBoard[3][3].color).toBe("white");
  });

  it("clears the board and its error", () => {
    const { store } = createTestStore();
    store.setState({ setupError: "someError" });

    store.getState().clearSetupBoard();

    expect(
      store
        .getState()
        .setupBoard.flat()
        .every((cell) => cell.color === null),
    ).toBe(true);
    expect(store.getState().setupError).toBeNull();
  });

  it("restores the initial board and player", () => {
    const { store } = createTestStore();
    store.getState().clearSetupBoard();
    store.setState({ setupCurrentPlayer: "white" });

    store.getState().resetSetupToInitial();

    expect(store.getState().setupCurrentPlayer).toBe("black");
    expect(store.getState().setupBoard[3][3].color).toBe("white");
    expect(store.getState().setupBoard[3][4].color).toBe("black");
  });

  it("cycles a cell through empty, black, and white", () => {
    const { store } = createTestStore();
    store.getState().clearSetupBoard();

    for (const color of ["black", "white", null] as const) {
      store.getState().setSetupCellColor(0, 0);
      expect(store.getState().setupBoard[0][0].color).toBe(color);
    }
  });
});

describe("setSetupTab", () => {
  it("resolves the selected input", () => {
    const { store } = createTestStore();
    store.setState({ transcriptInput: "F5D6" });

    store.getState().setSetupTab("transcript");

    expect(store.getState().setupTab).toBe("transcript");
    expect(store.getState().setupError).toBeNull();
    expect(store.getState().setupBoard[4][5].color).toBe("black");
  });

  it("keeps the selected tab and exposes parse errors", () => {
    const { store } = createTestStore();
    store.setState({ transcriptInput: "Z" });

    store.getState().setSetupTab("transcript");

    expect(store.getState().setupTab).toBe("transcript");
    expect(store.getState().setupError).not.toBeNull();
  });
});

describe("setTranscriptInput", () => {
  it("updates board and player on valid input", () => {
    const { store } = createTestStore();
    store.getState().setTranscriptInput("F5D6");
    const s = store.getState();
    expect(s.transcriptInput).toBe("F5D6");
    expect(s.setupError).toBeNull();
    expect(s.setupBoard[4][5].color).toBe("black");
    expect(s.setupCurrentPlayer).toBe("black");
  });

  it("sets setupError on invalid input", () => {
    const { store } = createTestStore();
    store.getState().setTranscriptInput("Z");
    const s = store.getState();
    expect(s.transcriptInput).toBe("Z");
    expect(s.setupError).not.toBeNull();
  });
});

describe("setBoardStringInput", () => {
  it("updates the board and clears a prior parse error", () => {
    const { store } = createTestStore();
    const boardString = "-".repeat(27) + "OX------XO" + "-".repeat(27);
    store.getState().setBoardStringInput("too-short");
    expect(store.getState().setupError).not.toBeNull();

    store.getState().setBoardStringInput(boardString);

    expect(store.getState().boardStringInput).toBe(boardString);
    expect(store.getState().setupError).toBeNull();
    expect(store.getState().setupBoard[3][3].color).toBe("white");
  });
});

describe("startFromSetup", () => {
  let store: TestStore;

  beforeEach(() => {
    ({ store } = createTestStore());
  });

  it("starts a playable game with the supplied settings", async () => {
    const started = await store.getState().startFromSetup({
      gameMode: "pvp",
      aiLevel: 10,
      aiMode: "level",
      gameTimeLimit: 150,
    });

    const state = store.getState();
    expect(started).toBe(true);
    expect(state).toMatchObject({
      gameStatus: "playing",
      gameOver: false,
      setupError: null,
      gameMode: "pvp",
      aiLevel: 10,
      aiMode: "level",
      gameTimeLimit: 150,
      aiRemainingTime: 150000,
    });
    expect(state.moveHistory.length).toBe(0);
    expect(state.validMoves).toContainEqual([2, 3]);
  });

  it("sets setupError when board validation fails", async () => {
    // Board with only black stones  EvalidateBoard returns "needBothColors"
    const board = createEmptyBoard();
    board[0][0] = { color: "black" };
    store.setState({ setupBoard: board });
    const started = await store.getState().startFromSetup();
    expect(started).toBe(false);
    expect(store.getState().setupError).toBe("needBothColors");
    expect(store.getState().gameStatus).not.toBe("playing");
  });

  it("does not abort the current game when the setup AI readiness check fails", async () => {
    ({ store } = createTestStore({
      ai: createMockAIService({
        checkReady: vi.fn().mockRejectedValue(new Error("check failed")),
      }),
    }));
    store.setState({
      gameStatus: "playing",
      gameMode: "ai-black",
      currentPlayer: "black",
      engineActivity: { kind: "ai-move", runId: 1 },
    });
    const abortSpy = vi.spyOn(store.getState(), "abortAIMove");

    const started = await store.getState().startFromSetup();

    expect(started).toBe(false);
    expect(store.getState().setupError).toBe("aiInitFailed");
    expect(abortSpy).not.toHaveBeenCalled();
    expect(store.getState().engineActivity.kind).toBe("ai-move");
    expect(store.getState().gameStatus).toBe("playing");
  });

  it("restores the current game when setup search reset fails", async () => {
    ({ store } = createTestStore({
      ai: createMockAIService({
        initialize: vi.fn().mockRejectedValue(new Error("init failed")),
      }),
    }));
    store.setState({
      gameStatus: "playing",
      gameMode: "ai-black",
      currentPlayer: "black",
      engineActivity: { kind: "ai-move", runId: 1 },
      aiLevel: 21,
      aiMode: "game-time",
      gameTimeLimit: 60,
    });
    const abortSpy = vi.spyOn(store.getState(), "abortAIMove");
    const makeAIMoveSpy = vi.spyOn(store.getState(), "makeAIMove").mockResolvedValue(undefined);

    const started = await store.getState().startFromSetup({
      gameMode: "pvp",
      aiLevel: 8,
      aiMode: "level",
      gameTimeLimit: 120,
    });

    expect(started).toBe(false);
    expect(store.getState().setupError).toBe("aiInitFailed");
    expect(abortSpy).toHaveBeenCalled();
    expect(makeAIMoveSpy).toHaveBeenCalled();
    expect(store.getState().gameStatus).toBe("playing");
    expect(store.getState().gameMode).toBe("ai-black");
    expect(store.getState().aiLevel).toBe(21);
    expect(store.getState().aiMode).toBe("game-time");
    expect(store.getState().gameTimeLimit).toBe(60);
  });

  it("exits solver mode after a successful setup game start", async () => {
    const test = createTestStore();
    store = test.store;
    const { services } = test;
    const solverBoard = store.getState().board;
    store.setState({
      isSolverActive: true,
      solverHistory: [{ board: solverBoard, player: "black", moveFrom: null }],
    });

    const started = await store.getState().startFromSetup();

    expect(started).toBe(true);
    expect(services.solver.abort).toHaveBeenCalledTimes(2);
    expect(store.getState().isSolverActive).toBe(false);
    expect(store.getState().solverHistory).toEqual([]);
    expect(store.getState().gameStatus).toBe("playing");
  });

  it("preserves solver state when setup game init fails", async () => {
    ({ store } = createTestStore({
      ai: createMockAIService({
        initialize: vi.fn().mockRejectedValue(new Error("init failed")),
      }),
    }));
    const solverBoard = store.getState().board;
    store.setState({
      isSolverActive: true,
      solverHistory: [{ board: solverBoard, player: "black", moveFrom: null }],
    });

    const started = await store.getState().startFromSetup();

    expect(started).toBe(false);
    expect(store.getState().isSolverActive).toBe(true);
    const state = store.getState();
    expect(state.solverHistory[state.solverHistory.length - 1]?.board).toBe(solverBoard);
    expect(store.getState().setupError).toBe("aiInitFailed");
  });

  it("calls abortAIMove when an AI move is active", async () => {
    store.setState({ engineActivity: { kind: "ai-move", runId: 1 } });
    const abortSpy = vi.spyOn(store.getState(), "abortAIMove");
    await store.getState().startFromSetup();
    expect(abortSpy).toHaveBeenCalled();
  });

  it("sets setupError when the selected player has no legal move", async () => {
    const board = createEmptyBoard();
    board[0][0].color = "black";
    board[0][1].color = "black";
    board[0][2].color = "black";
    board[6][4].color = "white";
    board[7][4].color = "black";
    store.setState({ setupBoard: board, setupCurrentPlayer: "white" });

    const started = await store.getState().startFromSetup();

    expect(started).toBe(false);
    expect(store.getState().setupError).toBe("currentPlayerNoMoves");
    expect(store.getState().gameStatus).not.toBe("playing");
  });

  it("sets setupError when neither player has valid moves", async () => {
    // Board with adjacent black and white but no empty cells adjacent that create flanks
    // Fill entire board: black on left half of row 0, white on right half
    const board = createEmptyBoard();
    for (let r = 0; r < 8; r++) {
      for (let c = 0; c < 8; c++) {
        board[r][c] = { color: r < 4 ? "black" : "white" };
      }
    }
    store.setState({ setupBoard: board, setupCurrentPlayer: "black" });

    const started = await store.getState().startFromSetup();
    expect(started).toBe(false);
    expect(store.getState().setupError).toBe("noValidMoves");
    expect(store.getState().gameStatus).not.toBe("playing");
  });
});
