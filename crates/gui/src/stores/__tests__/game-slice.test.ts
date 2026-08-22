import { afterAll, beforeEach, describe, expect, it, vi } from "vitest";
import { createMockAIService } from "@/services/mock-ai-service";
import { createMockSolverService } from "@/services/mock-solver-service";
import { createDeferred, createTestStore, type TestStore } from "./test-helpers";

const consoleErrorSpy = vi.spyOn(console, "error").mockImplementation(() => {});

afterAll(() => {
  consoleErrorSpy.mockRestore();
});

describe("triggerAutomation", () => {
  it("does nothing when gameStatus is not playing", () => {
    const { store } = createTestStore();
    const makeAIMoveSpy = vi.spyOn(store.getState(), "makeAIMove");
    const analyzeBoardSpy = vi.spyOn(store.getState(), "analyzeBoard");

    store.getState().triggerAutomation();

    expect(makeAIMoveSpy).not.toHaveBeenCalled();
    expect(analyzeBoardSpy).not.toHaveBeenCalled();
  });

  it("calls makeAIMove when it is AI's turn", async () => {
    const { store } = createTestStore();
    await store.getState().startGame();
    store.setState({ gameMode: "ai-black" });
    const makeAIMoveSpy = vi.spyOn(store.getState(), "makeAIMove");

    store.getState().triggerAutomation();

    expect(makeAIMoveSpy).toHaveBeenCalled();
  });

  it("calls analyzeBoard when hint mode is enabled", async () => {
    const { store } = createTestStore();
    await store.getState().startGame();
    store.setState({ isHintMode: true });
    const analyzeBoardSpy = vi.spyOn(store.getState(), "analyzeBoard");

    store.getState().triggerAutomation();

    expect(analyzeBoardSpy).toHaveBeenCalled();
  });

  it("does nothing when not AI turn and hint mode is off", async () => {
    const { store } = createTestStore();
    await store.getState().startGame();
    const makeAIMoveSpy = vi.spyOn(store.getState(), "makeAIMove");
    const analyzeBoardSpy = vi.spyOn(store.getState(), "analyzeBoard");

    store.getState().triggerAutomation();

    expect(makeAIMoveSpy).not.toHaveBeenCalled();
    expect(analyzeBoardSpy).not.toHaveBeenCalled();
  });

  it("does nothing while a search or game analysis is active", async () => {
    const { store } = createTestStore();
    await store.getState().startGame();
    store.setState({ gameMode: "ai-black", currentPlayer: "black" });
    const makeAIMoveSpy = vi.spyOn(store.getState(), "makeAIMove");
    const analyzeBoardSpy = vi.spyOn(store.getState(), "analyzeBoard");

    // triggerAutomation blocks while any in-game Engine Search is active.
    for (const kind of ["ai-move", "hint", "game-analysis"] as const) {
      makeAIMoveSpy.mockClear();
      analyzeBoardSpy.mockClear();
      store.setState({ engineActivity: { kind, runId: 1 } });

      store.getState().triggerAutomation();

      expect(makeAIMoveSpy).not.toHaveBeenCalled();
      expect(analyzeBoardSpy).not.toHaveBeenCalled();
    }
  });
});

describe("isAITurn", () => {
  it.each([
    ["ai-black", "black", false, true],
    ["ai-black", "white", false, false],
    ["ai-white", "white", false, true],
    ["ai-white", "black", false, false],
    ["pvp", "black", false, false],
    ["ai-black", "black", true, false],
  ] as const)(
    "returns %s for %s when gameOver=%s",
    async (gameMode, currentPlayer, gameOver, expected) => {
      const { store } = createTestStore();
      await store.getState().startGame();
      store.setState({ gameMode, currentPlayer, gameOver });
      expect(store.getState().isAITurn()).toBe(expected);
    },
  );
});

describe("isValidMove", () => {
  it("requires a playing game and a listed move", async () => {
    const { store } = createTestStore();
    expect(store.getState().isValidMove(2, 3)).toBe(false);

    await store.getState().startGame();
    expect(store.getState().isValidMove(2, 3)).toBe(true);
    expect(store.getState().isValidMove(0, 0)).toBe(false);
  });
});

describe("makeMove", () => {
  let store: TestStore;

  beforeEach(async () => {
    ({ store } = createTestStore());
    await store.getState().startGame();
  });

  it("commits one move transition and clears stale analysis", async () => {
    const move = { row: 2, col: 3, isAI: false };
    const movesBefore = store.getState().validMoves;
    store.setState({
      analyzeResults: new Map([["2,3", {} as never]]),
      gameAnalysisResult: [{ moveIndex: 0 } as never],
    });

    await store.getState().makeMove(move);

    const state = store.getState();
    expect(state.board[2][3].color).toBe("black");
    expect(state.board[3][3].color).toBe("black");
    expect(state.currentPlayer).toBe("white");
    expect(state.moveHistory.currentMoves[0]).toMatchObject({ player: "black", row: 2, col: 3 });
    expect(state.validMoves).not.toEqual(movesBefore);
    expect(state.validMoves.length).toBeGreaterThan(0);
    expect(state.lastMove).toEqual(move);
    expect(state.analyzeResults).toBeNull();
    expect(state.gameAnalysisResult).toBeNull();
  });

  it("a user move supersedes an in-flight hint analysis through the Engine Search", async () => {
    const hintRun = createDeferred<void>();
    const { store: s, services: svc } = createTestStore({
      ai: createMockAIService({
        analyze: vi.fn().mockReturnValue(hintRun.promise), // hint run stays live
      }),
    });
    await s.getState().startGame({
      gameMode: "pvp", // human turn so hint analysis is allowed
      aiLevel: 5,
      aiMode: "level",
      gameTimeLimit: 60,
    });

    s.getState().setHintMode(true);
    for (
      let i = 0;
      i < 10 && (svc.ai.analyze as ReturnType<typeof vi.fn>).mock.calls.length === 0;
      i++
    ) {
      await Promise.resolve();
    }
    expect(s.getState().engineActivity.kind).toBe("hint");

    await s.getState().makeMove({ row: 2, col: 3, isAI: false });

    // Aborted via the canonical hint path: dedupe guard set, backend told to
    // stop, and the Engine Search properly superseded (activity → idle) —
    // the old direct-poke path left the run un-superseded.
    expect(svc.ai.abortSearch).toHaveBeenCalled();
    expect(s.getState().hintAnalysisAbortPending).toBe(true);
    expect(s.getState().engineActivity.kind).toBe("idle");

    hintRun.resolve();
  });

  it("does not move while game analysis is active", async () => {
    store.setState({ engineActivity: { kind: "game-analysis", runId: 1 } });

    await store.getState().makeMove({ row: 2, col: 3, isAI: false });

    expect(store.getState().moveHistory.length).toBe(0);
    expect(store.getState().currentPlayer).toBe("black");
  });

  it("sets gameStatus to finished when game is over", async () => {
    const { createEmptyBoard } = await import("@/domain/game/game-logic");
    // Set up a board where the next move ends the game:
    // black at (0,0), white at (0,1), black plays (0,2)  Eflips (0,1) to black
    // After: only black stones remain, no valid moves for either player
    const board = createEmptyBoard();
    board[0][0].color = "black";
    board[0][1].color = "white";
    store.setState({
      board,
      currentPlayer: "black",
      validMoves: [[0, 2]],
    });

    await store.getState().makeMove({ row: 0, col: 2, isAI: false });
    expect(store.getState().gameOver).toBe(true);
    expect(store.getState().gameStatus).toBe("finished");
  });

  it("sets showPassNotification when opponent must pass", async () => {
    const { createEmptyBoard } = await import("@/domain/game/game-logic");
    // Pre-move board: (0,1)=W, (0,2)=B, (6,4)=W, (7,4)=B
    // Black plays (0,0): flips (0,1) W->B
    // Post-move: (0,0)=B, (0,1)=B, (0,2)=B, (6,4)=W, (7,4)=B
    // White has no valid moves: the only white stone (6,4) is not
    //   reachable through any chain of black stones from an empty cell.
    // Black can play (5,4): direction (1,0) -> (6,4)=W -> (7,4)=B flips (6,4).
    // -> showPassNotification should be set to "white"
    const board = createEmptyBoard();
    board[0][1].color = "white";
    board[0][2].color = "black";
    board[6][4].color = "white";
    board[7][4].color = "black";
    store.setState({
      board,
      currentPlayer: "black",
      validMoves: [[0, 0]],
    });

    await store.getState().makeMove({ row: 0, col: 0, isAI: false });
    const s = store.getState();
    expect(s.board[0][0].color).toBe("black");
    expect(s.showPassNotification).toBe("white");
    expect(s.currentPlayer).toBe("black");
    expect(s.moveHistory.length).toBe(2);
    expect(s.moveHistory.lastMove?.notation).toBe("Pass");
  });

  it("waits for the pass notification before letting AI play again", async () => {
    vi.useFakeTimers();
    try {
      const { createEmptyBoard } = await import("@/domain/game/game-logic");
      store.setState({ gameMode: "ai-black" });

      const board = createEmptyBoard();
      board[0][1].color = "white";
      board[0][2].color = "black";
      board[6][4].color = "white";
      board[7][4].color = "black";
      store.setState({
        board,
        currentPlayer: "black",
        validMoves: [[0, 0]],
      });

      const makeAIMoveSpy = vi.spyOn(store.getState(), "makeAIMove");
      await store.getState().makeMove({ row: 0, col: 0, isAI: true });

      expect(makeAIMoveSpy).not.toHaveBeenCalled();

      vi.advanceTimersByTime(1499);
      await Promise.resolve();
      expect(makeAIMoveSpy).not.toHaveBeenCalled();

      vi.advanceTimersByTime(1);
      await Promise.resolve();
      expect(makeAIMoveSpy).toHaveBeenCalledTimes(1);
    } finally {
      vi.useRealTimers();
    }
  });
});

describe("game over notification", () => {
  let store: TestStore;

  beforeEach(async () => {
    ({ store } = createTestStore());
    await store.getState().startGame();
    const { createEmptyBoard } = await import("@/domain/game/game-logic");
    const { cloneBoard } = await import("@/domain/game/store-helpers");
    // Black plays (0,2), flipping (0,1): only black stones remain, game over.
    const board = createEmptyBoard();
    board[0][0].color = "black";
    board[0][1].color = "white";
    store.setState({
      board,
      historyStartBoard: cloneBoard(board),
      currentPlayer: "black",
      validMoves: [[0, 2]],
    });
  });

  it("signals the notification when a played move ends the game", async () => {
    await store.getState().makeMove({ row: 0, col: 2, isAI: false });

    expect(store.getState().showGameOverNotification).toBe(true);
  });

  it("does not re-signal when history navigation lands on the terminal position", async () => {
    await store.getState().makeMove({ row: 0, col: 2, isAI: false });
    store.getState().hideGameOverNotification();

    store.setState({ gameStatus: "playing" });
    store.getState().undoMove();
    store.getState().redoMove();

    expect(store.getState().gameOver).toBe(true);
    expect(store.getState().showGameOverNotification).toBe(false);
  });

  it("clears a pending notification when a new game starts", async () => {
    await store.getState().makeMove({ row: 0, col: 2, isAI: false });

    await store.getState().startGame();

    expect(store.getState().showGameOverNotification).toBe(false);
  });
});

describe("undoMove", () => {
  let store: TestStore;

  beforeEach(async () => {
    ({ store } = createTestStore());
    await store.getState().startGame();
  });

  it("does nothing when no moves exist", () => {
    const stateBefore = store.getState();
    store.getState().undoMove();
    const stateAfter = store.getState();
    expect(stateAfter.moveHistory.length).toBe(0);
    expect(stateAfter.currentPlayer).toBe(stateBefore.currentPlayer);
  });

  it("does nothing when gameStatus is waiting", async () => {
    await store.getState().makeMove({ row: 2, col: 3, isAI: false });
    store.setState({ gameStatus: "waiting" });
    store.getState().undoMove();
    expect(store.getState().moveHistory.length).toBe(1);
  });

  it("restores the playable state after undo", async () => {
    const boardBeforeMove = JSON.stringify(store.getState().board);
    await store.getState().makeMove({ row: 2, col: 3, isAI: false });
    store.setState({
      gameStatus: "finished",
      gameOver: true,
      analyzeResults: new Map([["2,3", {} as never]]),
    });

    store.getState().undoMove();

    const state = store.getState();
    expect(state.moveHistory.length).toBe(0);
    expect(state.gameStatus).toBe("playing");
    expect(state.gameOver).toBe(false);
    expect(state.currentPlayer).toBe("black");
    expect(JSON.stringify(state.board)).toBe(boardBeforeMove);
    expect(state.analyzeResults).toBeNull();
  });

  it("does not re-apply a forced pass when undoing a pass move", async () => {
    const { createEmptyBoard } = await import("@/domain/game/game-logic");
    const { cloneBoard } = await import("@/domain/game/store-helpers");
    const board = createEmptyBoard();
    board[0][1].color = "white";
    board[0][2].color = "black";
    board[6][4].color = "white";
    board[7][4].color = "black";
    store.setState({
      board,
      historyStartBoard: cloneBoard(board),
      historyStartPlayer: "black",
      currentPlayer: "black",
      validMoves: [[0, 0]],
    });

    await store.getState().makeMove({ row: 0, col: 0, isAI: false });
    expect(store.getState().moveHistory.length).toBe(2);

    store.getState().undoMove();

    const s = store.getState();
    expect(s.moveHistory.length).toBe(1);
    expect(s.currentPlayer).toBe("white");
    expect(s.showPassNotification).toBeNull();
    expect(s.paused).toBe(false);
  });

  it("does nothing while game analysis is active", async () => {
    await store.getState().makeMove({ row: 2, col: 3, isAI: false });
    store.setState({
      engineActivity: { kind: "game-analysis", runId: 1 },
    });

    store.getState().undoMove();

    expect(store.getState().moveHistory.length).toBe(1);
  });

  it("does nothing while an AI-move search is active", async () => {
    await store.getState().makeMove({ row: 2, col: 3, isAI: false });
    store.setState({
      engineActivity: { kind: "ai-move", runId: 1 },
    });

    store.getState().undoMove();

    expect(store.getState().moveHistory.length).toBe(1);
  });
});

describe("redoMove", () => {
  let store: TestStore;

  beforeEach(async () => {
    ({ store } = createTestStore());
    await store.getState().startGame();
  });

  it("does nothing when no redo available", () => {
    const stateBefore = store.getState();
    store.getState().redoMove();
    expect(store.getState().moveHistory.length).toBe(stateBefore.moveHistory.length);
  });

  it("restores the playable state after redo", async () => {
    await store.getState().makeMove({ row: 2, col: 3, isAI: false });
    const colorsAfterMove = store.getState().board.map((row) => row.map((cell) => cell.color));
    const playerAfterMove = store.getState().currentPlayer;
    store.getState().undoMove();
    store.setState({
      gameStatus: "finished",
      analyzeResults: new Map([["2,3", {} as never]]),
    });

    store.getState().redoMove();

    const state = store.getState();
    expect(state.currentPlayer).toBe(playerAfterMove);
    expect(state.board.map((row) => row.map((cell) => cell.color))).toEqual(colorsAfterMove);
    expect(state.gameStatus).toBe("playing");
    expect(state.analyzeResults).toBeNull();
  });

  it("detects game-over condition after redo", async () => {
    const { createEmptyBoard } = await import("@/domain/game/game-logic");
    const { cloneBoard } = await import("@/domain/game/store-helpers");
    // Set up a board where black's move ends the game
    const board = createEmptyBoard();
    board[0][0].color = "black";
    board[0][1].color = "white";
    store.setState({
      board,
      historyStartBoard: cloneBoard(board),
      currentPlayer: "black",
      validMoves: [[0, 2]],
    });

    // Make the game-ending move, then undo, then redo
    await store.getState().makeMove({ row: 0, col: 2, isAI: false });
    expect(store.getState().gameOver).toBe(true);

    // undoMove requires gameStatus to be "playing"
    store.setState({ gameStatus: "playing" });
    store.getState().undoMove();
    expect(store.getState().gameOver).toBe(false);

    store.getState().redoMove();
    expect(store.getState().gameOver).toBe(true);
  });

  it("does nothing when gameStatus is waiting", async () => {
    await store.getState().makeMove({ row: 2, col: 3, isAI: false });
    store.getState().undoMove();
    store.setState({ gameStatus: "waiting" });
    store.getState().redoMove();
    expect(store.getState().moveHistory.length).toBe(0);
  });

  it("does nothing while game analysis is active", async () => {
    await store.getState().makeMove({ row: 2, col: 3, isAI: false });
    store.getState().undoMove();
    store.setState({
      engineActivity: { kind: "game-analysis", runId: 1 },
    });

    store.getState().redoMove();

    expect(store.getState().moveHistory.length).toBe(0);
  });
});

describe("goToMove", () => {
  let store: TestStore;

  beforeEach(async () => {
    ({ store } = createTestStore());
    await store.getState().startGame();
    store.setState({ gameMode: "pvp" });
  });

  it("does nothing while game analysis is active", async () => {
    await store.getState().makeMove({ row: 2, col: 3, isAI: false });
    store.setState({
      engineActivity: { kind: "game-analysis", runId: 1 },
    });

    store.getState().goToMove(0);

    expect(store.getState().moveHistory.length).toBe(1);
    expect(store.getState().currentPlayer).toBe("white");
  });
});

describe("startGame", () => {
  it("starts a playable game with the supplied settings", async () => {
    const { store } = createTestStore();
    const started = await store.getState().startGame({
      gameMode: "pvp",
      aiLevel: 12,
      aiMode: "level",
      gameTimeLimit: 180,
    });

    const state = store.getState();
    expect(started).toBe(true);
    expect(state).toMatchObject({
      gameStatus: "playing",
      currentPlayer: "black",
      gameOver: false,
      gameMode: "pvp",
      aiLevel: 12,
      aiMode: "level",
      gameTimeLimit: 180,
      aiRemainingTime: 180000,
    });
    expect(state.moveHistory.length).toBe(0);
    expect(state.validMoves).toEqual(
      expect.arrayContaining([
        [2, 3],
        [3, 2],
        [4, 5],
        [5, 4],
      ]),
    );
  });
});

describe("startInitialGame", () => {
  it("starts paused without auto-playing when the AI moves first (ai-black)", async () => {
    const { store } = createTestStore();
    store.setState({ gameMode: "ai-black" });
    const makeAIMoveSpy = vi.spyOn(store.getState(), "makeAIMove");

    const started = await store.getState().startInitialGame();

    expect(started).toBe(true);
    expect(store.getState().paused).toBe(true);
    expect(makeAIMoveSpy).not.toHaveBeenCalled();
  });

  it("resolves false instead of rejecting when the replacement fails unexpectedly", async () => {
    const { store } = createTestStore({
      solver: createMockSolverService({
        abort: vi.fn().mockRejectedValue(new Error("abort IPC failed")),
      }),
    });

    await expect(store.getState().startInitialGame()).resolves.toBe(false);
  });

  it("coalesces concurrent launch auto-start calls", async () => {
    const { store, services } = createTestStore();

    const firstStart = store.getState().startInitialGame();
    const secondStart = store.getState().startInitialGame();

    expect(secondStart).toBe(firstStart);
    await expect(Promise.all([firstStart, secondStart])).resolves.toEqual([true, true]);
    expect(services.ai.initialize).toHaveBeenCalledTimes(1);
    expect(services.ai.resizeTT).toHaveBeenCalledTimes(1);
  });
});

describe("resumeAI", () => {
  it("clears pause and immediately triggers an AI move on an AI turn", async () => {
    const { store } = createTestStore();
    await store.getState().startGame();
    const makeAIMoveSpy = vi.spyOn(store.getState(), "makeAIMove").mockResolvedValue(undefined);
    store.setState({ paused: true, gameMode: "ai-black", currentPlayer: "black" });

    store.getState().resumeAI();

    expect(store.getState().paused).toBe(false);
    expect(makeAIMoveSpy).toHaveBeenCalledTimes(1);
  });
});
