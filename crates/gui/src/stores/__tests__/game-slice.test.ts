import { describe, expect, it, vi, beforeEach } from "vitest";
import { create } from "zustand";
import { createGameSlice, triggerAutomation } from "@/stores/slices/game-slice";
import { createAISlice } from "@/stores/slices/ai-slice";
import { createUISlice } from "@/stores/slices/ui-slice";
import { createSettingsSlice } from "@/stores/slices/settings-slice";
import { createSetupSlice } from "@/stores/slices/setup-slice";
import type { ReversiState } from "@/stores/slices/types";

vi.mock("@/lib/ai", () => ({
  initializeAI: vi.fn().mockResolvedValue(undefined),
  abortAISearch: vi.fn().mockResolvedValue(undefined),
  getAIMove: vi.fn().mockResolvedValue(null),
  analyze: vi.fn().mockResolvedValue(undefined),
}));

vi.mock("@/lib/settings-store", () => ({
  saveSetting: vi.fn(),
  loadSettings: vi.fn().mockResolvedValue({}),
}));

type TestStore = ReturnType<typeof createTestStore>;

function createTestStore() {
  return create<ReversiState>()((...a) => ({
    ...createGameSlice(...a),
    ...createAISlice(...a),
    ...createUISlice(...a),
    ...createSettingsSlice(...a),
    ...createSetupSlice(...a),
  }));
}

describe("triggerAutomation", () => {
  it("does nothing when gameStatus is not playing", () => {
    const store = createTestStore();
    const makeAIMoveSpy = vi.spyOn(store.getState(), "makeAIMove");
    const analyzeBoardSpy = vi.spyOn(store.getState(), "analyzeBoard");

    triggerAutomation(store.getState);

    expect(makeAIMoveSpy).not.toHaveBeenCalled();
    expect(analyzeBoardSpy).not.toHaveBeenCalled();
  });

  it("calls makeAIMove when it is AI's turn", async () => {
    const store = createTestStore();
    await store.getState().startGame();
    store.setState({ gameMode: "ai-black" });
    const makeAIMoveSpy = vi.spyOn(store.getState(), "makeAIMove");

    triggerAutomation(store.getState);

    expect(makeAIMoveSpy).toHaveBeenCalled();
  });

  it("calls analyzeBoard when hint mode is enabled", async () => {
    const store = createTestStore();
    await store.getState().startGame();
    store.setState({ isHintMode: true });
    const analyzeBoardSpy = vi.spyOn(store.getState(), "analyzeBoard");

    triggerAutomation(store.getState);

    expect(analyzeBoardSpy).toHaveBeenCalled();
  });

  it("does nothing when not AI turn and hint mode is off", async () => {
    const store = createTestStore();
    await store.getState().startGame();
    const makeAIMoveSpy = vi.spyOn(store.getState(), "makeAIMove");
    const analyzeBoardSpy = vi.spyOn(store.getState(), "analyzeBoard");

    triggerAutomation(store.getState);

    expect(makeAIMoveSpy).not.toHaveBeenCalled();
    expect(analyzeBoardSpy).not.toHaveBeenCalled();
  });
});

describe("getScores", () => {
  it("returns { black: 2, white: 2 } for initial board", () => {
    const store = createTestStore();
    expect(store.getState().getScores()).toEqual({ black: 2, white: 2 });
  });
});

describe("isAITurn", () => {
  let store: TestStore;

  beforeEach(async () => {
    store = createTestStore();
    await store.getState().startGame();
  });

  it("returns true for ai-black mode when black's turn", () => {
    store.setState({ gameMode: "ai-black", currentPlayer: "black" });
    expect(store.getState().isAITurn()).toBe(true);
  });

  it("returns false for ai-black mode when white's turn", () => {
    store.setState({ gameMode: "ai-black", currentPlayer: "white" });
    expect(store.getState().isAITurn()).toBe(false);
  });

  it("returns true for ai-white mode when white's turn", () => {
    store.setState({ gameMode: "ai-white", currentPlayer: "white" });
    expect(store.getState().isAITurn()).toBe(true);
  });

  it("returns false for ai-white mode when black's turn", () => {
    store.setState({ gameMode: "ai-white", currentPlayer: "black" });
    expect(store.getState().isAITurn()).toBe(false);
  });

  it("returns false when game is over (ai-black, black's turn)", () => {
    store.setState({ gameMode: "ai-black", currentPlayer: "black", gameOver: true });
    expect(store.getState().isAITurn()).toBe(false);
  });

  it("returns false when game is over (ai-white, white's turn)", () => {
    store.setState({ gameMode: "ai-white", currentPlayer: "white", gameOver: true });
    expect(store.getState().isAITurn()).toBe(false);
  });
});

describe("isValidMove", () => {
  it("returns false when gameStatus is waiting", () => {
    const store = createTestStore();
    expect(store.getState().isValidMove(2, 3)).toBe(false);
  });

  it("returns true for a valid move coordinate during play", async () => {
    const store = createTestStore();
    await store.getState().startGame();
    // Initial board: black can play at (2,3), (3,2), (4,5), (5,4)
    expect(store.getState().isValidMove(2, 3)).toBe(true);
  });

  it("returns false for an invalid move coordinate during play", async () => {
    const store = createTestStore();
    await store.getState().startGame();
    expect(store.getState().isValidMove(0, 0)).toBe(false);
  });
});

describe("makeMove", () => {
  let store: TestStore;

  beforeEach(async () => {
    store = createTestStore();
    await store.getState().startGame();
  });

  it("places stone and updates board", async () => {
    // Black plays d3 (row=2, col=3)
    await store.getState().makeMove({ row: 2, col: 3, isAI: false });
    const s = store.getState();
    expect(s.board[2][3].color).toBe("black");
    // (3,3) was white and should be flipped to black
    expect(s.board[3][3].color).toBe("black");
  });

  it("switches currentPlayer after move", async () => {
    expect(store.getState().currentPlayer).toBe("black");
    await store.getState().makeMove({ row: 2, col: 3, isAI: false });
    expect(store.getState().currentPlayer).toBe("white");
  });

  it("appends record to moves and allMoves", async () => {
    await store.getState().makeMove({ row: 2, col: 3, isAI: false });
    const s = store.getState();
    expect(s.moves).toHaveLength(1);
    expect(s.allMoves).toHaveLength(1);
    expect(s.moves[0].player).toBe("black");
    expect(s.moves[0].row).toBe(2);
    expect(s.moves[0].col).toBe(3);
  });

  it("recalculates validMoves for next player", async () => {
    const movesBefore = store.getState().validMoves;
    await store.getState().makeMove({ row: 2, col: 3, isAI: false });
    const movesAfter = store.getState().validMoves;
    // Valid moves change after a move is made
    expect(movesAfter).not.toEqual(movesBefore);
    // White should have valid moves on the updated board
    expect(movesAfter.length).toBeGreaterThan(0);
  });

  it("sets lastMove to the played move", async () => {
    const move = { row: 2, col: 3, isAI: false };
    await store.getState().makeMove(move);
    const s = store.getState();
    expect(s.lastMove).toEqual(move);
  });

  it("clears analyzeResults", async () => {
    store.setState({ analyzeResults: new Map([["2,3", {} as never]]) });
    await store.getState().makeMove({ row: 2, col: 3, isAI: false });
    expect(store.getState().analyzeResults).toBeNull();
  });

  it("aborts analysis when a user move is made during analysis", async () => {
    const { abortAISearch } = await import("@/lib/ai");
    store.setState({ isAnalyzing: true });
    await store.getState().makeMove({ row: 2, col: 3, isAI: false });
    expect(store.getState().isAnalyzing).toBe(false);
    expect(abortAISearch).toHaveBeenCalled();
  });

  it("sets gameStatus to finished when game is over", async () => {
    const { createEmptyBoard } = await import("@/lib/game-logic");
    // Set up a board where the next move ends the game:
    // black at (0,0), white at (0,1), black plays (0,2) — flips (0,1) to black
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
    const { createEmptyBoard } = await import("@/lib/game-logic");
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
  });
});

describe("makePass", () => {
  let store: TestStore;

  beforeEach(async () => {
    store = createTestStore();
    await store.getState().startGame();
  });

  it("switches currentPlayer", () => {
    expect(store.getState().currentPlayer).toBe("black");
    store.getState().makePass();
    expect(store.getState().currentPlayer).toBe("white");
  });

  it("does not change the board content", () => {
    const boardBefore = JSON.stringify(store.getState().board);
    store.getState().makePass();
    const boardAfter = JSON.stringify(store.getState().board);
    expect(boardAfter).toBe(boardBefore);
  });

  it("appends pass record to moves and allMoves", () => {
    store.getState().makePass();
    const s = store.getState();
    expect(s.moves).toHaveLength(1);
    expect(s.allMoves).toHaveLength(1);
    expect(s.moves[0].row).toBe(-1);
    expect(s.moves[0].col).toBe(-1);
    expect(s.moves[0].notation).toBe("Pass");
  });

  it("sets isPass to true", () => {
    expect(store.getState().isPass).toBe(false);
    store.getState().makePass();
    expect(store.getState().isPass).toBe(true);
  });
});

describe("undoMove", () => {
  let store: TestStore;

  beforeEach(async () => {
    store = createTestStore();
    await store.getState().startGame();
  });

  it("does nothing when no moves exist", () => {
    const stateBefore = store.getState();
    store.getState().undoMove();
    const stateAfter = store.getState();
    expect(stateAfter.moves).toHaveLength(0);
    expect(stateAfter.currentPlayer).toBe(stateBefore.currentPlayer);
  });

  it("does nothing when gameStatus is not playing", async () => {
    await store.getState().makeMove({ row: 2, col: 3, isAI: false });
    store.setState({ gameStatus: "waiting" });
    store.getState().undoMove();
    expect(store.getState().moves).toHaveLength(1);
  });

  it("restores board and currentPlayer after undo", async () => {
    const boardBeforeMove = JSON.stringify(store.getState().board);
    await store.getState().makeMove({ row: 2, col: 3, isAI: false });
    expect(store.getState().currentPlayer).toBe("white");

    store.getState().undoMove();
    expect(store.getState().currentPlayer).toBe("black");
    expect(JSON.stringify(store.getState().board)).toBe(boardBeforeMove);
  });

  it("resets gameOver to false", async () => {
    await store.getState().makeMove({ row: 2, col: 3, isAI: false });
    store.setState({ gameOver: true });
    store.getState().undoMove();
    expect(store.getState().gameOver).toBe(false);
  });

  it("clears analyzeResults", async () => {
    await store.getState().makeMove({ row: 2, col: 3, isAI: false });
    store.setState({ analyzeResults: new Map([["2,3", {} as never]]) });
    store.getState().undoMove();
    expect(store.getState().analyzeResults).toBeNull();
  });
});

describe("redoMove", () => {
  let store: TestStore;

  beforeEach(async () => {
    store = createTestStore();
    await store.getState().startGame();
  });

  it("does nothing when moves equals allMoves", () => {
    const stateBefore = store.getState();
    store.getState().redoMove();
    expect(store.getState().moves.length).toBe(stateBefore.moves.length);
  });

  it("advances board and currentPlayer after redo", async () => {
    await store.getState().makeMove({ row: 2, col: 3, isAI: false });
    // Extract only colors so we ignore transient flags like isNew
    const colorsAfterMove = store
      .getState()
      .board.map((row) => row.map((cell) => cell.color));
    const playerAfterMove = store.getState().currentPlayer;

    store.getState().undoMove();
    expect(store.getState().currentPlayer).toBe("black");

    store.getState().redoMove();
    const colorsAfterRedo = store
      .getState()
      .board.map((row) => row.map((cell) => cell.color));
    expect(store.getState().currentPlayer).toBe(playerAfterMove);
    expect(colorsAfterRedo).toEqual(colorsAfterMove);
  });

  it("detects game-over condition after redo", async () => {
    const { createEmptyBoard } = await import("@/lib/game-logic");
    const { cloneBoard } = await import("@/lib/store-helpers");
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

  it("does nothing when gameStatus is not playing", async () => {
    await store.getState().makeMove({ row: 2, col: 3, isAI: false });
    store.getState().undoMove();
    store.setState({ gameStatus: "finished" });
    store.getState().redoMove();
    // Should still be undone state
    expect(store.getState().moves).toHaveLength(0);
  });

  it("clears analyzeResults", async () => {
    await store.getState().makeMove({ row: 2, col: 3, isAI: false });
    store.getState().undoMove();
    store.setState({ analyzeResults: new Map([["2,3", {} as never]]) });
    store.getState().redoMove();
    expect(store.getState().analyzeResults).toBeNull();
  });
});

describe("resetGame", () => {
  it("resets to initial board state", async () => {
    const store = createTestStore();
    await store.getState().startGame();
    await store.getState().makeMove({ row: 2, col: 3, isAI: false });

    await store.getState().resetGame();
    const s = store.getState();
    expect(s.gameStatus).toBe("waiting");
    expect(s.moves).toHaveLength(0);
    expect(s.allMoves).toHaveLength(0);
    expect(s.currentPlayer).toBe("black");
    expect(s.gameOver).toBe(false);
    expect(s.lastMove).toBeNull();
    expect(s.validMoves).toHaveLength(0);
    expect(s.isPass).toBe(false);
  });

  it("clears AI-related state", async () => {
    const store = createTestStore();
    await store.getState().startGame();
    store.setState({
      isAIThinking: true,
      analyzeResults: new Map([["2,3", {} as never]]),
      aiMoveProgress: { bestMove: "d3" } as never,
    });

    await store.getState().resetGame();
    const s = store.getState();
    expect(s.isAIThinking).toBe(false);
    expect(s.isAnalyzing).toBe(false);
    expect(s.analyzeResults).toBeNull();
    expect(s.aiMoveProgress).toBeNull();
    expect(s.lastAIMove).toBeNull();
  });

  it("calls abortAIMove when AI is thinking", async () => {
    const store = createTestStore();
    await store.getState().startGame();
    store.setState({ isAIThinking: true });
    const abortSpy = vi.spyOn(store.getState(), "abortAIMove");

    await store.getState().resetGame();
    expect(abortSpy).toHaveBeenCalled();
  });
});

describe("startGame", () => {
  it("sets gameStatus to playing on success", async () => {
    const store = createTestStore();
    await store.getState().startGame();

    const s = store.getState();
    expect(s.gameStatus).toBe("playing");
    expect(s.currentPlayer).toBe("black");
    expect(s.gameOver).toBe(false);
    expect(s.moves).toHaveLength(0);
  });

  it("computes validMoves for initial board", async () => {
    const store = createTestStore();
    await store.getState().startGame();

    const s = store.getState();
    expect(s.validMoves).toHaveLength(4);
    expect(s.validMoves).toContainEqual([2, 3]);
    expect(s.validMoves).toContainEqual([3, 2]);
    expect(s.validMoves).toContainEqual([4, 5]);
    expect(s.validMoves).toContainEqual([5, 4]);
  });

  it("does not change gameStatus when initializeAI fails", async () => {
    const { initializeAI } = await import("@/lib/ai");
    (initializeAI as ReturnType<typeof vi.fn>).mockRejectedValueOnce(new Error("init failed"));

    const store = createTestStore();
    await store.getState().startGame();

    expect(store.getState().gameStatus).toBe("waiting");
  });
});

describe("setGameStatus", () => {
  it("sets gameStatus to the given value", () => {
    const store = createTestStore();
    store.getState().setGameStatus("playing");
    expect(store.getState().gameStatus).toBe("playing");

    store.getState().setGameStatus("finished");
    expect(store.getState().gameStatus).toBe("finished");

    store.getState().setGameStatus("waiting");
    expect(store.getState().gameStatus).toBe("waiting");
  });
});
