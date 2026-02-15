import { beforeEach, describe, expect, it, vi } from "vitest";
import { create } from "zustand";
import { createGameSlice } from "@/stores/slices/game-slice";
import { createAISlice } from "@/stores/slices/ai-slice";
import { createUISlice } from "@/stores/slices/ui-slice";
import { createSettingsSlice } from "@/stores/slices/settings-slice";
import { createSetupSlice } from "@/stores/slices/setup-slice";
import { createEmptyBoard } from "@/lib/game-logic";
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

describe("resetSetup", () => {
  it("has correct initial state", () => {
    const store = createTestStore();
    const s = store.getState();
    expect(s.setupCurrentPlayer).toBe("black");
    expect(s.setupTab).toBe("manual");
    expect(s.transcriptInput).toBe("");
    expect(s.boardStringInput).toBe("");
    expect(s.setupError).toBeNull();
  });

  it("resets modified state back to initial", () => {
    const store = createTestStore();
    store.setState({
      setupTab: "transcript",
      setupCurrentPlayer: "white",
      transcriptInput: "F5",
      boardStringInput: "X".repeat(64),
      setupError: "someError",
    });

    store.getState().resetSetup();
    const s = store.getState();
    expect(s.setupCurrentPlayer).toBe("black");
    expect(s.setupTab).toBe("manual");
    expect(s.transcriptInput).toBe("");
    expect(s.boardStringInput).toBe("");
    expect(s.setupError).toBeNull();
  });
});

describe("setSetupCurrentPlayer", () => {
  it("changes player", () => {
    const store = createTestStore();
    store.getState().setSetupCurrentPlayer("white");
    expect(store.getState().setupCurrentPlayer).toBe("white");
  });

  it("clears setupError", () => {
    const store = createTestStore();
    store.setState({ setupError: "someError" });
    store.getState().setSetupCurrentPlayer("white");
    expect(store.getState().setupError).toBeNull();
  });
});

describe("setSetupBoard", () => {
  it("sets board directly", () => {
    const store = createTestStore();
    const board = createEmptyBoard();
    board[0][0] = { color: "black" };
    store.getState().setSetupBoard(board);
    expect(store.getState().setupBoard[0][0].color).toBe("black");
  });
});

describe("clearSetupBoard", () => {
  it("sets all cells to null", () => {
    const store = createTestStore();
    // Initial board has stones at center
    store.getState().clearSetupBoard();
    const board = store.getState().setupBoard;
    for (let r = 0; r < 8; r++) {
      for (let c = 0; c < 8; c++) {
        expect(board[r][c].color).toBeNull();
      }
    }
  });

  it("clears setupError", () => {
    const store = createTestStore();
    store.setState({ setupError: "someError" });
    store.getState().clearSetupBoard();
    expect(store.getState().setupError).toBeNull();
  });
});

describe("resetSetupToInitial", () => {
  it("restores initial Reversi position", () => {
    const store = createTestStore();
    store.getState().clearSetupBoard();
    store.getState().resetSetupToInitial();
    const board = store.getState().setupBoard;
    expect(board[3][3].color).toBe("white");
    expect(board[3][4].color).toBe("black");
    expect(board[4][3].color).toBe("black");
    expect(board[4][4].color).toBe("white");
    expect(board[0][0].color).toBeNull();
  });

  it("resets currentPlayer to black", () => {
    const store = createTestStore();
    store.setState({ setupCurrentPlayer: "white" });
    store.getState().resetSetupToInitial();
    expect(store.getState().setupCurrentPlayer).toBe("black");
  });
});

describe("setSetupCellColor", () => {
  it("cycles null to black", () => {
    const store = createTestStore();
    store.getState().clearSetupBoard();
    store.getState().setSetupCellColor(0, 0);
    expect(store.getState().setupBoard[0][0].color).toBe("black");
  });

  it("cycles black to white", () => {
    const store = createTestStore();
    store.getState().clearSetupBoard();
    store.getState().setSetupCellColor(0, 0); // null -> black
    store.getState().setSetupCellColor(0, 0); // black -> white
    expect(store.getState().setupBoard[0][0].color).toBe("white");
  });

  it("cycles white to null", () => {
    const store = createTestStore();
    store.getState().clearSetupBoard();
    store.getState().setSetupCellColor(0, 0); // null -> black
    store.getState().setSetupCellColor(0, 0); // black -> white
    store.getState().setSetupCellColor(0, 0); // white -> null
    expect(store.getState().setupBoard[0][0].color).toBeNull();
  });
});

describe("setSetupTab", () => {
  it("switches to manual tab and uses setupBoard as-is", () => {
    const store = createTestStore();
    const boardBefore = JSON.stringify(store.getState().setupBoard);
    store.getState().setSetupTab("manual");
    expect(store.getState().setupTab).toBe("manual");
    expect(JSON.stringify(store.getState().setupBoard)).toBe(boardBefore);
    expect(store.getState().setupError).toBeNull();
  });

  it("switches to transcript tab and resolves board from valid transcript", () => {
    const store = createTestStore();
    store.setState({ transcriptInput: "F5D6" });
    store.getState().setSetupTab("transcript");
    expect(store.getState().setupTab).toBe("transcript");
    expect(store.getState().setupError).toBeNull();
    // F5 = row 4 col 5, D6 = row 5 col 3
    expect(store.getState().setupBoard[4][5].color).toBe("black");
    expect(store.getState().setupBoard[5][3].color).toBe("white");
  });

  it("switches to transcript tab and sets error on invalid transcript", () => {
    const store = createTestStore();
    store.setState({ transcriptInput: "Z" });
    store.getState().setSetupTab("transcript");
    expect(store.getState().setupTab).toBe("transcript");
    expect(store.getState().setupError).not.toBeNull();
  });

  it("switches to boardString tab and resolves board from valid string", () => {
    const store = createTestStore();
    const boardStr = "X" + "-".repeat(63);
    store.setState({ boardStringInput: boardStr });
    store.getState().setSetupTab("boardString");
    expect(store.getState().setupTab).toBe("boardString");
    expect(store.getState().setupError).toBeNull();
    expect(store.getState().setupBoard[0][0].color).toBe("black");
  });

  it("switches to boardString tab and sets error on invalid string", () => {
    const store = createTestStore();
    store.setState({ boardStringInput: "invalid" });
    store.getState().setSetupTab("boardString");
    expect(store.getState().setupTab).toBe("boardString");
    expect(store.getState().setupError).not.toBeNull();
  });
});

describe("setTranscriptInput", () => {
  it("updates board and player on valid input", () => {
    const store = createTestStore();
    store.getState().setTranscriptInput("F5D6");
    const s = store.getState();
    expect(s.transcriptInput).toBe("F5D6");
    expect(s.setupError).toBeNull();
    expect(s.setupBoard[4][5].color).toBe("black");
    expect(s.setupCurrentPlayer).toBe("black");
  });

  it("sets setupError on invalid input", () => {
    const store = createTestStore();
    store.getState().setTranscriptInput("Z");
    const s = store.getState();
    expect(s.transcriptInput).toBe("Z");
    expect(s.setupError).not.toBeNull();
  });

  it("resets to initial board on empty string", () => {
    const store = createTestStore();
    store.getState().setTranscriptInput("F5D6");
    store.getState().setTranscriptInput("");
    const s = store.getState();
    expect(s.transcriptInput).toBe("");
    expect(s.setupError).toBeNull();
    // Initial board: center stones only
    expect(s.setupBoard[3][3].color).toBe("white");
    expect(s.setupBoard[3][4].color).toBe("black");
  });
});

describe("setBoardStringInput", () => {
  it("updates board on valid input", () => {
    const store = createTestStore();
    const boardStr = "-".repeat(27) + "OX------XO" + "-".repeat(27);
    store.getState().setBoardStringInput(boardStr);
    const s = store.getState();
    expect(s.boardStringInput).toBe(boardStr);
    expect(s.setupError).toBeNull();
    expect(s.setupBoard[3][3].color).toBe("white");
    expect(s.setupBoard[3][4].color).toBe("black");
  });

  it("sets setupError on invalid input", () => {
    const store = createTestStore();
    store.getState().setBoardStringInput("too-short");
    expect(store.getState().setupError).not.toBeNull();
  });

  it("clears previous error on valid input", () => {
    const store = createTestStore();
    store.getState().setBoardStringInput("too-short");
    expect(store.getState().setupError).not.toBeNull();
    store.getState().setBoardStringInput("-".repeat(64));
    expect(store.getState().setupError).toBeNull();
  });
});

describe("startFromSetup", () => {
  let store: TestStore;

  beforeEach(() => {
    store = createTestStore();
  });

  it("starts game from manual tab", async () => {
    // Default setupBoard is initial position — valid
    await store.getState().startFromSetup();
    const s = store.getState();
    expect(s.gameStatus).toBe("playing");
    expect(s.gameOver).toBe(false);
    expect(s.setupError).toBeNull();
    expect(s.moves).toHaveLength(0);
  });

  it("starts game from transcript tab", async () => {
    store.getState().setTranscriptInput("F5D6");
    store.setState({ setupTab: "transcript" });
    await store.getState().startFromSetup();
    const s = store.getState();
    expect(s.gameStatus).toBe("playing");
    expect(s.board[4][5].color).toBe("black");
    expect(s.setupError).toBeNull();
  });

  it("starts game from boardString tab", async () => {
    const boardStr = "-".repeat(27) + "OX------XO" + "-".repeat(27);
    store.getState().setBoardStringInput(boardStr);
    store.setState({ setupTab: "boardString" });
    await store.getState().startFromSetup();
    const s = store.getState();
    expect(s.gameStatus).toBe("playing");
    expect(s.setupError).toBeNull();
  });

  it("sets setupError when board validation fails", async () => {
    // Board with only black stones — validateBoard returns "needBothColors"
    const board = createEmptyBoard();
    board[0][0] = { color: "black" };
    store.setState({ setupBoard: board });
    await store.getState().startFromSetup();
    expect(store.getState().setupError).toBe("needBothColors");
    expect(store.getState().gameStatus).not.toBe("playing");
  });

  it("sets setupError to aiInitFailed when AI init fails", async () => {
    const { initializeAI } = await import("@/lib/ai");
    (initializeAI as ReturnType<typeof vi.fn>).mockRejectedValueOnce(new Error("init failed"));

    await store.getState().startFromSetup();
    expect(store.getState().setupError).toBe("aiInitFailed");
  });

  it("computes validMoves after game start", async () => {
    await store.getState().startFromSetup();
    const s = store.getState();
    expect(s.validMoves).toHaveLength(4);
    expect(s.validMoves).toContainEqual([2, 3]);
  });

  it("calls abortAIMove when AI is thinking", async () => {
    store.setState({ isAIThinking: true });
    const abortSpy = vi.spyOn(store.getState(), "abortAIMove");
    await store.getState().startFromSetup();
    expect(abortSpy).toHaveBeenCalled();
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

    await store.getState().startFromSetup();
    expect(store.getState().setupError).toBe("noValidMoves");
    expect(store.getState().gameStatus).not.toBe("playing");
  });
});
