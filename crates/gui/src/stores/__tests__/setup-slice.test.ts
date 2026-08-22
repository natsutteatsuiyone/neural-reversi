import { beforeEach, describe, expect, it } from "vitest";
import { createTestStore, type TestStore } from "./test-helpers";

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
});
