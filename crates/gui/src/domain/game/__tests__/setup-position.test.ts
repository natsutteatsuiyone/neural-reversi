import { describe, expect, it } from "vitest";
import { createEmptyBoard, initializeBoard } from "@/domain/game/game-logic";
import { resolveSetupPosition, resolveValidSetupPosition } from "@/domain/game/setup-position";

describe("resolveSetupPosition", () => {
  it("resolves transcript input into board and player", () => {
    const resolved = resolveSetupPosition({
      source: "transcript",
      board: initializeBoard(),
      currentPlayer: "black",
      transcriptInput: "F5",
      boardStringInput: "",
    });

    expect(resolved.ok).toBe(true);
    if (!resolved.ok) return;
    expect(resolved.board[4][5].color).toBe("black");
    expect(resolved.currentPlayer).toBe("white");
  });

  it("keeps the selected player for board strings", () => {
    const boardString = "-".repeat(27) + "OX------XO" + "-".repeat(27);
    const resolved = resolveSetupPosition({
      source: "boardString",
      board: initializeBoard(),
      currentPlayer: "white",
      transcriptInput: "",
      boardStringInput: boardString,
    });

    expect(resolved.ok).toBe(true);
    if (!resolved.ok) return;
    expect(resolved.board[3][3].color).toBe("white");
    expect(resolved.currentPlayer).toBe("white");
  });

  it("reports parse errors from the active source", () => {
    const resolved = resolveSetupPosition({
      source: "boardString",
      board: initializeBoard(),
      currentPlayer: "black",
      transcriptInput: "",
      boardStringInput: "too-short",
    });

    expect(resolved).toEqual({ ok: false, error: "invalidBoardLength" });
  });
});

describe("resolveValidSetupPosition", () => {
  it("accepts a playable manual setup", () => {
    const resolved = resolveValidSetupPosition({
      source: "manual",
      board: initializeBoard(),
      currentPlayer: "black",
      transcriptInput: "",
      boardStringInput: "",
    });

    expect(resolved.ok).toBe(true);
  });

  it("returns validation errors after parsing", () => {
    const resolved = resolveValidSetupPosition({
      source: "manual",
      board: createEmptyBoard(),
      currentPlayer: "black",
      transcriptInput: "",
      boardStringInput: "",
    });

    expect(resolved).toEqual({ ok: false, error: "needBothColors" });
  });
});
