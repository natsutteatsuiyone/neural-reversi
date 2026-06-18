import { describe, expect, it } from "vitest";
import { serializeBoardForAI } from "@/services/board-serialization";
import { boardToString } from "@/domain/game/board-parser";
import { createEmptyBoard } from "@/domain/game/game-logic";
import type { Board, Player } from "@/domain/game/types";

function setupBoard(stones: [number, number, Player | null][]): Board {
  const board = createEmptyBoard();
  for (const [r, c, color] of stones) {
    board[r][c].color = color;
  }
  return board;
}

describe("serializeBoardForAI", () => {
  it("maps an empty board to 64 dashes for either player", () => {
    const board = createEmptyBoard();
    expect(serializeBoardForAI(board, "black")).toBe("-".repeat(64));
    expect(serializeBoardForAI(board, "white")).toBe("-".repeat(64));
  });

  it("always emits exactly 64 characters", () => {
    const board = setupBoard([
      [0, 0, "black"],
      [3, 4, "white"],
      [7, 7, "black"],
    ]);
    expect(serializeBoardForAI(board, "black")).toHaveLength(64);
    expect(serializeBoardForAI(board, "white")).toHaveLength(64);
  });

  it("encodes cells relative to the player to move (X = player, O = opponent)", () => {
    const board = setupBoard([
      [0, 0, "black"],
      [0, 1, "white"],
    ]);
    expect(serializeBoardForAI(board, "black")).toBe("XO" + "-".repeat(62));
    expect(serializeBoardForAI(board, "white")).toBe("OX" + "-".repeat(62));
  });

  it("orders cells row-major (row 0 first, then row 1, ...)", () => {
    const board = setupBoard([[1, 0, "black"]]);
    const result = serializeBoardForAI(board, "black");
    expect(result[8]).toBe("X");
    expect(result.indexOf("X")).toBe(8);
  });

  it("is the polarity-inverse of boardToString (relative vs absolute color)", () => {
    const board = setupBoard([
      [0, 0, "black"],
      [0, 1, "white"],
      [4, 4, "black"],
      [5, 3, "white"],
    ]);
    const absolute = boardToString(board); // X = black, O = white
    expect(serializeBoardForAI(board, "black")).toBe(absolute);
    const swapped = absolute.replace(/[XO]/g, (ch) => (ch === "X" ? "O" : "X"));
    expect(serializeBoardForAI(board, "white")).toBe(swapped);
  });
});
