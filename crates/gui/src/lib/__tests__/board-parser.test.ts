import { describe, expect, it } from "vitest";
import {
  parseTranscript,
  parseBoardString,
  validateBoard,
  boardToString,
} from "@/lib/board-parser";
import {
  createEmptyBoard,
  initializeBoard,
} from "@/lib/game-logic";
import { BOARD_SIZE } from "@/lib/constants";
import type { Board, Player } from "@/types";

function setupBoard(stones: [number, number, Player | null][]): Board {
  const board = createEmptyBoard();
  for (const [r, c, color] of stones) {
    board[r][c].color = color;
  }
  return board;
}

describe("parseTranscript", () => {
  it("returns initial board for empty string", () => {
    const result = parseTranscript("");
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.currentPlayer).toBe("black");
    expect(result.board[3][3].color).toBe("white");
    expect(result.board[3][4].color).toBe("black");
    expect(result.board[4][3].color).toBe("black");
    expect(result.board[4][4].color).toBe("white");
  });

  it("returns initial board for whitespace-only input", () => {
    const result = parseTranscript("  ");
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.currentPlayer).toBe("black");
    expect(result.board[3][3].color).toBe("white");
  });

  it("parses single move F5 correctly", () => {
    const result = parseTranscript("F5");
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.currentPlayer).toBe("white");
    // F5 = col 5, row 4 → board[4][5] = black (new stone)
    expect(result.board[4][5].color).toBe("black");
    // E5 = (4,4) was white, flipped to black
    expect(result.board[4][4].color).toBe("black");
    // Unchanged initial stones
    expect(result.board[3][3].color).toBe("white");
    expect(result.board[3][4].color).toBe("black");
    expect(result.board[4][3].color).toBe("black");
  });

  it("parses multiple moves", () => {
    const result = parseTranscript("F5D6C3D3C4");
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    // 5 moves: black, white, black, white, black → next is white
    expect(result.currentPlayer).toBe("white");
  });

  it("is case-insensitive", () => {
    const upper = parseTranscript("F5D6");
    const lower = parseTranscript("f5d6");
    expect(upper.ok).toBe(true);
    expect(lower.ok).toBe(true);
    if (!upper.ok || !lower.ok) return;
    expect(upper.currentPlayer).toBe(lower.currentPlayer);
    for (let r = 0; r < BOARD_SIZE; r++) {
      for (let c = 0; c < BOARD_SIZE; c++) {
        expect(upper.board[r][c].color).toBe(lower.board[r][c].color);
      }
    }
  });

  it("returns invalidLength for odd-length input", () => {
    expect(parseTranscript("F5D")).toEqual({ ok: false, error: "invalidLength" });
  });

  it("returns invalidMove for out-of-range column (I1)", () => {
    expect(parseTranscript("I1")).toEqual({ ok: false, error: "invalidMove" });
  });

  it("returns invalidMove for out-of-range row (A9)", () => {
    expect(parseTranscript("A9")).toEqual({ ok: false, error: "invalidMove" });
  });

  it("returns invalidMove for row zero (A0)", () => {
    expect(parseTranscript("A0")).toEqual({ ok: false, error: "invalidMove" });
  });

  it("returns invalidMove for non-digit row character", () => {
    expect(parseTranscript("AX")).toEqual({ ok: false, error: "invalidMove" });
  });

  it("returns illegalMove for a move with no valid flips", () => {
    const result = parseTranscript("A1");
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.error).toBe("illegalMove:A1");
  });

  it("returns illegalMove for placing on an occupied cell", () => {
    const result = parseTranscript("F5E5");
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.error).toBe("illegalMove:E5");
  });
});

describe("parseBoardString", () => {
  it("returns empty board for empty string", () => {
    const result = parseBoardString("");
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.currentPlayer).toBe("black");
    for (let r = 0; r < BOARD_SIZE; r++) {
      for (let c = 0; c < BOARD_SIZE; c++) {
        expect(result.board[r][c].color).toBeNull();
      }
    }
  });

  it("parses 64 dashes as empty board", () => {
    const result = parseBoardString("-".repeat(64));
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    for (let r = 0; r < BOARD_SIZE; r++) {
      for (let c = 0; c < BOARD_SIZE; c++) {
        expect(result.board[r][c].color).toBeNull();
      }
    }
  });

  it("parses X/O/- format for initial position", () => {
    const input = "---------------------------OX------XO---------------------------";
    const result = parseBoardString(input);
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.board[3][3].color).toBe("white");
    expect(result.board[3][4].color).toBe("black");
    expect(result.board[4][3].color).toBe("black");
    expect(result.board[4][4].color).toBe("white");
    expect(result.board[0][0].color).toBeNull();
  });

  it("parses B/W format", () => {
    const input = "---------------------------WB------BW---------------------------";
    const result = parseBoardString(input);
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.board[3][3].color).toBe("white");
    expect(result.board[3][4].color).toBe("black");
    expect(result.board[4][3].color).toBe("black");
    expect(result.board[4][4].color).toBe("white");
  });

  it("parses */. format", () => {
    const input = "...........................O*......*O...........................";
    const result = parseBoardString(input);
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.board[3][3].color).toBe("white");
    expect(result.board[3][4].color).toBe("black");
    expect(result.board[4][3].color).toBe("black");
    expect(result.board[4][4].color).toBe("white");
  });

  it("is case-insensitive", () => {
    const input = "---------------------------ox------xo---------------------------";
    const result = parseBoardString(input);
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.board[3][3].color).toBe("white");
    expect(result.board[3][4].color).toBe("black");
  });

  it("strips whitespace and newlines before parsing", () => {
    const rows = [
      "--------",
      "--------",
      "--------",
      "---OX---",
      "---XO---",
      "--------",
      "--------",
      "--------",
    ];
    const input = rows.join("\n");
    const result = parseBoardString(input);
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.board[3][3].color).toBe("white");
    expect(result.board[3][4].color).toBe("black");
  });

  it("returns invalidBoardLength for too-short input", () => {
    expect(parseBoardString("-".repeat(63))).toEqual({
      ok: false,
      error: "invalidBoardLength",
    });
  });

  it("returns invalidBoardLength for too-long input", () => {
    expect(parseBoardString("-".repeat(65))).toEqual({
      ok: false,
      error: "invalidBoardLength",
    });
  });

  it("returns invalidCharacter for unrecognized character", () => {
    expect(parseBoardString("Z" + "-".repeat(63))).toEqual({
      ok: false,
      error: "invalidCharacter",
    });
  });
});

describe("validateBoard", () => {
  it("returns null for the initial board (valid)", () => {
    const board = initializeBoard();
    expect(validateBoard(board, "black")).toBeNull();
  });

  it("returns null for initial board with white to move", () => {
    const board = initializeBoard();
    expect(validateBoard(board, "white")).toBeNull();
  });

  it("returns needBothColors when only black stones exist", () => {
    const board = setupBoard([
      [0, 0, "black"],
      [0, 1, "black"],
    ]);
    expect(validateBoard(board, "black")).toBe("needBothColors");
  });

  it("returns needBothColors when only white stones exist", () => {
    const board = setupBoard([
      [0, 0, "white"],
      [0, 1, "white"],
    ]);
    expect(validateBoard(board, "white")).toBe("needBothColors");
  });

  it("returns needBothColors for empty board", () => {
    const board = createEmptyBoard();
    expect(validateBoard(board, "black")).toBe("needBothColors");
  });

  it("returns tooFewDiscs when fewer than 4 stones on board", () => {
    const board = setupBoard([
      [0, 0, "black"],
      [0, 1, "white"],
      [0, 2, "black"],
    ]);
    expect(validateBoard(board, "black")).toBe("tooFewDiscs");
  });

  it("returns noValidMoves when neither player can move", () => {
    const board = setupBoard([
      [0, 0, "black"],
      [0, 1, "black"],
      [7, 6, "white"],
      [7, 7, "white"],
    ]);
    expect(validateBoard(board, "black")).toBe("noValidMoves");
  });

  it("returns currentPlayerNoMoves when only opponent can move", () => {
    const board = setupBoard([
      [0, 0, "white"],
      [0, 1, "black"],
      [7, 6, "black"],
      [7, 7, "black"],
    ]);
    expect(validateBoard(board, "black")).toBe("currentPlayerNoMoves");
  });
});

describe("boardToString", () => {
  it("converts empty board to 64 dashes", () => {
    const board = createEmptyBoard();
    expect(boardToString(board)).toBe("-".repeat(64));
  });

  it("converts initial board to correct string", () => {
    const board = initializeBoard();
    const expected = "---------------------------OX------XO---------------------------";
    expect(boardToString(board)).toBe(expected);
    expect(expected).toHaveLength(64);
  });

  it("round-trips with parseBoardString", () => {
    const board = initializeBoard();
    const str = boardToString(board);
    const result = parseBoardString(str);
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    for (let r = 0; r < BOARD_SIZE; r++) {
      for (let c = 0; c < BOARD_SIZE; c++) {
        expect(result.board[r][c].color).toBe(board[r][c].color);
      }
    }
  });

  it("round-trips a custom board position", () => {
    const board = setupBoard([
      [0, 0, "black"],
      [0, 1, "white"],
      [3, 3, "black"],
      [7, 7, "white"],
    ]);
    const str = boardToString(board);
    const result = parseBoardString(str);
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.board[0][0].color).toBe("black");
    expect(result.board[0][1].color).toBe("white");
    expect(result.board[3][3].color).toBe("black");
    expect(result.board[7][7].color).toBe("white");
    expect(result.board[1][1].color).toBeNull();
  });
});
