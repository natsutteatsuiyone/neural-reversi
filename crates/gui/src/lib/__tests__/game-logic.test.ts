import { describe, expect, it } from "vitest";
import {
  createEmptyBoard,
  initializeBoard,
  getNotation,
  opponentPlayer,
  calculateScores,
  getWinner,
  getFlippedDiscs,
  getValidMoves,
} from "@/lib/game-logic";
import { BOARD_SIZE } from "@/lib/constants";
import type { Board, Player } from "@/types";

describe("createEmptyBoard", () => {
  it("returns an 8x8 board with all cells empty", () => {
    const board = createEmptyBoard();

    expect(board).toHaveLength(BOARD_SIZE);
    for (const row of board) {
      expect(row).toHaveLength(BOARD_SIZE);
      for (const cell of row) {
        expect(cell.color).toBeNull();
      }
    }
  });

  it("returns independent rows (not shared references)", () => {
    const board = createEmptyBoard();
    board[0][0].color = "black";
    expect(board[1][0].color).toBeNull();
  });
});

describe("initializeBoard", () => {
  it("places 4 initial stones at correct positions", () => {
    const board = initializeBoard();

    expect(board[3][3].color).toBe("white");
    expect(board[3][4].color).toBe("black");
    expect(board[4][3].color).toBe("black");
    expect(board[4][4].color).toBe("white");
  });

  it("leaves all other cells empty", () => {
    const board = initializeBoard();
    let emptyCount = 0;

    for (let r = 0; r < BOARD_SIZE; r++) {
      for (let c = 0; c < BOARD_SIZE; c++) {
        if (board[r][c].color === null) emptyCount++;
      }
    }

    expect(emptyCount).toBe(BOARD_SIZE * BOARD_SIZE - 4);
  });
});

describe("getNotation", () => {
  it("converts (0, 0) to 'a1'", () => {
    expect(getNotation(0, 0)).toBe("a1");
  });

  it("converts (7, 7) to 'h8'", () => {
    expect(getNotation(7, 7)).toBe("h8");
  });

  it("converts (2, 3) to 'd3'", () => {
    expect(getNotation(2, 3)).toBe("d3");
  });
});

describe("opponentPlayer", () => {
  it("returns white for black", () => {
    expect(opponentPlayer("black")).toBe("white");
  });

  it("returns black for white", () => {
    expect(opponentPlayer("white")).toBe("black");
  });
});

describe("calculateScores", () => {
  it("counts initial board as 2-2", () => {
    const board = initializeBoard();
    expect(calculateScores(board)).toEqual({ black: 2, white: 2 });
  });

  it("counts empty board as 0-0", () => {
    const board = createEmptyBoard();
    expect(calculateScores(board)).toEqual({ black: 0, white: 0 });
  });

  it("counts a board with custom stones", () => {
    const board = createEmptyBoard();
    board[0][0].color = "black";
    board[0][1].color = "black";
    board[0][2].color = "white";
    expect(calculateScores(board)).toEqual({ black: 2, white: 1 });
  });
});

describe("getWinner", () => {
  it("returns 'black' when black has more", () => {
    expect(getWinner({ black: 40, white: 24 })).toBe("black");
  });

  it("returns 'white' when white has more", () => {
    expect(getWinner({ black: 20, white: 44 })).toBe("white");
  });

  it("returns 'draw' when equal", () => {
    expect(getWinner({ black: 32, white: 32 })).toBe("draw");
  });
});

function setupBoard(stones: [number, number, Player | null][]): Board {
  const board = createEmptyBoard();
  for (const [r, c, color] of stones) {
    board[r][c].color = color;
  }
  return board;
}

describe("getFlippedDiscs", () => {
  it("flips one disc in a single direction", () => {
    const board = setupBoard([
      [3, 4, "black"],
      [3, 3, "white"],
    ]);
    const flipped = getFlippedDiscs(board, 3, 2, "black");
    expect(flipped).toEqual([[3, 3]]);
  });

  it("flips multiple discs in one direction", () => {
    const board = setupBoard([
      [3, 5, "black"],
      [3, 4, "white"],
      [3, 3, "white"],
    ]);
    const flipped = getFlippedDiscs(board, 3, 2, "black");
    expect(flipped).toContainEqual([3, 3]);
    expect(flipped).toContainEqual([3, 4]);
    expect(flipped).toHaveLength(2);
  });

  it("flips discs in multiple directions simultaneously", () => {
    // Set up a board where black at (3,3) flips in two directions:
    // horizontal: white at (3,4), black at (3,5)
    // vertical: white at (4,3), black at (5,3)
    const board = setupBoard([
      [3, 4, "white"],
      [3, 5, "black"],
      [4, 3, "white"],
      [5, 3, "black"],
    ]);
    const flipped = getFlippedDiscs(board, 3, 3, "black");
    expect(flipped).toContainEqual([3, 4]);
    expect(flipped).toContainEqual([4, 3]);
    expect(flipped).toHaveLength(2);
  });

  it("returns empty array when no opponent disc between placement and own disc", () => {
    const board = initializeBoard();
    const flipped = getFlippedDiscs(board, 3, 3, "black");
    expect(flipped).toEqual([]);
  });

  it("returns empty array when no opponent discs to flip", () => {
    const board = setupBoard([[3, 3, "black"]]);
    const flipped = getFlippedDiscs(board, 3, 4, "black");
    expect(flipped).toEqual([]);
  });

  it("does not flip when line is not terminated by own disc", () => {
    const board = setupBoard([[3, 3, "white"]]);
    const flipped = getFlippedDiscs(board, 3, 2, "black");
    expect(flipped).toEqual([]);
  });

  it("handles edge of board correctly", () => {
    const board = setupBoard([
      [0, 2, "black"],
      [0, 1, "white"],
    ]);
    const flipped = getFlippedDiscs(board, 0, 0, "black");
    expect(flipped).toEqual([[0, 1]]);
  });

  it("handles corner positions", () => {
    const board = setupBoard([
      [2, 0, "black"],
      [1, 0, "white"],
    ]);
    const flipped = getFlippedDiscs(board, 0, 0, "black");
    expect(flipped).toEqual([[1, 0]]);
  });

  it("handles diagonal flips", () => {
    const board = setupBoard([
      [5, 5, "black"],
      [4, 4, "white"],
    ]);
    const flipped = getFlippedDiscs(board, 3, 3, "black");
    expect(flipped).toEqual([[4, 4]]);
  });
});

describe("getValidMoves", () => {
  it("returns 4 valid moves for black on initial board", () => {
    const board = initializeBoard();
    const moves = getValidMoves(board, "black");

    expect(moves).toHaveLength(4);
    expect(moves).toContainEqual([2, 3]);
    expect(moves).toContainEqual([3, 2]);
    expect(moves).toContainEqual([4, 5]);
    expect(moves).toContainEqual([5, 4]);
  });

  it("returns 4 valid moves for white on initial board", () => {
    const board = initializeBoard();
    const moves = getValidMoves(board, "white");

    expect(moves).toHaveLength(4);
    expect(moves).toContainEqual([2, 4]);
    expect(moves).toContainEqual([3, 5]);
    expect(moves).toContainEqual([4, 2]);
    expect(moves).toContainEqual([5, 3]);
  });

  it("returns empty array when no valid moves exist", () => {
    const board = createEmptyBoard();
    board[0][0].color = "black";
    const moves = getValidMoves(board, "white");
    expect(moves).toEqual([]);
  });

  it("returns empty array on empty board", () => {
    const board = createEmptyBoard();
    expect(getValidMoves(board, "black")).toEqual([]);
    expect(getValidMoves(board, "white")).toEqual([]);
  });
});
