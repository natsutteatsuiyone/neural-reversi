import { describe, expect, it } from "vitest";
import {
  cloneBoard,
  createMoveRecord,
  applyMove,
  createPassMove,
  reconstructBoardFromMoves,
  checkGameOver,
  getUndoCount,
  getRedoCount,
} from "@/lib/store-helpers";
import type { Move } from "@/lib/store-helpers";
import { createEmptyBoard, getNotation, initializeBoard } from "@/lib/game-logic";
import type { Board, MoveRecord, Player } from "@/types";

function setupBoard(stones: [number, number, Player | null][]): Board {
  const board = createEmptyBoard();
  for (const [r, c, color] of stones) {
    board[r][c].color = color;
  }
  return board;
}

function makeMoveRecord(id: number, player: Player, row: number, col: number): MoveRecord {
  return { id, player, row, col, notation: getNotation(row, col) };
}

function makePassRecord(id: number, player: Player): MoveRecord {
  return { id, player, row: -1, col: -1, notation: "Pass" };
}

describe("cloneBoard", () => {
  it("clones initial board with identical content", () => {
    const original = initializeBoard();
    const clone = cloneBoard(original);

    for (let r = 0; r < 8; r++) {
      for (let c = 0; c < 8; c++) {
        expect(clone[r][c].color).toBe(original[r][c].color);
      }
    }
  });

  it("produces a deep copy (mutation independence)", () => {
    const original = initializeBoard();
    const clone = cloneBoard(original);

    clone[0][0].color = "black";
    expect(original[0][0].color).toBeNull();
  });

  it("preserves isNew property", () => {
    const board = createEmptyBoard();
    board[2][3] = { color: "black", isNew: true };
    const clone = cloneBoard(board);

    expect(clone[2][3].isNew).toBe(true);
    expect(clone[2][3].color).toBe("black");
  });
});

describe("createMoveRecord", () => {
  it("sets all fields correctly for a normal move", () => {
    const move: Move = { row: 2, col: 3, isAI: false, score: 4 };
    const record = createMoveRecord(1, "black", move, 120);

    expect(record).toEqual({
      id: 1,
      player: "black",
      row: 2,
      col: 3,
      notation: "d3",
      score: 4,
      isAI: false,
      remainingTime: 120,
    });
  });

  it("leaves score undefined when move has no score", () => {
    const move: Move = { row: 4, col: 5, isAI: true };
    const record = createMoveRecord(2, "white", move);

    expect(record.score).toBeUndefined();
    expect(record.isAI).toBe(true);
  });

  it("leaves remainingTime undefined when not provided", () => {
    const move: Move = { row: 0, col: 0, isAI: false };
    const record = createMoveRecord(3, "black", move);

    expect(record.remainingTime).toBeUndefined();
  });
});

describe("createPassMove", () => {
  it("creates a pass record with row=-1, col=-1, notation='Pass'", () => {
    const record = createPassMove(5, "white");

    expect(record).toEqual({
      id: 5,
      player: "white",
      row: -1,
      col: -1,
      notation: "Pass",
      remainingTime: undefined,
    });
  });

  it("includes remainingTime when provided", () => {
    const record = createPassMove(6, "black", 30);

    expect(record.remainingTime).toBe(30);
    expect(record.notation).toBe("Pass");
  });
});

describe("applyMove", () => {
  it("places stone and flips opponent discs", () => {
    // Black plays d3 (row=2, col=3) on initial board
    // Flips: (3,3) white → black (vertical down direction)
    const board = initializeBoard();
    const move: Move = { row: 2, col: 3, isAI: false };
    const result = applyMove(board, move, "black");

    expect(result[2][3].color).toBe("black");
    expect(result[3][3].color).toBe("black"); // flipped from white
    expect(result[4][4].color).toBe("white"); // unchanged
  });

  it("does not mutate the original board", () => {
    const board = initializeBoard();
    const move: Move = { row: 2, col: 3, isAI: false };

    applyMove(board, move, "black");

    expect(board[2][3].color).toBeNull();
    expect(board[3][3].color).toBe("white");
  });

  it("marks the placed stone with isNew: true", () => {
    const board = initializeBoard();
    const move: Move = { row: 2, col: 3, isAI: false };
    const result = applyMove(board, move, "black");

    expect(result[2][3].isNew).toBe(true);
  });

  it("flips discs in multiple directions", () => {
    // Board: (3,4)=W, (3,5)=B, (4,3)=W, (5,3)=B
    // Black plays (3,3): flips (3,4) horizontal and (4,3) vertical
    const board = setupBoard([
      [3, 4, "white"],
      [3, 5, "black"],
      [4, 3, "white"],
      [5, 3, "black"],
    ]);
    const move: Move = { row: 3, col: 3, isAI: false };
    const result = applyMove(board, move, "black");

    expect(result[3][3]).toEqual({ color: "black", isNew: true });
    expect(result[3][4].color).toBe("black"); // flipped
    expect(result[4][3].color).toBe("black"); // flipped
    expect(result[3][5].color).toBe("black"); // unchanged anchor
    expect(result[5][3].color).toBe("black"); // unchanged anchor
  });
});

describe("reconstructBoardFromMoves", () => {
  it("returns initial board and black when given empty moves", () => {
    const result = reconstructBoardFromMoves([]);

    expect(result.currentPlayer).toBe("black");
    expect(result.board[3][3].color).toBe("white");
    expect(result.board[3][4].color).toBe("black");
    expect(result.board[4][3].color).toBe("black");
    expect(result.board[4][4].color).toBe("white");
  });

  it("reconstructs board after a single move", () => {
    // Black plays F5 (row=4, col=5)
    // Flips (4,4) white → black
    const moves = [makeMoveRecord(1, "black", 4, 5)];
    const result = reconstructBoardFromMoves(moves);

    expect(result.board[4][5].color).toBe("black");
    expect(result.board[4][4].color).toBe("black"); // flipped
    expect(result.board[3][3].color).toBe("white"); // unchanged
    expect(result.currentPlayer).toBe("white");
  });

  it("skips pass moves without changing the board", () => {
    const moves = [makePassRecord(1, "black")];
    const result = reconstructBoardFromMoves(moves);

    // Board should be initial (pass doesn't change it)
    expect(result.board[3][3].color).toBe("white");
    expect(result.board[3][4].color).toBe("black");
    expect(result.currentPlayer).toBe("white");
  });

  it("returns correct currentPlayer after multiple moves", () => {
    const moves = [
      makeMoveRecord(1, "black", 4, 5),
      makeMoveRecord(2, "white", 5, 3),
    ];
    const result = reconstructBoardFromMoves(moves);

    expect(result.currentPlayer).toBe("black");
  });

  it("uses custom start board and player when provided", () => {
    const customBoard = setupBoard([
      [3, 3, "black"],
      [3, 4, "white"],
    ]);
    const result = reconstructBoardFromMoves([], customBoard, "white");

    expect(result.board[3][3].color).toBe("black");
    expect(result.board[3][4].color).toBe("white");
    expect(result.board[0][0].color).toBeNull();
    expect(result.currentPlayer).toBe("white");
  });
});

describe("checkGameOver", () => {
  it("returns gameOver=false, shouldPass=false when current player has moves", () => {
    const board = initializeBoard();
    expect(checkGameOver(board, "black")).toEqual({
      gameOver: false,
      shouldPass: false,
    });
  });

  it("returns shouldPass=true when current player has no moves but opponent does", () => {
    // white at (0,0), black at (0,1): black has no moves, white can play (0,2)
    const board = setupBoard([
      [0, 0, "white"],
      [0, 1, "black"],
    ]);
    expect(checkGameOver(board, "black")).toEqual({
      gameOver: false,
      shouldPass: true,
    });
  });

  it("returns gameOver=true when neither player has moves", () => {
    // Isolated stones: black at (0,0), white at (7,7) — no flips possible
    const board = setupBoard([
      [0, 0, "black"],
      [7, 7, "white"],
    ]);
    expect(checkGameOver(board, "black")).toEqual({
      gameOver: true,
      shouldPass: false,
    });
  });
});

describe("getUndoCount", () => {
  it("returns 0 for empty moves", () => {
    expect(getUndoCount([], "analyze")).toBe(0);
  });

  it("returns 1 in analyze mode", () => {
    const moves = [makeMoveRecord(1, "black", 4, 5), makeMoveRecord(2, "white", 5, 3)];
    expect(getUndoCount(moves, "analyze")).toBe(1);
  });

  it("returns 2 in AI mode when player's turn", () => {
    const moves = [
      makeMoveRecord(1, "black", 4, 5),
      makeMoveRecord(2, "white", 5, 3),
      makeMoveRecord(3, "black", 2, 2),
      makeMoveRecord(4, "white", 2, 3),
    ];
    expect(getUndoCount(moves, "ai-white")).toBe(2);
  });

  it("returns 1 in AI mode when AI's turn", () => {
    const moves = [
      makeMoveRecord(1, "black", 4, 5),
      makeMoveRecord(2, "white", 5, 3),
      makeMoveRecord(3, "black", 2, 2),
    ];
    expect(getUndoCount(moves, "ai-white")).toBe(1);
  });

  it("returns 0 when player's turn but only 1 move", () => {
    const moves = [makeMoveRecord(1, "black", 4, 5)];
    expect(getUndoCount(moves, "ai-black")).toBe(0);
  });
});

describe("getRedoCount", () => {
  it("returns 0 when at the end", () => {
    const moves = [makeMoveRecord(1, "black", 4, 5)];
    expect(getRedoCount(moves, moves, "analyze")).toBe(0);
  });

  it("returns 1 in analyze mode", () => {
    const allMoves = [
      makeMoveRecord(1, "black", 4, 5),
      makeMoveRecord(2, "white", 5, 3),
    ];
    expect(getRedoCount([allMoves[0]], allMoves, "analyze")).toBe(1);
  });

  it("returns count until player's turn in AI mode", () => {
    const allMoves = [
      makeMoveRecord(1, "black", 4, 5),
      makeMoveRecord(2, "white", 5, 3),
      makeMoveRecord(3, "black", 2, 2),
      makeMoveRecord(4, "white", 2, 3),
    ];
    expect(getRedoCount([allMoves[0]], allMoves, "ai-white")).toBe(1);
  });

  it("skips past pass moves in AI mode", () => {
    const allMoves = [
      makeMoveRecord(1, "black", 4, 5),
      makeMoveRecord(2, "white", 5, 3),
      makePassRecord(3, "black"),
      makeMoveRecord(4, "white", 2, 2),
    ];
    expect(getRedoCount([allMoves[0], allMoves[1]], allMoves, "ai-white")).toBe(2);
  });
});
