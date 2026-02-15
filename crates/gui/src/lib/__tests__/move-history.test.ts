import { describe, expect, it } from "vitest";
import { MoveHistory } from "@/lib/move-history";
import type { MoveRecord } from "@/types";

function makeRecord(id: number, player: "black" | "white", row: number, col: number): MoveRecord {
  return { id, player, row, col, notation: `${String.fromCharCode(97 + col)}${row + 1}` };
}

describe("MoveHistory", () => {
  describe("empty", () => {
    it("creates an empty history", () => {
      const h = MoveHistory.empty();
      expect(h.length).toBe(0);
      expect(h.totalLength).toBe(0);
      expect(h.currentMoves).toEqual([]);
      expect(h.lastMove).toBeUndefined();
      expect(h.canUndo).toBe(false);
      expect(h.canRedo).toBe(false);
    });
  });

  describe("append", () => {
    it("adds a move to empty history", () => {
      const h = MoveHistory.empty().append(makeRecord(0, "black", 2, 3));
      expect(h.length).toBe(1);
      expect(h.totalLength).toBe(1);
      expect(h.currentMoves).toHaveLength(1);
      expect(h.currentMoves[0].player).toBe("black");
      expect(h.lastMove?.row).toBe(2);
    });

    it("discards redo history when appending after undo", () => {
      const r1 = makeRecord(0, "black", 2, 3);
      const r2 = makeRecord(1, "white", 3, 2);
      const r3 = makeRecord(2, "black", 4, 5);

      const h = MoveHistory.empty().append(r1).append(r2).undo(1).append(r3);
      expect(h.length).toBe(2);
      expect(h.totalLength).toBe(2);
      expect(h.currentMoves[1]).toBe(r3);
      expect(h.canRedo).toBe(false);
    });

    it("does not mutate the original instance", () => {
      const h1 = MoveHistory.empty();
      const h2 = h1.append(makeRecord(0, "black", 2, 3));
      expect(h1.length).toBe(0);
      expect(h2.length).toBe(1);
    });
  });

  describe("undo", () => {
    it("moves cursor back by count", () => {
      const h = MoveHistory.empty()
        .append(makeRecord(0, "black", 2, 3))
        .append(makeRecord(1, "white", 3, 2))
        .undo(1);
      expect(h.length).toBe(1);
      expect(h.totalLength).toBe(2);
      expect(h.canRedo).toBe(true);
    });

    it("clamps to zero when undoing more than available", () => {
      const h = MoveHistory.empty()
        .append(makeRecord(0, "black", 2, 3))
        .undo(5);
      expect(h.length).toBe(0);
      expect(h.totalLength).toBe(1);
    });

    it("returns same instance when count is 0", () => {
      const h = MoveHistory.empty().append(makeRecord(0, "black", 2, 3));
      expect(h.undo(0)).toBe(h);
    });
  });

  describe("redo", () => {
    it("moves cursor forward by count", () => {
      const h = MoveHistory.empty()
        .append(makeRecord(0, "black", 2, 3))
        .append(makeRecord(1, "white", 3, 2))
        .undo(2)
        .redo(1);
      expect(h.length).toBe(1);
      expect(h.totalLength).toBe(2);
    });

    it("clamps to totalLength when redoing more than available", () => {
      const h = MoveHistory.empty()
        .append(makeRecord(0, "black", 2, 3))
        .undo(1)
        .redo(5);
      expect(h.length).toBe(1);
      expect(h.canRedo).toBe(false);
    });

    it("returns same instance when count is 0", () => {
      const h = MoveHistory.empty()
        .append(makeRecord(0, "black", 2, 3))
        .undo(1);
      expect(h.redo(0)).toBe(h);
    });
  });

  describe("redoMoves", () => {
    it("returns empty array when no redo available", () => {
      const h = MoveHistory.empty().append(makeRecord(0, "black", 2, 3));
      expect(h.redoMoves).toEqual([]);
    });

    it("returns future moves after undo", () => {
      const r1 = makeRecord(0, "black", 2, 3);
      const r2 = makeRecord(1, "white", 3, 2);
      const h = MoveHistory.empty().append(r1).append(r2).undo(1);
      expect(h.redoMoves).toEqual([r2]);
    });
  });

  describe("lastMove", () => {
    it("returns undefined for empty history", () => {
      expect(MoveHistory.empty().lastMove).toBeUndefined();
    });

    it("returns the last current move", () => {
      const r1 = makeRecord(0, "black", 2, 3);
      const r2 = makeRecord(1, "white", 3, 2);
      const h = MoveHistory.empty().append(r1).append(r2);
      expect(h.lastMove).toBe(r2);
    });

    it("returns correct move after undo", () => {
      const r1 = makeRecord(0, "black", 2, 3);
      const r2 = makeRecord(1, "white", 3, 2);
      const h = MoveHistory.empty().append(r1).append(r2).undo(1);
      expect(h.lastMove).toBe(r1);
    });
  });
});
