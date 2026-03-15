import { describe, expect, it } from "vitest";
import {
  cellToWorld,
  CELL_SIZE,
  BOARD_OFFSET,
  STONE_RADIUS,
  STONE_HEIGHT,
} from "../board3d-utils";

describe("board3d-utils", () => {
  describe("cellToWorld", () => {
    it("converts top-left cell (0,0) to world position", () => {
      const [x, z] = cellToWorld(0, 0);
      expect(x).toBe(-3.5);
      expect(z).toBe(-3.5);
    });
    it("converts center cell (3,3) to world position", () => {
      const [x, z] = cellToWorld(3, 3);
      expect(x).toBe(-0.5);
      expect(z).toBe(-0.5);
    });
    it("converts bottom-right cell (7,7) to world position", () => {
      const [x, z] = cellToWorld(7, 7);
      expect(x).toBe(3.5);
      expect(z).toBe(3.5);
    });
    it("converts cell (0,7) to world position", () => {
      const [x, z] = cellToWorld(0, 7);
      expect(x).toBe(3.5);
      expect(z).toBe(-3.5);
    });
  });
  describe("constants", () => {
    it("has correct cell size", () => {
      expect(CELL_SIZE).toBe(1);
    });
    it("has correct board offset", () => {
      expect(BOARD_OFFSET).toBe(3.5);
    });
    it("has stone radius ~40% of cell width", () => {
      expect(STONE_RADIUS).toBeCloseTo(0.4, 1);
    });
    it("has thin stone height", () => {
      expect(STONE_HEIGHT).toBeGreaterThan(0);
      expect(STONE_HEIGHT).toBeLessThan(0.2);
    });
  });
});
