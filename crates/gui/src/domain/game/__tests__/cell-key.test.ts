import { describe, expect, it } from "vitest";
import { cellKey } from "@/domain/game/cell-key";

describe("cellKey", () => {
  it("encodes row and col as a comma-joined key", () => {
    expect(cellKey(0, 0)).toBe("0,0");
    expect(cellKey(3, 5)).toBe("3,5");
    expect(cellKey(7, 7)).toBe("7,7");
  });

  it("is stable for the same cell", () => {
    expect(cellKey(2, 4)).toBe(cellKey(2, 4));
  });

  it("distinguishes transposed cells", () => {
    expect(cellKey(2, 4)).not.toBe(cellKey(4, 2));
  });
});
