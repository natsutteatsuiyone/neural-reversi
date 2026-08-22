import { describe, expect, it } from "vitest";
import { cellToWorld } from "../board3d-utils";

describe("cellToWorld", () => {
  it("maps columns to x and rows to z", () => {
    expect(cellToWorld(7, 0)).toEqual([-3.5, 3.5]);
  });
});
