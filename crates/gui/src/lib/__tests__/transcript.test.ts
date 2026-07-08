import { describe, expect, it } from "vitest";
import { buildTranscript } from "@/lib/transcript";
import type { MoveRecord } from "@/domain/game/types";

function move(notation: string, row = 0): MoveRecord {
  return { id: 0, player: "black", row, col: 0, notation };
}

describe("buildTranscript", () => {
  it("returns an empty string for no moves", () => {
    expect(buildTranscript([])).toBe("");
  });

  it("lowercases each move's notation and concatenates without separators", () => {
    expect(buildTranscript([move("F5"), move("D6"), move("C3")])).toBe("f5d6c3");
  });

  it("omits pass records (row < 0) while keeping order", () => {
    expect(buildTranscript([move("F5"), move("--", -1), move("D6")])).toBe("f5d6");
  });
});
