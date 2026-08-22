import { describe, expect, it } from "vitest";
import { buildTranscript } from "@/lib/transcript";
import type { MoveRecord } from "@/domain/game/types";

function move(notation: string, row = 0): MoveRecord {
  return { id: 0, player: "black", row, col: 0, notation };
}

describe("buildTranscript", () => {
  it("lowercases played moves and omits pass records", () => {
    expect(buildTranscript([move("F5"), move("--", -1), move("D6")])).toBe("f5d6");
  });
});
