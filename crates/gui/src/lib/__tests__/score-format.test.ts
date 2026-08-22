import { describe, expect, it } from "vitest";
import { formatScore } from "@/lib/score-format";

describe("formatScore", () => {
  it("removes the sign after rounding to zero", () => {
    expect(formatScore(-0.4, "whole")).toBe("0");
    expect(formatScore(-0.04, "tenth")).toBe("0.0");
  });
});
