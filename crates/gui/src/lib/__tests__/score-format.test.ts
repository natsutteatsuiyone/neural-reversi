import { describe, expect, it } from "vitest";
import { formatScore, scoreToneClass, formatDepth } from "@/lib/score-format";

describe("formatScore", () => {
  it("prefixes a positive score with +", () => {
    expect(formatScore(3, "raw")).toBe("+3");
    expect(formatScore(3, "whole")).toBe("+3");
    expect(formatScore(3, "tenth")).toBe("+3.0");
  });

  it("keeps the value's own minus for a negative score", () => {
    expect(formatScore(-3, "raw")).toBe("-3");
    expect(formatScore(-3, "whole")).toBe("-3");
    expect(formatScore(-3, "tenth")).toBe("-3.0");
  });

  it("renders zero with no sign", () => {
    expect(formatScore(0, "raw")).toBe("0");
    expect(formatScore(0, "whole")).toBe("0");
    expect(formatScore(0, "tenth")).toBe("0.0");
  });

  it("reduces before signing per the chosen rounding", () => {
    expect(formatScore(3.6, "raw")).toBe("+3.6");
    expect(formatScore(3.6, "whole")).toBe("+4");
    expect(formatScore(3.6, "tenth")).toBe("+3.6");
    expect(formatScore(-0.4, "whole")).toBe("0"); // rounds toward zero → unsigned
    expect(formatScore(0.4, "whole")).toBe("0"); // positive rounds to zero → still unsigned, not "+0"
    expect(formatScore(0.04, "tenth")).toBe("0.0"); // positive reduces to zero → "0.0", not "+0.0"
  });

  it("defaults to whole-disc rounding", () => {
    expect(formatScore(3.6)).toBe("+4");
  });
});

describe("scoreToneClass", () => {
  it("maps sign to a text-colour class", () => {
    expect(scoreToneClass(1)).toBe("text-primary");
    expect(scoreToneClass(-1)).toBe("text-destructive");
    expect(scoreToneClass(0)).toBe("text-foreground");
  });
});

describe("formatDepth", () => {
  it("shows bare depth at full accuracy", () => {
    expect(formatDepth(20, 100)).toBe("20");
  });

  it("shows depth@acc% below full accuracy", () => {
    expect(formatDepth(20, 95)).toBe("20@95%");
  });
});
