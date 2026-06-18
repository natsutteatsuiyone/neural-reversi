import { describe, expect, it } from "vitest";
import { sliderValueToNumber } from "@/components/ui/slider-value";

describe("sliderValueToNumber", () => {
  it("returns a scalar number unchanged (Base UI single-thumb runtime value)", () => {
    expect(sliderValueToNumber(7)).toBe(7);
    expect(sliderValueToNumber(0)).toBe(0);
  });

  it("returns the first element of an array value", () => {
    expect(sliderValueToNumber([12])).toBe(12);
    expect(sliderValueToNumber([3, 9])).toBe(3);
  });

  it("never yields NaN for a scalar (the regression this guards against)", () => {
    expect(Number.isNaN(sliderValueToNumber(0))).toBe(false);
  });
});
