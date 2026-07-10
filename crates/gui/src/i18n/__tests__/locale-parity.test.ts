import { describe, expect, it } from "vitest";
import en from "../locales/en.json";
import ja from "../locales/ja.json";

type LocaleLeaf = [path: string, value: unknown];

function collectLeaves(value: unknown, prefix = ""): LocaleLeaf[] {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return [[prefix, value]];
  }

  return Object.entries(value).flatMap(([key, child]) => {
    const path = prefix ? `${prefix}.${key}` : key;
    return collectLeaves(child, path);
  });
}

describe("locale files", () => {
  const enLeaves = collectLeaves(en);
  const jaLeaves = collectLeaves(ja);
  const enKeys = new Set(enLeaves.map(([path]) => path));
  const jaKeys = new Set(jaLeaves.map(([path]) => path));

  it("keeps English and Japanese leaf keys identical", () => {
    const onlyInEn = [...enKeys].filter((key) => !jaKeys.has(key)).sort();
    const onlyInJa = [...jaKeys].filter((key) => !enKeys.has(key)).sort();

    expect(
      onlyInEn.length === 0 && onlyInJa.length === 0,
      `locale key mismatch\nonly in en: ${onlyInEn.join(", ") || "(none)"}\nonly in ja: ${onlyInJa.join(", ") || "(none)"}`,
    ).toBe(true);
  });

  it.each([
    ["en", enLeaves],
    ["ja", jaLeaves],
  ] as const)("contains only non-empty string leaves in %s", (locale, leaves) => {
    for (const [path, value] of leaves) {
      expect(
        typeof value === "string" && value.trim().length > 0,
        `${locale}.${path} must be a non-empty string`,
      ).toBe(true);
    }
  });
});
