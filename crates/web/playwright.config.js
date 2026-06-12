import { defineConfig, devices } from "@playwright/test";
import { existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const configDir = dirname(fileURLToPath(import.meta.url));

const requiredWasmFiles = ["pkg/web.js", "pkg/web_bg.wasm"];

const missingWasmFiles = requiredWasmFiles.filter((file) => !existsSync(resolve(configDir, file)));

if (missingWasmFiles.length > 0) {
  throw new Error(
    [
      "Playwright E2E tests need generated WASM packages.",
      "Run `bun run build:wasm:dev` from crates/web before `bun run test:e2e`.",
      `Missing: ${missingWasmFiles.join(", ")}`,
    ].join("\n"),
  );
}

export default defineConfig({
  testDir: "./tests/e2e",
  timeout: 60_000,
  expect: {
    timeout: 10_000,
  },
  reporter: process.env.CI ? [["github"], ["html", { open: "never" }]] : "list",
  use: {
    baseURL: "http://127.0.0.1:8080",
    locale: "ja-JP",
    trace: "on-first-retry",
    viewport: {
      width: 1280,
      height: 900,
    },
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
  webServer: {
    command: "bun run dev:e2e",
    url: "http://127.0.0.1:8080",
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
});
