/// <reference types="node" />

import fs from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { TAURI_COMMAND, TAURI_EVENT } from "../tauri-contract";

const backendPath = path.resolve(__dirname, "../../../src-tauri/src/lib.rs");
const backendSource = fs.readFileSync(backendPath, "utf8");
const handlerMatch = backendSource.match(/generate_handler!\[([\s\S]*?)\]/);

if (!handlerMatch) {
  throw new Error("generate_handler! block not found in src-tauri/src/lib.rs");
}

const registeredCommands = handlerMatch[1]
  .split(",")
  .map((name) => name.trim())
  .filter(Boolean);

function escapeRegex(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

describe("Tauri IPC contract", () => {
  it("defines and registers every frontend command in the Rust backend", () => {
    for (const command of Object.values(TAURI_COMMAND)) {
      const commandPattern = escapeRegex(command);
      expect(
        new RegExp(`\\bfn\\s+${commandPattern}\\s*\\(`).test(backendSource),
        `command ${command} in tauri-contract.ts not found in lib.rs — renamed backend-side?`,
      ).toBe(true);
      expect(
        registeredCommands.includes(command),
        `command ${command} in tauri-contract.ts is not registered in generate_handler!`,
      ).toBe(true);
    }
  });

  it("lists every registered Rust command in the frontend contract", () => {
    const frontendCommands = Object.values(TAURI_COMMAND) as string[];
    for (const command of registeredCommands) {
      expect(
        frontendCommands.includes(command),
        `command ${command} in generate_handler! is missing from tauri-contract.ts`,
      ).toBe(true);
    }
  });

  it("defines every frontend event as a Rust emit target", () => {
    for (const event of Object.values(TAURI_EVENT)) {
      const eventPattern = escapeRegex(event);
      expect(
        new RegExp(`\\.emit\\(\\s*"${eventPattern}"`).test(backendSource),
        `event ${event} in tauri-contract.ts not found as an emit target in lib.rs — renamed backend-side?`,
      ).toBe(true);
    }
  });
});
