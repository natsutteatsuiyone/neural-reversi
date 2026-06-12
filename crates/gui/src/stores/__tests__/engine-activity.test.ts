import { describe, expect, it } from "vitest";
import type { EngineActivityKind } from "@/domain/engine/engine-search";
import {
  IDLE_ENGINE_ACTIVITY,
  engineActivityPatch,
  idleEngineActivityPatch,
  isGameSearchActive,
} from "@/stores/engine-activity";

const ALL_KINDS: EngineActivityKind[] = ["idle", "ai-move", "hint", "game-analysis", "solver"];

// The four busy booleans are views of `engineActivity.kind` and this is their
// single source (CONTEXT.md → Engine Activity). These tests pin that mapping
// so a fifth kind cannot silently desync a hand-listed reset.
describe("engineActivityPatch", () => {
  it("sets exactly the one boolean that matches the kind", () => {
    const expected: Record<
      EngineActivityKind,
      {
        isAIThinking: boolean;
        isAnalyzing: boolean;
        isGameAnalyzing: boolean;
        isSolverSearching: boolean;
      }
    > = {
      idle: {
        isAIThinking: false,
        isAnalyzing: false,
        isGameAnalyzing: false,
        isSolverSearching: false,
      },
      "ai-move": {
        isAIThinking: true,
        isAnalyzing: false,
        isGameAnalyzing: false,
        isSolverSearching: false,
      },
      hint: {
        isAIThinking: false,
        isAnalyzing: true,
        isGameAnalyzing: false,
        isSolverSearching: false,
      },
      "game-analysis": {
        isAIThinking: false,
        isAnalyzing: false,
        isGameAnalyzing: true,
        isSolverSearching: false,
      },
      solver: {
        isAIThinking: false,
        isAnalyzing: false,
        isGameAnalyzing: false,
        isSolverSearching: true,
      },
    };

    for (const kind of ALL_KINDS) {
      const patch = engineActivityPatch({ kind, runId: 7 });
      expect(patch.engineActivity).toEqual({ kind, runId: 7 });
      expect({
        isAIThinking: patch.isAIThinking,
        isAnalyzing: patch.isAnalyzing,
        isGameAnalyzing: patch.isGameAnalyzing,
        isSolverSearching: patch.isSolverSearching,
      }).toEqual(expected[kind]);
    }
  });
});

describe("idleEngineActivityPatch", () => {
  it("resets engineActivity AND every busy boolean in one patch", () => {
    const patch = idleEngineActivityPatch();
    expect(patch.engineActivity).toBe(IDLE_ENGINE_ACTIVITY);
    expect(patch.isAIThinking).toBe(false);
    expect(patch.isAnalyzing).toBe(false);
    expect(patch.isGameAnalyzing).toBe(false);
    expect(patch.isSolverSearching).toBe(false);
  });

  it("is the same projection engineActivityPatch produces for idle", () => {
    expect(idleEngineActivityPatch()).toEqual(engineActivityPatch(IDLE_ENGINE_ACTIVITY));
  });
});

describe("isGameSearchActive", () => {
  it("is true for in-game searches, false for idle and solver", () => {
    expect(isGameSearchActive({ kind: "idle", runId: 0 })).toBe(false);
    expect(isGameSearchActive({ kind: "ai-move", runId: 1 })).toBe(true);
    expect(isGameSearchActive({ kind: "hint", runId: 1 })).toBe(true);
    expect(isGameSearchActive({ kind: "game-analysis", runId: 1 })).toBe(true);
    expect(isGameSearchActive({ kind: "solver", runId: 1 })).toBe(false);
  });
});
