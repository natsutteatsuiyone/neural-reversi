import { beforeEach, describe, expect, it, vi } from "vitest";
import { createTestStore, type TestStore } from "./test-helpers";
import type { EngineActivity } from "@/domain/engine/engine-search";

/**
 * The navigation guard is now concentrated in the history-navigation seam
 * (beginNavigation): undo/redo/goToMove are one-line delegations, so they all
 * share the same canonical "blocked while an in-game Engine Search is active"
 * predicate. These tests pin that single rule through the slice surface.
 */
describe("history-navigation guard (beginNavigation)", () => {
  let store: TestStore;

  beforeEach(async () => {
    ({ store } = createTestStore());
    await store.getState().startGame();
    store.setState({ gameMode: "pvp" });
    await store.getState().makeMove({ row: 2, col: 3, isAI: false });
  });

  const activity = (kind: EngineActivity["kind"]): EngineActivity => ({
    kind,
    runId: 1,
  });

  it.each(["ai-move", "hint", "game-analysis"] as const)(
    "blocks undo/redo/goToMove while a %s search is active",
    (kind) => {
      store.setState({ engineActivity: activity(kind) });
      const lengthBefore = store.getState().moveHistory.length;

      store.getState().undoMove();
      store.getState().goToMove(0);

      expect(store.getState().moveHistory.length).toBe(lengthBefore);
    },
  );

  it("allows navigation when the engine is idle", () => {
    store.setState({ engineActivity: activity("idle") });
    expect(store.getState().moveHistory.length).toBe(1);

    store.getState().undoMove();

    expect(store.getState().moveHistory.length).toBe(0);
  });

  it("does not block on the separate solver activity", () => {
    store.setState({ engineActivity: activity("solver") });

    store.getState().undoMove();

    expect(store.getState().moveHistory.length).toBe(0);
  });

  it("cancels the pending auto-play step on a real navigation but not when blocked", () => {
    const cancelAutomation = vi.fn();
    store.setState({ cancelAutomation });

    store.setState({ engineActivity: activity("ai-move") });
    store.getState().undoMove();
    expect(cancelAutomation).not.toHaveBeenCalled();

    store.setState({ engineActivity: activity("idle") });
    store.getState().undoMove();
    expect(cancelAutomation).toHaveBeenCalledTimes(1);
  });
});
