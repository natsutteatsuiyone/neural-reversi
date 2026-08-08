import type { EngineActivity } from "@/domain/engine/engine-search";

/**
 * Whether the engine is running an in-game search — every Engine Activity
 * kind except `idle` and the separate `solver` mode. The game's automation
 * and history-navigation guards block while one of these is in flight; this
 * is the single predicate they share (CONTEXT.md → Engine Activity), so a new
 * kind only has to be classified here, not at every guard.
 */
export function isGameSearchActive(activity: EngineActivity): boolean {
  return (
    activity.kind === "ai-move" || activity.kind === "hint" || activity.kind === "game-analysis"
  );
}

/**
 * The Engine Activity of a store with nothing in flight: the initial state
 * and every freshly-(re)started game/position. `runId` 0 is informational;
 * guards read only `.kind`.
 */
export const IDLE_ENGINE_ACTIVITY: EngineActivity = { kind: "idle", runId: 0 };
