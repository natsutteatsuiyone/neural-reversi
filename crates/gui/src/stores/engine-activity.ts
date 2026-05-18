import type { EngineActivity } from "@/domain/engine/engine-search";
import type { ReversiState } from "./slices/types";

/**
 * Whether the engine is running an in-game search — every Engine Activity
 * kind except `idle` and the separate `solver` mode. The game's automation
 * and history-navigation guards block while one of these is in flight; this
 * is the single predicate they share (CONTEXT.md → Engine Activity), so a new
 * kind only has to be classified here, not at every guard.
 */
export function isGameSearchActive(activity: EngineActivity): boolean {
  return (
    activity.kind === "ai-move" ||
    activity.kind === "hint" ||
    activity.kind === "game-analysis"
  );
}

/**
 * The single mapping from an Engine Activity to a store patch (CONTEXT.md →
 * Engine Activity). The four feature "busy" booleans are views of
 * `engineActivity.kind` — they are added here, one feature at a time, as each
 * feature stops writing its own flag and deletes its generation counter.
 *
 * All four feature "busy" booleans are now views of `engineActivity.kind`;
 * nothing else writes them.
 */
export function engineActivityPatch(
  activity: EngineActivity,
): Partial<ReversiState> {
  return {
    engineActivity: activity,
    isAIThinking: activity.kind === "ai-move",
    isAnalyzing: activity.kind === "hint",
    isGameAnalyzing: activity.kind === "game-analysis",
    isSolverSearching: activity.kind === "solver",
  };
}

/**
 * The Engine Activity of a store with nothing in flight: the initial state
 * and every freshly-(re)started game/position. The single idle sentinel —
 * `runId` 0 is informational only (nothing reads `engineActivity.runId`;
 * guards read `.kind`), matching the store's cold-start value.
 */
export const IDLE_ENGINE_ACTIVITY: EngineActivity = { kind: "idle", runId: 0 };

/**
 * The idle Engine Activity projection. A "fresh game state" builder spreads
 * this instead of re-deriving the kind→busy-booleans mapping by hand, so the
 * mapping (and a future fifth kind) lives in exactly one place — closing the
 * desync where a hand-listed reset forgets a boolean (CONTEXT.md → Engine
 * Activity).
 */
export function idleEngineActivityPatch(): Partial<ReversiState> {
  return engineActivityPatch(IDLE_ENGINE_ACTIVITY);
}
