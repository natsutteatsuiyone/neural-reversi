import type { Services } from "@/services/types";
import type { ReversiState } from "./slices/types";

/**
 * Game Replacement (CONTEXT.md → Game Replacement): the shared choreography
 * for swapping the live game/position for a new one. It is its own seam — the
 * game, setup, and solver slices all depend on it as a peer, instead of two
 * of them reaching into the third slice's module for these internals.
 */

/**
 * Abort the in-game Engine Searches in flight (ai-move / hint via
 * `abortAIMove`, game-analysis via `abortGameAnalysis`) — exactly the
 * `isGameSearchActive` set. The solver is the fourth Engine Search but is
 * *not* an in-game search; freeing it is `prepareGameReplacement`'s job (it
 * only matters when the backend is re-initialised), so a reset that does not
 * re-init does not touch it. The AI-move countdown timer is cancelled
 * synchronously inside `abortAIMove` (its teardown entry point); no external
 * timer reach-in.
 *
 * `hintAnalysisAbortPending` is part of "hint engine-active" here even though
 * `isAnalyzing` is already false: a hint abort-then-restart stamps Engine
 * Activity back to idle synchronously while its backend abort + restart are
 * still in flight. Without this, replacement would skip the abort and let the
 * pending hint teardown restart analysis on the just-reinitialised backend.
 * Routing it through `abortAIMove` supersedes that pending run via
 * EngineSearch, so its stale restart is dropped (CONTEXT.md → Engine Search).
 */
export async function abortInFlightGameSearches(get: () => ReversiState): Promise<void> {
  const state = get();
  const aborts: Promise<void>[] = [];
  if (state.isAIThinking || state.isAnalyzing || state.hintAnalysisAbortPending) {
    aborts.push(state.abortAIMove());
  }
  if (state.isGameAnalyzing) {
    aborts.push(state.abortGameAnalysis());
  }
  await Promise.all(aborts);
}

/**
 * Prepare the backend to replace the live game: verify AI readiness, free
 * the shared engine of every in-flight Engine Search, and re-initialise it.
 * Returns whether the caller may now install the new position.
 *
 * "Free the shared engine" covers all four Engine Searches: the in-game ones
 * (`abortInFlightGameSearches`) and the solver. The solver contends for the
 * same backend engine/mutex (CONTEXT.md → Engine Search), so `services.ai
 * .initialize()` below would deadlock on it if a solver search were still
 * running. Aborting it here — once, in the seam — is what lets every starter
 * (new game, setup, solver) just call `prepareGameReplacement` without
 * re-deriving the deadlock constraint. The solver's *session state* teardown
 * (`exitSolver`) stays caller-specific: `startGame` defers it to after
 * success so a failed init leaves the user's solver session intact.
 *
 * On failure it restores the prior activity exactly — resuming a superseded
 * game analysis (and queueing the deferred automation step) or otherwise
 * re-triggering automation — and restores `paused`, so a failed replacement
 * leaves the current game intact.
 */
export async function prepareGameReplacement(
  services: Services,
  get: () => ReversiState,
  set: (partial: Partial<ReversiState>) => void,
): Promise<boolean> {
  if (!(await get().checkAIReady())) {
    return false;
  }

  const shouldResumeGameAnalysis = get().isGameAnalyzing;
  const wasPaused = get().paused;

  get().cancelAutomation();
  await abortInFlightGameSearches(get);
  // Free the solver search too (it holds the same backend engine/mutex
  // re-init contends for). Idempotent: a no-op when no solver is running.
  await services.solver.abort();

  try {
    await services.ai.initialize();
    await services.ai.resizeTT(get().hashSize);
    return true;
  } catch (error) {
    console.error("Failed to prepare AI for a new position:", error);
    if (shouldResumeGameAnalysis) {
      void get().analyzeGame();
      set({ paused: wasPaused });
      get().queueResumeAutomation();
    } else {
      set({ paused: wasPaused });
      get().triggerAutomation();
    }
    return false;
  }
}
