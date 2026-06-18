import type { Services } from "@/services/types";
import type { Board, Player } from "@/domain/game/types";
import { initializeBoard } from "@/domain/game/game-logic";
import { cloneBoard, createGameStartState } from "@/domain/game/store-helpers";
import { idleEngineActivityPatch } from "@/stores/engine-activity";
import {
  createNewGamePatch,
  persistNewGameSettings,
  resolveNewGameSettings,
} from "@/stores/new-game";
import type { NewGameSettings, ReversiState, SetState, SolverConfig } from "./slices/types";

type SolverStarter = (board: Board, player: Player) => Promise<void>;

export type GameReplacementTarget =
  | { kind: "new-game"; settings?: NewGameSettings }
  | { kind: "setup-game"; settings?: NewGameSettings }
  | {
      kind: "solver-position";
      board: Board;
      player: Player;
      config?: SolverConfig;
      startSolver: SolverStarter;
    }
  | { kind: "setup-solver"; config?: SolverConfig; startSolver: SolverStarter };

/**
 * Game Replacement (CONTEXT.md → Game Replacement): the shared choreography
 * for swapping the live game/position for a new one. It is its own seam — the
 * game, setup, and solver slices all depend on it as a peer, instead of two
 * of them reaching into the third slice's module for these internals.
 *
 * The public interface is a transaction over a target workflow. Callers name
 * the target; this module owns preparation, install, failure restore, Solver
 * session survival, New Game Settings persistence, setup errors, and
 * Automation resume/trigger rules.
 */

export async function runGameReplacement(
  services: Services,
  get: () => ReversiState,
  set: SetState,
  target: GameReplacementTarget,
): Promise<boolean> {
  switch (target.kind) {
    case "new-game":
      return replaceWithGame(services, get, set, {
        settings: target.settings,
        position: { board: initializeBoard(), currentPlayer: "black" },
      });

    case "setup-game": {
      const setup = resolveSetupForReplacement(get, set);
      if (!setup) {
        return false;
      }

      const replaced = await replaceWithGame(services, get, set, {
        settings: target.settings,
        position: { board: cloneBoard(setup.board), currentPlayer: setup.currentPlayer },
      });
      set({ setupError: replaced ? null : "aiInitFailed" });
      return replaced;
    }

    case "solver-position":
      return replaceWithSolver(services, get, set, {
        board: target.board,
        player: target.player,
        config: target.config,
        startSolver: target.startSolver,
      });

    case "setup-solver": {
      const setup = resolveSetupForReplacement(get, set);
      if (!setup) {
        return false;
      }

      const replaced = await replaceWithSolver(services, get, set, {
        board: setup.board,
        player: setup.currentPlayer,
        config: target.config,
        startSolver: target.startSolver,
      });
      set({ setupError: replaced ? null : "aiInitFailed" });
      return replaced;
    }
  }
}

/**
 * Resolve the validated setup position both setup targets need, recording the
 * validation error and bailing when invalid. The single guard shared by the
 * `setup-game` and `setup-solver` targets, so the resolve-or-fail rule lives in
 * one place instead of being repeated per arm.
 */
function resolveSetupForReplacement(
  get: () => ReversiState,
  set: SetState,
): { board: Board; currentPlayer: Player } | null {
  const resolved = get().resolveValidSetup();
  if (!resolved.ok) {
    set({ setupError: resolved.error });
    return null;
  }
  return { board: resolved.board, currentPlayer: resolved.currentPlayer };
}

/**
 * Abort the in-game Engine Searches in flight (ai-move / hint via
 * `abortAIMove`, game-analysis via `abortGameAnalysis`) — exactly the
 * `isGameSearchActive` set. The solver is the fourth Engine Search but is
 * *not* an in-game search; freeing it is part of the replacement preparation
 * step (it only matters when the backend is re-initialised), so a reset that
 * does not re-init does not touch it. The AI-move countdown timer is cancelled
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
 * Internal preparation step for replacing the live game: verify AI readiness,
 * free the shared engine of every in-flight Engine Search, and re-initialise
 * it. Returns whether the replacement target may now install the new position.
 *
 * "Free the shared engine" covers all four Engine Searches: the in-game ones
 * (`abortInFlightGameSearches`) and the solver. The solver contends for the
 * same backend engine/mutex (CONTEXT.md → Engine Search), so `services.ai
 * .initialize()` below would deadlock on it if a solver search were still
 * running. Aborting it here is what lets every target enter through
 * `runGameReplacement` without re-deriving the deadlock constraint. Solver
 * *session state* is cleared only after a successful Game install, so a failed
 * init leaves the user's solver session intact.
 *
 * On failure it restores the prior activity exactly — resuming a superseded
 * game analysis (and queueing the deferred automation step) or otherwise
 * re-triggering automation — and restores `paused`, so a failed replacement
 * leaves the current game intact.
 */
async function prepareReplacementBackend(
  services: Services,
  get: () => ReversiState,
  set: SetState,
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

async function replaceWithGame(
  services: Services,
  get: () => ReversiState,
  set: SetState,
  target: {
    settings?: NewGameSettings;
    position: { board: Board; currentPlayer: Player };
  },
): Promise<boolean> {
  const settings = resolveNewGameSettings(get(), target.settings);
  const wasSolverActive = get().isSolverActive;

  if (!(await prepareReplacementBackend(services, get, set))) {
    return false;
  }

  // A Game Replacement that installs a Game leaves Solver mode, but only on
  // the success path. Failed backend init preserves the user's Solver session.
  if (wasSolverActive) {
    await get().exitSolver();
  }

  set(createNewGamePatch(settings, target.position));
  persistNewGameSettings(services, settings);
  get().triggerAutomation();
  return true;
}

async function replaceWithSolver(
  services: Services,
  get: () => ReversiState,
  set: SetState,
  target: {
    board: Board;
    player: Player;
    config?: SolverConfig;
    startSolver: SolverStarter;
  },
): Promise<boolean> {
  if (!(await prepareReplacementBackend(services, get, set))) {
    return false;
  }

  installWaitingGameShell(get, set);

  if (target.config) {
    commitSolverConfig(services, set, target.config);
  }

  set({ isSolverModalOpen: false });
  await target.startSolver(target.board, target.player);
  return true;
}

function installWaitingGameShell(get: () => ReversiState, set: SetState): void {
  const board = initializeBoard();
  set({
    ...createGameStartState(board, "black", "waiting", get().gameTimeLimit * 1000),
    ...idleEngineActivityPatch(),
  });
}

function commitSolverConfig(services: Services, set: SetState, config: SolverConfig): void {
  set({
    targetSelectivity: config.selectivity,
    solverMode: config.mode,
  });
  void services.settings.saveSetting("solverTargetSelectivity", config.selectivity);
  void services.settings.saveSetting("solverMode", config.mode);
}
