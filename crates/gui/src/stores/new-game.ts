import type { Board, Player } from "@/domain/game/types";
import { createGameStartState } from "@/domain/game/store-helpers";
import { IDLE_ENGINE_ACTIVITY } from "@/stores/engine-activity";
import type { NewGameSettings, ReversiState } from "./slices/types";

/**
 * Resolve the settings a new game starts with: each field falls back to the
 * store's current value when the caller leaves it unspecified.
 *
 * The single expression of "which fields define a new game and how an unset
 * override falls back" (CONTEXT.md → New Game Settings). Every starter (new
 * game, setup position) resolves through here, so a fifth setting cannot be
 * added to one starter and silently forgotten in another.
 */
export function resolveNewGameSettings(
  defaults: NewGameSettings,
  overrides?: NewGameSettings,
): NewGameSettings {
  return {
    gameMode: overrides?.gameMode ?? defaults.gameMode,
    aiLevel: overrides?.aiLevel ?? defaults.aiLevel,
    aiMode: overrides?.aiMode ?? defaults.aiMode,
    gameTimeLimit: overrides?.gameTimeLimit ?? defaults.gameTimeLimit,
  };
}

/**
 * Build the full store patch for a new game, including idle Engine Activity.
 */
export function createNewGamePatch(
  settings: NewGameSettings,
  position: { board: Board; currentPlayer: Player },
): Partial<ReversiState> {
  return {
    ...createGameStartState(
      position.board,
      position.currentPlayer,
      "playing",
      settings.gameTimeLimit * 1000,
    ),
    engineActivity: IDLE_ENGINE_ACTIVITY,
    gameMode: settings.gameMode,
    aiLevel: settings.aiLevel,
    aiMode: settings.aiMode,
    gameTimeLimit: settings.gameTimeLimit,
  };
}
