import { describe, expect, it } from "vitest";
import {
  createNewGamePatch,
  persistNewGameSettings,
  resolveNewGameSettings,
} from "@/stores/new-game";
import { initializeBoard } from "@/domain/game/game-logic";
import { createMockSettingsService } from "@/services/mock-settings-service";
import type { Services } from "@/services/types";
import type { NewGameSettings } from "@/stores/slices/types";

const DEFAULTS: NewGameSettings = {
  gameMode: "ai-black",
  aiLevel: 10,
  aiMode: "level",
  gameTimeLimit: 30,
};

describe("resolveNewGameSettings", () => {
  it("falls back to every default when no overrides are given", () => {
    expect(resolveNewGameSettings(DEFAULTS)).toEqual(DEFAULTS);
  });

  it("takes each provided override and falls back per-field for the rest", () => {
    const overrides: NewGameSettings = {
      gameMode: "pvp",
      aiLevel: 21,
      aiMode: "game-time",
      gameTimeLimit: 60,
    };
    expect(resolveNewGameSettings(DEFAULTS, overrides)).toEqual(overrides);
  });

  it("does not mutate the defaults", () => {
    const snapshot = { ...DEFAULTS };
    resolveNewGameSettings(DEFAULTS, { ...DEFAULTS, aiLevel: 5 });
    expect(DEFAULTS).toEqual(snapshot);
  });
});

describe("createNewGamePatch", () => {
  it("spreads the resolved settings into the patch", () => {
    const patch = createNewGamePatch(DEFAULTS, {
      board: initializeBoard(),
      currentPlayer: "black",
    });
    expect(patch.gameMode).toBe("ai-black");
    expect(patch.aiLevel).toBe(10);
    expect(patch.aiMode).toBe("level");
    expect(patch.gameTimeLimit).toBe(30);
  });

  it("converts gameTimeLimit seconds to the ms clock once", () => {
    const patch = createNewGamePatch(DEFAULTS, {
      board: initializeBoard(),
      currentPlayer: "black",
    });
    // createGameStartState seeds the remaining-time clock in ms.
    expect(patch.aiRemainingTime).toBe(30_000);
  });

  it("starts a playing game at the given position with an idle Engine Activity", () => {
    const board = initializeBoard();
    const patch = createNewGamePatch(DEFAULTS, { board, currentPlayer: "white" });
    expect(patch.gameStatus).toBe("playing");
    expect(patch.currentPlayer).toBe("white");
    expect(patch.board).toBe(board);
    expect(patch.engineActivity).toEqual({ kind: "idle", runId: 0 });
    expect(patch.isAIThinking).toBe(false);
    expect(patch.isAnalyzing).toBe(false);
    expect(patch.isGameAnalyzing).toBe(false);
    expect(patch.isSolverSearching).toBe(false);
  });
});

describe("persistNewGameSettings", () => {
  it("saves exactly the four New Game Settings fields to disk", () => {
    const settings = createMockSettingsService();
    const services = { settings } as unknown as Services;

    persistNewGameSettings(services, {
      gameMode: "pvp",
      aiLevel: 7,
      aiMode: "game-time",
      gameTimeLimit: 45,
    });

    expect(settings.saveSetting).toHaveBeenCalledWith("gameMode", "pvp");
    expect(settings.saveSetting).toHaveBeenCalledWith("aiLevel", 7);
    expect(settings.saveSetting).toHaveBeenCalledWith("aiMode", "game-time");
    expect(settings.saveSetting).toHaveBeenCalledWith("gameTimeLimit", 45);
    expect(settings.saveSetting).toHaveBeenCalledTimes(4);
  });
});
