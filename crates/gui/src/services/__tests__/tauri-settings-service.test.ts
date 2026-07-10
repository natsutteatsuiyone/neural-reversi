import { beforeEach, describe, expect, it, vi } from "vitest";
import { DEFAULT_SETTINGS } from "../types";

const { getMock, loadMock, setMock } = vi.hoisted(() => ({
  getMock: vi.fn(),
  loadMock: vi.fn(),
  setMock: vi.fn(),
}));

vi.mock("@tauri-apps/plugin-store", () => ({
  load: loadMock,
}));

import { TauriSettingsService } from "../tauri-settings-service";

async function loadSettings(stored: Record<string, unknown>) {
  getMock.mockImplementation((key: string) => Promise.resolve(stored[key]));
  return new TauriSettingsService().loadSettings();
}

describe("TauriSettingsService", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    loadMock.mockResolvedValue({ get: getMock, set: setMock });
  });

  it("validates engine levels against 0..=30", async () => {
    const outOfRange = await loadSettings({
      aiLevel: -1,
      hintLevel: 31,
      gameAnalysisLevel: Number.POSITIVE_INFINITY,
    });
    expect(outOfRange.aiLevel).toBe(DEFAULT_SETTINGS.aiLevel);
    expect(outOfRange.hintLevel).toBe(DEFAULT_SETTINGS.hintLevel);
    expect(outOfRange.gameAnalysisLevel).toBe(DEFAULT_SETTINGS.gameAnalysisLevel);

    const wrongType = await loadSettings({ aiLevel: "10", hintLevel: null, gameAnalysisLevel: {} });
    expect(wrongType.aiLevel).toBe(DEFAULT_SETTINGS.aiLevel);
    expect(wrongType.hintLevel).toBe(DEFAULT_SETTINGS.hintLevel);
    expect(wrongType.gameAnalysisLevel).toBe(DEFAULT_SETTINGS.gameAnalysisLevel);

    const valid = await loadSettings({ aiLevel: 0, hintLevel: 30, gameAnalysisLevel: 12 });
    expect(valid.aiLevel).toBe(0);
    expect(valid.hintLevel).toBe(30);
    expect(valid.gameAnalysisLevel).toBe(12);
  });

  it("validates time limits as positive finite integers", async () => {
    const invalid = await loadSettings({ timeLimit: 0, gameTimeLimit: 1.5 });
    expect(invalid.timeLimit).toBe(DEFAULT_SETTINGS.timeLimit);
    expect(invalid.gameTimeLimit).toBe(DEFAULT_SETTINGS.gameTimeLimit);

    const wrongType = await loadSettings({ timeLimit: "5", gameTimeLimit: null });
    expect(wrongType.timeLimit).toBe(DEFAULT_SETTINGS.timeLimit);
    expect(wrongType.gameTimeLimit).toBe(DEFAULT_SETTINGS.gameTimeLimit);

    const valid = await loadSettings({ timeLimit: 5, gameTimeLimit: 300 });
    expect(valid.timeLimit).toBe(5);
    expect(valid.gameTimeLimit).toBe(300);
  });

  it("validates hash size against 1..=16384", async () => {
    expect((await loadSettings({ hashSize: 0 })).hashSize).toBe(DEFAULT_SETTINGS.hashSize);
    expect((await loadSettings({ hashSize: 16385 })).hashSize).toBe(DEFAULT_SETTINGS.hashSize);
    expect((await loadSettings({ hashSize: "1024" })).hashSize).toBe(DEFAULT_SETTINGS.hashSize);
    expect((await loadSettings({ hashSize: 1024 })).hashSize).toBe(1024);
  });

  it("validates game and AI modes against their literal unions", async () => {
    const unsupported = await loadSettings({ gameMode: "online", aiMode: "instant" });
    expect(unsupported.gameMode).toBe(DEFAULT_SETTINGS.gameMode);
    expect(unsupported.aiMode).toBe(DEFAULT_SETTINGS.aiMode);

    const wrongType = await loadSettings({ gameMode: 1, aiMode: null });
    expect(wrongType.gameMode).toBe(DEFAULT_SETTINGS.gameMode);
    expect(wrongType.aiMode).toBe(DEFAULT_SETTINGS.aiMode);

    const valid = await loadSettings({ gameMode: "pvp", aiMode: "time" });
    expect(valid.gameMode).toBe("pvp");
    expect(valid.aiMode).toBe("time");
  });
});
