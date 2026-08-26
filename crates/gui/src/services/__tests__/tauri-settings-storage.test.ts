import { beforeEach, describe, expect, it, vi } from "vitest";

const { deleteMock, getMock, loadMock, setMock } = vi.hoisted(() => ({
  deleteMock: vi.fn(),
  getMock: vi.fn(),
  loadMock: vi.fn(),
  setMock: vi.fn(),
}));

vi.mock("@tauri-apps/plugin-store", () => ({
  load: loadMock,
}));

import { createTauriSettingsStorage } from "../tauri-settings-storage";

const STORAGE_KEY = "neural-reversi-settings";

describe("Tauri settings storage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    loadMock.mockResolvedValue({
      delete: deleteMock,
      get: getMock,
      set: setMock,
    });
  });

  it("reads and writes the Zustand payload through Tauri Store", async () => {
    const payload = JSON.stringify({ version: 2, state: { aiLevel: 12 } });
    getMock.mockResolvedValue(payload);
    const storage = createTauriSettingsStorage();

    await expect(storage.getItem(STORAGE_KEY)).resolves.toBe(payload);
    await storage.setItem(STORAGE_KEY, payload);

    expect(loadMock).toHaveBeenCalledTimes(1);
    expect(getMock).toHaveBeenCalledWith(STORAGE_KEY);
    expect(setMock).toHaveBeenCalledWith(STORAGE_KEY, payload);
  });

  it("exposes legacy per-setting values as a version-zero payload", async () => {
    const legacy = {
      gameMode: "pvp",
      aiLevel: 8,
      solverTargetSelectivity: 95,
    } as const;
    getMock.mockImplementation((key: keyof typeof legacy) => Promise.resolve(legacy[key]));
    const storage = createTauriSettingsStorage();

    const raw = await storage.getItem(STORAGE_KEY);

    expect(JSON.parse(raw as string)).toEqual({
      version: 0,
      state: {
        gameMode: "pvp",
        aiLevel: 8,
        targetSelectivity: 95,
      },
    });
  });

  it("removes legacy keys after the migrated payload is saved", async () => {
    getMock.mockImplementation((key: string) => Promise.resolve(key === "aiLevel" ? 8 : undefined));
    const storage = createTauriSettingsStorage();

    await storage.getItem(STORAGE_KEY);
    await storage.setItem(STORAGE_KEY, JSON.stringify({ version: 2, state: { aiLevel: 8 } }));

    expect(deleteMock).toHaveBeenCalledTimes(13);
    expect(deleteMock).toHaveBeenCalledWith("aiLevel");
    expect(deleteMock).toHaveBeenCalledWith("solverTargetSelectivity");
  });
});
