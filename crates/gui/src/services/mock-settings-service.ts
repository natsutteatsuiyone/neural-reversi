import { vi } from "vitest";
import { DEFAULT_SETTINGS, type SettingsService } from "./types";

export function createMockSettingsService(overrides?: Partial<SettingsService>): SettingsService {
  return {
    loadSettings: vi.fn().mockResolvedValue(DEFAULT_SETTINGS),
    saveSetting: vi.fn().mockResolvedValue(true),
    ...overrides,
  };
}
