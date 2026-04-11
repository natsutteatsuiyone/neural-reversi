import { vi } from "vitest";
import type { SolverService } from "./types";

export function createMockSolverService(overrides?: Partial<SolverService>): SolverService {
  return {
    startSearch: vi.fn().mockResolvedValue(undefined),
    abort: vi.fn().mockResolvedValue(undefined),
    onProgress: vi.fn().mockResolvedValue(() => {}),
    ...overrides,
  };
}
