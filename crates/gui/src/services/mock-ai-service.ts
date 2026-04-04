import { vi } from "vitest";
import type { AIService } from "./types";

export function createMockAIService(overrides?: Partial<AIService>): AIService {
  return {
    getAIMove: vi.fn().mockResolvedValue(null),
    analyze: vi.fn().mockResolvedValue(undefined),
    analyzeGame: vi.fn().mockResolvedValue(undefined),
    initialize: vi.fn().mockResolvedValue(undefined),
    resizeTT: vi.fn().mockResolvedValue(undefined),
    abortSearch: vi.fn().mockResolvedValue(undefined),
    abortGameAnalysis: vi.fn().mockResolvedValue(undefined),
    ...overrides,
  };
}
