import { createReversiStore } from "@/stores/use-reversi-store";
import { createMockAIService } from "@/services/mock-ai-service";
import { createMockSettingsService } from "@/services/mock-settings-service";
import type { Services } from "@/services/types";

export function createTestStore(overrides?: Partial<Services>) {
  const services: Services = {
    ai: createMockAIService(),
    settings: createMockSettingsService(),
    ...overrides,
  };
  return { store: createReversiStore(services), services };
}

export type TestStore = ReturnType<typeof createTestStore>["store"];
