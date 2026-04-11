import { createReversiStore } from "@/stores/use-reversi-store";
import { createMockAIService } from "@/services/mock-ai-service";
import { createMockSettingsService } from "@/services/mock-settings-service";
import { createMockSolverService } from "@/services/mock-solver-service";
import type { Services } from "@/services/types";

export function createTestStore(overrides?: Partial<Services>) {
  const services: Services = {
    ai: createMockAIService(),
    settings: createMockSettingsService(),
    solver: createMockSolverService(),
    ...overrides,
  };
  return { store: createReversiStore(services), services };
}

export type TestStore = ReturnType<typeof createTestStore>["store"];

export function createDeferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}
