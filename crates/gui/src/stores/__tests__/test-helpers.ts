import { createReversiStore } from "@/stores/use-reversi-store";
import { createMockAIService } from "@/services/mock-ai-service";
import { createMockSolverService } from "@/services/mock-solver-service";
import type { Services } from "@/services/types";
import type { StateStorage } from "zustand/middleware";

export function createMemoryStorage(): StateStorage {
  const values = new Map<string, string>();
  return {
    getItem: (name) => values.get(name) ?? null,
    setItem: (name, value) => {
      values.set(name, value);
    },
    removeItem: (name) => {
      values.delete(name);
    },
  };
}

export function createTestStore(
  overrides?: Partial<Services>,
  storage: StateStorage = createMemoryStorage(),
) {
  const services: Services = {
    ai: createMockAIService(),
    solver: createMockSolverService(),
    ...overrides,
  };
  return { store: createReversiStore(services, storage), services, storage };
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
