import { describe, expect, it, vi } from "vitest";
import { SearchOperation } from "@/services/search-operation";

describe("SearchOperation", () => {
  it("creates the next run id from its private generation counter", () => {
    const operation = new SearchOperation({ initialRunId: 3 });

    const next = operation.nextRun();
    expect(next.id).toBe(4);
    expect(next.isCurrent()).toBe(false);

    const started = operation.startRun();
    expect(started.id).toBe(4);
    expect(started.isCurrent()).toBe(true);
    expect(operation.accepts(started.id)).toBe(true);
    expect(operation.accepts(started.id - 1)).toBe(false);
  });

  it("runs current cleanup only for the current run", async () => {
    const operation = new SearchOperation({ initialRunId: 1 });
    const cleanup = vi.fn();

    await operation.runCurrent({
      run: operation.createRun(1),
      start: async () => {
        operation.invalidate();
      },
      onCurrentFinally: cleanup,
    });

    expect(cleanup).not.toHaveBeenCalled();
  });

  it("starts a run and passes the run to the start callback", () => {
    const operation = new SearchOperation({ initialRunId: 3 });
    const onStart = vi.fn();

    const run = operation.startRun(onStart);

    expect(run.id).toBe(4);
    expect(run.isCurrent()).toBe(true);
    expect(onStart).toHaveBeenCalledWith(run);
  });

  it("gates callbacks to the current run", () => {
    const operation = new SearchOperation({ initialRunId: 1 });
    const callback = vi.fn();
    const run = operation.createRun(1);
    const currentOnly = operation.currentOnly(run, callback);

    currentOnly("accepted");
    operation.invalidate();
    currentOnly("stale");

    expect(callback).toHaveBeenCalledTimes(1);
    expect(callback).toHaveBeenCalledWith("accepted");
  });

  it("reports errors and still cleans up current runs", async () => {
    const operation = new SearchOperation({ initialRunId: 1 });
    const onError = vi.fn();
    const cleanup = vi.fn();
    const error = new Error("boom");

    await operation.runCurrent({
      run: operation.createRun(1),
      start: async () => {
        throw error;
      },
      onError,
      onCurrentFinally: cleanup,
    });

    expect(onError).toHaveBeenCalledWith(error);
    expect(cleanup).toHaveBeenCalledTimes(1);
  });

  it("starts a lifecycle run with current-only progress", async () => {
    const operation = new SearchOperation({ initialRunId: 4 });
    let committedRunId = 0;
    const progress = vi.fn();
    const cleanup = vi.fn();

    await operation.startCurrent<[string]>({
      commitStart: (run) => {
        committedRunId = run.id;
      },
      start: async (acceptProgress) => {
        acceptProgress("accepted");
        operation.invalidate();
        acceptProgress("stale");
      },
      onProgress: progress,
      onCurrentFinally: cleanup,
    });

    expect(committedRunId).toBe(5);
    expect(progress).toHaveBeenCalledTimes(1);
    expect(progress).toHaveBeenCalledWith("accepted");
    expect(cleanup).not.toHaveBeenCalled();
  });

  it("runs abort cleanup only while the abort run is current", async () => {
    const operation = new SearchOperation({ initialRunId: 1 });
    const cleanup = vi.fn();

    await operation.abortCurrent({
      run: operation.createRun(1),
      abort: async () => {
        operation.invalidate();
      },
      onCurrentFinally: cleanup,
    });

    expect(cleanup).not.toHaveBeenCalled();
  });

  it("invalidates and aborts the latest run in one lifecycle step", async () => {
    const operation = new SearchOperation({ initialRunId: 2 });
    const abort = vi.fn().mockResolvedValue(undefined);
    const cleanup = vi.fn();
    const commitAbort = vi.fn();

    await operation.abortLatest({
      commitAbort,
      abort,
      onCurrentFinally: cleanup,
    });

    expect(commitAbort).toHaveBeenCalledWith(expect.objectContaining({ id: 3 }));
    expect(abort).toHaveBeenCalledTimes(1);
    expect(cleanup).toHaveBeenCalledTimes(1);
  });
});
