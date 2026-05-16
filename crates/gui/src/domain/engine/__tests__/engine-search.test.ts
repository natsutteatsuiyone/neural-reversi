import { describe, expect, it, vi } from "vitest";
import { createEngineSearch } from "@/domain/engine/engine-search";

function deferred<T>() {
  let resolve!: (v: T | PromiseLike<T>) => void;
  let reject!: (e?: unknown) => void;
  const promise = new Promise<T>((res, rej) => { resolve = res; reject = rej; });
  return { promise, resolve, reject };
}

describe("EngineSearch", () => {
  it("S1: accepts only the current monotonically increasing generation", async () => {
    const es = createEngineSearch({ initialRunId: 3 });
    let firstId = 0;
    await es.start({
      onStart: (r) => { firstId = r.id; },
      run: async () => {},
      abort: async () => {},
    });
    expect(firstId).toBe(4);
    expect(es.accepts(4)).toBe(true);
    expect(es.accepts(3)).toBe(false);
  });

  it("S2: ok path runs onStart, onResult, then onTeardown(ok)", async () => {
    const es = createEngineSearch();
    const order: string[] = [];
    await es.start<never, string>({
      onStart: () => order.push("start"),
      run: async () => "R",
      abort: async () => {},
      onResult: (r) => order.push(`result:${r}`),
      onTeardown: (o) => order.push(`teardown:${o.status}`),
    });
    expect(order).toEqual(["start", "result:R", "teardown:ok"]);
  });

  it("S3: progress is delivered only while current", async () => {
    const es = createEngineSearch();
    const seen: number[] = [];
    const d = deferred<void>();
    const first = es.start<number, void>({
      run: async (accept) => { accept(1); await d.promise; accept(2); },
      abort: async () => {},
      onProgress: (p) => seen.push(p),
    });
    await Promise.resolve();
    await es.start({ run: async () => {}, abort: async () => {} }); // supersede
    d.resolve();
    await first;
    expect(seen).toEqual([1]);
  });

  it("S4: supersede aborts+tears-down prior exactly once before new onStart", async () => {
    const es = createEngineSearch();
    const order: string[] = [];
    const d = deferred<void>();
    const first = es.start({
      run: async () => { await d.promise; },
      abort: async () => { order.push("abort-1"); },
      onTeardown: (o) => order.push(`teardown-1:${o.status}`),
    });
    await Promise.resolve();
    await es.start({
      onStart: () => order.push("start-2"),
      run: async () => {},
      abort: async () => {},
      onTeardown: () => order.push("teardown-2:ok"),
    });
    d.resolve();
    await first;
    expect(order).toEqual([
      "abort-1",
      "teardown-1:superseded",
      "start-2",
      "teardown-2:ok",
    ]);
  });

  it("S5: superseded run's late resolution does not re-fire its callbacks", async () => {
    const es = createEngineSearch();
    const teardown = vi.fn();
    const onResult = vi.fn();
    const d = deferred<string>();
    const first = es.start<never, string>({
      run: () => d.promise,
      abort: async () => {},
      onResult,
      onTeardown: teardown,
    });
    await Promise.resolve();
    await es.start({ run: async () => {}, abort: async () => {} });
    d.resolve("late");
    await first;
    expect(onResult).not.toHaveBeenCalled();
    expect(teardown).toHaveBeenCalledTimes(1);
    expect(teardown).toHaveBeenCalledWith({ status: "superseded" });
  });

  it("S6: run rejection routes onError then onTeardown(error); start resolves", async () => {
    const es = createEngineSearch();
    const order: string[] = [];
    const err = new Error("boom");
    await expect(
      es.start({
        run: async () => { throw err; },
        abort: async () => {},
        onError: (e) => order.push(`error:${(e as Error).message}`),
        onTeardown: (o) => order.push(`teardown:${o.status}`),
      }),
    ).resolves.toBeUndefined();
    expect(order).toEqual(["error:boom", "teardown:error"]);
  });

  it("S7: abort supersedes prior, runs onAbort, abort, onSettled once", async () => {
    const es = createEngineSearch();
    const order: string[] = [];
    const d = deferred<void>();
    const first = es.start({
      run: async () => { await d.promise; },
      abort: async () => {},
      onTeardown: (o) => order.push(`teardown-1:${o.status}`),
    });
    await Promise.resolve();
    await es.abort({
      onAbort: () => order.push("onAbort"),
      abort: async () => order.push("backend-abort"),
      onSettled: () => order.push("settled"),
    });
    d.resolve();
    await first;
    expect(order).toEqual([
      "teardown-1:superseded",
      "onAbort",
      "backend-abort",
      "settled",
    ]);
  });

  it("S8: abort backend rejection routes onError; onSettled still runs", async () => {
    const es = createEngineSearch();
    const order: string[] = [];
    await es.abort({
      abort: async () => { throw new Error("x"); },
      onError: () => order.push("error"),
      onSettled: () => order.push("settled"),
    });
    expect(order).toEqual(["error", "settled"]);
  });

  it("S9: a superseded run whose abort rejects does not poison the superseding run", async () => {
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    try {
      const es = createEngineSearch();
      const order: string[] = [];
      const d = deferred<void>();
      const first = es.start({
        run: async () => { await d.promise; },
        abort: async () => { throw new Error("prior abort failed"); },
        onTeardown: (o) => order.push(`teardown-1:${o.status}`),
      });
      await Promise.resolve();
      await expect(
        es.start({
          onStart: () => order.push("start-2"),
          run: async () => {},
          abort: async () => {},
          onTeardown: () => order.push("teardown-2:ok"),
        }),
      ).resolves.toBeUndefined();
      d.resolve();
      await first;
      expect(order).toEqual([
        "teardown-1:superseded",
        "start-2",
        "teardown-2:ok",
      ]);
      expect(errorSpy).toHaveBeenCalled();
    } finally {
      errorSpy.mockRestore();
    }
  });

  it("S10: a superseded run stops delivering progress immediately, before its slow abort resolves", async () => {
    const es = createEngineSearch();
    const seen: number[] = [];
    const runGate = deferred<void>();
    const abortGate = deferred<void>();
    const first = es.start<number, void>({
      run: async (accept) => { accept(1); await runGate.promise; accept(2); },
      abort: () => abortGate.promise, // slow abort: keeps supersede() awaiting
      onProgress: (p) => seen.push(p),
    });
    await Promise.resolve();
    const second = es.start({ run: async () => {}, abort: async () => {} });
    runGate.resolve();
    await Promise.resolve();
    abortGate.resolve();
    await second;
    await first;
    expect(seen).toEqual([1]);
  });

  it("S11: onClaim runs synchronously before supersede/onStart/run", () => {
    const es = createEngineSearch();
    const order: string[] = [];
    void es.start({
      onClaim: () => order.push("claim"),
      onStart: () => order.push("start"),
      run: async () => { order.push("run"); },
      abort: async () => {},
    });
    expect(order).toEqual(["claim"]);
  });

  it("S12: accepts(priorId) is false synchronously on supersede, before the slow abort resolves", async () => {
    const es = createEngineSearch();
    const abortGate = deferred<void>();
    let priorId = 0;
    const first = es.start<never, void>({
      onStart: (r) => { priorId = r.id; },
      run: async () => { await abortGate.promise; },
      abort: () => abortGate.promise, // slow abort: keeps supersede() awaiting
    });
    await Promise.resolve();
    expect(es.accepts(priorId)).toBe(true);
    const second = es.start({ run: async () => {}, abort: async () => {} });
    expect(es.accepts(priorId)).toBe(false);
    abortGate.resolve();
    await second;
    await first;
  });

  it("S14: abort superseded mid-flight skips onSettled but still runs onTeardown once", async () => {
    const es = createEngineSearch();
    const order: string[] = [];
    const priorAbort = deferred<void>();
    // A live run whose abort is slow, so the abort() below stalls in claim().
    const first = es.start({
      run: async () => { await priorAbort.promise; },
      abort: () => priorAbort.promise,
    });
    await Promise.resolve();

    const aborted = es.abort({
      onAbort: () => order.push("onAbort"),
      abort: async () => order.push("backend-abort"),
      onSettled: () => order.push("settled"),
      onTeardown: () => order.push("teardown"),
    });
    await Promise.resolve();
    // A newer start supersedes the in-flight abort during its slow prior abort.
    const newer = es.start({ run: async () => {}, abort: async () => {} });
    priorAbort.resolve();
    await Promise.all([first, aborted, newer]);

    expect(order).toContain("teardown");
    expect(order.filter((e) => e === "teardown")).toHaveLength(1);
    expect(order).not.toContain("settled");
    expect(order).not.toContain("onAbort");
    expect(order).not.toContain("backend-abort");
  });

  it("S14: abort superseded during its own slow backend abort skips onSettled, still runs onTeardown once", async () => {
    const es = createEngineSearch();
    const order: string[] = [];
    const backendAbort = deferred<void>();
    // Nothing live → claim()'s `superseded` resolves false, so abort proceeds
    // past the bail and into its slow `spec.abort()`.
    const aborted = es.abort({
      onAbort: () => order.push("onAbort"),
      abort: () => backendAbort.promise, // slow backend abort
      onSettled: () => order.push("settled"),
      onTeardown: () => order.push("teardown"),
    });
    await Promise.resolve();
    // A newer start claims while the backend abort is still in flight.
    const newer = es.start({ run: async () => {}, abort: async () => {} });
    backendAbort.resolve();
    await Promise.all([aborted, newer]);

    expect(order).toEqual(["onAbort", "teardown"]);
    expect(order).not.toContain("settled");
  });

  it("S14: a normally-settled abort runs onTeardown exactly once, after onSettled", async () => {
    const es = createEngineSearch();
    const order: string[] = [];
    await es.abort({
      onAbort: () => order.push("onAbort"),
      abort: async () => order.push("backend-abort"),
      onSettled: () => order.push("settled"),
      onTeardown: () => order.push("teardown"),
    });
    expect(order).toEqual(["onAbort", "backend-abort", "settled", "teardown"]);
  });

  it("S13: a slow prior abort cannot let an older start overtake a newer one", async () => {
    const es = createEngineSearch();
    const order: string[] = [];
    const l0Abort = deferred<void>();

    // L0: the initial live run; its abort is slow so the next claim's
    // teardown await is held open.
    const l0 = es.start({
      onStart: (r) => order.push(`L0-start:${r.id}`),
      run: async () => { await l0Abort.promise; },
      abort: () => l0Abort.promise, // slow abort
      onTeardown: (o) => order.push(`L0-teardown:${o.status}`),
    });
    await Promise.resolve();

    // A: its claim must abort L0; it stalls on L0's slow abort.
    let aId = 0;
    const a = es.start({
      onStart: (r) => { aId = r.id; order.push(`A-start:${r.id}`); },
      run: async () => {},
      abort: async () => {},
      onResult: () => order.push("A-result"),
      onTeardown: (o) => order.push(`A-teardown:${o.status}`),
    });
    await Promise.resolve();

    // B: starts while A is still blocked on L0's slow abort.
    let bId = 0;
    const b = es.start({
      onStart: (r) => { bId = r.id; order.push(`B-start:${r.id}`); },
      run: async () => {},
      abort: async () => {},
      onResult: () => order.push("B-result"),
      onTeardown: (o) => order.push(`B-teardown:${o.status}`),
    });

    l0Abort.resolve();
    await Promise.all([l0, a, b]);

    // B was issued after A, so B must remain the current run; the older A
    // must not overtake it once its slow prior abort finally resolves.
    expect(bId).toBeGreaterThan(aId);
    expect(es.accepts(bId)).toBe(true);
    expect(es.accepts(aId)).toBe(false);
    // L0 is torn down exactly once, as superseded.
    expect(order.filter((e) => e.startsWith("L0-teardown"))).toEqual([
      "L0-teardown:superseded",
    ]);
    // The overtaken older A neither delivers a result nor an ok teardown,
    // but it IS torn down exactly once as superseded.
    expect(order).not.toContain("A-result");
    expect(order).not.toContain("A-teardown:ok");
    expect(order.filter((e) => e.startsWith("A-teardown"))).toEqual([
      "A-teardown:superseded",
    ]);
  });

  it("S15: a claimed start superseded before install still runs onTeardown to undo onClaim", async () => {
    const es = createEngineSearch();
    const order: string[] = [];
    const l0Abort = deferred<void>();

    // L0: initial live run with a slow abort so the next claim's teardown
    // await is held open long enough for a newer op to overtake.
    const l0 = es.start({
      onStart: () => order.push("L0-start"),
      run: async () => { await l0Abort.promise; },
      abort: () => l0Abort.promise, // slow abort
      onTeardown: (o) => order.push(`L0-teardown:${o.status}`),
    });
    await Promise.resolve();

    // A: commits a synchronous breadcrumb in onClaim (like analyzeGame's
    // `isGameAnalyzing: true`). Its claim stalls on L0's slow abort.
    const a = es.start({
      onClaim: () => order.push("A-claim"),
      onStart: () => order.push("A-start"),
      run: async () => { order.push("A-run"); },
      abort: async () => {},
      onResult: () => order.push("A-result"),
      onTeardown: (o) => order.push(`A-teardown:${o.status}`),
    });
    await Promise.resolve();

    // B: a newer op overtakes A while A is still blocked on L0's slow abort.
    const b = es.start({
      onStart: () => order.push("B-start"),
      run: async () => {},
      abort: async () => {},
      onTeardown: (o) => order.push(`B-teardown:${o.status}`),
    });

    l0Abort.resolve();
    await Promise.all([l0, a, b]);

    // A's onClaim mutated state, then A was superseded before it could
    // install — it must NOT start/run/resolve, but its onTeardown MUST
    // still run (as superseded) so the onClaim breadcrumb is undone.
    expect(order).toContain("A-claim");
    expect(order).not.toContain("A-start");
    expect(order).not.toContain("A-run");
    expect(order).not.toContain("A-result");
    expect(order.filter((e) => e.startsWith("A-teardown"))).toEqual([
      "A-teardown:superseded",
    ]);
  });
});
