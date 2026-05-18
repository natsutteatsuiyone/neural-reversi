import { describe, expect, it, vi } from "vitest";
import { runGuardedStart, type GuardedStartGate } from "@/lib/guarded-start";

function makeGate(initialBusy = false): GuardedStartGate & { busy: boolean } {
  const gate = {
    busy: initialBusy,
    isBusy() {
      return gate.busy;
    },
    setBusy(busy: boolean) {
      gate.busy = busy;
    },
  };
  return gate;
}

describe("runGuardedStart", () => {
  it("does nothing when already busy (re-entry guard)", async () => {
    const gate = makeGate(true);
    const start = vi.fn().mockResolvedValue(true);
    const onStarted = vi.fn();
    const onError = vi.fn();

    await runGuardedStart(gate, start, { onStarted, onError });

    expect(start).not.toHaveBeenCalled();
    expect(onStarted).not.toHaveBeenCalled();
    expect(onError).not.toHaveBeenCalled();
  });

  it("marks busy, runs onStarted, then clears busy when start resolves true", async () => {
    const gate = makeGate();
    const seenBusyDuringStart = vi.fn();
    const start = vi.fn().mockImplementation(async () => {
      seenBusyDuringStart(gate.busy);
      return true;
    });
    const onStarted = vi.fn();
    const onError = vi.fn();

    await runGuardedStart(gate, start, { onStarted, onError });

    expect(seenBusyDuringStart).toHaveBeenCalledWith(true);
    expect(onStarted).toHaveBeenCalledTimes(1);
    expect(onError).not.toHaveBeenCalled();
    expect(gate.busy).toBe(false);
  });

  it("does not run onStarted when start resolves false", async () => {
    const gate = makeGate();
    const onStarted = vi.fn();
    const onError = vi.fn();

    await runGuardedStart(gate, vi.fn().mockResolvedValue(false), {
      onStarted,
      onError,
    });

    expect(onStarted).not.toHaveBeenCalled();
    expect(onError).not.toHaveBeenCalled();
    expect(gate.busy).toBe(false);
  });

  it("routes a thrown error to onError and still clears busy", async () => {
    const gate = makeGate();
    const error = new Error("boom");
    const onStarted = vi.fn();
    const onError = vi.fn();

    await runGuardedStart(gate, vi.fn().mockRejectedValue(error), {
      onStarted,
      onError,
    });

    expect(onError).toHaveBeenCalledWith(error);
    expect(onStarted).not.toHaveBeenCalled();
    expect(gate.busy).toBe(false);
  });

  it("serialises overlapping calls: the second is dropped while the first runs", async () => {
    const gate = makeGate();
    let resolveFirst!: (v: boolean) => void;
    const first = vi.fn().mockImplementation(
      () => new Promise<boolean>((res) => { resolveFirst = res; }),
    );
    const second = vi.fn().mockResolvedValue(true);
    const onError = vi.fn();

    const p1 = runGuardedStart(gate, first, { onError });
    // second click while first is in flight
    await runGuardedStart(gate, second, { onError });
    expect(second).not.toHaveBeenCalled();

    resolveFirst(true);
    await p1;
    expect(first).toHaveBeenCalledTimes(1);
    expect(gate.busy).toBe(false);
  });
});
