export type RunId = number;

/**
 * What the single Engine Search is currently doing (CONTEXT.md → Engine
 * Activity). A projection of the current run id, owned here so no feature
 * keeps its own generation counter.
 */
export type EngineActivityKind = "idle" | "ai-move" | "hint" | "game-analysis" | "solver";

export interface EngineActivity {
  readonly kind: EngineActivityKind;
  /** The run id whose claim produced this activity. */
  readonly runId: RunId;
}

export type RunOutcome =
  | { status: "ok" }
  | { status: "error"; error: unknown }
  | { status: "superseded" };

export interface RunHandle {
  readonly id: RunId;
  isCurrent(): boolean;
}

export interface StartSpec<P, R> {
  /** Which Engine Activity this run represents. Set synchronously at claim
   *  (before `await supersede()`) so a rapidly-following guard reads it, and
   *  returned to `idle` on this run's teardown — but only while it is still
   *  the current run, so a superseding run's activity is never clobbered.
   *  Omit only for the engine's own contract tests. */
  kind?: EngineActivityKind;
  /** Optional SYNCHRONOUS hook run at call time, before `await supersede()` /
   *  onStart/run — for callers that must commit breadcrumb state synchronously
   *  so a rapidly-following call reads it. onClaim MUST NOT throw — it runs
   *  before `await supersede()`, so a throw would skip the prior run's teardown
   *  and reject `start`, violating the "resolve, never reject" contract. */
  onClaim?: () => void;
  onStart?: (run: RunHandle) => void;
  run: (accept: (progress: P) => void, run: RunHandle) => Promise<R>;
  abort: () => Promise<unknown>;
  onProgress?: (progress: P) => void;
  onResult?: (result: R, run: RunHandle) => void;
  onError?: (error: unknown) => void;
  /** Guaranteed-once terminal hook: runs exactly once on every path —
   *  ok, error, or superseded. Both superseded entry points fire it: a
   *  run superseded *after* install (via `claim()`'s `prior.teardown`)
   *  and one superseded *before* install while still queued behind a
   *  slow prior abort (the post-await bail). A synchronous breadcrumb
   *  committed by the caller in `onClaim` must be undone here, since
   *  `onStart`/`run` may never run on the superseded path. (Spec S13/S15.) */
  onTeardown?: (outcome: RunOutcome) => void;
}

export interface AbortSpec {
  /** Optional SYNCHRONOUS hook run at call time, before `await supersede()` /
   *  onAbort — for callers that must commit breadcrumb state synchronously so
   *  a rapidly-following call reads it. onClaim MUST NOT throw — it runs
   *  before `await supersede()`, so a throw would skip the prior run's teardown
   *  and reject `abort`, violating the "resolve, never reject" contract. */
  onClaim?: () => void;
  onAbort?: () => void;
  abort: () => Promise<unknown>;
  onError?: (error: unknown) => void;
  onSettled?: () => void;
  /** Guaranteed-once terminal hook: runs exactly once on every path —
   *  whether this abort settles as current OR is superseded by a newer
   *  start/abort during its (possibly slow) prior abort. `onSettled` only
   *  runs on the settled path; a synchronous breadcrumb committed by the
   *  caller before `abort()` (e.g. an abort-pending guard) must be cleared
   *  here, not in `onSettled`, or a superseding start would strand it. */
  onTeardown?: () => void;
}

export interface EngineSearch {
  start<P, R>(spec: StartSpec<P, R>): Promise<void>;
  abort(spec: AbortSpec): Promise<void>;
  accepts(id: RunId): boolean;
}

interface LiveRun {
  readonly id: RunId;
  abort: () => Promise<unknown>;
  teardown: (outcome: RunOutcome) => void;
  settled: boolean;
}

/**
 * The single backend engine's current search generation (CONTEXT.md →
 * Engine Search). At most one run is live; `start`/`abort` supersede the
 * prior run — its `abort()` then `onTeardown` run exactly once, before the
 * new run's `onStart`. Progress is gated to the current generation; a
 * superseded run's late resolution and progress are dropped. `start` and
 * `abort` resolve (never reject).
 */
export function createEngineSearch(
  opts: {
    initialRunId?: RunId;
    /** Notified on every Engine Activity transition. The store mirrors this
     *  into `engineActivity`; the four feature "busy" booleans are views of
     *  `engineActivity.kind` (CONTEXT.md → Engine Activity). */
    onActivityChange?: (activity: EngineActivity) => void;
  } = {},
): EngineSearch {
  let currentId: RunId = opts.initialRunId ?? 0;
  let live: LiveRun | null = null;

  const isCurrent = (id: RunId) => id === currentId;

  /**
   * Emit an Engine Activity transition. Claim-time emits are ordered by
   * invocation (claim() bumps `currentId` synchronously), so the last claim
   * wins — matching the supersede ordering. A run only returns the activity
   * to `idle` while it is still current, so a superseding run's activity is
   * never clobbered by a slow predecessor's teardown.
   */
  const emitActivity = (kind: EngineActivityKind, id: RunId) =>
    opts.onActivityChange?.({ kind, runId: id });

  /**
   * Take this operation's generation. SYNCHRONOUS up to the returned
   * `superseded` promise: it bumps `currentId` and detaches the prior live
   * run before yielding, so concurrent `start`/`abort` calls are ordered by
   * invocation, not by when their (possibly slow) prior abort resolves.
   *
   * - S12: `accepts(prior.id)` is false immediately — the prior generation
   *   is invalidated before awaiting the backend abort, so the solver's
   *   id-queried `solver-progress` path drops late prior-run progress.
   * - S9:  a rejection from the prior run's `abort()` is swallowed (logged)
   *   so it can never cancel the superseding op; `onTeardown({superseded})`
   *   still runs via `finally`, exactly once.
   * - S13: `superseded` resolves to `true` when a newer `start`/`abort`
   *   claimed a higher generation while we awaited the slow prior abort.
   *   The caller must then bail without installing — otherwise the older,
   *   later-resuming op would overwrite the newer run and become current.
   */
  function claim(): { id: RunId; superseded: Promise<boolean> } {
    const prior = live;
    if (prior) prior.settled = true;
    live = null;
    const id = ++currentId;
    const superseded = (async () => {
      if (prior) {
        try {
          await prior.abort();
        } catch (error) {
          console.error("EngineSearch: superseded search abort failed:", error);
        } finally {
          prior.teardown({ status: "superseded" });
        }
      }
      return !isCurrent(id);
    })();
    return { id, superseded };
  }

  async function start<P, R>(spec: StartSpec<P, R>): Promise<void> {
    spec.onClaim?.();
    const { id, superseded } = claim();
    // Stamp this run's activity synchronously at claim (ordered by invocation,
    // last claim wins) so a rapidly-following guard reads it. On the superseded
    // bail below we do NOT return to idle: the superseding run already claimed
    // and stamped its own activity after us, and clobbering it would strand a
    // stale `idle`. (Spec S13/S15.)
    if (spec.kind) emitActivity(spec.kind, id);
    // A newer start/abort claimed a higher generation while we awaited the
    // prior abort; bail without installing. We never installed a record nor
    // ran `spec.run`, and the prior we captured was already torn down inside
    // `claim()`. But `onClaim` may have synchronously committed caller state
    // (e.g. analyzeGame's `isGameAnalyzing: true`); the guaranteed-once
    // `onTeardown` MUST still run — as superseded — to undo it, exactly as
    // the `abort()` superseded path does. (Spec S13/S15.)
    if (await superseded) {
      spec.onTeardown?.({ status: "superseded" });
      return;
    }
    const handle: RunHandle = { id, isCurrent: () => isCurrent(id) };
    const record: LiveRun = {
      id,
      abort: spec.abort,
      teardown: (o) => spec.onTeardown?.(o),
      settled: false,
    };
    live = record;
    spec.onStart?.(handle);

    const accept = (p: P) => {
      // Gate on `record.settled` too, not just `isCurrent(id)`: a run that
      // finalizes *normally* stays the current generation, so late progress
      // emitted by the backend after `spec.run` resolved would otherwise fire
      // after onResult/onTeardown. `settled` is also set synchronously by a
      // superseding claim(), closing the supersede window as well. (Spec S10.)
      if (!record.settled && isCurrent(id)) spec.onProgress?.(p);
    };

    let result: R;
    try {
      result = await spec.run(accept, handle);
    } catch (error) {
      if (record.settled) return;
      record.settled = true;
      if (live === record) live = null;
      spec.onError?.(error);
      if (spec.kind && isCurrent(id)) emitActivity("idle", id);
      spec.onTeardown?.({ status: "error", error });
      return;
    }
    if (record.settled) return;
    record.settled = true;
    if (live === record) live = null;
    spec.onResult?.(result, handle);
    if (spec.kind && isCurrent(id)) emitActivity("idle", id);
    spec.onTeardown?.({ status: "ok" });
  }

  async function abort(spec: AbortSpec): Promise<void> {
    spec.onClaim?.();
    const { id, superseded } = claim();
    // An abort returns the engine to idle. Stamped synchronously at claim
    // (ordered by invocation); if a newer start supersedes this abort, its
    // later claim re-stamps its own kind and wins. On the superseded bail we
    // do not re-emit, mirroring `start`. (S13/S14.)
    emitActivity("idle", id);
    // Superseded by a newer start/abort during the prior abort: the newer op
    // governs the lifecycle now, so skip onAbort/abort/onSettled. The prior
    // run we captured was already aborted+torn down inside `claim()`. The
    // guaranteed-once onTeardown still runs so a caller breadcrumb is not
    // stranded by the supersede. (S13/S14.)
    if (await superseded) {
      spec.onTeardown?.();
      return;
    }
    spec.onAbort?.();
    try {
      await spec.abort();
    } catch (error) {
      spec.onError?.(error);
    }
    if (isCurrent(id)) spec.onSettled?.();
    spec.onTeardown?.();
  }

  return {
    start,
    abort,
    accepts: (id) => isCurrent(id),
  };
}
