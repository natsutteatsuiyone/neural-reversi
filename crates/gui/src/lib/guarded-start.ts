/**
 * The single-flight async-start choreography shared by the New Game and
 * Solver modals: a start action returning whether it started, guarded against
 * re-entry (double submit), reporting errors, and flipping a busy flag — with
 * the React/unmount details injected so this core stays pure and testable.
 *
 * The two modals are its adapters. Before this seam, New Game owned the full
 * guard while Solver hand-rolled a thinner version with no re-entry guard,
 * so a fast double-click could launch two solver replacements.
 */
export interface GuardedStartGate {
  /** True while a start is already in flight (synchronous re-entry guard). */
  isBusy(): boolean;
  /** Flip the busy flag. Its implementation may no-op the `false` write when
   *  the caller has unmounted, so a React state setter is never called late. */
  setBusy(busy: boolean): void;
}

export interface GuardedStartHandlers {
  /** Run when the start action resolved `true`. */
  onStarted?: () => void;
  /** Run when the start action threw. */
  onError: (error: unknown) => void;
}

/**
 * Run `start` at most once concurrently. If already busy, do nothing.
 * Otherwise mark busy, await `start()`, dispatch `onStarted`/`onError`, and
 * always clear busy in `finally`.
 */
export async function runGuardedStart(
  gate: GuardedStartGate,
  start: () => Promise<boolean>,
  handlers: GuardedStartHandlers,
): Promise<void> {
  if (gate.isBusy()) return;
  gate.setBusy(true);
  try {
    const started = await start();
    if (started) handlers.onStarted?.();
  } catch (error) {
    handlers.onError(error);
  } finally {
    gate.setBusy(false);
  }
}
