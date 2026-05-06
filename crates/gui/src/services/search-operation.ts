export interface SearchRun {
  readonly id: number;
  isCurrent: () => boolean;
}

interface SearchOperationOptions {
  initialRunId?: number;
}

interface RunCurrentOptions {
  run: SearchRun;
  start: () => Promise<void>;
  onError?: (error: unknown) => void;
  onCurrentFinally: () => void;
}

interface StartCurrentOptions<TArgs extends unknown[]> {
  commitStart?: (run: SearchRun) => void;
  start: (acceptProgress: (...args: TArgs) => void) => Promise<void>;
  onProgress: (...args: TArgs) => void;
  onError?: (error: unknown) => void;
  onCurrentFinally: () => void;
}

interface AbortCurrentOptions {
  run: SearchRun;
  abort: () => Promise<void>;
  onError?: (error: unknown) => void;
  onCurrentFinally: () => void;
}

interface AbortLatestOptions {
  commitAbort?: (run: SearchRun) => void;
  abort: () => Promise<void>;
  onError?: (error: unknown) => void;
  onCurrentFinally: () => void;
}

/**
 * Owns the frontend run-id mechanics shared by engine-backed searches.
 *
 * Slices still decide how accepted progress changes state, but this module
 * keeps stale-run filtering, invalidation, and current-only cleanup local.
 */
export class SearchOperation {
  private currentRunId: number;

  constructor({ initialRunId = 0 }: SearchOperationOptions = {}) {
    this.currentRunId = initialRunId;
  }

  createRun(id: number): SearchRun {
    return {
      id,
      isCurrent: () => this.isCurrent(id),
    };
  }

  nextRun(): SearchRun {
    return this.createRun(this.currentRunId + 1);
  }

  startRun(onStart?: (run: SearchRun) => void): SearchRun {
    const run = this.nextRun();
    this.currentRunId = run.id;
    onStart?.(run);
    return run;
  }

  invalidate(onInvalidate?: (run: SearchRun) => void): SearchRun {
    return this.startRun(onInvalidate);
  }

  isCurrent(runId: number): boolean {
    return this.currentRunId === runId;
  }

  accepts(runId: number): boolean {
    return this.isCurrent(runId);
  }

  currentOnly<TArgs extends unknown[]>(
    run: SearchRun,
    callback: (...args: TArgs) => void,
  ): (...args: TArgs) => void {
    return (...args: TArgs) => {
      if (run.isCurrent()) {
        callback(...args);
      }
    };
  }

  finishCurrent(run: SearchRun, cleanup: () => void): void {
    if (run.isCurrent()) {
      cleanup();
    }
  }

  async runCurrent({
    run,
    start,
    onError,
    onCurrentFinally,
  }: RunCurrentOptions): Promise<void> {
    try {
      await start();
    } catch (error) {
      onError?.(error);
    } finally {
      this.finishCurrent(run, onCurrentFinally);
    }
  }

  async startCurrent<TArgs extends unknown[]>({
    commitStart,
    start,
    onProgress,
    onError,
    onCurrentFinally,
  }: StartCurrentOptions<TArgs>): Promise<void> {
    const run = this.startRun(commitStart);
    const acceptProgress = this.currentOnly(run, onProgress);
    await this.runCurrent({
      run,
      start: () => start(acceptProgress),
      onError,
      onCurrentFinally,
    });
  }

  async abortCurrent({
    run,
    abort,
    onError,
    onCurrentFinally,
  }: AbortCurrentOptions): Promise<void> {
    try {
      await abort();
    } catch (error) {
      onError?.(error);
    } finally {
      this.finishCurrent(run, onCurrentFinally);
    }
  }

  async abortLatest({
    commitAbort,
    abort,
    onError,
    onCurrentFinally,
  }: AbortLatestOptions): Promise<void> {
    const run = this.invalidate(commitAbort);
    await this.abortCurrent({
      run,
      abort,
      onError,
      onCurrentFinally,
    });
  }
}
