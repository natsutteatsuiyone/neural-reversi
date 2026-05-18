import type { Board, Player } from "@/domain/game/types";
import type { EngineSearch, RunOutcome } from "@/domain/engine/engine-search";
import type {
  SolverCandidate,
  SolverMode,
  SolverProgressPayload,
  SolverService,
  SolverSelectivity,
} from "@/services/types";
import { applySolverProgress } from "./solver-candidates";
import {
  advanceSolverPosition,
  createSolverRootEntry,
  type SolverHistoryEntry,
} from "./solver-navigation";
import { SolverResultCache } from "./solver-result-cache";

export interface SolverSessionState {
  isSolverActive: boolean;
  solverRootBoard: Board | null;
  solverRootPlayer: Player | null;
  solverHistory: SolverHistoryEntry[];
  solverCurrentBoard: Board | null;
  solverCurrentPlayer: Player | null;
  targetSelectivity: SolverSelectivity;
  solverMode: SolverMode;
  solverCandidates: Map<string, SolverCandidate>;
  isSolverSearching: boolean;
  isSolverStopped: boolean;
}

export type SolverSessionPatch = Partial<SolverSessionState>;

export type SolverSessionCommit = (
  partial:
    | SolverSessionPatch
    | ((state: SolverSessionState) => SolverSessionPatch),
) => void;

interface SolverSessionOptions {
  solver: SolverService;
  read: () => SolverSessionState;
  commit: SolverSessionCommit;
  engineSearch: EngineSearch;
}

/**
 * Owns Solver Session lifecycle: current position, navigation history,
 * candidate cache, stop/resume state, and stale Engine Search filtering.
 * Pure position/candidate/cache logic lives behind `solver-navigation`,
 * `solver-candidates`, and `solver-result-cache`; this class is the
 * Engine Search choreography only.
 */
export class SolverSession {
  private readonly solver: SolverService;
  private readonly read: () => SolverSessionState;
  private readonly commit: SolverSessionCommit;
  private readonly engineSearch: EngineSearch;
  private readonly cache = new SolverResultCache();

  constructor({ solver, read, commit, engineSearch }: SolverSessionOptions) {
    this.solver = solver;
    this.read = read;
    this.commit = commit;
    this.engineSearch = engineSearch;
  }

  async subscribeProgress(): Promise<() => void> {
    return this.solver.onProgress((payload) => {
      this.applyProgress(payload);
    });
  }

  async start(board: Board, player: Player): Promise<void> {
    const rootEntry = createSolverRootEntry(board, player);
    const { targetSelectivity, solverMode } = this.read();
    await this.runSearch(
      board,
      player,
      () => this.commit({
        isSolverActive: true,
        solverRootBoard: board,
        solverRootPlayer: player,
        solverCurrentBoard: board,
        solverCurrentPlayer: player,
        solverHistory: [rootEntry],
        solverCandidates: new Map<string, SolverCandidate>(),
        isSolverStopped: false,
      }),
      targetSelectivity,
      solverMode,
    );
  }

  async exit(): Promise<void> {
    await this.engineSearch.abort({
      onClaim: () => this.commit({
        isSolverActive: false,
        solverRootBoard: null,
        solverRootPlayer: null,
        solverHistory: [],
        solverCurrentBoard: null,
        solverCurrentPlayer: null,
        solverCandidates: new Map<string, SolverCandidate>(),
        isSolverStopped: false,
      }),
      abort: () => this.solver.abort(),
    });
  }

  async advance(row: number, col: number): Promise<void> {
    const current = this.read();
    const currentBoard = current.solverCurrentBoard;
    const currentPlayer = current.solverCurrentPlayer;
    if (!currentBoard || !currentPlayer) {
      return;
    }

    const nextPosition = advanceSolverPosition(currentBoard, currentPlayer, row, col);

    if (nextPosition.gameOver) {
      // Game-over: no new search, but still supersede + abort the prior run so
      // its stale progress is filtered and the backend is stopped before the
      // final committed candidates land. Commit the breadcrumb synchronously
      // (onClaim) so a rapidly-following navigation reads this position.
      await this.engineSearch.abort({
        onClaim: () => this.commit((state) => ({
          solverHistory: [...state.solverHistory, nextPosition.entry],
          solverCurrentBoard: nextPosition.board,
          solverCurrentPlayer: nextPosition.player,
          solverCandidates: new Map<string, SolverCandidate>(),
          isSolverStopped: false,
        })),
        abort: () => this.solver.abort(),
      });
      return;
    }

    const { targetSelectivity, solverMode } = current;
    const cached = this.cache.get(
      nextPosition.board,
      nextPosition.player,
      targetSelectivity,
      solverMode,
    );

    if (cached) {
      // Cache-hit: no new search, but still supersede + abort the prior run so
      // its late progress cannot overwrite the committed cached candidates.
      await this.engineSearch.abort({
        onClaim: () => this.commit((state) => ({
          solverHistory: [...state.solverHistory, nextPosition.entry],
          solverCurrentBoard: nextPosition.board,
          solverCurrentPlayer: nextPosition.player,
          solverCandidates: cached,
          isSolverStopped: false,
        })),
        abort: () => this.solver.abort(),
      });
      return;
    }

    await this.runSearch(
      nextPosition.board,
      nextPosition.player,
      () => this.commit((state) => ({
        solverHistory: [...state.solverHistory, nextPosition.entry],
        solverCurrentBoard: nextPosition.board,
        solverCurrentPlayer: nextPosition.player,
        solverCandidates: new Map<string, SolverCandidate>(),
        isSolverStopped: false,
      })),
      targetSelectivity,
      solverMode,
    );
  }

  async undo(): Promise<void> {
    const initial = this.read();
    if (initial.solverHistory.length <= 1) {
      return;
    }

    const newHistory = initial.solverHistory.slice(0, -1);
    const prevEntry = newHistory[newHistory.length - 1];

    await this.repoint(prevEntry.board, prevEntry.player, initial.targetSelectivity, initial.solverMode, {
      solverHistory: newHistory,
      solverCurrentBoard: prevEntry.board,
      solverCurrentPlayer: prevEntry.player,
    });
  }

  async repointCurrent(): Promise<void> {
    const state = this.read();
    const board = state.solverCurrentBoard;
    const player = state.solverCurrentPlayer;
    if (!state.isSolverActive || !board || !player) {
      return;
    }

    await this.repoint(board, player, state.targetSelectivity, state.solverMode);
  }

  async stop(): Promise<void> {
    const state = this.read();
    if (!state.isSolverActive || !state.isSolverSearching) {
      return;
    }

    await this.engineSearch.abort({
      onClaim: () => this.commit({
        isSolverStopped: true,
      }),
      abort: () => this.solver.abort(),
    });
  }

  async resume(): Promise<void> {
    const state = this.read();
    if (!state.isSolverActive || state.isSolverSearching || !state.isSolverStopped) {
      return;
    }

    const board = state.solverCurrentBoard;
    const player = state.solverCurrentPlayer;
    if (!board || !player) {
      return;
    }

    const cached = this.cache.get(
      board,
      player,
      state.targetSelectivity,
      state.solverMode,
    );

    if (cached) {
      // Cache-hit: no new search, but still supersede + abort the prior run so
      // its late progress cannot overwrite the committed cached candidates.
      await this.engineSearch.abort({
        onClaim: () => this.commit({
          solverCandidates: cached,
          isSolverStopped: false,
        }),
        abort: () => this.solver.abort(),
      });
      return;
    }

    await this.runSearch(
      board,
      player,
      () => this.commit({
        isSolverStopped: false,
      }),
      state.targetSelectivity,
      state.solverMode,
    );
  }

  applyProgress(payload: SolverProgressPayload): void {
    const state = this.read();
    if (!state.isSolverActive) {
      return;
    }

    if (!this.engineSearch.accepts(payload.runId)) {
      return;
    }

    const solverCandidates = applySolverProgress(
      state.solverCandidates,
      payload,
      state.targetSelectivity,
      state.solverMode,
    );
    if (solverCandidates === state.solverCandidates) {
      return;
    }
    this.commit({ solverCandidates });
  }

  private async repoint(
    board: Board,
    player: Player,
    selectivity: SolverSelectivity,
    mode: SolverMode,
    extra: SolverSessionPatch = {},
  ): Promise<void> {
    const cached = this.cache.get(board, player, selectivity, mode);

    if (cached) {
      // Cache-hit: no new search, but still supersede + abort the prior run so
      // its late progress cannot overwrite the committed cached candidates.
      await this.engineSearch.abort({
        onClaim: () => this.commit({
          ...extra,
          solverCandidates: cached,
          isSolverStopped: false,
        }),
        abort: () => this.solver.abort(),
      });
      return;
    }

    await this.runSearch(
      board,
      player,
      () => this.commit({
        ...extra,
        solverCandidates: new Map<string, SolverCandidate>(),
        isSolverStopped: false,
      }),
      selectivity,
      mode,
    );
  }

  /**
   * Cache the completed result. `isSolverSearching` is no longer cleared
   * here — it is a view of the Engine Activity (CONTEXT.md → Engine
   * Activity): this run's teardown returns the activity to `idle` only while
   * it is still the current run, which is exactly the old generation check.
   */
  private solverTeardown(
    board: Board,
    player: Player,
    selectivity: SolverSelectivity,
    mode: SolverMode,
  ): (outcome: RunOutcome) => void {
    return (outcome: RunOutcome) => {
      // Cache ONLY the run that completed naturally. A superseded run's teardown
      // runs in supersede()'s finally — AFTER the superseding onClaim already
      // committed the NEW position's solverCandidates — so caching
      // this.read().solverCandidates for a superseded/error run would store the
      // wrong board's candidates under this run's (board,player) key.
      if (outcome.status === "ok") {
        this.cache.storeIfComplete(board, player, selectivity, mode, this.read().solverCandidates);
      }
    };
  }

  private async runSearch(
    board: Board,
    player: Player,
    onClaim: () => void,
    selectivity = this.read().targetSelectivity,
    mode = this.read().solverMode,
  ): Promise<void> {
    await this.engineSearch.start<never, void>({
      kind: "solver",
      // Commit the breadcrumb synchronously (onClaim) so a rapidly-following
      // navigation reads this position before `await supersede()` defers it.
      onClaim,
      // Abort the prior backend search unconditionally, then bail if a racing
      // navigation already superseded this run (faithful to the old
      // startRun + await solver.abort() + run.isCurrent() sequence).
      run: async (_accept, run) => {
        await this.solver.abort();
        if (!run.isCurrent()) {
          return;
        }
        await this.solver.startSearch(board, player, selectivity, mode, run.id);
      },
      abort: () => this.solver.abort(),
      onError: (error) => {
        console.error("Failed to start solver search:", error);
      },
      onTeardown: this.solverTeardown(board, player, selectivity, mode),
    });
  }
}
