import { boardToString } from "@/domain/game/board-parser";
import { getNotation, opponentPlayer } from "@/domain/game/game-logic";
import { applyMove, checkGameOver, type Move } from "@/domain/game/store-helpers";
import type { Board, Player } from "@/domain/game/types";
import { SearchOperation, type SearchRun } from "@/services/search-operation";
import type {
  SolverCandidate,
  SolverMode,
  SolverProgressPayload,
  SolverService,
  SolverSelectivity,
} from "@/services/types";

export interface SolverHistoryEntry {
  board: Board;
  player: Player;
  moveFrom: string | null;
}

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
}

export type SolverResultCache = Map<string, Map<string, SolverCandidate>>;

export function createSolverResultCache(): SolverResultCache {
  return new Map();
}

export function createSolverRootEntry(board: Board, player: Player): SolverHistoryEntry {
  return { board, player, moveFrom: null };
}

const SOLVER_CACHE_MAX_ENTRIES = 64;

function solverCacheKey(
  board: Board,
  player: Player,
  selectivity: SolverSelectivity,
  mode: SolverMode,
): string {
  return `${boardToString(board)}:${player}:${selectivity}:${mode}`;
}

export function getCachedSolverResult(
  solverResultCache: SolverResultCache,
  board: Board,
  player: Player,
  selectivity: SolverSelectivity,
  mode: SolverMode,
): Map<string, SolverCandidate> | null {
  const key = solverCacheKey(board, player, selectivity, mode);
  const entry = solverResultCache.get(key);
  if (!entry) return null;
  solverResultCache.delete(key);
  solverResultCache.set(key, entry);
  // Fresh copy so store-side mutations cannot leak back into the cache.
  return new Map(entry);
}

export function isCompleteSolverResult(candidates: Map<string, SolverCandidate>): boolean {
  if (candidates.size === 0) return false;
  for (const candidate of candidates.values()) {
    if (!candidate.isComplete) return false;
  }
  return true;
}

export function cacheSolverResult(
  solverResultCache: SolverResultCache,
  board: Board,
  player: Player,
  selectivity: SolverSelectivity,
  mode: SolverMode,
  candidates: Map<string, SolverCandidate>,
): void {
  const key = solverCacheKey(board, player, selectivity, mode);
  solverResultCache.delete(key);
  // applySolverProgress always spreads into a fresh Map, so this ref is frozen.
  solverResultCache.set(key, candidates);
  if (solverResultCache.size > SOLVER_CACHE_MAX_ENTRIES) {
    const oldest = solverResultCache.keys().next().value;
    if (oldest !== undefined) solverResultCache.delete(oldest);
  }
}

export function cacheCompleteSolverResult(
  solverResultCache: SolverResultCache,
  board: Board,
  player: Player,
  selectivity: SolverSelectivity,
  mode: SolverMode,
  candidates: Map<string, SolverCandidate>,
): void {
  if (isCompleteSolverResult(candidates)) {
    cacheSolverResult(solverResultCache, board, player, selectivity, mode, candidates);
  }
}

interface AdvanceSolverPositionResult {
  board: Board;
  player: Player;
  entry: SolverHistoryEntry;
  gameOver: boolean;
}

export function advanceSolverPosition(
  board: Board,
  player: Player,
  row: number,
  col: number,
): AdvanceSolverPositionResult {
  const move: Move = { row, col, isAI: false, score: 0 };
  const nextBoard = applyMove(board, move, player);
  let nextPlayer = opponentPlayer(player);

  // Collapse an implicit pass into this solver step so one undo unwinds it.
  const { gameOver, shouldPass } = checkGameOver(nextBoard, nextPlayer);
  if (shouldPass) {
    nextPlayer = opponentPlayer(nextPlayer);
  }

  return {
    board: nextBoard,
    player: nextPlayer,
    entry: {
      board: nextBoard,
      player: nextPlayer,
      moveFrom: getNotation(row, col),
    },
    gameOver,
  };
}

export function applySolverProgress(
  candidates: Map<string, SolverCandidate>,
  payload: SolverProgressPayload,
  targetSelectivity: SolverSelectivity,
  mode: SolverMode,
): Map<string, SolverCandidate> {
  const isComplete = payload.isEndgame
    ? payload.acc >= targetSelectivity
    : payload.depth >= payload.targetDepth;
  const key = `${payload.row},${payload.col}`;
  const existing = candidates.get(key);

  if (
    existing &&
    existing.score === payload.score &&
    existing.depth === payload.depth &&
    existing.targetDepth === payload.targetDepth &&
    existing.acc === payload.acc &&
    existing.isEndgame === payload.isEndgame &&
    existing.isComplete === isComplete &&
    existing.pvLine === payload.pvLine
  ) {
    return candidates;
  }

  const candidate: SolverCandidate = {
    move: payload.bestMove,
    row: payload.row,
    col: payload.col,
    score: payload.score,
    depth: payload.depth,
    targetDepth: payload.targetDepth,
    acc: payload.acc,
    pvLine: payload.pvLine,
    isEndgame: payload.isEndgame,
    isComplete,
  };

  if (mode === "bestOnly") {
    return new Map([[key, candidate]]);
  }

  const next = new Map(candidates);
  next.set(key, candidate);
  return next;
}

/**
 * Owns Solver Session lifecycle: current position, navigation history,
 * candidate cache, stop/resume state, and stale Search Operation filtering.
 */
export class SolverSession {
  private readonly solver: SolverService;
  private readonly read: () => SolverSessionState;
  private readonly commit: SolverSessionCommit;
  private readonly searchOperation: SearchOperation;
  private readonly solverResultCache = createSolverResultCache();

  constructor({ solver, read, commit }: SolverSessionOptions) {
    this.solver = solver;
    this.read = read;
    this.commit = commit;
    this.searchOperation = new SearchOperation();
  }

  async subscribeProgress(): Promise<() => void> {
    return this.solver.onProgress((payload) => {
      this.applyProgress(payload);
    });
  }

  async start(board: Board, player: Player): Promise<void> {
    const rootEntry = createSolverRootEntry(board, player);
    const run = this.searchOperation.startRun(() => this.commit({
      isSolverActive: true,
      solverRootBoard: board,
      solverRootPlayer: player,
      solverCurrentBoard: board,
      solverCurrentPlayer: player,
      solverHistory: [rootEntry],
      solverCandidates: new Map<string, SolverCandidate>(),
      isSolverSearching: true,
      isSolverStopped: false,
    }));

    await this.runSearch(board, player, run);
  }

  async exit(): Promise<void> {
    await this.solver.abort();
    this.searchOperation.invalidate(() => this.commit({
      isSolverActive: false,
      solverRootBoard: null,
      solverRootPlayer: null,
      solverHistory: [],
      solverCurrentBoard: null,
      solverCurrentPlayer: null,
      solverCandidates: new Map<string, SolverCandidate>(),
      isSolverSearching: false,
      isSolverStopped: false,
    }));
  }

  async advance(row: number, col: number): Promise<void> {
    const initial = this.read();
    if (!initial.solverCurrentBoard || !initial.solverCurrentPlayer) {
      return;
    }

    // Claim the run id before awaiting abort so racing navigation reads the
    // latest breadcrumb state and stale queued progress is filtered.
    const run = this.searchOperation.startRun(() => this.commit({
      isSolverSearching: true,
      isSolverStopped: false,
    }));

    await this.solver.abort();

    if (!run.isCurrent()) {
      return;
    }

    const current = this.read();
    const currentBoard = current.solverCurrentBoard;
    const currentPlayer = current.solverCurrentPlayer;
    if (!currentBoard || !currentPlayer) {
      return;
    }

    const nextPosition = advanceSolverPosition(currentBoard, currentPlayer, row, col);

    if (nextPosition.gameOver) {
      this.commit((state) => ({
        solverHistory: [...state.solverHistory, nextPosition.entry],
        solverCurrentBoard: nextPosition.board,
        solverCurrentPlayer: nextPosition.player,
        solverCandidates: new Map<string, SolverCandidate>(),
        isSolverSearching: false,
      }));
      return;
    }

    const { targetSelectivity, solverMode } = this.read();
    const cached = getCachedSolverResult(
      this.solverResultCache,
      nextPosition.board,
      nextPosition.player,
      targetSelectivity,
      solverMode,
    );

    if (cached) {
      this.commit((state) => ({
        solverHistory: [...state.solverHistory, nextPosition.entry],
        solverCurrentBoard: nextPosition.board,
        solverCurrentPlayer: nextPosition.player,
        solverCandidates: cached,
        isSolverSearching: false,
        isSolverStopped: false,
      }));
      return;
    }

    this.commit((state) => ({
      solverHistory: [...state.solverHistory, nextPosition.entry],
      solverCurrentBoard: nextPosition.board,
      solverCurrentPlayer: nextPosition.player,
      solverCandidates: new Map<string, SolverCandidate>(),
    }));

    await this.runSearch(nextPosition.board, nextPosition.player, run, targetSelectivity, solverMode);
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

    await this.solver.abort();

    this.searchOperation.invalidate(() => this.commit({
      isSolverSearching: false,
      isSolverStopped: true,
    }));
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

    const cached = getCachedSolverResult(
      this.solverResultCache,
      board,
      player,
      state.targetSelectivity,
      state.solverMode,
    );

    if (cached) {
      this.searchOperation.invalidate(() => this.commit({
        solverCandidates: cached,
        isSolverSearching: false,
        isSolverStopped: false,
      }));
      return;
    }

    const run = this.searchOperation.startRun(() => this.commit({
      isSolverSearching: true,
      isSolverStopped: false,
    }));

    await this.runSearch(board, player, run, state.targetSelectivity, state.solverMode);
  }

  applyProgress(payload: SolverProgressPayload): void {
    const state = this.read();
    if (!state.isSolverActive) {
      return;
    }

    if (!this.searchOperation.accepts(payload.runId)) {
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
    const cached = getCachedSolverResult(this.solverResultCache, board, player, selectivity, mode);
    // Commit the new destination before aborting so a second navigation starts
    // from this updated position instead of from a stale pre-abort snapshot.
    const run = this.searchOperation.startRun(() => this.commit({
      ...extra,
      solverCandidates: cached ?? new Map<string, SolverCandidate>(),
      isSolverSearching: !cached,
      isSolverStopped: false,
    }));

    await this.solver.abort();

    if (cached || !run.isCurrent()) {
      return;
    }

    await this.runSearch(board, player, run, selectivity, mode);
  }

  private async runSearch(
    board: Board,
    player: Player,
    run: SearchRun,
    selectivity = this.read().targetSelectivity,
    mode = this.read().solverMode,
  ): Promise<void> {
    await this.searchOperation.runCurrent({
      run,
      start: () => this.solver.startSearch(board, player, selectivity, mode, run.id),
      onError: (error) => {
        console.error("Failed to start solver search:", error);
      },
      onCurrentFinally: () => {
        cacheCompleteSolverResult(
          this.solverResultCache,
          board,
          player,
          selectivity,
          mode,
          this.read().solverCandidates,
        );
        this.commit({ isSolverSearching: false });
      },
    });
  }
}
