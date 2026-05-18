import { boardToString } from "@/domain/game/board-parser";
import type { Board, Player } from "@/domain/game/types";
import type { SolverCandidate, SolverMode, SolverSelectivity } from "@/services/types";
import { isCompleteSolverResult } from "./solver-candidates";

const SOLVER_CACHE_MAX_ENTRIES = 64;

/**
 * LRU cache of *completed* solver results, keyed by
 * (board, player, selectivity, mode).
 *
 * The interface is two methods; everything below the seam is hidden:
 *
 * - The key format never escapes — callers pass the position, not a string.
 * - {@link get} touches the entry's recency and returns a fresh copy, so
 *   store-side mutation of a returned map cannot leak back into the cache.
 * - {@link storeIfComplete} is the only writer and is a no-op for an
 *   incomplete/empty result, so a partial search can never poison a key.
 *
 * Producers (`applySolverProgress`) always spread into a fresh Map, so the
 * stored ref is frozen and `get()`'s defensive copy is the only mutation
 * guard a caller needs.
 */
export class SolverResultCache {
  private readonly entries = new Map<string, Map<string, SolverCandidate>>();

  get(
    board: Board,
    player: Player,
    selectivity: SolverSelectivity,
    mode: SolverMode,
  ): Map<string, SolverCandidate> | null {
    const key = SolverResultCache.key(board, player, selectivity, mode);
    const entry = this.entries.get(key);
    if (!entry) return null;
    // LRU touch: re-insert so this key becomes the most-recently-used.
    this.entries.delete(key);
    this.entries.set(key, entry);
    return new Map(entry);
  }

  /** Store only when every candidate is complete; a no-op otherwise. */
  storeIfComplete(
    board: Board,
    player: Player,
    selectivity: SolverSelectivity,
    mode: SolverMode,
    candidates: Map<string, SolverCandidate>,
  ): void {
    if (!isCompleteSolverResult(candidates)) return;
    const key = SolverResultCache.key(board, player, selectivity, mode);
    this.entries.delete(key);
    this.entries.set(key, candidates);
    if (this.entries.size > SOLVER_CACHE_MAX_ENTRIES) {
      const oldest = this.entries.keys().next().value;
      if (oldest !== undefined) this.entries.delete(oldest);
    }
  }

  private static key(
    board: Board,
    player: Player,
    selectivity: SolverSelectivity,
    mode: SolverMode,
  ): string {
    return `${boardToString(board)}:${player}:${selectivity}:${mode}`;
  }
}
