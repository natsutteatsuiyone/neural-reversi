/**
 * The trail of recent cells the AI search has been probing, shown on the
 * board while it thinks. Presentation view-state (not store state): the pure
 * reducer lives here so the ring-buffer rule is tested directly instead of
 * being trapped in a `useBoardScene` effect. The hook owns the React
 * `useState` and the `useEffect` that feeds progress in; this owns *what the
 * next trail is*.
 */

export interface AIProgressTrailCell {
  row: number;
  col: number;
  timestamp: number;
}

export const AI_PROGRESS_TRAIL_SIZE = 3;

/** Stable empty trail — a shared reference so non-AI render paths don't churn. */
export const EMPTY_AI_PROGRESS_TRAIL: AIProgressTrailCell[] = [];

/**
 * The next trail given the current one and the latest progress `cell`:
 *
 * - no `cell` and still the AI's turn → unchanged (same reference);
 * - no `cell` and no longer the AI's turn → cleared;
 * - `cell` repeats the head → unchanged (same reference);
 * - a new `cell` → prepended, capped at {@link AI_PROGRESS_TRAIL_SIZE}.
 *
 * Returning the same reference for the no-change cases lets React skip the
 * re-render, matching the prior inline effect.
 */
export function nextAIProgressTrail(
  prev: AIProgressTrailCell[],
  cell: { row: number; col: number } | null,
  isAITurn: boolean,
  now: number,
): AIProgressTrailCell[] {
  if (!cell) {
    return isAITurn ? prev : EMPTY_AI_PROGRESS_TRAIL;
  }
  if (prev.length > 0 && prev[0].row === cell.row && prev[0].col === cell.col) {
    return prev;
  }
  return [{ row: cell.row, col: cell.col, timestamp: now }, ...prev].slice(
    0,
    AI_PROGRESS_TRAIL_SIZE,
  );
}
