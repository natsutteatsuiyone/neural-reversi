import { cellKey, type CellKey } from "@/domain/game/cell-key";
import type { AIMoveProgress } from "@/domain/game/types";
import type {
  SolverCandidate,
  SolverMode,
  SolverProgressPayload,
  SolverSelectivity,
} from "@/services/types";

/**
 * Pure transforms over a solver candidate map keyed by {@link cellKey}
 * (CONTEXT.md → Cell Key). No Engine Search / store coupling: a candidate
 * map is the only thing in scope here, so every rule below is unit-testable
 * in isolation.
 */

export function isCompleteSolverResult(candidates: Map<CellKey, SolverCandidate>): boolean {
  if (candidates.size === 0) return false;
  for (const candidate of candidates.values()) {
    if (!candidate.isComplete) return false;
  }
  return true;
}

export function applySolverProgress(
  candidates: Map<CellKey, SolverCandidate>,
  payload: SolverProgressPayload,
  targetSelectivity: SolverSelectivity,
  mode: SolverMode,
): Map<CellKey, SolverCandidate> {
  const isComplete = payload.isEndgame
    ? payload.acc >= targetSelectivity
    : payload.depth >= payload.targetDepth;
  const key = cellKey(payload.row, payload.col);
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
 * Project solver candidates into the board's analysis-overlay shape
 * ({@link AIMoveProgress}), which the renderer shares with hint analysis.
 * Forces `acc=100` on a completed endgame candidate so non-100 target
 * selectivities still trip the overlay's "done" branch. Pure so the board
 * component stays a thin consumer and this rule is unit-testable.
 */
export function solverCandidatesToAnalysisResults(
  candidates: Map<CellKey, SolverCandidate>,
): Map<CellKey, AIMoveProgress> {
  const map = new Map<CellKey, AIMoveProgress>();
  for (const [key, candidate] of candidates) {
    map.set(key, {
      bestMove: candidate.move,
      row: candidate.row,
      col: candidate.col,
      score: candidate.score,
      depth: candidate.depth,
      targetDepth: candidate.targetDepth,
      acc: candidate.isEndgame && candidate.isComplete ? 100 : candidate.acc,
      nodes: 0,
      pvLine: candidate.pvLine,
      isEndgame: candidate.isEndgame,
    });
  }
  return map;
}
