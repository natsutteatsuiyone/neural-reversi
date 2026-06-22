import type { SolverCandidate } from "@/services/types";

export function sortSolverCandidates(candidates: Iterable<SolverCandidate>): SolverCandidate[] {
  const sorted = Array.from(candidates);
  sorted.sort((a, b) => b.score - a.score);
  return sorted;
}

export function getBestRoundedScore(candidates: readonly SolverCandidate[]): number | null {
  return candidates.length > 0 ? Math.round(candidates[0].score) : null;
}
