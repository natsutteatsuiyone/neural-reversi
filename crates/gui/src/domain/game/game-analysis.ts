import { getNotation } from "@/domain/game/game-logic";
import type { AIMoveProgress, GameAnalysisProgress } from "@/services/types";
import type { MoveRecord, Player } from "@/domain/game/types";

export type HintAnalysisResults = Map<string, AIMoveProgress>;

export interface MoveAnalysis {
  moveIndex: number;
  player: Player;
  playedMove: string;
  playedScore: number;
  bestMove: string;
  bestScore: number;
  scoreLoss: number;
  depth: number;
}

export function applyHintAnalysisProgress(
  results: HintAnalysisResults,
  progress: AIMoveProgress,
): HintAnalysisResults | null {
  if (progress.row === undefined || progress.col === undefined) {
    return null;
  }

  const key = `${progress.row},${progress.col}`;
  const existing = results.get(key);
  if (existing && isSameHintAnalysisProgress(existing, progress)) {
    return null;
  }

  const next = new Map(results);
  next.set(key, progress);
  return next;
}

export function createGameAnalysisMoveList(moves: readonly MoveRecord[]): string[] {
  return moves.map((move) =>
    move.row < 0 ? "--" : getNotation(move.row, move.col)
  );
}

export function appendGameAnalysisProgress(
  results: readonly MoveAnalysis[],
  moves: readonly MoveRecord[],
  progress: GameAnalysisProgress,
): MoveAnalysis[] {
  return [
    ...results,
    createMoveAnalysis(moves[progress.moveIndex], progress),
  ];
}

function createMoveAnalysis(move: MoveRecord, progress: GameAnalysisProgress): MoveAnalysis {
  return {
    moveIndex: progress.moveIndex,
    player: move.player,
    playedMove: move.notation,
    playedScore: progress.playedScore,
    bestMove: progress.bestMove,
    bestScore: progress.bestScore,
    scoreLoss: progress.scoreLoss,
    depth: progress.depth,
  };
}

function isSameHintAnalysisProgress(a: AIMoveProgress, b: AIMoveProgress): boolean {
  return (
    a.score === b.score &&
    a.depth === b.depth &&
    a.targetDepth === b.targetDepth &&
    a.acc === b.acc &&
    a.isEndgame === b.isEndgame &&
    a.pvLine === b.pvLine
  );
}
