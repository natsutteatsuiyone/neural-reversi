import type { MoveAnalysis } from "@/domain/game/game-analysis";
import type { MoveRecord } from "@/domain/game/types";

export interface ChartDataItem {
  move: number;
  timelineIndex: number;
  score: number | null;
  scoreDisplay: string | null;
  notation: string | null;
  analysis?: MoveAnalysis;
}

const DEFAULT_Y_AXIS_DOMAIN: [number, number] = [-8, 8];
const Y_AXIS_TICK_UNIT = 4;

export function createEvaluationChartData(
  moves: readonly MoveRecord[],
  analysisResults: readonly MoveAnalysis[] | null,
): ChartDataItem[] {
  const analysisByMoveIndex = createAnalysisByMoveIndex(analysisResults);
  const items: ChartDataItem[] = [];
  let moveNumber = 0;

  for (let timelineIndex = 0; timelineIndex < moves.length; timelineIndex++) {
    const move = moves[timelineIndex];
    if (move.row < 0) {
      continue;
    }

    moveNumber++;
    const analysis = analysisByMoveIndex.get(timelineIndex);
    const score = resolveMoveScore(move, analysis);

    items.push({
      move: moveNumber,
      timelineIndex,
      score,
      scoreDisplay: formatScore(score),
      notation: move.notation,
      analysis,
    });
  }

  return items;
}

export function resolveCursorMoveNumber(
  chartData: readonly ChartDataItem[],
  cursorPosition: number,
  totalLength: number,
): number | null {
  if (cursorPosition >= totalLength) {
    return null;
  }

  let lastMoveNumber: number | null = null;
  for (const item of chartData) {
    if (item.timelineIndex >= cursorPosition) break;
    lastMoveNumber = item.move;
  }
  return lastMoveNumber;
}

export function resolveYAxisDomain(chartData: readonly ChartDataItem[]): [number, number] {
  const scores = chartData.flatMap((item) => item.score === null ? [] : [item.score]);
  if (scores.length === 0) {
    return DEFAULT_Y_AXIS_DOMAIN;
  }

  const dataMin = Math.min(...scores);
  const dataMax = Math.max(...scores);

  if (Math.max(Math.abs(dataMin), Math.abs(dataMax)) <= Y_AXIS_TICK_UNIT) {
    return DEFAULT_Y_AXIS_DOMAIN;
  }

  const padding = Math.max(
    Y_AXIS_TICK_UNIT,
    Math.ceil(Math.abs(dataMax - dataMin) * 0.08 / Y_AXIS_TICK_UNIT) * Y_AXIS_TICK_UNIT,
  );
  let lo = Math.floor((dataMin - padding) / Y_AXIS_TICK_UNIT) * Y_AXIS_TICK_UNIT;
  let hi = Math.ceil((dataMax + padding) / Y_AXIS_TICK_UNIT) * Y_AXIS_TICK_UNIT;

  lo = Math.min(lo, 0);
  hi = Math.max(hi, 0);

  const dominant = Math.max(Math.abs(lo), hi);
  const minOpposite = Math.max(8, Math.ceil(dominant * 0.3 / Y_AXIS_TICK_UNIT) * Y_AXIS_TICK_UNIT);
  if (Math.abs(lo) > hi) hi = Math.max(hi, minOpposite);
  else if (hi > Math.abs(lo)) lo = Math.min(lo, -minOpposite);

  return [Math.max(lo, -64), Math.min(hi, 64)];
}

export function resolveYAxisTicks(domain: readonly [number, number]): number[] {
  const [min, max] = domain;
  const range = max - min;
  const tickStep = Math.max(4, Math.ceil(range / 32 / 4) * 4);
  const ticks: number[] = [];
  for (let value = min; value <= max; value += tickStep) {
    ticks.push(value);
  }
  return ticks;
}

function createAnalysisByMoveIndex(
  analysisResults: readonly MoveAnalysis[] | null,
): Map<number, MoveAnalysis> {
  const analysisByMoveIndex = new Map<number, MoveAnalysis>();
  if (!analysisResults) {
    return analysisByMoveIndex;
  }

  for (const analysis of analysisResults) {
    analysisByMoveIndex.set(analysis.moveIndex, analysis);
  }
  return analysisByMoveIndex;
}

function resolveMoveScore(move: MoveRecord, analysis?: MoveAnalysis): number | null {
  const isWhite = move.player === "white";
  if (analysis) {
    return isWhite ? -analysis.playedScore : analysis.playedScore;
  }

  if (move.isAI && move.score !== undefined) {
    return isWhite ? -move.score : move.score;
  }

  return null;
}

function formatScore(score: number | null): string | null {
  if (score === null) {
    return null;
  }

  return score > 0 ? `+${score.toFixed(1)}` : `${score.toFixed(1)}`;
}
