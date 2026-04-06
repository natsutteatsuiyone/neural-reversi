import { useCallback, useMemo } from "react";
import { useTranslation } from "react-i18next";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  ResponsiveContainer,
  CartesianGrid,
  ReferenceLine,
  ReferenceArea,
  Tooltip,
  type TooltipProps,
  type MouseHandlerDataParam,
  useYAxisScale,
  usePlotArea,
} from "recharts";
import { useReversiStore } from "@/stores/use-reversi-store";
import type { MoveAnalysis } from "@/stores/slices/types";

interface ChartDataItem {
  move: number;
  timelineIndex: number;
  score: number | null;
  scoreDisplay: string | null;
  hasData: boolean;
  notation: string | null;
  analysis?: MoveAnalysis;
}

const DUBIOUS_THRESHOLD = 2;
const BLUNDER_THRESHOLD = 6;
const STYLE = getComputedStyle(document.documentElement);
const LINE_COLOR = STYLE.getPropertyValue("--primary").trim() || "#3d9970";
const BLUNDER_COLOR = STYLE.getPropertyValue("--chart-blunder").trim() || "#ef4444";
const DUBIOUS_COLOR = STYLE.getPropertyValue("--chart-dubious").trim() || "#eab308";

function getMarkerColor(scoreLoss: number): string | null {
  if (scoreLoss > BLUNDER_THRESHOLD) return BLUNDER_COLOR;
  if (scoreLoss > DUBIOUS_THRESHOLD) return DUBIOUS_COLOR;
  return null;
}

function DiscIndicatorBar() {
  const yScale = useYAxisScale();
  const plotArea = usePlotArea();
  if (!yScale || !plotArea) return null;

  const zeroY = yScale(0) as number;
  const barWidth = 8;
  const barX = 4;
  const r = 2;

  return (
    <g>
      <rect x={barX} y={plotArea.y} width={barWidth} height={zeroY - plotArea.y} rx={r} ry={r} fill="var(--stone-black-to)" stroke="rgba(255,255,255,0.3)" strokeWidth={0.5} />
      <rect x={barX} y={zeroY} width={barWidth} height={plotArea.y + plotArea.height - zeroY} rx={r} ry={r} fill="var(--stone-white-to)" stroke="rgba(255,255,255,0.15)" strokeWidth={0.5} />
    </g>
  );
}

function CustomTooltip({
  active,
  payload,
}: TooltipProps<number, string> & { payload?: Array<{ payload: ChartDataItem }> }) {
  const { t } = useTranslation();
  if (active && payload && payload.length) {
    const dataPoint = payload[0].payload;
    const analysis = dataPoint.analysis;
    return (
      <div className="bg-popover text-popover-foreground px-3 py-2 rounded-lg shadow-lg border border-white/20 text-sm">
        <p className="text-xs text-foreground-muted mb-1">#{dataPoint.move}</p>
        {typeof dataPoint.score === "number" && (
          <p className="font-semibold text-foreground">
            {dataPoint.notation}: {dataPoint.scoreDisplay}
          </p>
        )}
        {analysis && (
          <div className="mt-1 space-y-0.5 text-xs">
            {analysis.bestMove !== analysis.playedMove && (
              <p className="text-foreground-muted">
                {t('analysis.bestMoveLabel', {
                  move: analysis.bestMove,
                  score: analysis.bestScore > 0 ? `+${analysis.bestScore}` : analysis.bestScore,
                })}
              </p>
            )}
            {analysis.scoreLoss > DUBIOUS_THRESHOLD && (
              <p className={analysis.scoreLoss > BLUNDER_THRESHOLD ? "text-red-400" : "text-yellow-400"}>
                {t('analysis.lossLabel', { loss: analysis.scoreLoss.toFixed(1) })}
              </p>
            )}
          </div>
        )}
      </div>
    );
  }
  return null;
}

export function EvaluationChart() {
  const moveHistory = useReversiStore((state) => state.moveHistory);
  const gameStatus = useReversiStore((state) => state.gameStatus);
  const goToMove = useReversiStore((state) => state.goToMove);
  const gameAnalysisResult = useReversiStore((state) => state.gameAnalysisResult);

  const allMoves = moveHistory.allMoves;
  const cursorPosition = moveHistory.length;

  const chartData = useMemo(() => {
    const analysisMap = new Map<number, MoveAnalysis>();
    if (gameAnalysisResult) {
      for (const a of gameAnalysisResult) {
        analysisMap.set(a.moveIndex, a);
      }
    }

    const items: ChartDataItem[] = [];
    let moveNumber = 0;

    for (let i = 0; i < allMoves.length; i++) {
      const currentMove = allMoves[i];
      if (currentMove.row < 0) {
        continue;
      }

      moveNumber++;
      const analysis = analysisMap.get(i);

      const isWhite = currentMove.player === "white";
      let score: number | null = null;
      if (analysis) {
        score = isWhite ? -analysis.playedScore : analysis.playedScore;
      } else if (currentMove.isAI && currentMove.score !== undefined) {
        score = isWhite ? -currentMove.score : currentMove.score;
      }

      items.push({
        move: moveNumber,
        timelineIndex: i,
        score,
        scoreDisplay: score !== null ? (score > 0 ? `+${score.toFixed(1)}` : `${score.toFixed(1)}`) : null,
        hasData: score !== null,
        notation: currentMove.notation,
        analysis,
      });
    }
    return items;
  }, [allMoves, gameAnalysisResult]);

  const cursorMoveNumber = useMemo(() => {
    if (cursorPosition >= moveHistory.totalLength) return null;

    let lastMoveNumber: number | null = null;
    for (const item of chartData) {
      if (item.timelineIndex >= cursorPosition) break;
      lastMoveNumber = item.move;
    }
    return lastMoveNumber;
  }, [cursorPosition, moveHistory.totalLength, chartData]);

  const yAxisDomain = useMemo((): [number, number] => {
    const defaultRange: [number, number] = [-8, 8];
    const tickUnit = 4;

    const scores = chartData
      .filter((d) => d.hasData)
      .map((d) => d.score as number);
    if (scores.length === 0) return defaultRange;

    const dataMin = Math.min(...scores);
    const dataMax = Math.max(...scores);

    if (Math.max(Math.abs(dataMin), Math.abs(dataMax)) <= tickUnit) {
      return defaultRange;
    }

    const padding = Math.max(tickUnit, Math.ceil(Math.abs(dataMax - dataMin) * 0.08 / tickUnit) * tickUnit);
    let lo = Math.floor((dataMin - padding) / tickUnit) * tickUnit;
    let hi = Math.ceil((dataMax + padding) / tickUnit) * tickUnit;

    lo = Math.min(lo, 0);
    hi = Math.max(hi, 0);

    // Ensure the opposite side has at least 30% of the dominant side (min 8)
    const dominant = Math.max(Math.abs(lo), hi);
    const minOpposite = Math.max(8, Math.ceil(dominant * 0.3 / tickUnit) * tickUnit);
    if (Math.abs(lo) > hi) hi = Math.max(hi, minOpposite);
    else if (hi > Math.abs(lo)) lo = Math.min(lo, -minOpposite);

    return [Math.max(lo, -64), Math.min(hi, 64)];
  }, [chartData]);

  const yAxisTicks = useMemo(() => {
    const [min, max] = yAxisDomain;
    const range = max - min;
    const tickStep = Math.max(4, Math.ceil(range / 32 / 4) * 4);
    const ticks = [];
    for (let v = min; v <= max; v += tickStep) {
      ticks.push(v);
    }
    return ticks;
  }, [yAxisDomain]);

  const handleChartClick = useCallback(
    (data: MouseHandlerDataParam) => {
      if (data.activeTooltipIndex == null) return;
      const item = chartData[data.activeTooltipIndex as number];
      if (!item) return;
      goToMove(item.timelineIndex + 1);
    },
    [goToMove, chartData],
  );

  const renderDot = useCallback(
    (props: { cx?: number; cy?: number; index?: number }) => {
      if (props.cx == null || props.cy == null || props.index == null) return null;
      const markerColor = getMarkerColor(chartData[props.index]?.analysis?.scoreLoss ?? 0);
      const r = markerColor ? 5 : 3;
      const fill = markerColor ?? LINE_COLOR;
      const stroke = markerColor ? "rgba(255,255,255,0.4)" : "rgba(255,255,255,0.2)";
      const sw = markerColor ? 1.5 : 1;
      return (
        <circle
          key={`dot-${props.index}`}
          cx={props.cx}
          cy={props.cy}
          r={r}
          fill={fill}
          stroke={stroke}
          strokeWidth={sw}
        />
      );
    },
    [chartData],
  );

  if (gameStatus === "waiting") {
    return null;
  }

  const hasAnalysis = gameAnalysisResult !== null;

  return (
    <div className="bg-white/5 rounded-lg p-2 border border-white/10">
      <ResponsiveContainer width="100%" height={180}>
        <LineChart
          data={chartData}
          margin={{ top: 5, right: 10, left: 14, bottom: 5 }}
          onClick={handleChartClick}
          style={{ cursor: "pointer" }}
        >
          <DiscIndicatorBar />
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="rgba(255,255,255,0.1)"
            horizontal={true}
            vertical={false}
          />

          <ReferenceArea y1={0} y2={yAxisDomain[1]} fill="rgba(0,0,0,0.18)" />
          <ReferenceArea y1={yAxisDomain[0]} y2={0} fill="rgba(255,255,255,0.08)" />

          <ReferenceLine
            y={0}
            stroke="rgba(255,255,255,0.3)"
            strokeWidth={1}
            strokeDasharray="4 4"
          />

          {cursorMoveNumber !== null && (
            <ReferenceLine
              x={cursorMoveNumber}
              stroke="rgba(255,255,255,0.5)"
              strokeWidth={1}
              strokeDasharray="4 4"
            />
          )}

          <XAxis
            dataKey="move"
            axisLine={false}
            tickLine={false}
            tick={{
              fontSize: 11,
              fill: "rgba(255,255,255,0.5)",
            }}
            tickMargin={8}
            domain={[1, 60]}
            type="number"
            scale="linear"
            ticks={[1, 10, 20, 30, 40, 50, 60]}
          />

          <YAxis
            axisLine={false}
            tickLine={false}
            tick={{
              fontSize: 11,
              fill: "rgba(255,255,255,0.5)",
            }}
            tickMargin={4}
            domain={yAxisDomain}
            ticks={yAxisTicks}
            width={30}
          />

          <Line
            type="monotone"
            dataKey="score"
            stroke={LINE_COLOR}
            strokeWidth={2}
            dot={hasAnalysis ? renderDot : {
              fill: LINE_COLOR,
              strokeWidth: 1,
              stroke: "rgba(255,255,255,0.2)",
              r: 3,
            }}
            activeDot={{
              r: 5,
              fill: LINE_COLOR,
              stroke: "rgba(255,255,255,0.5)",
              strokeWidth: 2,
            }}
            connectNulls={true}
            isAnimationActive={false}
          />

          <Tooltip
            content={<CustomTooltip />}
            cursor={{
              stroke: "rgba(255,255,255,0.2)",
              strokeWidth: 1,
              strokeDasharray: "3 3",
            }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
