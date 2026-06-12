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
import { formatScore } from "@/lib/score-format";
import {
  createEvaluationChartData,
  resolveCursorMoveNumber,
  resolveYAxisDomain,
  resolveYAxisTicks,
  type ChartDataItem,
} from "./evaluation-chart-model";

const DUBIOUS_THRESHOLD = 2;
const BLUNDER_THRESHOLD = 6;

let _cachedColors: { line: string; blunder: string; dubious: string } | null = null;
function getChartColors() {
  if (!_cachedColors) {
    const style = getComputedStyle(document.documentElement);
    _cachedColors = {
      line: style.getPropertyValue("--primary").trim() || "#3d9970",
      blunder: style.getPropertyValue("--chart-blunder").trim() || "#ef4444",
      dubious: style.getPropertyValue("--chart-dubious").trim() || "#eab308",
    };
  }
  return _cachedColors;
}

function getMarkerColor(scoreLoss: number): string | null {
  const colors = getChartColors();
  if (scoreLoss > BLUNDER_THRESHOLD) return colors.blunder;
  if (scoreLoss > DUBIOUS_THRESHOLD) return colors.dubious;
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
      <rect
        x={barX}
        y={plotArea.y}
        width={barWidth}
        height={zeroY - plotArea.y}
        rx={r}
        ry={r}
        fill="var(--stone-black-to)"
        stroke="rgba(255,255,255,0.3)"
        strokeWidth={0.5}
      />
      <rect
        x={barX}
        y={zeroY}
        width={barWidth}
        height={plotArea.y + plotArea.height - zeroY}
        rx={r}
        ry={r}
        fill="var(--stone-white-to)"
        stroke="rgba(255,255,255,0.15)"
        strokeWidth={0.5}
      />
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
                {t("analysis.bestMoveLabel", {
                  move: analysis.bestMove,
                  score: formatScore(analysis.bestScore, "raw"),
                })}
              </p>
            )}
            {analysis.scoreLoss > DUBIOUS_THRESHOLD && (
              <p
                className={
                  analysis.scoreLoss > BLUNDER_THRESHOLD ? "text-red-400" : "text-yellow-400"
                }
              >
                {t("analysis.lossLabel", { loss: analysis.scoreLoss.toFixed(1) })}
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
  const lineColor = getChartColors().line;
  const moveHistory = useReversiStore((state) => state.moveHistory);
  const gameStatus = useReversiStore((state) => state.gameStatus);
  const goToMove = useReversiStore((state) => state.goToMove);
  const gameAnalysisResult = useReversiStore((state) => state.gameAnalysisResult);
  const isGameAnalyzing = useReversiStore((state) => state.isGameAnalyzing);

  const allMoves = moveHistory.allMoves;
  const cursorPosition = moveHistory.length;

  const chartData = useMemo(() => {
    return createEvaluationChartData(allMoves, gameAnalysisResult);
  }, [allMoves, gameAnalysisResult]);

  const cursorMoveNumber = useMemo(() => {
    return resolveCursorMoveNumber(chartData, cursorPosition, moveHistory.totalLength);
  }, [cursorPosition, moveHistory.totalLength, chartData]);

  const yAxisDomain = useMemo((): [number, number] => {
    return resolveYAxisDomain(chartData);
  }, [chartData]);

  const yAxisTicks = useMemo(() => {
    return resolveYAxisTicks(yAxisDomain);
  }, [yAxisDomain]);

  const handleChartClick = useCallback(
    (data: MouseHandlerDataParam) => {
      if (data.activeTooltipIndex == null) return;
      if (isGameAnalyzing) return;
      const item = chartData[data.activeTooltipIndex as number];
      if (!item) return;
      goToMove(item.timelineIndex + 1);
    },
    [goToMove, chartData, isGameAnalyzing],
  );

  const renderDot = useCallback(
    (props: { cx?: number; cy?: number; index?: number }) => {
      if (props.cx == null || props.cy == null || props.index == null) return null;
      const markerColor = getMarkerColor(chartData[props.index]?.analysis?.scoreLoss ?? 0);
      const r = markerColor ? 5 : 3;
      const fill = markerColor ?? lineColor;
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
    [chartData, lineColor],
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
          style={{ cursor: isGameAnalyzing ? "default" : "pointer" }}
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
            stroke={lineColor}
            strokeWidth={2}
            dot={
              hasAnalysis
                ? renderDot
                : {
                    fill: lineColor,
                    strokeWidth: 1,
                    stroke: "rgba(255,255,255,0.2)",
                    r: 3,
                  }
            }
            activeDot={{
              r: 5,
              fill: lineColor,
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
