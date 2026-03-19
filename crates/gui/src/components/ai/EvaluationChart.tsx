import { useCallback, useMemo } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  ResponsiveContainer,
  CartesianGrid,
  ReferenceLine,
  Tooltip,
  type TooltipProps,
  type MouseHandlerDataParam,
} from "recharts";
import { useReversiStore } from "@/stores/use-reversi-store";

interface ChartDataItem {
  move: number;
  timelineIndex: number;
  score: number | null;
  scoreDisplay: string | null;
  hasData: boolean;
  notation: string | null;
}

function CustomTooltip({
  active,
  payload,
}: TooltipProps<number, string> & { payload?: Array<{ payload: ChartDataItem }> }) {
  if (active && payload && payload.length) {
    const dataPoint = payload[0].payload;
    return (
      <div className="bg-popover text-popover-foreground px-3 py-2 rounded-lg shadow-lg border border-white/20 text-sm">
        <p className="text-xs text-foreground-muted mb-1">#{dataPoint.move}</p>
        {typeof dataPoint.score === "number" && (
          <p className="font-semibold text-foreground">
            {dataPoint.notation}: {dataPoint.scoreDisplay}
          </p>
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

  const allMoves = moveHistory.allMoves;
  const cursorPosition = moveHistory.length;

  const chartData = useMemo(() => {
    const items: ChartDataItem[] = [];
    let moveNumber = 0;

    for (let i = 0; i < allMoves.length; i++) {
      const currentMove = allMoves[i];
      if (currentMove.row < 0) {
        continue;
      }

      moveNumber++;
      const score =
        currentMove.isAI && currentMove.score !== undefined
          ? currentMove.score
          : null;

      items.push({
        move: moveNumber,
        timelineIndex: i,
        score: score !== null ? Math.round(score) : null,
        scoreDisplay: score !== null ? (score > 0 ? `+${score}` : `${score}`) : null,
        hasData: Boolean(currentMove.isAI && currentMove.score !== undefined),
        notation: currentMove.notation,
      });
    }
    return items;
  }, [allMoves]);

  const cursorMoveNumber = useMemo(() => {
    if (cursorPosition >= moveHistory.totalLength) return null;

    // Find the chart data point at or just before the cursor position
    let lastMoveNumber: number | null = null;
    for (const item of chartData) {
      if (item.timelineIndex >= cursorPosition) break;
      lastMoveNumber = item.move;
    }
    return lastMoveNumber;
  }, [cursorPosition, moveHistory.totalLength, chartData]);

  const yAxisDomain = useMemo(() => {
    const defaultRange = [-8, 8];
    if (chartData.length === 0) {
      return defaultRange;
    }

    const scores = chartData
      .filter((d) => d.hasData)
      .map((d) => d.score as number);
    if (scores.length === 0) {
      return defaultRange;
    }

    const dataMin = Math.min(...scores);
    const dataMax = Math.max(...scores);
    const maxAbsValue = Math.max(Math.abs(dataMin), Math.abs(dataMax));

    if (maxAbsValue <= 4) {
      return defaultRange;
    }

    const adjustedMin = Math.floor(dataMin / 4) * 4;
    const adjustedMax = Math.ceil(dataMax / 4) * 4;

    return [adjustedMin, adjustedMax];
  }, [chartData]);

  const yAxisTicks = useMemo(() => {
    const [min, max] = yAxisDomain;
    const ticks = [];
    for (let i = min; i <= max; i += 4) {
      ticks.push(i);
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

  if (gameStatus === "waiting") {
    return null;
  }

  // Use a bright color for visibility
  const lineColor = "#3d9970"; // primary green

  return (
    <div className="bg-white/5 rounded-lg p-2 border border-white/10">
      <ResponsiveContainer width="100%" height={180}>
        <LineChart
          data={chartData}
          margin={{ top: 5, right: 10, left: 0, bottom: 5 }}
          onClick={handleChartClick}
          style={{ cursor: "pointer" }}
        >
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="rgba(255,255,255,0.1)"
            horizontal={true}
            vertical={false}
          />

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
            dot={{
              fill: lineColor,
              strokeWidth: 1,
              stroke: "rgba(255,255,255,0.2)",
              r: 3,
            }}
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
