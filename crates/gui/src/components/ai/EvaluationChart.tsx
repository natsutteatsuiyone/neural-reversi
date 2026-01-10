import { useMemo } from "react";
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
} from "recharts";
import { useReversiStore } from "@/stores/use-reversi-store";

interface ChartDataItem {
  move: number;
  score: number | null;
  scoreDisplay: string | null;
  hasData: boolean;
  notation: string | null;
}

export function EvaluationChart() {
  const moves = useReversiStore((state) => state.moves);
  const gameMode = useReversiStore((state) => state.gameMode);

  const chartData = useMemo(() => {
    const allMoves: ChartDataItem[] = [];

    for (let i = 0; i < moves.length; i++) {
      const currentMove = moves[i];
      if (currentMove.notation.toLowerCase() === "pass") {
        continue;
      }

      const score =
        currentMove.isAI && currentMove.score !== undefined
          ? currentMove.score
          : null;

      allMoves.push({
        move: i + 1,
        score: score !== null ? Math.round(score) : null,
        scoreDisplay: score !== null ? (score > 0 ? `+${score}` : `${score}`) : null,
        hasData: Boolean(currentMove.isAI && currentMove.score !== undefined),
        notation: currentMove.notation,
      });
    }
    return allMoves;
  }, [moves]);

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

  if (gameMode !== "ai-black" && gameMode !== "ai-white") {
    return null;
  }

  // Use a bright color for visibility
  const lineColor = "#3d9970"; // primary green

  const CustomTooltip = ({
    active,
    payload,
  }: TooltipProps<number, string> & { payload?: Array<{ payload: ChartDataItem }> }) => {
    if (active && payload && payload.length) {
      const dataPoint = payload[0].payload;
      return (
        <div className="bg-popover text-popover-foreground px-3 py-2 rounded-lg shadow-lg border border-white/20 text-sm">
          <p className="text-xs text-foreground-muted mb-1">Move {dataPoint.move}</p>
          {typeof dataPoint.score === "number" && (
            <p className="font-semibold text-foreground">
              {dataPoint.notation}: {dataPoint.scoreDisplay}
            </p>
          )}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="h-full bg-white/5 rounded-lg p-2 border border-white/10">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart
          data={chartData}
          margin={{ top: 5, right: 10, left: 0, bottom: 5 }}
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

          <XAxis
            dataKey="move"
            axisLine={false}
            tickLine={false}
            tick={{
              fontSize: 11,
              fill: "rgba(255,255,255,0.5)",
            }}
            tickMargin={8}
            domain={[0, 60]}
            type="number"
            scale="linear"
            ticks={[0, 10, 20, 30, 40, 50, 60]}
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
