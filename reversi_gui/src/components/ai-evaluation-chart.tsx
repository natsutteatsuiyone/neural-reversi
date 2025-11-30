"use client";

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
  score_org: string | null;
  hasData: boolean;
  notation: string | null;
}

export function AIEvaluationChart() {
  const moves = useReversiStore((state) => state.moves);
  const gameMode = useReversiStore((state) => state.gameMode);

  const chartData = useMemo(() => {
    // Use ChartDataItem for allMoves
    const allMoves: ChartDataItem[] = [];

    for (let i = 0; i < moves.length; i++) {
      const currentMove = moves[i];
      if (currentMove.notation.toLowerCase() === "pass") {
        continue;
      }

      const score = currentMove.isAI && currentMove.score !== undefined
        ? currentMove.score
        : null;

      allMoves.push({
        move: i + 1,
        score: score !== null ? Math.round(score) : null,
        score_org: (score ?? 0) > 0 ? `+${score}` : `${score}`,
        hasData: Boolean(currentMove.isAI && currentMove.score !== undefined),
        notation: currentMove.notation, // Add notation here
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

  const xAxisDomain = useMemo(() => {
    return [0, 60];
  }, []);

  if (gameMode !== "ai-black" && gameMode !== "ai-white") {
    return null;
  }

  const playerColor = gameMode === "ai-black" ? "#1b1b1b" : "#fff";

  const CustomTooltip = ({
    active,
    payload,
  }: TooltipProps<number, string> & { payload?: any[] }) => {
    if (active && payload && payload.length) {
      const dataPoint = payload[0].payload as ChartDataItem;
      return (
        <div className="bg-black/70 backdrop-blur-sm text-white p-3 rounded-md shadow-lg border border-white/30 text-sm">
          <p className="mb-1 text-xs">{`Move: ${dataPoint.move}`}</p>
          {typeof dataPoint.score === "number" && (
            <p className="mb-0.5">{`${dataPoint.notation} (${dataPoint.score_org})`}</p>
          )}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="bg-white/10 rounded-lg p-2">
      <div className="text-sm text-white/80 mb-1">Evaluation</div>
      <div className="h-36 bg-black/10 rounded-md pt-4">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={chartData}
            margin={{ top: 10, right: 15, left: 5, bottom: 5 }}
          >
            <CartesianGrid
              strokeDasharray="2 4"
              stroke="rgba(255,255,255,0.2)"
              horizontal={true}
              vertical={false}
            />

            <ReferenceLine
              y={0}
              stroke="rgba(255,255,255,0.5)"
              strokeWidth={2}
              strokeDasharray="4 2"
            />

            <XAxis
              dataKey="move"
              axisLine={false}
              tickLine={false}
              tick={{
                fontSize: 12,
                fill: "rgba(255,255,255,0.9)",
                fontWeight: 400,
              }}
              tickMargin={8}
              domain={xAxisDomain}
              type="number"
              scale="linear"
              ticks={[0, 10, 20, 30, 40, 50, 60]}
            />

            <YAxis
              axisLine={false}
              tickLine={false}
              tick={{
                fontSize: 14,
                fill: "rgba(255,255,255,0.9)",
                fontWeight: 600,
              }}
              tickMargin={6}
              domain={yAxisDomain}
              ticks={yAxisTicks}
              width={40}
            />

            {/* ライン */}
            <Line
              type="monotone"
              dataKey="score"
              stroke={playerColor}
              strokeWidth={2}
              dot={{
                fill: playerColor,
                strokeWidth: 1.5,
                stroke: "rgba(255,255,255,0.2)",
                r: 3,
              }}
              activeDot={{
                r: 5,
                fill: playerColor,
                stroke: "rgba(255,255,255,0.8)",
                strokeWidth: 2,
              }}
              connectNulls={true}
              isAnimationActive={false}
            />
            {/* ツールチップ */}
            <Tooltip
              content={<CustomTooltip />}
              cursor={{
                stroke: "rgba(255,255,255,0.4)",
                strokeWidth: 1,
                strokeDasharray: "3 3",
              }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
