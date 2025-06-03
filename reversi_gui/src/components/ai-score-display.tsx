import type { AIMoveProgress } from "@/lib/ai";
import type { Board } from "@/types";
import { calculateScores } from "@/lib/game-logic";

interface AIScoreDisplayProps {
  rowIndex: number;
  colIndex: number;
  analyzeResults: Map<string, AIMoveProgress> | null;
  gameMode: string;
  maxScore: number | null;
  aiLevel: number;
  gameOver: boolean;
  board: Board;
}

export function AIScoreDisplay({
  rowIndex,
  colIndex,
  analyzeResults,
  gameMode,
  maxScore,
  aiLevel,
  gameOver,
  board,
}: AIScoreDisplayProps) {
  if (gameMode !== "analyze" || !analyzeResults || gameOver) {
    return null;
  }

  const key = `${rowIndex},${colIndex}`;
  const result = analyzeResults.get(key);

  if (!result) {
    return null;
  }

  const { score, depth, acc } = result;
  const { black, white } = calculateScores(board);
  const emptyCount = 64 - black - white;

  const displayScore = score > 0 ? `+${score.toFixed(1)}` : score.toFixed(1);
  const isMaxScore = maxScore !== null && score === maxScore;
  const textColor = isMaxScore ? "text-green-500" : "text-white";
  const showDepthInfo = depth !== aiLevel && depth !== emptyCount;

  return (
    <div className="absolute inset-0 flex flex-col items-center justify-center z-20 pointer-events-none">
      <div className={`text-lg font-bold ${textColor} px-1 rounded`}>
        {displayScore}
      </div>
      {showDepthInfo && (
        <div className="text-xs text-gray-200 px-1 rounded mt-1">
          {depth}@{acc}%
        </div>
      )}
    </div>
  );
}