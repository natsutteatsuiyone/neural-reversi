import type { AIMoveProgress } from "@/lib/ai";
import { cn } from "@/lib/utils";

interface HintScoreDisplayProps {
  rowIndex: number;
  colIndex: number;
  analyzeResults: Map<string, AIMoveProgress> | null;
  maxScore: number | null;
  hintLevel: number;
  gameOver: boolean;
}

export function HintScoreDisplay({
  rowIndex,
  colIndex,
  analyzeResults,
  maxScore,
  hintLevel,
  gameOver,
}: HintScoreDisplayProps) {
  if (!analyzeResults || gameOver) {
    return null;
  }

  const key = `${rowIndex},${colIndex}`;
  const result = analyzeResults.get(key);

  if (!result) {
    return null;
  }

  const { score, depth } = result;

  const roundedScore = Math.round(score);
  const displayScore = roundedScore > 0 ? `+${roundedScore}` : `${roundedScore}`;
  const isMaxScore = maxScore !== null && roundedScore === maxScore;

  // Search progress: show until target depth reached
  const isSearchComplete = depth >= hintLevel;
  const searchProgress = Math.min((depth / hintLevel) * 100, 100);

  return (
    <div className="absolute inset-0 flex flex-col items-center justify-center z-20 pointer-events-none p-1">
      {/* Score */}
      <div
        className={cn(
          "text-xl font-bold px-1 rounded",
          isMaxScore
            ? "text-emerald-400"
            : "text-white"
        )}

      >
        {displayScore}
      </div>
      
      {/* Search progress bar - hide when complete */}
      {!isSearchComplete && (
        <div className="w-[70%] h-1 bg-black/40 rounded-full mt-0.5 overflow-hidden">
          <div
            className="h-full rounded-full bg-cyan-400 transition-all"
            style={{ width: `${searchProgress}%` }}
          />
        </div>
      )}
    </div>
  );
}
