import type { AIMoveProgress } from "@/lib/ai";
import { cn } from "@/lib/utils";

interface HintScoreDisplayProps {
  rowIndex: number;
  colIndex: number;
  analyzeResults: Map<string, AIMoveProgress> | null;
  maxScore: number | null;
  gameOver: boolean;
  isValidMoveCell: boolean;
}

export function HintScoreDisplay({
  rowIndex,
  colIndex,
  analyzeResults,
  maxScore,
  gameOver,
  isValidMoveCell,
}: HintScoreDisplayProps) {
  if (!analyzeResults || gameOver) {
    return null;
  }

  const key = `${rowIndex},${colIndex}`;
  const result = analyzeResults.get(key);

  // Show waiting progress bar for valid move cells without results yet
  if (!result) {
    if (!isValidMoveCell) {
      return null;
    }
    // Show 0% progress bar for unsearched valid moves
    return (
      <div className="absolute inset-0 flex flex-col items-center justify-center z-20 pointer-events-none p-1">
        <div className="w-[70%] h-1 bg-black/40 rounded-full mt-0.5 overflow-hidden">
          <div
            className="h-full rounded-full bg-cyan-400 transition-all"
            style={{ width: "0%" }}
          />
        </div>
      </div>
    );
  }

  const { score, depth, targetDepth, acc, isEndgame } = result;

  const roundedScore = Math.round(score);
  const displayScore = roundedScore > 0 ? `+${roundedScore}` : `${roundedScore}`;
  const isMaxScore = maxScore !== null && roundedScore === maxScore;

  // Search progress calculation:
  // - Midgame: iterates by depth (1 → 2 → ... → targetDepth)
  // - Endgame: iterates by selectivity (73% → 87% → 95% → 98% → 99% → 100%)
  const isSearchComplete = isEndgame ? acc === 100 : depth >= targetDepth;
  const searchProgress = isEndgame
    ? acc // Endgame: use selectivity probability directly
    : Math.min((depth / targetDepth) * 100, 100); // Midgame: depth-based progress

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
