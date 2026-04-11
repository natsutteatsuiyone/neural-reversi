import { memo } from "react";
import type { SolverCandidate } from "@/services/types";
import { cn } from "@/lib/utils";

interface SolverCandidateRowProps {
  candidate: SolverCandidate;
  isBest: boolean;
  onClick: (row: number, col: number) => void;
}

export const SolverCandidateRow = memo(function SolverCandidateRow({
  candidate,
  isBest,
  onClick,
}: SolverCandidateRowProps) {
  const roundedScore = Math.round(candidate.score);
  const displayScore = roundedScore > 0 ? `+${roundedScore}` : `${roundedScore}`;
  const pvDisplay = candidate.pvLine.split(" ").slice(0, 8).join(" ");

  return (
    <button
      type="button"
      onClick={() => onClick(candidate.row, candidate.col)}
      className={cn(
        "w-full flex items-center gap-3 px-3 py-2 rounded-md text-left cursor-pointer",
        "hover:bg-white/5 transition-colors",
        isBest && "bg-emerald-500/10",
      )}
    >
      <span
        className={cn(
          "font-mono font-semibold text-sm min-w-[2.5rem]",
          isBest ? "text-emerald-400" : "text-foreground",
        )}
      >
        {candidate.move}
      </span>
      <span
        className={cn(
          "font-mono text-sm min-w-[3rem] text-right",
          isBest ? "text-emerald-400" : "text-foreground-secondary",
        )}
      >
        {displayScore}
      </span>
      <span className="text-xs text-foreground-muted min-w-[2.5rem] text-right">
        {candidate.acc}%
      </span>
      <span className="flex-1 text-xs text-foreground-muted font-mono truncate">
        {pvDisplay}
      </span>
    </button>
  );
});
