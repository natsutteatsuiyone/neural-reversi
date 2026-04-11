import { useCallback, useMemo } from "react";
import { useTranslation } from "react-i18next";
import { Activity, X } from "lucide-react";
import { useReversiStore } from "@/stores/use-reversi-store";
import { cn } from "@/lib/utils";
import { Stone } from "@/components/board";
import { Button } from "@/components/ui/button";
import { SolverControls } from "./SolverControls";
import { SolverCandidateRow } from "./SolverCandidateRow";

export function SolverPanel() {
  const { t } = useTranslation();
  const isSolverActive = useReversiStore((s) => s.isSolverActive);
  const isSolverSearching = useReversiStore((s) => s.isSolverSearching);
  const solverCandidates = useReversiStore((s) => s.solverCandidates);
  const solverCurrentPlayer = useReversiStore((s) => s.solverCurrentPlayer);
  const advanceSolver = useReversiStore((s) => s.advanceSolver);
  const exitSolver = useReversiStore((s) => s.exitSolver);

  const sortedCandidates = useMemo(() => {
    const arr = Array.from(solverCandidates.values());
    arr.sort((a, b) => b.score - a.score);
    return arr;
  }, [solverCandidates]);

  const bestScore =
    sortedCandidates.length > 0 ? Math.round(sortedCandidates[0].score) : null;

  const handleCandidateClick = useCallback(
    (row: number, col: number) => {
      void advanceSolver(row, col);
    },
    [advanceSolver],
  );

  if (!isSolverActive) {
    return null;
  }

  return (
    <aside className="flex h-full min-h-0 min-w-0 flex-col bg-background-secondary">
      <div className="flex items-center gap-3 px-4 py-3 border-b border-white/10">
        <Activity
          className={cn(
            "w-4 h-4",
            isSolverSearching ? "text-accent-blue animate-pulse" : "text-foreground-muted",
          )}
        />
        <span className="text-sm font-medium text-foreground">
          {t("solver.title")}
        </span>
        {isSolverSearching && (
          <span className="text-xs bg-accent-blue/20 text-accent-blue px-2 py-0.5 rounded-full">
            {t("solver.searching")}
          </span>
        )}
        <div className="flex-1" />
        <Button
          variant="ghost"
          size="sm"
          onClick={() => void exitSolver()}
          className="gap-1 text-foreground-secondary hover:text-foreground"
        >
          <X className="w-4 h-4" />
          {t("solver.exit")}
        </Button>
      </div>

      <SolverControls />

      <div className="flex flex-1 min-h-0 flex-col overflow-y-auto">
        <div className="flex items-center gap-3 px-4 py-4">
          <span className="text-xs font-medium text-foreground-muted uppercase tracking-wide">
            {t("solver.candidates")}
          </span>
          {solverCurrentPlayer && (
            <div className="flex items-center gap-2">
              <Stone color={solverCurrentPlayer} size="sm" />
              <span className="text-sm font-medium text-foreground">
                {t(`colors.${solverCurrentPlayer}`)}
              </span>
            </div>
          )}
        </div>
        {sortedCandidates.length === 0 ? (
          <div className="px-4 py-6 text-sm text-foreground-muted text-center">
            {isSolverSearching ? t("solver.searching") : t("solver.noCandidates")}
          </div>
        ) : (
          <div className="flex flex-col gap-1 px-2 pb-3">
            {sortedCandidates.map((candidate) => (
              <SolverCandidateRow
                key={`${candidate.row},${candidate.col}`}
                candidate={candidate}
                isBest={bestScore !== null && Math.round(candidate.score) === bestScore}
                onClick={handleCandidateClick}
              />
            ))}
          </div>
        )}
      </div>
    </aside>
  );
}
