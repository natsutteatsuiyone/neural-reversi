import { useTranslation } from "react-i18next";
import { ArrowLeft, Home, Play, Square } from "lucide-react";
import { useReversiStore } from "@/stores/use-reversi-store";
import { Button } from "@/components/ui/button";
import { SolverSelectivitySelector } from "./SolverSelectivitySelector";

export function SolverControls() {
  const { t } = useTranslation();
  const solverHistory = useReversiStore((s) => s.solverHistory);
  const isSolverSearching = useReversiStore((s) => s.isSolverSearching);
  const isSolverStopped = useReversiStore((s) => s.isSolverStopped);
  const undoSolver = useReversiStore((s) => s.undoSolver);
  const resetSolverToRoot = useReversiStore((s) => s.resetSolverToRoot);
  const stopSolverSearch = useReversiStore((s) => s.stopSolverSearch);
  const resumeSolverSearch = useReversiStore((s) => s.resumeSolverSearch);

  const canGoBack = solverHistory.length > 1;

  return (
    <div className="flex flex-col gap-3 px-4 py-3 border-b border-white/10">
      <div className="flex gap-2">
        <Button
          variant="outline"
          size="sm"
          onClick={() => void undoSolver()}
          disabled={!canGoBack}
          className="gap-1"
        >
          <ArrowLeft className="w-4 h-4" />
          {t("solver.back")}
        </Button>
        <Button
          variant="outline"
          size="sm"
          onClick={() => void resetSolverToRoot()}
          disabled={!canGoBack}
          className="gap-1"
        >
          <Home className="w-4 h-4" />
          {t("solver.backToStart")}
        </Button>
        <div className="flex-1" />
        {isSolverSearching && (
          <Button
            variant="soft"
            size="sm"
            onClick={() => void stopSolverSearch()}
            className="gap-1 bg-red-500/20 text-red-400 hover:bg-red-500/30"
          >
            <Square className="w-4 h-4" />
            {t("solver.stop")}
          </Button>
        )}
        {!isSolverSearching && isSolverStopped && (
          <Button
            variant="outline"
            size="sm"
            onClick={() => void resumeSolverSearch()}
            className="gap-1"
          >
            <Play className="w-4 h-4" />
            {t("solver.resume")}
          </Button>
        )}
      </div>

      <SolverSelectivitySelector />
    </div>
  );
}
