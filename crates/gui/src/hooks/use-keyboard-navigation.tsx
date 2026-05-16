import { useEffect } from "react";
import { useReversiStore } from "@/stores/use-reversi-store";

export function useKeyboardNavigation() {
  const gameStatus = useReversiStore((state) => state.gameStatus);
  const undoMove = useReversiStore((state) => state.undoMove);
  const redoMove = useReversiStore((state) => state.redoMove);
  const isAIThinking = useReversiStore((state) => state.isAIThinking);
  const isAnalyzing = useReversiStore((state) => state.isAnalyzing);
  const isGameAnalyzing = useReversiStore((state) => state.isGameAnalyzing);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (gameStatus === "waiting") return;
      
      // Don't navigate while a search is tied to the current position/history.
      if (isAIThinking || isAnalyzing || isGameAnalyzing) return;

      switch (e.key) {
        case "ArrowLeft":
          e.preventDefault();
          undoMove();
          break;
        case "ArrowRight":
          e.preventDefault();
          redoMove();
          break;
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [gameStatus, undoMove, redoMove, isAIThinking, isAnalyzing, isGameAnalyzing]);
}
