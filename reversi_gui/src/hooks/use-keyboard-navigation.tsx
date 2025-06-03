import { useEffect } from "react";
import { useReversiStore } from "@/stores/use-reversi-store";

export function useKeyboardNavigation() {
  const gameStatus = useReversiStore((state) => state.gameStatus);
  const undoMove = useReversiStore((state) => state.undoMove);
  const redoMove = useReversiStore((state) => state.redoMove);
  const isAIThinking = useReversiStore((state) => state.isAIThinking);
  const isAnalyzing = useReversiStore((state) => state.isAnalyzing);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Only work during playing status
      if (gameStatus !== "playing") return;
      
      // Don't work while AI is thinking or analyzing
      if (isAIThinking || isAnalyzing) return;

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
  }, [gameStatus, undoMove, redoMove, isAIThinking, isAnalyzing]);
}