import { useEffect } from "react";
import { useReversiStore } from "@/stores/use-reversi-store";
import { isGameSearchActive } from "@/stores/engine-activity";

export function useKeyboardNavigation() {
  const gameStatus = useReversiStore((state) => state.gameStatus);
  const undoMove = useReversiStore((state) => state.undoMove);
  const redoMove = useReversiStore((state) => state.redoMove);
  const gameSearchActive = useReversiStore((state) => isGameSearchActive(state.engineActivity));

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (gameStatus === "waiting") return;
      
      // Don't navigate while a search is tied to the current position/history.
      if (gameSearchActive) return;

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
  }, [gameStatus, undoMove, redoMove, gameSearchActive]);
}
