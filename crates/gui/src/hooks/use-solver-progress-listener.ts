import { useEffect } from "react";
import { useReversiStore } from "@/stores/use-reversi-store";

export function useSolverProgressListener(): void {
  const subscribeSolverProgress = useReversiStore((state) => state.subscribeSolverProgress);

  useEffect(() => {
    let unlisten: (() => void) | null = null;
    let cancelled = false;

    (async () => {
      try {
        const fn = await subscribeSolverProgress();
        if (cancelled) {
          fn();
        } else {
          unlisten = fn;
        }
      } catch (error) {
        console.error("Failed to subscribe to solver-progress:", error);
      }
    })();

    return () => {
      cancelled = true;
      if (unlisten) {
        unlisten();
        unlisten = null;
      }
    };
  }, [subscribeSolverProgress]);
}
