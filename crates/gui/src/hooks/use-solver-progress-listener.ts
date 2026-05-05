import { useEffect } from "react";
import { useReversiStore } from "@/stores/use-reversi-store";
import { defaultServices } from "@/services";

export function useSolverProgressListener(): void {
  const applySolverProgress = useReversiStore((state) => state.applySolverProgress);

  useEffect(() => {
    let unlisten: (() => void) | null = null;
    let cancelled = false;

    (async () => {
      try {
        const fn = await defaultServices.solver.onProgress((payload) => {
          applySolverProgress(payload);
        });
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
  }, [applySolverProgress]);
}
