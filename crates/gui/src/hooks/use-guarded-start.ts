import { useCallback, useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { useTranslation } from "react-i18next";
import { runGuardedStart } from "@/lib/guarded-start";

/**
 * React adapter for the guarded async-start choreography
 * (`@/lib/guarded-start`). Owns the re-entry ref, the `isStarting` UI flag,
 * and unmount safety: the `isStarting` setter and the error toast never fire
 * after the host component unmounts. Both setup modals adapt to this one
 * seam instead of each hand-rolling the guard.
 */
export function useGuardedStart(errorMessageKey: string) {
  const { t } = useTranslation();
  const [isStarting, setIsStarting] = useState(false);
  const inFlight = useRef(false);
  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  const run = useCallback(
    (start: () => Promise<boolean>, onStarted?: () => void) =>
      runGuardedStart(
        {
          isBusy: () => inFlight.current,
          setBusy: (busy) => {
            inFlight.current = busy;
            // Setting busy happens during a render-safe event; clearing it may
            // resolve after unmount — only then is the setState skipped.
            if (busy || mounted.current) setIsStarting(busy);
          },
        },
        start,
        {
          onStarted,
          onError: (error) => {
            console.error("Failed to start:", error);
            if (mounted.current) toast.error(t(errorMessageKey));
          },
        },
      ),
    [t, errorMessageKey],
  );

  return { isStarting, run };
}
