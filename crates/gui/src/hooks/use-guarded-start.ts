import { useCallback, useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { useTranslation } from "react-i18next";

/** Guards setup-modal starts against double submits and updates after unmount. */
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
    async (start: () => Promise<boolean>, onStarted?: () => void) => {
      if (inFlight.current) return;
      inFlight.current = true;
      setIsStarting(true);
      try {
        if (await start()) onStarted?.();
      } catch (error) {
        console.error("Failed to start:", error);
        if (mounted.current) toast.error(t(errorMessageKey));
      } finally {
        inFlight.current = false;
        if (mounted.current) setIsStarting(false);
      }
    },
    [t, errorMessageKey],
  );

  return { isStarting, run };
}
