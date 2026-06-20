import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { useReversiStore } from "@/stores/use-reversi-store";
import { Play } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { useGuardedStart } from "@/hooks/use-guarded-start";
import { SetupTabs } from "@/components/setup/SetupTabs";
import { SolverSelectivitySelector } from "./SolverSelectivitySelector";
import { SolverModeSelector } from "./SolverModeSelector";

export function SolverModal() {
  const { t } = useTranslation();

  const isOpen = useReversiStore((s) => s.isSolverModalOpen);
  const closeSolverModal = useReversiStore((s) => s.closeSolverModal);
  const setupError = useReversiStore((s) => s.setupError);
  const startSolverFromSetup = useReversiStore((s) => s.startSolverFromSetup);

  // Draft kept local so twiddling radios in the modal does not bleed into an
  // active solver session; startSolverFromSetup commits it only after setup
  // validation and solver replacement succeed.
  const [draftSelectivity, setDraftSelectivity] = useState(
    () => useReversiStore.getState().targetSelectivity,
  );
  const [draftMode, setDraftMode] = useState(() => useReversiStore.getState().solverMode);

  useEffect(() => {
    if (isOpen) {
      const s = useReversiStore.getState();
      setDraftSelectivity(s.targetSelectivity);
      setDraftMode(s.solverMode);
    }
  }, [isOpen]);

  const handleOpenChange = useCallback(
    (open: boolean) => {
      if (!open) {
        closeSolverModal();
      }
    },
    [closeSolverModal],
  );

  const { isStarting, run } = useGuardedStart("notification.startGameFailed");
  const handleStart = () =>
    run(
      () =>
        startSolverFromSetup({
          selectivity: draftSelectivity,
          mode: draftMode,
        }),
      () => closeSolverModal(),
    );

  return (
    <Dialog open={isOpen} onOpenChange={handleOpenChange}>
      <DialogContent
        aria-describedby={undefined}
        className="bg-card border-white/10 sm:max-w-lg"
        showCloseButton={false}
      >
        <DialogHeader>
          <DialogTitle className="text-xl text-foreground">{t("solver.title")}</DialogTitle>
        </DialogHeader>

        <SetupTabs />

        <div className="flex flex-row flex-wrap gap-6">
          <SolverSelectivitySelector
            idPrefix="solver-modal-selectivity"
            value={draftSelectivity}
            onValueChange={setDraftSelectivity}
          />
          <SolverModeSelector
            idPrefix="solver-modal-mode"
            value={draftMode}
            onValueChange={setDraftMode}
          />
        </div>

        <DialogFooter className="flex-row items-center gap-2">
          <Button
            variant="ghost"
            onClick={closeSolverModal}
            disabled={isStarting}
            className="text-foreground-secondary hover:text-foreground hover:bg-white/10"
          >
            {t("solver.cancel")}
          </Button>
          <div className="flex-1" />
          <Button
            onClick={() => void handleStart()}
            disabled={isStarting || !!setupError}
            className="gap-2 bg-primary text-primary-foreground hover:bg-primary-hover"
          >
            <Play className="w-4 h-4" />
            {t("solver.start")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
