import { Button } from "@/components/ui/button";
import { SetupMiniBoard } from "./SetupMiniBoard";
import { TurnSelector } from "./TurnSelector";
import { SetupError } from "./SetupError";
import { useReversiStore } from "@/stores/use-reversi-store";
import { useTranslation } from "react-i18next";

export function ManualSetupTab() {
  const { t } = useTranslation();
  const setupBoard = useReversiStore((state) => state.setupBoard);
  const setupCurrentPlayer = useReversiStore((state) => state.setupCurrentPlayer);
  const setSetupCurrentPlayer = useReversiStore((state) => state.setSetupCurrentPlayer);
  const setSetupCellColor = useReversiStore((state) => state.setSetupCellColor);
  const clearSetupBoard = useReversiStore((state) => state.clearSetupBoard);
  const resetSetupToInitial = useReversiStore((state) => state.resetSetupToInitial);
  const setupError = useReversiStore((state) => state.setupError);

  return (
    <div className="flex gap-4 items-start">
      <SetupMiniBoard
        board={setupBoard}
        editable
        onCellClick={setSetupCellColor}
      />

      <div className="flex flex-col gap-4 min-w-0 flex-1">
        <TurnSelector
          currentPlayer={setupCurrentPlayer}
          onPlayerChange={setSetupCurrentPlayer}
        />

        <div className="flex flex-col gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={clearSetupBoard}
            className="text-foreground-secondary border-white/20 hover:bg-white/10"
          >
            {t("setup.clear")}
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={resetSetupToInitial}
            className="text-foreground-secondary border-white/20 hover:bg-white/10"
          >
            {t("setup.initialPosition")}
          </Button>
        </div>
        <SetupError error={setupError} />
      </div>
    </div>
  );
}
