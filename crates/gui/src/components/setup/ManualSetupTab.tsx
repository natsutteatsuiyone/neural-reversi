import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { SetupMiniBoard } from "./SetupMiniBoard";
import { Stone } from "@/components/board/Stone";
import { useReversiStore } from "@/stores/use-reversi-store";
import { useTranslation } from "react-i18next";
import type { Player } from "@/types";

export function ManualSetupTab() {
  const { t } = useTranslation();
  const setupBoard = useReversiStore((state) => state.setupBoard);
  const setupCurrentPlayer = useReversiStore((state) => state.setupCurrentPlayer);
  const setSetupCurrentPlayer = useReversiStore((state) => state.setSetupCurrentPlayer);
  const setSetupCellColor = useReversiStore((state) => state.setSetupCellColor);
  const clearSetupBoard = useReversiStore((state) => state.clearSetupBoard);
  const resetSetupToInitial = useReversiStore((state) => state.resetSetupToInitial);
  const setupError = useReversiStore((state) => state.setupError);

  const errorMessage = setupError
    ? setupError.startsWith("illegalMove:")
      ? t("setup.error.illegalMove", { move: setupError.split(":")[1] })
      : t(`setup.error.${setupError}`)
    : null;

  return (
    <div className="flex gap-4 items-start">
      <SetupMiniBoard
        board={setupBoard}
        editable
        onCellClick={setSetupCellColor}
      />

      <div className="flex flex-col gap-4">
        {/* Turn selection */}
        <div className="space-y-2">
          <Label className="text-sm font-medium text-foreground-secondary">
            {t("setup.turn")}
          </Label>
          <RadioGroup
            value={setupCurrentPlayer}
            onValueChange={(v) => setSetupCurrentPlayer(v as Player)}
            className="flex gap-3"
          >
            <div className="flex items-center gap-1.5">
              <RadioGroupItem value="black" id="turn-black" />
              <Label htmlFor="turn-black" className="flex items-center gap-1 cursor-pointer text-foreground-secondary">
                <Stone color="black" size="sm" />
                {t("colors.black")}
              </Label>
            </div>
            <div className="flex items-center gap-1.5">
              <RadioGroupItem value="white" id="turn-white" />
              <Label htmlFor="turn-white" className="flex items-center gap-1 cursor-pointer text-foreground-secondary">
                <Stone color="white" size="sm" />
                {t("colors.white")}
              </Label>
            </div>
          </RadioGroup>
        </div>

        {/* Action buttons */}
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
        {errorMessage && (
          <p className="text-sm text-red-400">{errorMessage}</p>
        )}
      </div>
    </div>
  );
}
