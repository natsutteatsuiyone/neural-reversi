import { Label } from "@/components/ui/label";
import { SetupMiniBoard } from "./SetupMiniBoard";
import { TurnSelector } from "./TurnSelector";
import { SetupError } from "./SetupError";
import { useReversiStore } from "@/stores/use-reversi-store";
import { useTranslation } from "react-i18next";

export function BoardStringTab() {
  const { t } = useTranslation();
  const setupBoard = useReversiStore((state) => state.setupBoard);
  const setupCurrentPlayer = useReversiStore((state) => state.setupCurrentPlayer);
  const setSetupCurrentPlayer = useReversiStore((state) => state.setSetupCurrentPlayer);
  const boardStringInput = useReversiStore((state) => state.boardStringInput);
  const setBoardStringInput = useReversiStore((state) => state.setBoardStringInput);
  const setupError = useReversiStore((state) => state.setupError);

  return (
    <div className="flex gap-4 items-start">
      <SetupMiniBoard board={setupBoard} />

      <div className="flex flex-col gap-4 min-w-0 flex-1">
        <div className="space-y-2">
          <Label className="text-sm font-medium text-foreground-secondary">
            {t("setup.tabs.boardString")}
          </Label>
          <textarea
            value={boardStringInput}
            onChange={(e) => setBoardStringInput(e.target.value)}
            placeholder={t("setup.boardStringPlaceholder")}
            className="flex w-full rounded-md border border-white/20 bg-white/5 px-3 py-2 text-sm font-mono text-foreground placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring h-20 resize-none"
            spellCheck={false}
          />
          <SetupError error={setupError} />
        </div>

        <TurnSelector
          currentPlayer={setupCurrentPlayer}
          onPlayerChange={setSetupCurrentPlayer}
        />
      </div>
    </div>
  );
}
