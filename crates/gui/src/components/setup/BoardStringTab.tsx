import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { SetupMiniBoard } from "./SetupMiniBoard";
import { Stone } from "@/components/board/Stone";
import { useReversiStore } from "@/stores/use-reversi-store";
import { useTranslation } from "react-i18next";
import type { Player } from "@/types";

export function BoardStringTab() {
  const { t } = useTranslation();
  const setupBoard = useReversiStore((state) => state.setupBoard);
  const setupCurrentPlayer = useReversiStore((state) => state.setupCurrentPlayer);
  const setSetupCurrentPlayer = useReversiStore((state) => state.setSetupCurrentPlayer);
  const boardStringInput = useReversiStore((state) => state.boardStringInput);
  const setBoardStringInput = useReversiStore((state) => state.setBoardStringInput);
  const setupError = useReversiStore((state) => state.setupError);

  return (
    <div className="space-y-4">
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
        {setupError && (
          <p className="text-sm text-red-400">
            {setupError.startsWith("illegalMove:")
              ? t("setup.error.illegalMove", { move: setupError.split(":")[1] })
              : t(`setup.error.${setupError}`)}
          </p>
        )}
      </div>

      <div className="flex gap-4 items-start">
        <SetupMiniBoard board={setupBoard} />
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
              <RadioGroupItem value="black" id="bs-turn-black" />
              <Label
                htmlFor="bs-turn-black"
                className="flex items-center gap-1 cursor-pointer text-foreground-secondary"
              >
                <Stone color="black" size="sm" />
                {t("colors.black")}
              </Label>
            </div>
            <div className="flex items-center gap-1.5">
              <RadioGroupItem value="white" id="bs-turn-white" />
              <Label
                htmlFor="bs-turn-white"
                className="flex items-center gap-1 cursor-pointer text-foreground-secondary"
              >
                <Stone color="white" size="sm" />
                {t("colors.white")}
              </Label>
            </div>
          </RadioGroup>
        </div>
      </div>
    </div>
  );
}
