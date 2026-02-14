import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { SetupMiniBoard } from "./SetupMiniBoard";
import { Stone } from "@/components/board/Stone";
import { useReversiStore } from "@/stores/use-reversi-store";
import { useTranslation } from "react-i18next";

export function TranscriptTab() {
  const { t } = useTranslation();
  const setupBoard = useReversiStore((state) => state.setupBoard);
  const setupCurrentPlayer = useReversiStore((state) => state.setupCurrentPlayer);
  const transcriptInput = useReversiStore((state) => state.transcriptInput);
  const setTranscriptInput = useReversiStore((state) => state.setTranscriptInput);
  const setupError = useReversiStore((state) => state.setupError);

  const errorMessage = setupError
    ? setupError.startsWith("illegalMove:")
      ? t("setup.error.illegalMove", { move: setupError.split(":")[1] })
      : t(`setup.error.${setupError}`)
    : null;

  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <Label className="text-sm font-medium text-foreground-secondary">
          {t("setup.tabs.transcript")}
        </Label>
        <Input
          value={transcriptInput}
          onChange={(e) => setTranscriptInput(e.target.value)}
          placeholder={t("setup.transcriptPlaceholder")}
          className="font-mono bg-white/5 border-white/20 text-foreground"
          spellCheck={false}
        />
        {errorMessage && (
          <p className="text-sm text-red-400">{errorMessage}</p>
        )}
      </div>

      <div className="flex gap-4 items-start">
        <SetupMiniBoard board={setupBoard} />
        <div className="space-y-2">
          <Label className="text-sm font-medium text-foreground-secondary">
            {t("setup.turn")}
          </Label>
          <div className="flex items-center gap-1.5 text-foreground-secondary">
            <Stone color={setupCurrentPlayer} size="sm" />
            {t(`colors.${setupCurrentPlayer}`)}
          </div>
        </div>
      </div>
    </div>
  );
}
