import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { SetupMiniBoard } from "./SetupMiniBoard";
import { TurnSelector } from "./TurnSelector";
import { SetupError } from "./SetupError";
import { useReversiStore } from "@/stores/use-reversi-store";
import { useTranslation } from "react-i18next";

export function TranscriptTab() {
  const { t } = useTranslation();
  const setupBoard = useReversiStore((state) => state.setupBoard);
  const setupCurrentPlayer = useReversiStore((state) => state.setupCurrentPlayer);
  const transcriptInput = useReversiStore((state) => state.transcriptInput);
  const setTranscriptInput = useReversiStore((state) => state.setTranscriptInput);
  const setupError = useReversiStore((state) => state.setupError);

  return (
    <div className="flex gap-4 items-start">
      <SetupMiniBoard board={setupBoard} />

      <div className="flex flex-col gap-4 min-w-0 flex-1">
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
          <SetupError error={setupError} />
        </div>

        <TurnSelector currentPlayer={setupCurrentPlayer} readOnly />
      </div>
    </div>
  );
}
