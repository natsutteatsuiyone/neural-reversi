import { Play, StopCircle, Lightbulb } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { useReversiStore } from "@/stores/use-reversi-store";
import { useTranslation } from "react-i18next";

export function GameControls() {
  const { t } = useTranslation();
  const isHintMode = useReversiStore((state) => state.isHintMode);
  const setHintMode = useReversiStore((state) => state.setHintMode);
  const gameStatus = useReversiStore((state) => state.gameStatus);
  const isAIThinking = useReversiStore((state) => state.isAIThinking);
  const abortAIMove = useReversiStore((state) => state.abortAIMove);
  const setNewGameModalOpen = useReversiStore((state) => state.setNewGameModalOpen);

  return (
    <div className="p-4 flex items-center gap-3">
      <Button
        size="sm"
        onClick={() => setNewGameModalOpen(true)}
        className="gap-2 bg-primary text-primary-foreground hover:bg-primary-hover"
      >
        <Play className="w-4 h-4" />
        {t('game.newGame')}
      </Button>

      {isAIThinking && (
        <Button
          variant="outline"
          size="sm"
          onClick={() => void abortAIMove()}
          className="gap-2 text-accent-gold border-accent-gold/30 hover:bg-accent-gold/10"
        >
          <StopCircle className="w-4 h-4" />
          {t('game.stop')}
        </Button>
      )}

      <div className="flex-1" />

      <div className="flex items-center gap-2">
        <Lightbulb className={`w-4 h-4 ${isHintMode ? 'text-accent-gold' : 'text-foreground-muted'}`} />
        <Switch
          id="hint-mode"
          checked={isHintMode}
          onCheckedChange={setHintMode}
          disabled={gameStatus !== "playing"}
        />
        <Label
          htmlFor="hint-mode"
          className="text-sm font-medium text-foreground-secondary cursor-pointer"
        >
          {t('hint.hint')}
        </Label>
      </div>
    </div>
  );
}
