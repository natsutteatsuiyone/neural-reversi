import { cn } from "@/lib/utils";
import { Bot, Loader2, User, Clock, StopCircle } from "lucide-react";
import { Stone } from "@/components/board/Stone";
import { Button } from "@/components/ui/button";
import { useTranslation } from "react-i18next";

interface PlayerCardProps {
  color: "black" | "white";
  score: number;
  isCurrent: boolean;
  isAIControlled: boolean;
  aiLevel: number;
  isThinking: boolean;
  aiMode: string;
  aiRemainingTime: number;
  onStop?: () => void;
}

function formatTime(ms: number): string {
  const totalSeconds = Math.max(0, Math.floor(ms / 1000));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
}

export function PlayerCard({
  color,
  score,
  isCurrent,
  isAIControlled,
  aiLevel,
  isThinking,
  aiMode,
  aiRemainingTime,
  onStop,
}: PlayerCardProps) {
  const { t } = useTranslation();
  const showTimer = isAIControlled && aiMode === "game-time";

  return (
    <div
      className={cn(
        "rounded-xl p-3 transition-all duration-200 border",
        isCurrent
          ? "bg-white/15 border-primary/50 shadow-lg shadow-primary/10"
          : "bg-white/5 border-white/10"
      )}
    >
      {/* Main row */}
      <div className="flex items-center gap-3">
        {/* Stone icon */}
        <div className="shrink-0">
          <Stone color={color} size="lg" />
        </div>

        {/* Score */}
        <div className="text-3xl font-bold text-foreground">{score}</div>

        {/* Player label */}
        <div className="flex items-center gap-1.5">
          {isAIControlled ? (
            <>
              <Bot className="w-4 h-4 text-accent-blue" />
              <span className="text-sm font-medium text-foreground-secondary">
                {t('player.ai')} {aiMode === "level" && `${t('player.level')}${aiLevel}`}
              </span>
            </>
          ) : (
            <>
              <User className="w-4 h-4 text-foreground-secondary" />
              <span className="text-sm font-medium text-foreground-secondary">
                {t('player.you')}
              </span>
            </>
          )}
        </div>

        <div className="flex-1" />

        {/* Timer */}
        {showTimer && (
          <div className="flex items-center gap-1.5 px-2 py-1 rounded text-sm font-mono font-medium text-foreground-muted bg-white/5">
            <Clock className="w-3.5 h-3.5" />
            {formatTime(aiRemainingTime)}
          </div>
        )}

        {/* Turn indicator (when not thinking) */}
        {isCurrent && !isThinking && (
          <div className="w-2.5 h-2.5 rounded-full bg-primary animate-pulse" />
        )}
      </div>

      {/* Thinking row */}
      {isThinking && (
        <div className="flex items-center justify-between mt-2 pt-2 border-t border-white/10">
          <div className="flex items-center gap-1.5 text-cyan-400">
            <Loader2 className="w-4 h-4 animate-spin" />
            <span className="text-sm font-medium">{t('ai.thinking')}</span>
          </div>
          {onStop && (
            <Button
              variant="outline"
              size="sm"
              onClick={onStop}
              className="gap-1.5 h-7 px-2.5 text-destructive border-destructive/40 hover:bg-destructive/10 hover:text-destructive"
            >
              <StopCircle className="w-3.5 h-3.5" />
              {t('game.stop')}
            </Button>
          )}
        </div>
      )}
    </div>
  );
}
