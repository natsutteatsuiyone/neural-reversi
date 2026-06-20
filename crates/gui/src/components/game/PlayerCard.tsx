import { cn } from "@/lib/utils";
import { Bot, Loader2, User, Clock, StopCircle, Play } from "lucide-react";
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
  playerLabel?: string;
  onStop?: () => void;
  onResume?: () => void;
  /** Label the resume action as "Start" instead of "Resume" before any move. */
  resumeIsStart?: boolean;
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
  playerLabel,
  onStop,
  onResume,
  resumeIsStart,
}: PlayerCardProps) {
  const { t } = useTranslation();
  const showTimer = isAIControlled && aiMode === "game-time" && !onResume;

  return (
    <div
      className={cn(
        "rounded-xl p-3 transition-all duration-200 border-2",
        isCurrent
          ? "bg-primary/15 border-primary shadow-md shadow-primary/20"
          : "bg-white/5 border-card-border shadow-xs",
      )}
    >
      {/* Main row */}
      <div className="flex items-center gap-3">
        {/* Stone icon */}
        <div className="shrink-0">
          <Stone color={color} size="lg" />
        </div>

        {/* Score */}
        <div className="shrink-0 text-2xl font-bold text-foreground tabular-nums leading-none">
          {score}
        </div>

        {/* Player label / thinking status */}
        <div className="flex min-w-0 items-center gap-1.5">
          {isAIControlled ? (
            <>
              {isThinking ? (
                <Loader2 className="w-4 h-4 shrink-0 animate-spin text-accent-blue" />
              ) : (
                <Bot className="w-4 h-4 shrink-0 text-accent-blue" />
              )}
              <span
                className={cn(
                  "truncate text-xs font-medium",
                  isThinking ? "text-accent-blue" : "text-foreground-secondary",
                )}
              >
                {isThinking
                  ? t("ai.thinking")
                  : `${t("player.ai")}${aiMode === "level" ? ` ${t("player.level")}${aiLevel}` : ""}`}
              </span>
            </>
          ) : (
            <>
              <User className="w-4 h-4 shrink-0 text-foreground-secondary" />
              <span className="truncate text-xs font-medium text-foreground-secondary">
                {playerLabel ?? t("player.you")}
              </span>
            </>
          )}
        </div>

        <div className="flex-1" />

        {/* Right action: stop while thinking, resume while paused. Distinct
            keys force a fresh mount on swap so the Button's `transition-all`
            does not animate the color from the previous state (Resume's primary
            into Stop's destructive, and vice versa). */}
        {isThinking && onStop ? (
          <Button
            key="stop"
            variant="outline"
            size="sm"
            onClick={onStop}
            className="shrink-0 gap-1.5 h-7 px-2.5 text-destructive border-destructive/40 hover:bg-destructive/10 hover:text-destructive hover:shadow-sm"
          >
            <StopCircle className="w-3.5 h-3.5" />
            {t("game.stop")}
          </Button>
        ) : onResume ? (
          <Button
            key="resume"
            variant="outline"
            size="sm"
            onClick={onResume}
            className="shrink-0 gap-1.5 h-7 px-2.5 text-primary border-primary/40 hover:bg-primary/10 hover:text-primary hover:shadow-sm"
          >
            <Play className="w-3.5 h-3.5" />
            {resumeIsStart ? t("game.start") : t("game.resume")}
          </Button>
        ) : null}

        {/* Timer */}
        {showTimer && (
          <div className="flex shrink-0 items-center gap-1.5 rounded border border-card-border bg-white/5 px-2 py-0.5 text-xs font-mono font-medium tabular-nums text-foreground-muted">
            <Clock className="w-3.5 h-3.5" />
            {formatTime(aiRemainingTime)}
          </div>
        )}
      </div>
    </div>
  );
}
