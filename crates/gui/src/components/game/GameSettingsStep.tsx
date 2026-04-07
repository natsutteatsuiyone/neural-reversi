import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Timer, Zap, Bot, Users } from "lucide-react";
import { cn } from "@/lib/utils";
import type { AIMode, GameMode } from "@/types";
import { Stone } from "@/components/board/Stone";
import { useTranslation } from "react-i18next";

export interface GameSettings {
  gameMode: GameMode;
  aiMode: AIMode;
  aiLevel: number;
  gameTimeLimit: number;
}

interface GameSettingsStepProps {
  settings: GameSettings;
  onChange: (partial: Partial<GameSettings>) => void;
}

function formatTime(seconds: number) {
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  if (secs === 0) return `${mins}m`;
  return `${mins}m ${secs}s`;
}

export function GameSettingsStep({ settings, onChange }: GameSettingsStepProps) {
  const { t } = useTranslation();
  const isPvP = settings.gameMode === "pvp";

  return (
    <div className="space-y-6 py-4">
      <Tabs
        value={isPvP ? "pvp" : "ai"}
        onValueChange={(v) => {
          if (v === "pvp") {
            onChange({ gameMode: "pvp" });
          } else {
            onChange({ gameMode: "ai-white" });
          }
        }}
      >
        <TabsList className="w-full">
          <TabsTrigger value="ai" className="flex-1 gap-2">
            <Bot className="w-4 h-4" />
            {t('game.vsAI')}
          </TabsTrigger>
          <TabsTrigger value="pvp" className="flex-1 gap-2">
            <Users className="w-4 h-4" />
            {t('game.vsHuman')}
          </TabsTrigger>
        </TabsList>
      </Tabs>

      {!isPvP && (
        <>
          <div className="space-y-3">
            <Label className="text-sm font-medium text-foreground-secondary">{t('game.youPlay')}</Label>
            <div className="grid grid-cols-2 gap-3" role="radiogroup" aria-label={t('game.youPlay')}>
              <GameModeOption
                selected={settings.gameMode === "ai-white"}
                onClick={() => onChange({ gameMode: "ai-white" })}
                playerColor="black"
                label={t('colors.black')}
              />
              <GameModeOption
                selected={settings.gameMode === "ai-black"}
                onClick={() => onChange({ gameMode: "ai-black" })}
                playerColor="white"
                label={t('colors.white')}
              />
            </div>
          </div>

          <div className="space-y-3">
            <Label className="text-sm font-medium text-foreground-secondary">{t('ai.mode')}</Label>
            <Tabs
              value={settings.aiMode}
              onValueChange={(v) => onChange({ aiMode: v as AIMode })}
            >
              <TabsList className="w-full">
                <TabsTrigger value="game-time" className="flex-1 gap-2">
                  <Timer className="w-4 h-4" />
                  {t('ai.timed')}
                </TabsTrigger>
                <TabsTrigger value="level" className="flex-1 gap-2">
                  <Zap className="w-4 h-4" />
                  {t('ai.level')}
                </TabsTrigger>
              </TabsList>
            </Tabs>
          </div>

          {settings.aiMode === "level" ? (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <Label className="text-sm font-medium text-foreground-secondary">{t('ai.level')}</Label>
                <span className="text-sm font-mono font-semibold text-primary">
                  {settings.aiLevel}
                </span>
              </div>
              <Slider
                value={[settings.aiLevel]}
                min={1}
                max={30}
                step={1}
                onValueChange={([value]: number[]) => onChange({ aiLevel: value })}
              />
              <div className="flex justify-between text-xs text-foreground-muted">
                <span>{t('ai.easy')}</span>
                <span>{t('ai.hard')}</span>
              </div>
            </div>
          ) : (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <Label className="text-sm font-medium text-foreground-secondary">{t('ai.timePerGame')}</Label>
                <span className="text-sm font-mono font-semibold text-primary">
                  {formatTime(settings.gameTimeLimit)}
                </span>
              </div>
              <Slider
                value={[settings.gameTimeLimit]}
                min={30}
                max={600}
                step={30}
                onValueChange={([value]: number[]) => onChange({ gameTimeLimit: value })}
              />
              <div className="relative h-4 text-xs text-foreground-muted">
                <span className="absolute left-0">30s</span>
                <span className="absolute -translate-x-1/2" style={{ left: "47.4%" }}>5m</span>
                <span className="absolute right-0">10m</span>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

function GameModeOption({
  selected,
  onClick,
  playerColor,
  label,
}: {
  selected: boolean;
  onClick: () => void;
  playerColor: "black" | "white";
  label: string;
}) {
  return (
    <button
      type="button"
      role="radio"
      aria-checked={selected}
      onClick={onClick}
      className={cn(
        "flex flex-col items-center gap-2 p-5 rounded-xl border-2 transition-all cursor-pointer",
        selected
          ? "border-primary bg-primary/10"
          : "border-white/20 hover:border-white/30 hover:bg-white/5"
      )}
    >
      <Stone color={playerColor} size="lg" />
      <span className="text-sm font-semibold text-foreground">{label}</span>
    </button>
  );
}
