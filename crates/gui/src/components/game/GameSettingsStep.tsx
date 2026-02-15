import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Timer, Zap } from "lucide-react";
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

  return (
    <div className="space-y-6 py-4">
      {/* Game Mode Selection */}
      <div className="space-y-3">
        <Label className="text-sm font-medium text-foreground-secondary">{t('game.youPlay')}</Label>
        <div className="grid grid-cols-2 gap-3">
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

      {/* AI Mode Tabs */}
      <div className="space-y-3">
        <Label className="text-sm font-medium text-foreground-secondary">{t('ai.mode')}</Label>
        <Tabs
          value={settings.aiMode}
          onValueChange={(v) => onChange({ aiMode: v as AIMode })}
        >
          <TabsList className="w-full bg-white/10">
            <TabsTrigger value="game-time" className="flex-1 gap-2 data-[state=active]:bg-white/15 data-[state=active]:text-foreground text-foreground-secondary">
              <Timer className="w-4 h-4" />
              {t('ai.timed')}
            </TabsTrigger>
            <TabsTrigger value="level" className="flex-1 gap-2 data-[state=active]:bg-white/15 data-[state=active]:text-foreground text-foreground-secondary">
              <Zap className="w-4 h-4" />
              {t('ai.level')}
            </TabsTrigger>
          </TabsList>
        </Tabs>
      </div>

      {/* Settings based on AI mode */}
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
            max={24}
            step={1}
            onValueChange={([value]) => onChange({ aiLevel: value })}
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
            onValueChange={([value]) => onChange({ gameTimeLimit: value })}
          />
          <div className="flex justify-between text-xs text-foreground-muted">
            <span>30s</span>
            <span>10m</span>
          </div>
        </div>
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
      onClick={onClick}
      className={cn(
        "flex flex-col items-center gap-2 p-5 rounded-xl border-2 transition-all",
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
