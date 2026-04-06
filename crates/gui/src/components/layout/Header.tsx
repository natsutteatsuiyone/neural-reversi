import { Menu, Play, Lightbulb, Globe, HardDrive } from "lucide-react";
import { ANALYSIS_LEVELS } from "@/types";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuTrigger,
  DropdownMenuSub,
  DropdownMenuSubTrigger,
  DropdownMenuSubContent,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
} from "@/components/ui/dropdown-menu";
import { useReversiStore } from "@/stores/use-reversi-store";
import { useTranslation } from "react-i18next";
import { useLanguage } from "@/hooks/use-language";

export function Header() {
  const { t } = useTranslation();
  const { language, setLanguage } = useLanguage();
  const gameStatus = useReversiStore((state) => state.gameStatus);
  const hintLevel = useReversiStore((state) => state.hintLevel);
  const setHintLevel = useReversiStore((state) => state.setHintLevel);
  const hashSize = useReversiStore((state) => state.hashSize);
  const setHashSize = useReversiStore((state) => state.setHashSize);
  const setNewGameModalOpen = useReversiStore((state) => state.setNewGameModalOpen);
  const isHintMode = useReversiStore((state) => state.isHintMode);
  const setHintMode = useReversiStore((state) => state.setHintMode);

  return (
    <header className="flex min-h-12 shrink-0 flex-wrap items-center justify-between gap-2 border-b border-white/10 bg-background-secondary px-3 py-2 sm:h-12 sm:flex-nowrap sm:px-4 sm:py-0">
      {/* Left: New Game */}
      <div className="flex items-center gap-2">
        <Button
          size="sm"
          onClick={() => setNewGameModalOpen(true)}
          aria-label={t("game.newGame")}
          className="shrink-0 gap-2 bg-primary text-primary-foreground hover:bg-primary-hover max-[420px]:px-2"
        >
          <Play className="w-4 h-4" />
          <span className="max-[420px]:hidden">{t("game.newGame")}</span>
        </Button>
      </div>

      {/* Right: Controls */}
      <div className="flex flex-wrap items-center justify-end gap-2 sm:gap-3">
        {/* Hint */}
        <div className="flex shrink-0 items-center gap-2 rounded-lg border border-white/10 bg-white/5 px-2 py-1">
          <Lightbulb className={`w-4 h-4 ${isHintMode ? "text-accent-gold" : "text-foreground-muted"}`} />
          <Switch
            id="hint-mode"
            checked={isHintMode}
            onCheckedChange={setHintMode}
            disabled={gameStatus !== "playing"}
          />
          <Label
            htmlFor="hint-mode"
            className="cursor-pointer text-sm font-medium text-foreground-secondary max-[520px]:hidden"
          >
            {t("hint.hint")}
          </Label>
        </div>

        {/* Menu */}
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button
              variant="ghost"
              size="icon-sm"
              className="shrink-0 text-foreground-secondary hover:bg-white/10 hover:text-foreground"
            >
              <Menu className="w-5 h-5" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-48">
            <DropdownMenuSub>
              <DropdownMenuSubTrigger>
                <Lightbulb className="w-4 h-4 mr-2" />
                {t('hint.hintLevel')}
              </DropdownMenuSubTrigger>
              <DropdownMenuSubContent>
                <DropdownMenuRadioGroup
                  value={hintLevel.toString()}
                  onValueChange={(v) => setHintLevel(parseInt(v))}
                >
                  {ANALYSIS_LEVELS.map((level) => (
                    <DropdownMenuRadioItem
                      key={level}
                      value={level.toString()}
                    >
                      {t('player.level')}{level}
                    </DropdownMenuRadioItem>
                  ))}
                </DropdownMenuRadioGroup>
              </DropdownMenuSubContent>
            </DropdownMenuSub>

            <DropdownMenuSub>
              <DropdownMenuSubTrigger>
                <HardDrive className="w-4 h-4 mr-2" />
                {t('settings.hashSize')}
              </DropdownMenuSubTrigger>
              <DropdownMenuSubContent>
                <DropdownMenuRadioGroup
                  value={hashSize.toString()}
                  onValueChange={(v) => setHashSize(parseInt(v))}
                >
                  {[64, 128, 256, 512, 1024, 2048, 4096].map((size) => (
                    <DropdownMenuRadioItem
                      key={size}
                      value={size.toString()}
                    >
                      {size} MB
                    </DropdownMenuRadioItem>
                  ))}
                </DropdownMenuRadioGroup>
              </DropdownMenuSubContent>
            </DropdownMenuSub>

            <DropdownMenuSub>
              <DropdownMenuSubTrigger>
                <Globe className="w-4 h-4 mr-2" />
                {t('settings.language')}
              </DropdownMenuSubTrigger>
              <DropdownMenuSubContent>
                <DropdownMenuRadioGroup
                  value={language}
                  onValueChange={(v) => void setLanguage(v)}
                >
                  <DropdownMenuRadioItem value="auto">
                    {t('settings.auto')}
                  </DropdownMenuRadioItem>
                  <DropdownMenuRadioItem value="en">
                    {t('settings.english')}
                  </DropdownMenuRadioItem>
                  <DropdownMenuRadioItem value="ja">
                    {t('settings.japanese')}
                  </DropdownMenuRadioItem>
                </DropdownMenuRadioGroup>
              </DropdownMenuSubContent>
            </DropdownMenuSub>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </header>
  );
}
