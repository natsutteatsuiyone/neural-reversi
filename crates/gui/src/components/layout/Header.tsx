import { Menu, Play, Lightbulb, Globe } from "lucide-react";
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
  const setNewGameModalOpen = useReversiStore((state) => state.setNewGameModalOpen);
  const isHintMode = useReversiStore((state) => state.isHintMode);
  const setHintMode = useReversiStore((state) => state.setHintMode);

  return (
    <header className="h-12 border-b border-white/10 bg-background-secondary flex items-center justify-between px-4 shrink-0">
      {/* Left: New Game */}
      <div className="flex items-center gap-2">
        <Button
          size="sm"
          onClick={() => setNewGameModalOpen(true)}
          className="gap-2 bg-primary text-primary-foreground hover:bg-primary-hover"
        >
          <Play className="w-4 h-4" />
          {t('game.newGame')}
        </Button>
      </div>

      {/* Right: Controls */}
      <div className="flex items-center gap-3">
        {/* Hint */}
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

        {/* Menu */}
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" size="icon-sm" className="text-foreground-secondary hover:text-foreground hover:bg-white/10">
              <Menu className="w-5 h-5" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-48 bg-popover border-white/20">
            <DropdownMenuSub>
              <DropdownMenuSubTrigger className="text-foreground-secondary hover:text-foreground focus:text-foreground focus:bg-white/10">
                <Lightbulb className="w-4 h-4 mr-2" />
                {t('hint.hintLevel')}
              </DropdownMenuSubTrigger>
              <DropdownMenuSubContent className="bg-popover border-white/20">
                <DropdownMenuRadioGroup
                  value={hintLevel.toString()}
                  onValueChange={(v) => setHintLevel(parseInt(v))}
                >
                  {[1, 4, 8, 12, 16, 20, 22, 24].map((level) => (
                    <DropdownMenuRadioItem
                      key={level}
                      value={level.toString()}
                      className="text-foreground-secondary hover:text-foreground focus:text-foreground focus:bg-white/10"
                    >
                      {t('player.level')}{level}
                    </DropdownMenuRadioItem>
                  ))}
                </DropdownMenuRadioGroup>
              </DropdownMenuSubContent>
            </DropdownMenuSub>

            <DropdownMenuSub>
              <DropdownMenuSubTrigger className="text-foreground-secondary hover:text-foreground focus:text-foreground focus:bg-white/10">
                <Globe className="w-4 h-4 mr-2" />
                {t('settings.language')}
              </DropdownMenuSubTrigger>
              <DropdownMenuSubContent className="bg-popover border-white/20">
                <DropdownMenuRadioGroup
                  value={language}
                  onValueChange={(v) => void setLanguage(v)}
                >
                  <DropdownMenuRadioItem
                    value="auto"
                    className="text-foreground-secondary hover:text-foreground focus:text-foreground focus:bg-white/10"
                  >
                    {t('settings.auto')}
                  </DropdownMenuRadioItem>
                  <DropdownMenuRadioItem
                    value="en"
                    className="text-foreground-secondary hover:text-foreground focus:text-foreground focus:bg-white/10"
                  >
                    {t('settings.english')}
                  </DropdownMenuRadioItem>
                  <DropdownMenuRadioItem
                    value="ja"
                    className="text-foreground-secondary hover:text-foreground focus:text-foreground focus:bg-white/10"
                  >
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
