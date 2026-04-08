import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useReversiStore } from "@/stores/use-reversi-store";
import { Play, ChevronRight, ChevronLeft } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { cn } from "@/lib/utils";
import { useTranslation } from "react-i18next";
import { ManualSetupTab } from "@/components/setup/ManualSetupTab";
import { TranscriptTab } from "@/components/setup/TranscriptTab";
import { BoardStringTab } from "@/components/setup/BoardStringTab";
import { GameSettingsStep } from "@/components/game/GameSettingsStep";
import type { GameSettings } from "@/components/game/GameSettingsStep";
import type { SetupTab } from "@/stores/slices/types";

export function NewGameModal() {
  const { t } = useTranslation();
  const gameMode = useReversiStore((state) => state.gameMode);
  const aiLevel = useReversiStore((state) => state.aiLevel);
  const aiMode = useReversiStore((state) => state.aiMode);
  const gameTimeLimit = useReversiStore((state) => state.gameTimeLimit);

  const setGameMode = useReversiStore((state) => state.setGameMode);
  const setAILevelChange = useReversiStore((state) => state.setAILevelChange);
  const setAIMode = useReversiStore((state) => state.setAIMode);
  const setGameTimeLimit = useReversiStore((state) => state.setGameTimeLimit);
  const isNewGameModalOpen = useReversiStore((state) => state.isNewGameModalOpen);
  const setNewGameModalOpen = useReversiStore((state) => state.setNewGameModalOpen);
  const startGame = useReversiStore((state) => state.startGame);

  // Setup slice
  const setupTab = useReversiStore((state) => state.setupTab);
  const setSetupTab = useReversiStore((state) => state.setSetupTab);
  const setupError = useReversiStore((state) => state.setupError);
  const startFromSetup = useReversiStore((state) => state.startFromSetup);
  const resetSetup = useReversiStore((state) => state.resetSetup);

  // Local state for pending AI settings
  const [settings, setSettings] = useState<GameSettings>({
    gameMode,
    aiMode,
    aiLevel,
    gameTimeLimit,
  });

  // Step management
  const [step, setStep] = useState<1 | 2>(1);

  const handleSettingsChange = useCallback((partial: Partial<GameSettings>) => {
    setSettings((prev) => ({ ...prev, ...partial }));
  }, []);

  // Sync local state and reset step when modal opens
  useEffect(() => {
    if (isNewGameModalOpen) {
      setSettings({ gameMode, aiMode, aiLevel, gameTimeLimit });
      setStep(1);
      resetSetup();
    }
  }, [isNewGameModalOpen, gameMode, aiLevel, aiMode, gameTimeLimit, resetSetup]);

  const persistAISettings = (nextSettings: GameSettings) => {
    setGameMode(nextSettings.gameMode);
    setAILevelChange(nextSettings.aiLevel);
    setAIMode(nextSettings.aiMode);
    setGameTimeLimit(nextSettings.gameTimeLimit);
  };

  const handleStartGame = async () => {
    try {
      const started = await startGame(settings);
      if (started) {
        persistAISettings(settings);
        setNewGameModalOpen(false);
      }
    } catch (error) {
      console.error("Failed to start game:", error);
    }
  };

  const handleStartFromSetup = async () => {
    try {
      const started = await startFromSetup(settings);
      if (started) {
        persistAISettings(settings);
        setNewGameModalOpen(false);
      }
    } catch (error) {
      console.error("Failed to start game from setup:", error);
    }
  };

  return (
    <Dialog open={isNewGameModalOpen} onOpenChange={setNewGameModalOpen}>
      <DialogContent
        aria-describedby={undefined}
        className={cn(
          "bg-card border-white/10",
          step === 1 ? "sm:max-w-md" : "sm:max-w-lg"
        )}
        showCloseButton={false}
      >
        <DialogHeader>
          <DialogTitle className="text-xl text-foreground">
            {step === 1 ? t('game.newGame') : t('setup.title')}
          </DialogTitle>
        </DialogHeader>

        {step === 1 ? (
          <GameSettingsStep settings={settings} onChange={handleSettingsChange} />
        ) : (
          /* Step 2: Board Setup */
          <Tabs
            value={setupTab}
            onValueChange={(v) => setSetupTab(v as SetupTab)}
            className="py-4"
          >
            <TabsList className="w-full">
              <TabsTrigger value="manual" className="flex-1">
                {t("setup.tabs.manual")}
              </TabsTrigger>
              <TabsTrigger value="transcript" className="flex-1">
                {t("setup.tabs.transcript")}
              </TabsTrigger>
              <TabsTrigger value="boardString" className="flex-1">
                {t("setup.tabs.boardString")}
              </TabsTrigger>
            </TabsList>

            <div className="grid mt-4">
              <TabsContent forceMount value="manual" className="col-start-1 row-start-1 data-[state=inactive]:invisible">
                <ManualSetupTab />
              </TabsContent>
              <TabsContent forceMount value="transcript" className="col-start-1 row-start-1 data-[state=inactive]:invisible">
                <TranscriptTab />
              </TabsContent>
              <TabsContent forceMount value="boardString" className="col-start-1 row-start-1 data-[state=inactive]:invisible">
                <BoardStringTab />
              </TabsContent>
            </div>
          </Tabs>
        )}

        <DialogFooter className="flex-row items-center gap-2">
          {step === 1 ? (
            <>
              <Button
                variant="ghost"
                onClick={() => setNewGameModalOpen(false)}
                className="text-foreground-secondary hover:text-foreground hover:bg-white/10"
              >
                {t('game.cancel')}
              </Button>
              <div className="flex-1" />
              <Button
                variant="outline"
                onClick={() => setStep(2)}
                className="gap-1 text-foreground-secondary border-white/20 hover:text-foreground hover:bg-white/10"
              >
                {t('game.nextStep')}
                <ChevronRight className="w-4 h-4" />
              </Button>
              <Button
                onClick={() => void handleStartGame()}
                className="gap-2 bg-primary text-primary-foreground hover:bg-primary-hover"
              >
                <Play className="w-4 h-4" />
                {t('game.startGame')}
              </Button>
            </>
          ) : (
            <>
              <Button
                variant="ghost"
                onClick={() => setNewGameModalOpen(false)}
                className="text-foreground-secondary hover:text-foreground hover:bg-white/10"
              >
                {t('game.cancel')}
              </Button>
              <div className="flex-1" />
              <Button
                variant="ghost"
                onClick={() => setStep(1)}
                className="gap-1 text-foreground-secondary hover:text-foreground hover:bg-white/10"
              >
                <ChevronLeft className="w-4 h-4" />
                {t('game.previousStep')}
              </Button>
              <Button
                onClick={() => void handleStartFromSetup()}
                disabled={!!setupError}
                className="gap-2 bg-primary text-primary-foreground hover:bg-primary-hover"
              >
                <Play className="w-4 h-4" />
                {t('game.startGame')}
              </Button>
            </>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
