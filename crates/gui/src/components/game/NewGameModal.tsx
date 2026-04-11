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
import { useCallback, useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import { useTranslation } from "react-i18next";
import { ManualSetupTab } from "@/components/setup/ManualSetupTab";
import { TranscriptTab } from "@/components/setup/TranscriptTab";
import { BoardStringTab } from "@/components/setup/BoardStringTab";
import { GameSettingsStep } from "@/components/game/GameSettingsStep";
import type { GameSettings } from "@/components/game/GameSettingsStep";
import type { NewGameSettings, SetupTab } from "@/stores/slices/types";

interface NewGameModalContentProps {
  initialSettings: GameSettings;
  setupTab: SetupTab;
  setSetupTab: (tab: SetupTab) => void;
  setupError: string | null;
  startGame: (settings?: NewGameSettings) => Promise<boolean>;
  startFromSetup: (settings?: NewGameSettings) => Promise<boolean>;
  persistAISettings: (settings: GameSettings) => void;
  closeNewGameModal: () => void;
}

function NewGameModalContent({
  initialSettings,
  setupTab,
  setSetupTab,
  setupError,
  startGame,
  startFromSetup,
  persistAISettings,
  closeNewGameModal,
}: NewGameModalContentProps) {
  const { t } = useTranslation();

  const [settings, setSettings] = useState<GameSettings>(initialSettings);
  const [step, setStep] = useState<1 | 2>(1);
  const activeRef = useRef(true);
  useEffect(() => {
    activeRef.current = true;
    return () => { activeRef.current = false; };
  }, []);

  const handleSettingsChange = useCallback((partial: Partial<GameSettings>) => {
    setSettings((prev) => ({ ...prev, ...partial }));
  }, []);

  const handleStart = async (action: (s?: NewGameSettings) => Promise<boolean>) => {
    try {
      const started = await action(settings);
      if (started) {
        persistAISettings(settings);
      }
      if (!activeRef.current) return;
      if (started) {
        closeNewGameModal();
      }
    } catch (error) {
      console.error("Failed to start game:", error);
      if (activeRef.current) {
        toast.error(t("notification.startGameFailed"));
      }
    }
  };

  return (
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
          {step === 1 ? t("game.newGame") : t("setup.title")}
        </DialogTitle>
      </DialogHeader>

      {step === 1 ? (
        <GameSettingsStep settings={settings} onChange={handleSettingsChange} />
      ) : (
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
        <Button
          variant="ghost"
          onClick={closeNewGameModal}
          className="text-foreground-secondary hover:text-foreground hover:bg-white/10"
        >
          {t("game.cancel")}
        </Button>
        <div className="flex-1" />
        {step === 1 ? (
          <>
            <Button
              variant="outline"
              onClick={() => setStep(2)}
              className="gap-1 text-foreground-secondary border-white/20 hover:text-foreground hover:bg-white/10"
            >
              {t("game.nextStep")}
              <ChevronRight className="w-4 h-4" />
            </Button>
            <Button
              onClick={() => void handleStart(startGame)}
              className="gap-2 bg-primary text-primary-foreground hover:bg-primary-hover"
            >
              <Play className="w-4 h-4" />
              {t("game.startGame")}
            </Button>
          </>
        ) : (
          <>
            <Button
              variant="ghost"
              onClick={() => setStep(1)}
              className="gap-1 text-foreground-secondary hover:text-foreground hover:bg-white/10"
            >
              <ChevronLeft className="w-4 h-4" />
              {t("game.previousStep")}
            </Button>
            <Button
              onClick={() => void handleStart(startFromSetup)}
              disabled={!!setupError}
              className="gap-2 bg-primary text-primary-foreground hover:bg-primary-hover"
            >
              <Play className="w-4 h-4" />
              {t("game.startGame")}
            </Button>
          </>
        )}
      </DialogFooter>
    </DialogContent>
  );
}

export function NewGameModal() {
  const gameMode = useReversiStore((state) => state.gameMode);
  const aiLevel = useReversiStore((state) => state.aiLevel);
  const aiMode = useReversiStore((state) => state.aiMode);
  const gameTimeLimit = useReversiStore((state) => state.gameTimeLimit);

  const setGameMode = useReversiStore((state) => state.setGameMode);
  const setAILevelChange = useReversiStore((state) => state.setAILevelChange);
  const setAIMode = useReversiStore((state) => state.setAIMode);
  const setGameTimeLimit = useReversiStore((state) => state.setGameTimeLimit);
  const isNewGameModalOpen = useReversiStore((state) => state.isNewGameModalOpen);
  const newGameModalSession = useReversiStore((state) => state.newGameModalSession);
  const closeNewGameModal = useReversiStore((state) => state.closeNewGameModal);
  const startGame = useReversiStore((state) => state.startGame);

  // Setup slice
  const setupTab = useReversiStore((state) => state.setupTab);
  const setSetupTab = useReversiStore((state) => state.setSetupTab);
  const setupError = useReversiStore((state) => state.setupError);
  const startFromSetup = useReversiStore((state) => state.startFromSetup);

  const persistAISettings = (nextSettings: GameSettings) => {
    setGameMode(nextSettings.gameMode);
    setAILevelChange(nextSettings.aiLevel);
    setAIMode(nextSettings.aiMode);
    setGameTimeLimit(nextSettings.gameTimeLimit);
  };

  const initialSettings: GameSettings = {
    gameMode,
    aiMode,
    aiLevel,
    gameTimeLimit,
  };

  const handleOpenChange = useCallback((open: boolean) => {
    if (!open) {
      closeNewGameModal();
    }
  }, [closeNewGameModal]);

  return (
    <Dialog open={isNewGameModalOpen} onOpenChange={handleOpenChange}>
      {isNewGameModalOpen && (
        <NewGameModalContent
          key={newGameModalSession}
          initialSettings={initialSettings}
          setupTab={setupTab}
          setSetupTab={setSetupTab}
          setupError={setupError}
          startGame={startGame}
          startFromSetup={startFromSetup}
          persistAISettings={persistAISettings}
          closeNewGameModal={closeNewGameModal}
        />
      )}
    </Dialog>
  );
}
