import { GameLayout } from "@/components/layout/GameLayout";
import { useReversiStore } from "@/stores/use-reversi-store";
import { loadSettings } from "@/lib/settings-store";
import { useEffect, useState } from "react";
import { toast } from "sonner";
import { Trophy, Info } from "lucide-react";
import { getWinner } from "@/lib/game-logic";
import { useKeyboardNavigation } from "@/hooks/use-keyboard-navigation";
import { useTranslation } from "react-i18next";
import "./App.css";

function App() {
  const { t } = useTranslation();
  const [isInitialized, setIsInitialized] = useState(false);

  const showPassNotification = useReversiStore((state) => state.showPassNotification);
  const hidePassNotification = useReversiStore((state) => state.hidePassNotification);
  const gameOver = useReversiStore((state) => state.gameOver);
  const getScores = useReversiStore((state) => state.getScores);
  const setGameMode = useReversiStore((state) => state.setGameMode);
  const setAILevelChange = useReversiStore((state) => state.setAILevelChange);
  const setAIMode = useReversiStore((state) => state.setAIMode);
  const setTimeLimit = useReversiStore((state) => state.setTimeLimit);
  const setGameTimeLimit = useReversiStore((state) => state.setGameTimeLimit);
  const setHintLevel = useReversiStore((state) => state.setHintLevel);
  const setAIAnalysisPanelOpen = useReversiStore((state) => state.setAIAnalysisPanelOpen);
  const startGame = useReversiStore((state) => state.startGame);

  const scores = getScores();
  const winner = gameOver ? getWinner(scores) : null;

  // Enable keyboard navigation
  useKeyboardNavigation();

  // Load settings and auto-start game on mount
  useEffect(() => {
    const initApp = async () => {
      const settings = await loadSettings();
      setGameMode("ai-white"); // Always start with player black / AI white
      setAILevelChange(settings.aiLevel);
      setAIMode(settings.aiMode as "level" | "time" | "game-time");
      setTimeLimit(settings.timeLimit);
      setGameTimeLimit(settings.gameTimeLimit);
      setHintLevel(settings.hintLevel);
      setAIAnalysisPanelOpen(settings.aiAnalysisPanelOpen);

      await startGame();
      setIsInitialized(true);
    };
    void initApp();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Game over notification
  useEffect(() => {
    if (gameOver && winner) {
      const message =
        winner === "draw"
          ? t("notification.draw")
          : t("notification.wins", { color: t(`colors.${winner}`) });

      const description = t("notification.score", { black: scores.black, white: scores.white });

      toast(message, {
        description: description,
        icon: <Trophy className="w-4 h-4 text-accent-gold" />,
        duration: 5000,
      });
    }
  }, [gameOver, winner, scores.black, scores.white, t]);

  // Pass notification
  useEffect(() => {
    if (showPassNotification) {
      toast(t("notification.passingTurn", { color: t(`colors.${showPassNotification}`) }), {
        icon: <Info className="w-4 h-4 text-accent-blue" />,
        duration: 1500,
        onDismiss: hidePassNotification,
        onAutoClose: hidePassNotification,
      });
    }
  }, [showPassNotification, hidePassNotification, t]);

  // Loading screen
  if (!isInitialized) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-background">
        <div className="flex flex-col items-center gap-4">
          <div className="w-12 h-12 border-4 border-primary border-t-transparent rounded-full animate-spin" />
          <p className="text-foreground-muted">{t("game.loading")}</p>
        </div>
      </div>
    );
  }

  return <GameLayout />;
}

export default App;
