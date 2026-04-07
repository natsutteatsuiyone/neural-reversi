import { GameLayout } from "@/components/layout/GameLayout";
import { useReversiStore } from "@/stores/use-reversi-store";
import type { AppSettings } from "@/services";
import { useEffect, useState } from "react";
import { toast } from "sonner";
import { Trophy, Info } from "lucide-react";
import { getWinner } from "@/lib/game-logic";
import { useKeyboardNavigation } from "@/hooks/use-keyboard-navigation";
import { useTranslation } from "react-i18next";
import { PASS_NOTIFICATION_DURATION_MS } from "@/stores/slices/game-slice";
import "./App.css";

interface AppProps {
  initialSettings: AppSettings;
}

function App({ initialSettings }: AppProps) {
  const { t } = useTranslation();
  const [initStatus, setInitStatus] = useState<"loading" | "ready" | "error">("loading");

  const showPassNotification = useReversiStore((state) => state.showPassNotification);
  const hidePassNotification = useReversiStore((state) => state.hidePassNotification);
  const gameOver = useReversiStore((state) => state.gameOver);
  const getScores = useReversiStore((state) => state.getScores);
  const hydrateSettings = useReversiStore((state) => state.hydrateSettings);
  const startGame = useReversiStore((state) => state.startGame);

  const scores = getScores();
  const winner = gameOver ? getWinner(scores) : null;

  // Enable keyboard navigation
  useKeyboardNavigation();

  // Load settings and auto-start game on mount
  useEffect(() => {
    const initApp = async () => {
      hydrateSettings({ ...initialSettings, gameMode: "ai-white" });
      const started = await startGame();
      setInitStatus(started ? "ready" : "error");
    };
    void initApp();
  }, [hydrateSettings, initialSettings, startGame]);

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
        duration: PASS_NOTIFICATION_DURATION_MS,
        onDismiss: hidePassNotification,
        onAutoClose: hidePassNotification,
      });
    }
  }, [showPassNotification, hidePassNotification, t]);

  // Loading screen
  if (initStatus === "loading") {
    return (
      <div className="flex items-center justify-center min-h-screen bg-background">
        <div className="flex flex-col items-center gap-4">
          <div className="w-12 h-12 border-4 border-primary border-t-transparent rounded-full animate-spin" />
          <p className="text-foreground-muted">{t("game.loading")}</p>
        </div>
      </div>
    );
  }

  if (initStatus === "error") {
    return (
      <div className="flex items-center justify-center min-h-screen bg-background px-6">
        <div className="max-w-md rounded-xl border border-white/10 bg-card p-6 text-center shadow-lg">
          <h1 className="text-xl font-semibold text-foreground">{t("setup.error.aiInitFailed")}</h1>
          <p className="mt-2 text-sm text-foreground-muted">{t("setup.error.unexpectedError")}</p>
        </div>
      </div>
    );
  }

  return <GameLayout />;
}

export default App;
