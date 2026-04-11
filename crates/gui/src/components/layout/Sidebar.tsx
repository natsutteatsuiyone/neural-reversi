import { PlayerCard } from "@/components/game/PlayerCard";
import { MoveHistory } from "@/components/game/MoveHistory";
import { useReversiStore } from "@/stores/use-reversi-store";
import { useTranslation } from "react-i18next";

export function Sidebar() {
  const { t } = useTranslation();
  const currentPlayer = useReversiStore((state) => state.currentPlayer);
  const gameMode = useReversiStore((state) => state.gameMode);
  const gameStatus = useReversiStore((state) => state.gameStatus);
  const aiLevel = useReversiStore((state) => state.aiLevel);
  const aiMode = useReversiStore((state) => state.aiMode);
  const aiRemainingTime = useReversiStore((state) => state.aiRemainingTime);
  const getScores = useReversiStore((state) => state.getScores);
  const abortAIMove = useReversiStore((state) => state.abortAIMove);
  const isAIThinking = useReversiStore((state) => state.isAIThinking);
  const paused = useReversiStore((state) => state.paused);
  const resumeAI = useReversiStore((state) => state.resumeAI);

  const scores = getScores();
  const isPvP = gameMode === "pvp";
  const blackIsAI = gameMode === "ai-black";
  const whiteIsAI = gameMode === "ai-white";

  const handleStop = () => {
    void abortAIMove();
  };

  const isAITurn = gameStatus === "playing" && !isPvP && (
    (blackIsAI && currentPlayer === "black") ||
    (whiteIsAI && currentPlayer === "white")
  );
  const showResume = paused && isAITurn;

  return (
    <aside className="flex h-full min-h-0 min-w-0 flex-col bg-background-secondary">
      {/* Player Cards */}
      <div className="p-4 space-y-3 border-b border-white/10">
        <PlayerCard
          color="black"
          score={scores.black}
          isCurrent={currentPlayer === "black"}
          isAIControlled={blackIsAI}
          aiLevel={aiLevel}
          isThinking={isAIThinking && blackIsAI && currentPlayer === "black"}
          aiMode={aiMode}
          aiRemainingTime={aiRemainingTime}
          playerLabel={isPvP ? t('colors.black') : undefined}
          onStop={blackIsAI ? handleStop : undefined}
          onResume={blackIsAI && showResume ? resumeAI : undefined}
        />
        <PlayerCard
          color="white"
          score={scores.white}
          isCurrent={currentPlayer === "white"}
          isAIControlled={whiteIsAI}
          aiLevel={aiLevel}
          isThinking={isAIThinking && whiteIsAI && currentPlayer === "white"}
          aiMode={aiMode}
          aiRemainingTime={aiRemainingTime}
          playerLabel={isPvP ? t('colors.white') : undefined}
          onStop={whiteIsAI ? handleStop : undefined}
          onResume={whiteIsAI && showResume ? resumeAI : undefined}
        />
      </div>

      {/* Move History */}
      <div className="flex-1 min-h-0">
        <MoveHistory />
      </div>
    </aside>
  );
}
