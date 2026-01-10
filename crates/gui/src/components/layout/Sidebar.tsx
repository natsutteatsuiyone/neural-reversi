import { PlayerCard } from "@/components/game/PlayerCard";
import { MoveHistory } from "@/components/game/MoveHistory";
import { useReversiStore } from "@/stores/use-reversi-store";

export function Sidebar() {
  const currentPlayer = useReversiStore((state) => state.currentPlayer);
  const gameMode = useReversiStore((state) => state.gameMode);
  const gameStatus = useReversiStore((state) => state.gameStatus);
  const aiLevel = useReversiStore((state) => state.aiLevel);
  const aiMode = useReversiStore((state) => state.aiMode);
  const aiRemainingTime = useReversiStore((state) => state.aiRemainingTime);
  const getScores = useReversiStore((state) => state.getScores);
  const abortAIMove = useReversiStore((state) => state.abortAIMove);

  const scores = getScores();
  const blackIsAI = gameMode === "ai-black";
  const whiteIsAI = gameMode === "ai-white";

  const handleStop = () => {
    void abortAIMove();
  };

  return (
    <aside className="w-80 border-l border-white/10 bg-background-secondary flex flex-col shrink-0">
      {/* Player Cards */}
      <div className="p-4 space-y-3 border-b border-white/10">
        <PlayerCard
          color="black"
          score={scores.black}
          isCurrent={currentPlayer === "black"}
          isAIControlled={blackIsAI}
          aiLevel={aiLevel}
          isThinking={blackIsAI && currentPlayer === "black" && gameStatus === "playing"}
          aiMode={aiMode}
          aiRemainingTime={aiRemainingTime}
          onStop={blackIsAI ? handleStop : undefined}
        />
        <PlayerCard
          color="white"
          score={scores.white}
          isCurrent={currentPlayer === "white"}
          isAIControlled={whiteIsAI}
          aiLevel={aiLevel}
          isThinking={whiteIsAI && currentPlayer === "white" && gameStatus === "playing"}
          aiMode={aiMode}
          aiRemainingTime={aiRemainingTime}
          onStop={whiteIsAI ? handleStop : undefined}
        />
      </div>

      {/* Move History */}
      <div className="flex-1 min-h-0">
        <MoveHistory />
      </div>
    </aside>
  );
}
