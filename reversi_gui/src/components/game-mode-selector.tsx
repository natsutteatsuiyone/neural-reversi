"use client";

import { useReversiStore } from "@/stores/use-reversi-store";
import { GameModeOptions } from "./game-mode/game-mode-options";
import { AIModeTabs } from "./game-mode/ai-mode-tabs";
import { AISettings } from "./game-mode/ai-settings";
import { TimeSettings } from "./game-mode/time-settings";

interface GameModeSelectorProps {
  disabled?: boolean;
}

export function GameModeSelector({ disabled }: GameModeSelectorProps) {
  const gameMode = useReversiStore((state) => state.gameMode);
  const aiLevel = useReversiStore((state) => state.aiLevel);
  const aiMode = useReversiStore((state) => state.aiMode);
  const gameTimeLimit = useReversiStore((state) => state.gameTimeLimit);
  const setGameMode = useReversiStore((state) => state.setGameMode);
  const setAILevelChange = useReversiStore((state) => state.setAILevelChange);
  const setAIMode = useReversiStore((state) => state.setAIMode);
  const setGameTimeLimit = useReversiStore((state) => state.setGameTimeLimit);

  return (
    <div className="space-y-2">
      <h2 className="text-lg font-medium text-white/90">Game Mode</h2>
      <GameModeOptions
        gameMode={gameMode}
        setGameMode={setGameMode}
        disabled={disabled}
      />
      <div className="space-y-3 pt-3 border-t border-white/10">
        {gameMode !== "analyze" && (
          <AIModeTabs
            aiMode={aiMode}
            setAIMode={setAIMode}
            disabled={disabled}
          />
        )}

        {gameMode === "analyze" || aiMode === "level" ? (
          <AISettings
            aiLevel={aiLevel}
            setAILevelChange={setAILevelChange}
            disabled={disabled}
          />
        ) : (
          <TimeSettings
            aiMode={aiMode}
            gameTimeLimit={gameTimeLimit}
            setGameTimeLimit={setGameTimeLimit}
            disabled={disabled}
          />
        )}
      </div>
    </div>
  );
}
