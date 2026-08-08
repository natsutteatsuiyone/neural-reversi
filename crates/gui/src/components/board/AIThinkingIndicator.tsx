import type { CSSProperties } from "react";
import { Bot } from "lucide-react";
import type { AIMoveProgress } from "@/services/types";

interface MoveHistoryItem {
  row: number;
  col: number;
  timestamp: number;
}

interface AIThinkingIndicatorProps {
  rowIndex: number;
  colIndex: number;
  aiMoveProgress: AIMoveProgress | null;
  moveHistory: MoveHistoryItem[];
  lastAIMove: { row: number; col: number; timestamp: number } | null;
}

const RIPPLE_DELAYS = ["0.3s", "0.6s", "0.9s"] as const;

function ThinkingRipple() {
  return (
    <>
      {RIPPLE_DELAYS.map((animationDelay) => (
        <div
          key={animationDelay}
          className="absolute inset-0 rounded-sm border-2 border-accent-ai/70 animate-pulse-ring"
          style={{ animationDelay }}
        />
      ))}
    </>
  );
}

export function AIThinkingIndicator({
  rowIndex,
  colIndex,
  aiMoveProgress,
  moveHistory,
  lastAIMove,
}: AIThinkingIndicatorProps) {
  const isRecentAIMove =
    lastAIMove &&
    lastAIMove.row === rowIndex &&
    lastAIMove.col === colIndex &&
    Date.now() - lastAIMove.timestamp < 1500;

  if (!aiMoveProgress && !lastAIMove) {
    return null;
  }

  if (isRecentAIMove) {
    return null;
  }

  const isCurrentThinkingCell =
    aiMoveProgress && aiMoveProgress.row === rowIndex && aiMoveProgress.col === colIndex;

  if (isCurrentThinkingCell) {
    return (
      <div
        data-ai-thinking="current"
        data-board-cell={`${rowIndex},${colIndex}`}
        className="absolute inset-0 flex items-center justify-center z-10"
      >
        <ThinkingRipple />
        <div className="relative animate-thinking-bot">
          <Bot className="text-accent-ai drop-shadow-[0_2px_4px_rgba(0,0,0,0.5)]" size={22} />
        </div>
      </div>
    );
  }

  const historyIndex = moveHistory.findIndex(
    (move) => move.row === rowIndex && move.col === colIndex,
  );

  if (historyIndex !== -1 && historyIndex < 3) {
    const opacity = 0.7 - historyIndex * 0.15;
    const size = 18 - historyIndex * 3;
    const scale = size / 22;

    return (
      <div
        data-ai-thinking="trail"
        data-board-cell={`${rowIndex},${colIndex}`}
        data-ai-trail-index={historyIndex}
        className="absolute inset-0 flex items-center justify-center z-5"
      >
        <div
          className="animate-scale-in"
          style={
            {
              "--scale-in-from": 0.5,
              "--scale-in-overshoot": scale * 1.05,
              "--scale-in-to": scale,
              "--scale-in-opacity": opacity,
            } as CSSProperties
          }
        >
          <Bot className="text-accent-ai/70 drop-shadow-[0_1px_2px_rgba(0,0,0,0.4)]" size={22} />
        </div>
      </div>
    );
  }

  return null;
}
