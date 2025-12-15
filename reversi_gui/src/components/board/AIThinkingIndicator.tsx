import { motion } from "framer-motion";
import { Bot } from "lucide-react";
import type { AIMoveProgress } from "@/lib/ai";

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

function ThinkingRipple() {
  return (
    <>
      {[1, 2, 3].map((i) => (
        <motion.div
          key={i}
          className="absolute inset-0 rounded-sm border-2 border-cyan-400/70"
          initial={{ opacity: 0.7, scale: 0.5 }}
          animate={{
            opacity: 0,
            scale: 1.1,
          }}
          transition={{
            duration: 1.5,
            ease: "easeOut",
            repeat: Number.POSITIVE_INFINITY,
            delay: i * 0.3,
          }}
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
    aiMoveProgress &&
    aiMoveProgress.row === rowIndex &&
    aiMoveProgress.col === colIndex;

  if (isCurrentThinkingCell) {
    return (
      <div className="absolute inset-0 flex items-center justify-center z-10">
        <ThinkingRipple />
        <motion.div
          animate={{
            rotate: [0, 10, 0, -10, 0],
            scale: [1, 1.05, 1, 1.05, 1],
          }}
          transition={{
            duration: 2,
            repeat: Number.POSITIVE_INFINITY,
            ease: "easeInOut",
          }}
          className="relative"
        >
          <Bot className="text-cyan-400 drop-shadow-[0_2px_4px_rgba(0,0,0,0.5)]" size={22} />
        </motion.div>
      </div>
    );
  }

  const historyIndex = moveHistory.findIndex(
    (move) => move.row === rowIndex && move.col === colIndex
  );

  if (historyIndex !== -1 && historyIndex < 3) {
    const opacity = 0.7 - historyIndex * 0.15;
    const size = 18 - historyIndex * 3;

    return (
      <div className="absolute inset-0 flex items-center justify-center z-5">
        <motion.div
          initial={{ opacity: 0, scale: 0.5 }}
          animate={{
            opacity,
            scale: size / 22,
          }}
          transition={{
            type: "spring",
            stiffness: 200,
            damping: 10,
          }}
        >
          <Bot className="text-cyan-400/70 drop-shadow-[0_1px_2px_rgba(0,0,0,0.4)]" size={22} />
        </motion.div>
      </div>
    );
  }

  return null;
}
