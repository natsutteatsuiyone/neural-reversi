import { motion } from "framer-motion";
import { Bot, Zap } from "lucide-react";
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

function ThinkingRippleEffect() {
  return (
    <>
      {[1, 2, 3].map((i) => (
        <motion.div
          key={i}
          className="absolute inset-0 rounded-sm border-2 border-cyan-400"
          initial={{ opacity: 0.7, scale: 0.3 }}
          animate={{
            opacity: 0,
            scale: 1.2,
          }}
          transition={{
            duration: 2,
            ease: "easeOut",
            repeat: Number.POSITIVE_INFINITY,
            delay: i * 0.4,
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
        <ThinkingRippleEffect />
        <motion.div
          animate={{
            rotate: [0, 20, 0, -20, 0],
            scale: [1, 1.1, 1, 1.1, 1],
          }}
          transition={{
            duration: 2,
            repeat: Number.POSITIVE_INFINITY,
            ease: "easeInOut",
          }}
          className="relative"
        >
          <Bot className="text-cyan-400 drop-shadow-glow" size={24} />
          <motion.div
            animate={{ opacity: [0, 1, 0] }}
            transition={{ duration: 0.8, repeat: Number.POSITIVE_INFINITY }}
            className="absolute -top-2 -right-2"
          >
            <Zap className="text-yellow-300 fill-yellow-300" size={10} />
          </motion.div>
        </motion.div>
      </div>
    );
  }

  const historyIndex = moveHistory.findIndex(
    (move) => move.row === rowIndex && move.col === colIndex
  );

  if (historyIndex !== -1 && historyIndex < 3) {
    const opacity = 0.8 - historyIndex * 0.2;
    const size = 20 - historyIndex * 4;

    return (
      <div className="absolute inset-0 flex items-center justify-center z-5">
        <motion.div
          initial={{ opacity: 0, scale: 0.5 }}
          animate={{
            opacity,
            scale: size / 24,
            rotate: historyIndex * 15,
          }}
          transition={{
            type: "spring",
            stiffness: 200,
            damping: 10,
          }}
        >
          <Bot className="text-cyan-300/70" size={24} />
        </motion.div>
      </div>
    );
  }

  return null;
}