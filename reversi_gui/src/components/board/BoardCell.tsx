import { memo } from "react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import { Stone } from "./Stone";
import { AIThinkingIndicator } from "./AIThinkingIndicator";
import { HintScoreDisplay } from "./HintScoreDisplay";
import { COLUMN_LABELS, ROW_LABELS } from "@/lib/constants";
import type { AIMoveProgress } from "@/lib/ai";
import type { Cell } from "@/types";

interface MoveHistoryItem {
  row: number;
  col: number;
  timestamp: number;
}

interface BoardCellProps {
  rowIndex: number;
  colIndex: number;
  cell: Cell;
  lastMove?: {
    row: number;
    col: number;
    isAI?: boolean;
  } | null;
  aiMoveProgress: AIMoveProgress | null;
  lastAIMove: {
    row: number;
    col: number;
    timestamp: number;
  } | null;
  moveHistory: MoveHistoryItem[];
  gameOver: boolean;
  isValidMove: (row: number, col: number) => boolean;
  isAITurn: () => boolean;
  onCellClick: (row: number, col: number) => void;
  analyzeResults: Map<string, AIMoveProgress> | null;
  maxScore: number | null;
  hintLevel: number;
}

export const BoardCell = memo(function BoardCell({
  rowIndex,
  colIndex,
  cell,
  lastMove,
  aiMoveProgress,
  lastAIMove,
  moveHistory,
  gameOver,
  isValidMove,
  isAITurn,
  onCellClick,
  analyzeResults,
  maxScore,
  hintLevel,
}: BoardCellProps) {
  const isLastMove = lastMove?.row === rowIndex && lastMove?.col === colIndex;
  const isAIMoveCell =
    (aiMoveProgress?.row === rowIndex && aiMoveProgress?.col === colIndex) ||
    (lastAIMove?.row === rowIndex && lastAIMove?.col === colIndex);
  const isValidMoveCell = isValidMove(rowIndex, colIndex);
  const showValidMoveIndicator =
    !cell.color &&
    isValidMoveCell &&
    !isAITurn() &&
    !(analyzeResults && analyzeResults.has(`${rowIndex},${colIndex}`));

  const cellAriaLabel = `${COLUMN_LABELS[colIndex]}${ROW_LABELS[rowIndex]} - ${
    cell.color
      ? `${cell.color} piece`
      : isValidMoveCell && !isAITurn()
        ? "valid move"
        : "empty"
  }`;

  return (
    <motion.button
      className={cn(
        "w-full pt-[100%] relative rounded-sm cell-shadow",
        "bg-board-cell transition-colors duration-150",
        "hover:bg-board-cell-hover",
        "focus:outline-none focus:ring-2 focus:ring-inset focus:ring-white/30",
        isLastMove && "last-move-highlight",
        isAIMoveCell && "bg-board-cell-hover"
      )}
      onClick={() => onCellClick(rowIndex, colIndex)}
      disabled={gameOver || !isValidMoveCell}
      whileHover={!gameOver && isValidMoveCell ? { scale: 0.96 } : {}}
      whileTap={!gameOver && isValidMoveCell ? { scale: 0.92 } : {}}
      transition={{ type: "spring", stiffness: 400, damping: 20 }}
      aria-label={cellAriaLabel}
    >
      <div className="absolute inset-0 flex items-center justify-center">
        {cell.color && <Stone color={cell.color} />}

        {showValidMoveIndicator && (
          <div className="w-[24%] h-[24%] rounded-full bg-white/25 shadow-inner" />
        )}

        <AIThinkingIndicator
          rowIndex={rowIndex}
          colIndex={colIndex}
          aiMoveProgress={aiMoveProgress}
          moveHistory={moveHistory}
          lastAIMove={lastAIMove}
        />

        <HintScoreDisplay
          rowIndex={rowIndex}
          colIndex={colIndex}
          analyzeResults={analyzeResults}
          maxScore={maxScore}
          hintLevel={hintLevel}
          gameOver={gameOver}
        />
      </div>
    </motion.button>
  );
});
