"use client";

import { motion } from "framer-motion";
import { memo } from "react";
import { cn } from "@/lib/utils";
import { GamePiece } from "./game-piece";
import { AIThinkingIndicator } from "./ai-thinking-indicator";
import { AIScoreDisplay } from "./ai-score-display";
import { COLUMN_LABELS, ROW_LABELS } from "@/lib/constants";
import type { AIMoveProgress } from "@/lib/ai";
import type { Cell, Board } from "@/types";

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
  gameMode: string;
  maxScore: number | null;
  aiLevel: number;
  board: Board;
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
  gameMode,
  maxScore,
  aiLevel,
  board,
}: BoardCellProps) {
  const isLastMove = lastMove?.row === rowIndex && lastMove?.col === colIndex;
  const isAIMoveCell = (aiMoveProgress?.row === rowIndex && aiMoveProgress?.col === colIndex) ||
                      (lastAIMove?.row === rowIndex && lastAIMove?.col === colIndex);
  const isValidMoveCell = isValidMove(rowIndex, colIndex);
  const showValidMoveIndicator = !cell.color && isValidMoveCell && !isAITurn() &&
    !(gameMode === "analyze" && analyzeResults && analyzeResults.has(`${rowIndex},${colIndex}`));

  const cellClasses = cn(
    "w-full pt-[100%] relative bg-[#0e7250] rounded-sm",
    "hover:bg-[#0f8259] transition-colors",
    "focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-[#11936a]",
    isLastMove && "ring-2 ring-[#c8b45c]/70",
    "shadow-[inset_1px_1px_1px_rgba(255,255,255,0.1),inset_-1px_-1px_1px_rgba(0,0,0,0.1)]",
    isAIMoveCell && "bg-[#0e7d58]"
  );

  const cellAriaLabel = `${COLUMN_LABELS[colIndex]}${ROW_LABELS[rowIndex]} - ${
    cell.color
      ? `${cell.color} piece`
      : isValidMoveCell && !isAITurn()
      ? "valid move"
      : "empty"
  }`;

  return (
    <motion.button
      key={`${COLUMN_LABELS[colIndex]}${ROW_LABELS[rowIndex]}`}
      className={cellClasses}
      onClick={() => onCellClick(rowIndex, colIndex)}
      disabled={gameOver || !isValidMoveCell}
      whileHover={
        !gameOver && isValidMoveCell ? { scale: 0.95 } : {}
      }
      transition={{ type: "spring", stiffness: 400, damping: 17 }}
      aria-label={cellAriaLabel}
    >
      <div className="absolute inset-0 flex items-center justify-center">
        {cell.color && (
          <GamePiece color={cell.color} isNew={cell.isNew} />
        )}

        {showValidMoveIndicator && (
          <div className="w-[20%] h-[20%] rounded-full bg-emerald-100 opacity-20" />
        )}

        <AIThinkingIndicator
          rowIndex={rowIndex}
          colIndex={colIndex}
          aiMoveProgress={aiMoveProgress}
          moveHistory={moveHistory}
          lastAIMove={lastAIMove}
        />

        <AIScoreDisplay
          rowIndex={rowIndex}
          colIndex={colIndex}
          analyzeResults={analyzeResults}
          gameMode={gameMode}
          maxScore={maxScore}
          aiLevel={aiLevel}
          gameOver={gameOver}
          board={board}
        />
      </div>
    </motion.button>
  );
});
