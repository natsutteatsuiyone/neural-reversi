"use client";

import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import { GamePiece } from "./game-piece";
import { AIThinkingIndicator } from "./ai-thinking-indicator";
import { AIScoreDisplay } from "./ai-score-display";
import { COLUMN_LABELS, ROW_LABELS } from "@/lib/constants";
import { useReversiStore } from "@/stores/use-reversi-store";
import { useEffect, useMemo, useState } from "react";

interface MoveHistoryItem {
  row: number;
  col: number;
  timestamp: number;
}

export function GameBoard() {
  const board = useReversiStore((state) => state.board);
  const gameOver = useReversiStore((state) => state.gameOver);
  const lastMove = useReversiStore((state) => state.lastMove);
  const isAITurn = useReversiStore((state) => state.isAITurn);
  const isValidMove = useReversiStore((state) => state.isValidMove);
  const makeMove = useReversiStore((state) => state.makeMove);
  const aiMoveProgress = useReversiStore((state) => state.aiMoveProgress);
  const analyzeResults = useReversiStore((state) => state.analyzeResults);
  const gameMode = useReversiStore((state) => state.gameMode);
  const aiLevel = useReversiStore((state) => state.aiLevel);

  const [moveHistory, setMoveHistory] = useState<MoveHistoryItem[]>([]);
  const [lastAIMove, setLastAIMove] = useState<{ row: number; col: number; timestamp: number } | null>(null);

  const maxScore = useMemo(() => {
    if (!analyzeResults || analyzeResults.size === 0) return null;

    let highest = Number.NEGATIVE_INFINITY;
    for (const result of analyzeResults.values()) {
      if (result.score > highest) {
        highest = result.score;
      }
    }
    return highest;
  }, [analyzeResults]);

  useEffect(() => {
    if (
      aiMoveProgress &&
      aiMoveProgress.row !== undefined &&
      aiMoveProgress.col !== undefined
    ) {
      const newMove = {
        row: aiMoveProgress.row,
        col: aiMoveProgress.col,
        timestamp: Date.now(),
      };

      setMoveHistory((prev) => {
        if (
          prev.length > 0 &&
          prev[0].row === newMove.row &&
          prev[0].col === newMove.col
        ) {
          return prev;
        }

        return [newMove, ...prev].slice(0, 3);
      });
    } else if (!isAITurn()) {
      setMoveHistory([]);
    }
  }, [aiMoveProgress, isAITurn]);

  useEffect(() => {
    if (lastMove?.isAI) {
      setLastAIMove({
        row: lastMove.row,
        col: lastMove.col,
        timestamp: Date.now()
      });

      const timer = setTimeout(() => {
        setLastAIMove(null);
      }, 1200);

      return () => clearTimeout(timer);
    }
  }, [lastMove]);

  function onCellClick(row: number, col: number) {
    if (isAITurn() || gameOver || !isValidMove(row, col)) {
      return;
    }

    makeMove({
      row,
      col,
      isAI: false,
    });
  }

  return (
    <div className="w-full max-w-2xl lg:w-[calc(100vh-2rem)] lg:max-w-[calc(100vh-2rem)] shrink-0 p-2">
      {/* Column labels */}
      <div className="flex mb-2 ms-12 me-4">
        {COLUMN_LABELS.map((label) => (
          <div
            key={label}
            className="flex-1 text-center text-xl font-bold text-emerald-50"
            aria-hidden="true"
          >
            {label}
          </div>
        ))}
      </div>

      <div className="flex">
        {/* Row labels */}
        <div className="flex flex-col justify-around pr-4 w-8 mb-4 mt-3">
          {ROW_LABELS.map((label) => (
            <div
              key={label}
              className="text-center text-xl font-bold text-emerald-50"
              aria-hidden="true"
            >
              {label}
            </div>
          ))}
        </div>

        {/* Board */}
        <div className="flex-1 bg-[#0d6245] p-4 rounded-lg shadow-[inset_0_2px_12px_rgba(0,0,0,0.3)]">
          <div className="grid grid-cols-8 gap-1">
            {board.map((row, rowIndex) =>
              row.map((cell, colIndex) => (
                <motion.button
                  key={`${COLUMN_LABELS[colIndex]}${ROW_LABELS[rowIndex]}`}
                  className={cn(
                    "w-full pt-[100%] relative bg-[#0e7250] rounded-sm",
                    "hover:bg-[#0f8259] transition-colors",
                    "focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-[#11936a]",
                    lastMove &&
                      lastMove.row === rowIndex &&
                      lastMove.col === colIndex &&
                      "ring-2 ring-[#c8b45c]/70",
                    "shadow-[inset_1px_1px_1px_rgba(255,255,255,0.1),inset_-1px_-1px_1px_rgba(0,0,0,0.1)]",
                    (aiMoveProgress?.row === rowIndex && aiMoveProgress?.col === colIndex) ||
                    (lastAIMove?.row === rowIndex && lastAIMove?.col === colIndex)
                      ? "bg-[#0e7d58]"
                      : ""
                  )}
                  onClick={() => onCellClick(rowIndex, colIndex)}
                  disabled={gameOver || !isValidMove(rowIndex, colIndex)}
                  whileHover={
                    !gameOver && isValidMove(rowIndex, colIndex)
                      ? { scale: 0.95 }
                      : {}
                  }
                  transition={{ type: "spring", stiffness: 400, damping: 17 }}
                  aria-label={`${COLUMN_LABELS[colIndex]}${
                    ROW_LABELS[rowIndex]
                  } - ${
                    cell.color
                      ? `${cell.color} piece`
                      : isValidMove(rowIndex, colIndex) && !isAITurn()
                      ? "valid move"
                      : "empty"
                  }`}
                >
                  <div className="absolute inset-0 flex items-center justify-center">
                    {cell.color && (
                        <GamePiece color={cell.color} isNew={cell.isNew} />
                    )}

                    {!cell.color &&
                      isValidMove(rowIndex, colIndex) &&
                      !isAITurn() &&
                      !(gameMode === "analyze" && analyzeResults && analyzeResults.has(`${rowIndex},${colIndex}`)) && (
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
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
