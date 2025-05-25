"use client";

import { motion } from "framer-motion";
import { Bot, Zap } from "lucide-react";
import { cn } from "@/lib/utils";
import { GamePiece } from "./game-piece";
import { COLUMN_LABELS, ROW_LABELS } from "@/lib/constants";
import { useReversiStore } from "@/stores/use-reversi-store";
import { useEffect, useMemo, useState } from "react";
import type { AIMoveProgress } from "@/lib/ai";
import type { Board } from "@/types";
import { calculateScores } from "@/lib/game-logic";

interface MoveHistoryItem {
  row: number;
  col: number;
  timestamp: number;
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

function AIScoreDisplay({
  rowIndex,
  colIndex,
  analyzeResults,
  gameMode,
  maxScore,
  aiLevel,
  gameOver,
  board,
}: {
  rowIndex: number;
  colIndex: number;
  analyzeResults: Map<string, AIMoveProgress> | null;
  gameMode: string;
  maxScore: number | null;
  aiLevel: number;
  gameOver: boolean;
  board: Board;
}) {
  if (gameMode !== "analyze" || !analyzeResults || gameOver) {
    return null;
  }

  const key = `${rowIndex},${colIndex}`;
  const result = analyzeResults.get(key);

  if (result === undefined) {
    return null;
  }

  const score = result.score;
  const depth = result.depth;
  const acc = result.acc;

  const { black, white } = calculateScores(board);
  const emptyCount = 64 - black - white;

  const displayScore = score > 0 ? `+${score.toFixed(1)}` : score.toFixed(1);
  const textColor = maxScore !== null && score === maxScore ? "text-green-500" : "text-white";

  return (
    <div className="absolute inset-0 flex flex-col items-center justify-center z-20 pointer-events-none">
      <div className={`text-lg font-bold ${textColor} px-1 rounded`}>
        {displayScore}
      </div>
      {depth !== aiLevel && depth !== emptyCount && (
        <div className="text-xs text-gray-200 px-1 rounded mt-1">
          {depth}@{acc}%
        </div>
      )}
    </div>
  );
}

function AIThinkingIndicator({
  rowIndex,
  colIndex,
  aiMoveProgress,
  moveHistory,
  lastAIMove,
}: {
  rowIndex: number;
  colIndex: number;
  aiMoveProgress: { row: number; col: number; score: number } | null;
  moveHistory: MoveHistoryItem[];
  lastAIMove: { row: number; col: number; timestamp: number } | null;
}) {
  if (
    (!aiMoveProgress && !lastAIMove) ||
    (lastAIMove &&
     lastAIMove.row === rowIndex &&
     lastAIMove.col === colIndex &&
     Date.now() - lastAIMove.timestamp < 1500)
  ) {
    return null;
  }

  if (aiMoveProgress && aiMoveProgress.row === rowIndex && aiMoveProgress.col === colIndex) {
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
          <Bot
            className="text-cyan-400 drop-shadow-glow"
            size={24}
          />
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
    const opacity = 0.8 - historyIndex * 0.2; // 0.8, 0.6, 0.4
    const size = 20 - historyIndex * 4; // 20px, 16px, 12px

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
  const [lastAIMove, setLastAIMove] = useState<{row: number; col: number; timestamp: number} | null>(null);

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
