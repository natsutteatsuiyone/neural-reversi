"use client";

import { BoardCell } from "./board-cell";
import type { AIMoveProgress } from "@/lib/ai";
import { COLUMN_LABELS, ROW_LABELS } from "@/lib/constants";
import { useReversiStore } from "@/stores/use-reversi-store";
import { useEffect, useMemo, useState } from "react";

interface MoveHistoryItem {
  row: number;
  col: number;
  timestamp: number;
}

const AI_MOVE_HIGHLIGHT_DURATION = 1200;
const MAX_MOVE_HISTORY_SIZE = 3;

export function GameBoard() {
  const board = useReversiStore((state) => state.board);
  const gameOver = useReversiStore((state) => state.gameOver);
  const lastMove = useReversiStore((state) => state.lastMove);
  const isAITurn = useReversiStore((state) => state.isAITurn);
  const isValidMove = useReversiStore((state) => state.isValidMove);
  const makeMove = useReversiStore((state) => state.makeMove);
  const aiMoveProgress = useReversiStore((state) => state.aiMoveProgress) as AIMoveProgress | null;
  const analyzeResults = useReversiStore((state) => state.analyzeResults) as Map<string, AIMoveProgress> | null;
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

        return [newMove, ...prev].slice(0, MAX_MOVE_HISTORY_SIZE);
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
      }, AI_MOVE_HIGHLIGHT_DURATION);

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
                <BoardCell
                  key={`${COLUMN_LABELS[colIndex]}${ROW_LABELS[rowIndex]}`}
                  rowIndex={rowIndex}
                  colIndex={colIndex}
                  cell={cell}
                  lastMove={lastMove}
                  aiMoveProgress={aiMoveProgress}
                  lastAIMove={lastAIMove}
                  moveHistory={moveHistory}
                  gameOver={gameOver}
                  isValidMove={isValidMove}
                  isAITurn={isAITurn}
                  onCellClick={onCellClick}
                  analyzeResults={analyzeResults}
                  gameMode={gameMode}
                  maxScore={maxScore}
                  aiLevel={aiLevel}
                  board={board}
                />
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
