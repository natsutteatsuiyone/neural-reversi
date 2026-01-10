import { BoardCell } from "./BoardCell";
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

export function Board() {
  const board = useReversiStore((state) => state.board);
  const gameOver = useReversiStore((state) => state.gameOver);
  const lastMove = useReversiStore((state) => state.lastMove);
  const isAITurn = useReversiStore((state) => state.isAITurn);
  const isValidMove = useReversiStore((state) => state.isValidMove);
  const makeMove = useReversiStore((state) => state.makeMove);
  const aiMoveProgress = useReversiStore((state) => state.aiMoveProgress) as AIMoveProgress | null;
  const analyzeResults = useReversiStore((state) => state.analyzeResults) as Map<string, AIMoveProgress> | null;

  const [moveHistory, setMoveHistory] = useState<MoveHistoryItem[]>([]);
  const [lastAIMove, setLastAIMove] = useState<{ row: number; col: number; timestamp: number } | null>(null);

  const maxScore = useMemo(() => {
    if (!analyzeResults || analyzeResults.size === 0) return null;

    let highest = Number.NEGATIVE_INFINITY;
    for (const result of analyzeResults.values()) {
      const rounded = Math.round(result.score);
      if (rounded > highest) {
        highest = rounded;
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
        timestamp: Date.now(),
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
    <div className="h-full w-full flex flex-col items-center justify-center">
      {/* Board container - maintains square aspect ratio */}
      <div className="h-full max-h-[min(calc(100vh-200px),600px)] max-w-full aspect-square flex flex-col">
        {/* Column labels */}
        <div className="flex ml-7 shrink-0 h-6">
          {COLUMN_LABELS.map((label) => (
            <div
              key={label}
              className="flex-1 text-center text-sm font-semibold text-foreground-secondary uppercase"
              aria-hidden="true"
            >
              {label}
            </div>
          ))}
        </div>

        <div className="flex flex-1 min-h-0">
          {/* Row labels */}
          <div className="flex flex-col justify-around w-7 shrink-0">
            {ROW_LABELS.map((label) => (
              <div
                key={label}
                className="text-center text-sm font-semibold text-foreground-secondary"
                aria-hidden="true"
              >
                {label}
              </div>
            ))}
          </div>

          {/* Board */}
          <div className="flex-1 bg-board-surface p-1.5 rounded-xl shadow-xl">
            <div className="grid grid-cols-8 grid-rows-8 gap-0.5 h-full w-full">
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
                    maxScore={maxScore}
                  />
                ))
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
