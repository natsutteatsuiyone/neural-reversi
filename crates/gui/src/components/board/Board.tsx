import { Canvas } from "@react-three/fiber";
import { Board3DScene } from "./Board3DScene";
import type { AIMoveProgress } from "@/services/types";
import { useReversiStore } from "@/stores/use-reversi-store";
import { Suspense, useCallback, useEffect, useMemo, useState } from "react";

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
  const skipAnimation = useReversiStore((state) => state.skipAnimation);

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

  const onCellClick = useCallback((row: number, col: number) => {
    if (isAITurn() || gameOver || !isValidMove(row, col)) {
      return;
    }

    makeMove({
      row,
      col,
      isAI: false,
    });
  }, [isAITurn, gameOver, isValidMove, makeMove]);

  return (
    <div className="h-full w-full">
      <Canvas frameloop="demand" resize={{ debounce: { scroll: 50, resize: 100 } }}>
        <Suspense fallback={null}>
          <Board3DScene
            board={board}
            lastMove={lastMove}
            gameOver={gameOver}
            isValidMove={isValidMove}
            isAITurn={isAITurn}
            onCellClick={onCellClick}
            aiMoveProgress={aiMoveProgress}
            lastAIMove={lastAIMove}
            moveHistory={moveHistory}
            analyzeResults={analyzeResults}
            maxScore={maxScore}
            skipAnimation={skipAnimation}
          />
        </Suspense>
      </Canvas>
    </div>
  );
}
