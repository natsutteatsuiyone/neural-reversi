import { Canvas } from "@react-three/fiber";
import { Board3DScene } from "./Board3DScene";
import { useReversiStore } from "@/stores/use-reversi-store";
import { getValidMoves } from "@/lib/game-logic";
import type { AIMoveProgress } from "@/services/types";
import { Suspense, useCallback, useEffect, useMemo, useState } from "react";

interface MoveHistoryItem {
  row: number;
  col: number;
  timestamp: number;
}

const AI_MOVE_HIGHLIGHT_DURATION = 1200;
const MAX_MOVE_HISTORY_SIZE = 3;
const EMPTY_MOVE_HISTORY: MoveHistoryItem[] = [];

export function Board() {
  const board = useReversiStore((state) => state.board);
  const gameOver = useReversiStore((state) => state.gameOver);
  const lastMove = useReversiStore((state) => state.lastMove);
  const isAITurn = useReversiStore((state) => state.isAITurn);
  const isValidMove = useReversiStore((state) => state.isValidMove);
  const makeMove = useReversiStore((state) => state.makeMove);
  const aiMoveProgress = useReversiStore((state) => state.aiMoveProgress);
  const analyzeResults = useReversiStore((state) => state.analyzeResults);
  const skipAnimation = useReversiStore((state) => state.skipAnimation);

  const isSolverActive = useReversiStore((state) => state.isSolverActive);
  const solverCurrentBoard = useReversiStore((state) => state.solverCurrentBoard);
  const solverCurrentPlayer = useReversiStore((state) => state.solverCurrentPlayer);
  const solverCandidates = useReversiStore((state) => state.solverCandidates);
  const solverMode = useReversiStore((state) => state.solverMode);
  const advanceSolver = useReversiStore((state) => state.advanceSolver);

  const [moveHistory, setMoveHistory] = useState<MoveHistoryItem[]>([]);
  const [lastAIMove, setLastAIMove] = useState<{ row: number; col: number; timestamp: number } | null>(null);

  const solverLegalMoves = useMemo(() => {
    if (!isSolverActive || !solverCurrentBoard || !solverCurrentPlayer) {
      return null;
    }
    return getValidMoves(solverCurrentBoard, solverCurrentPlayer);
  }, [isSolverActive, solverCurrentBoard, solverCurrentPlayer]);

  const isValidSolverMove = useCallback(
    (row: number, col: number) => {
      if (!solverLegalMoves) return false;
      return solverLegalMoves.some(([r, c]) => r === row && c === col);
    },
    [solverLegalMoves],
  );

  // Force acc=100 on endgame completion so non-100 target selectivities still trip the "done" branch.
  const solverAnalyzeResults = useMemo(() => {
    if (!isSolverActive) return null;
    const map = new Map<string, AIMoveProgress>();
    for (const [key, candidate] of solverCandidates) {
      map.set(key, {
        bestMove: candidate.move,
        row: candidate.row,
        col: candidate.col,
        score: candidate.score,
        depth: candidate.depth,
        targetDepth: candidate.targetDepth,
        acc: candidate.isEndgame && candidate.isComplete ? 100 : candidate.acc,
        nodes: 0,
        pvLine: candidate.pvLine,
        isEndgame: candidate.isEndgame,
      });
    }
    return map;
  }, [isSolverActive, solverCandidates]);

  const maxScore = useMemo(() => {
    const source = isSolverActive ? solverAnalyzeResults : analyzeResults;
    if (!source || source.size === 0) return null;

    let highest = Number.NEGATIVE_INFINITY;
    for (const result of source.values()) {
      const rounded = Math.round(result.score);
      if (rounded > highest) {
        highest = rounded;
      }
    }
    return highest;
  }, [isSolverActive, solverAnalyzeResults, analyzeResults]);

  const isAITurnNow = useReversiStore((state) => state.isAITurn());

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
    } else if (!isAITurnNow) {
      setMoveHistory([]);
    }
  }, [aiMoveProgress, isAITurnNow]);

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
    setLastAIMove(null);
  }, [lastMove]);

  const onCellClick = useCallback(
    (row: number, col: number) => {
      if (isSolverActive) {
        if (!isValidSolverMove(row, col)) return;
        void advanceSolver(row, col);
        return;
      }
      if (isAITurn() || gameOver || !isValidMove(row, col)) {
        return;
      }

      makeMove({
        row,
        col,
        isAI: false,
      });
    },
    [isSolverActive, isValidSolverMove, advanceSolver, isAITurn, gameOver, isValidMove, makeMove],
  );

  const activeBoard = isSolverActive && solverCurrentBoard ? solverCurrentBoard : board;
  const activeIsValidMove = isSolverActive ? isValidSolverMove : isValidMove;
  const activeIsAITurn = useCallback(
    () => (isSolverActive ? false : isAITurn()),
    [isSolverActive, isAITurn],
  );
  const activeGameOver = isSolverActive ? false : gameOver;
  const activeLastMove = isSolverActive ? null : lastMove;
  const activeAiMoveProgress = isSolverActive ? null : aiMoveProgress;
  const activeLastAIMove = isSolverActive ? null : lastAIMove;
  const activeMoveHistory = isSolverActive ? EMPTY_MOVE_HISTORY : moveHistory;
  const activeAnalyzeResults = isSolverActive ? solverAnalyzeResults : analyzeResults;
  const activeSkipAnimation = isSolverActive ? true : skipAnimation;

  return (
    <div className="h-full w-full">
      <Canvas frameloop="demand" shadows resize={{ debounce: { scroll: 50, resize: 100 } }}>
        <Suspense fallback={null}>
          <Board3DScene
            board={activeBoard}
            lastMove={activeLastMove}
            gameOver={activeGameOver}
            isValidMove={activeIsValidMove}
            isAITurn={activeIsAITurn}
            onCellClick={onCellClick}
            aiMoveProgress={activeAiMoveProgress}
            lastAIMove={activeLastAIMove}
            moveHistory={activeMoveHistory}
            analyzeResults={activeAnalyzeResults}
            maxScore={maxScore}
            skipAnimation={activeSkipAnimation}
            showHintWaitingBar={!(isSolverActive && solverMode === "bestOnly")}
          />
        </Suspense>
      </Canvas>
    </div>
  );
}
