import { useCallback, useEffect, useMemo, useState } from "react";
import { getValidMoves } from "@/domain/game/game-logic";
import { solverCandidatesToAnalysisResults } from "@/domain/solver/solver-candidates";
import { AI_MOVE_HIGHLIGHT_DURATION_MS } from "@/lib/timing";
import { useReversiStore } from "@/stores/use-reversi-store";
import type { Board3DSceneProps } from "./Board3DScene";
import {
  EMPTY_AI_PROGRESS_TRAIL,
  nextAIProgressTrail,
  type AIProgressTrailCell,
} from "./ai-progress-trail";

/**
 * Resolves the single set of {@link Board3DSceneProps} the renderer needs,
 * choosing between the live game and the active Solver Session.
 *
 * This is the one place the game-vs-solver fork lives. Every rendered
 * property is derived here behind a one-object interface, so the board
 * component is a thin shell and a new solver-aware property can no longer
 * be added to one branch and silently forgotten in another. The
 * AI-move-highlight view-state (which is presentation, not store state)
 * is owned here too.
 */
export function useBoardScene(): Board3DSceneProps {
  const board = useReversiStore((state) => state.board);
  const gameOver = useReversiStore((state) => state.gameOver);
  const lastMove = useReversiStore((state) => state.lastMove);
  const isAITurn = useReversiStore((state) => state.isAITurn);
  const isValidMove = useReversiStore((state) => state.isValidMove);
  const makeMove = useReversiStore((state) => state.makeMove);
  const aiMoveProgress = useReversiStore((state) => state.aiMoveProgress);
  const analyzeResults = useReversiStore((state) => state.analyzeResults);
  const skipAnimation = useReversiStore((state) => state.skipAnimation);
  const isGameAnalyzing = useReversiStore((state) => state.isGameAnalyzing);

  const isSolverActive = useReversiStore((state) => state.isSolverActive);
  const solverCurrentBoard = useReversiStore((state) => state.solverCurrentBoard);
  const solverCurrentPlayer = useReversiStore((state) => state.solverCurrentPlayer);
  const solverCandidates = useReversiStore((state) => state.solverCandidates);
  const solverMode = useReversiStore((state) => state.solverMode);
  const advanceSolver = useReversiStore((state) => state.advanceSolver);

  const [moveHistory, setMoveHistory] = useState<AIProgressTrailCell[]>(
    EMPTY_AI_PROGRESS_TRAIL,
  );
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

  const solverAnalyzeResults = useMemo(() => {
    if (!isSolverActive) return null;
    return solverCandidatesToAnalysisResults(solverCandidates);
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
    const cell =
      aiMoveProgress &&
      aiMoveProgress.row !== undefined &&
      aiMoveProgress.col !== undefined
        ? { row: aiMoveProgress.row, col: aiMoveProgress.col }
        : null;
    setMoveHistory((prev) =>
      nextAIProgressTrail(prev, cell, isAITurnNow, Date.now()),
    );
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
      }, AI_MOVE_HIGHLIGHT_DURATION_MS);

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
      if (isGameAnalyzing || isAITurn() || gameOver || !isValidMove(row, col)) {
        return;
      }

      makeMove({
        row,
        col,
        isAI: false,
      });
    },
    [
      isSolverActive,
      isValidSolverMove,
      advanceSolver,
      isGameAnalyzing,
      isAITurn,
      gameOver,
      isValidMove,
      makeMove,
    ],
  );

  const isValidGameMove = useCallback(
    (row: number, col: number) => !isGameAnalyzing && isValidMove(row, col),
    [isGameAnalyzing, isValidMove],
  );
  const activeIsAITurn = useCallback(
    () => (isSolverActive ? false : isAITurn()),
    [isSolverActive, isAITurn],
  );

  return {
    board: isSolverActive && solverCurrentBoard ? solverCurrentBoard : board,
    lastMove: isSolverActive ? null : lastMove,
    gameOver: isSolverActive ? false : gameOver,
    isValidMove: isSolverActive ? isValidSolverMove : isValidGameMove,
    isAITurn: activeIsAITurn,
    onCellClick,
    aiMoveProgress: isSolverActive ? null : aiMoveProgress,
    lastAIMove: isSolverActive ? null : lastAIMove,
    moveHistory: isSolverActive ? EMPTY_AI_PROGRESS_TRAIL : moveHistory,
    analyzeResults: isSolverActive ? solverAnalyzeResults : analyzeResults,
    maxScore,
    skipAnimation: isSolverActive ? true : skipAnimation,
    showHintWaitingBar: !(isSolverActive && solverMode === "bestOnly"),
  };
}
