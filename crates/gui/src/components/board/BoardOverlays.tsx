import { useMemo } from "react";
import type { AIMoveProgress } from "@/services/types";
import { cellKey, type CellKey } from "@/domain/game/cell-key";
import { AIThinkingIndicator } from "./AIThinkingIndicator";
import { CellHtmlOverlay } from "./CellHtmlOverlay";
import { HintScoreDisplay } from "./HintScoreDisplay";
import type { AIProgressTrailCell } from "./ai-progress-trail";

interface BoardOverlaysProps {
  cellPixelSize: number;
  aiMoveProgress: AIMoveProgress | null;
  lastAIMove: { row: number; col: number; timestamp: number } | null;
  moveHistory: AIProgressTrailCell[];
  analyzeResults: Map<CellKey, AIMoveProgress> | null;
  maxScore: number | null;
  gameOver: boolean;
  isValidMove: (row: number, col: number) => boolean;
  showHintWaitingBar: boolean;
}

/**
 * The per-cell HTML overlays (AI thinking indicator + hint score). Computes,
 * once per relevant-input change, which cells need an overlay mounted, so the
 * 8×8 mount gate is no longer re-scanned (with an uncached `isValidMove`) on
 * every unrelated re-render. The leaves own what to draw; this owns only which
 * cells get one.
 */
export function BoardOverlays({
  cellPixelSize,
  aiMoveProgress,
  lastAIMove,
  moveHistory,
  analyzeResults,
  maxScore,
  gameOver,
  isValidMove,
  showHintWaitingBar,
}: BoardOverlaysProps) {
  const overlayCells = useMemo(() => {
    const cells: { row: number; col: number; isValidMoveCell: boolean }[] = [];
    for (let row = 0; row < 8; row++) {
      for (let col = 0; col < 8; col++) {
        const isThinkingCell = aiMoveProgress?.row === row && aiMoveProgress?.col === col;
        const isHistoryCell = moveHistory.some((m) => m.row === row && m.col === col);
        const isRecentAIMove = lastAIMove?.row === row && lastAIMove?.col === col;
        const isValidMoveCell = isValidMove(row, col);
        const hasHint =
          analyzeResults?.has(cellKey(row, col)) ||
          (!gameOver && isValidMoveCell && analyzeResults !== null);

        if (isThinkingCell || isHistoryCell || isRecentAIMove || hasHint) {
          cells.push({ row, col, isValidMoveCell });
        }
      }
    }
    return cells;
  }, [aiMoveProgress, lastAIMove, moveHistory, analyzeResults, gameOver, isValidMove]);

  return (
    <>
      {overlayCells.map(({ row, col, isValidMoveCell }) => (
        <CellHtmlOverlay
          key={`overlay-${row}-${col}`}
          row={row}
          col={col}
          cellPixelSize={cellPixelSize}
        >
          <div className="absolute inset-0 flex items-center justify-center">
            <AIThinkingIndicator
              rowIndex={row}
              colIndex={col}
              aiMoveProgress={aiMoveProgress}
              moveHistory={moveHistory}
              lastAIMove={lastAIMove}
            />
            <HintScoreDisplay
              rowIndex={row}
              colIndex={col}
              analyzeResults={analyzeResults}
              maxScore={maxScore}
              gameOver={gameOver}
              isValidMoveCell={isValidMoveCell}
              showWaitingBar={showHintWaitingBar}
            />
          </div>
        </CellHtmlOverlay>
      ))}
    </>
  );
}
