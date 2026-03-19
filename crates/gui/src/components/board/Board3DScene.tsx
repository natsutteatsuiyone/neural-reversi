import { OrthographicCamera } from "@react-three/drei";
import { useThree, useFrame } from "@react-three/fiber";
import { useRef, useMemo } from "react";
import type { OrthographicCamera as ThreeOrthographicCamera } from "three";
import type { AIMoveProgress } from "@/lib/ai";
import type { Board } from "@/types";
import { AIThinkingIndicator } from "./AIThinkingIndicator";
import { BoardFrame } from "./BoardFrame";
import { BoardLabels } from "./BoardLabels";
import { BoardSurface } from "./BoardSurface";
import { CellHtmlOverlay } from "./CellHtmlOverlay";
import { CellInteraction } from "./CellInteraction";
import { HintScoreDisplay } from "./HintScoreDisplay";
import { MoveIndicators } from "./MoveIndicators";
import { Disc3D } from "./Disc3D";
import { CELL_SIZE, FRAME_WIDTH } from "./board3d-utils";

interface MoveHistoryItem {
  row: number;
  col: number;
  timestamp: number;
}

interface Board3DSceneProps {
  board: Board;
  lastMove: { row: number; col: number; isAI?: boolean } | null;
  gameOver: boolean;
  isValidMove: (row: number, col: number) => boolean;
  isAITurn: () => boolean;
  onCellClick: (row: number, col: number) => void;
  aiMoveProgress: AIMoveProgress | null;
  lastAIMove: { row: number; col: number; timestamp: number } | null;
  moveHistory: MoveHistoryItem[];
  analyzeResults: Map<string, AIMoveProgress> | null;
  maxScore: number | null;
  skipAnimation: boolean;
}

const TOTAL_SIZE = CELL_SIZE * 8 + FRAME_WIDTH * 2 + 1.2;

export function Board3DScene({
  board, lastMove, gameOver, isValidMove, isAITurn, onCellClick,
  aiMoveProgress, lastAIMove, moveHistory, analyzeResults, maxScore, skipAnimation,
}: Board3DSceneProps) {
  const { size } = useThree();
  const cameraRef = useRef<ThreeOrthographicCamera>(null);
  const lastAppliedSize = useRef({ width: 0, height: 0 });

  useFrame(({ size: frameSize }) => {
    const cam = cameraRef.current;
    if (!cam) return;

    const { width, height } = frameSize;
    if (width === lastAppliedSize.current.width && height === lastAppliedSize.current.height) return;
    lastAppliedSize.current = { width, height };

    const aspect = width / height;
    const boardHalf = TOTAL_SIZE / 2;
    cam.left = -(aspect >= 1 ? boardHalf * aspect : boardHalf);
    cam.right = aspect >= 1 ? boardHalf * aspect : boardHalf;
    cam.top = aspect >= 1 ? boardHalf : boardHalf / aspect;
    cam.bottom = -(aspect >= 1 ? boardHalf : boardHalf / aspect);
    cam.updateProjectionMatrix();
  });

  const aspect = size.width / size.height;
  const cellPixelSize = aspect >= 1
    ? (size.height / TOTAL_SIZE) * CELL_SIZE
    : (size.width / TOTAL_SIZE) * CELL_SIZE;

  const aiTurnActive = isAITurn();

  const validMoves = useMemo(() => {
    if (gameOver || aiTurnActive) return [];
    const moves: { row: number; col: number }[] = [];
    for (let row = 0; row < 8; row++) {
      for (let col = 0; col < 8; col++) {
        const cell = board[row][col];
        if (
          !cell.color &&
          isValidMove(row, col) &&
          !(analyzeResults && analyzeResults.has(`${row},${col}`))
        ) {
          moves.push({ row, col });
        }
      }
    }
    return moves;
  }, [board, gameOver, aiTurnActive, isValidMove, analyzeResults]);

  // Track previous board state to detect which stones actually changed color
  const prevBoard = useRef(board);
  const flipDelays = useMemo(() => {
    const delays = new Map<string, number>();
    for (let row = 0; row < 8; row++) {
      for (let col = 0; col < 8; col++) {
        const cell = board[row][col];
        const prevCell = prevBoard.current[row]?.[col];
        if (cell.color && prevCell?.color && cell.color !== prevCell.color && !cell.isNew) {
          delays.set(`${row},${col}`, 0);
        }
      }
    }
    prevBoard.current = board;
    return delays;
  }, [board]);

  return (
    <>
      <OrthographicCamera
        ref={cameraRef}
        makeDefault
        position={[0, 10, 0]}
        zoom={1}
        left={-TOTAL_SIZE / 2}
        right={TOTAL_SIZE / 2}
        top={TOTAL_SIZE / 2}
        bottom={-TOTAL_SIZE / 2}
        near={0.1}
        far={100}
        rotation={[-Math.PI / 2, 0, 0]}
      />

      <ambientLight intensity={1.2} />
      <directionalLight position={[2, 10, 2]} intensity={0.5} />
      <directionalLight position={[-2, 8, -2]} intensity={0.2} />

      <BoardFrame />
      <BoardSurface />
      <BoardLabels />

      {board.map((row, rowIndex) =>
        row.map((cell, colIndex) =>
          cell.color ? (
            <Disc3D
              key={`stone-${rowIndex}-${colIndex}`}
              row={rowIndex}
              col={colIndex}
              color={cell.color}
              isNew={cell.isNew}
              flipDelay={flipDelays.get(`${rowIndex},${colIndex}`) ?? 0}
              skipAnimation={skipAnimation}
            />
          ) : null
        )
      )}

      <MoveIndicators validMoves={validMoves} lastMove={lastMove} />

      <CellInteraction
        onCellClick={onCellClick}
        isValidMove={isValidMove}
        disabled={gameOver || aiTurnActive}
      />

      {board.map((row, rowIndex) =>
        row.map((_, colIndex) => {
          const isThinkingCell =
            aiMoveProgress?.row === rowIndex && aiMoveProgress?.col === colIndex;
          const isHistoryCell = moveHistory.some(
            (m) => m.row === rowIndex && m.col === colIndex
          );
          const isRecentAIMove =
            lastAIMove?.row === rowIndex && lastAIMove?.col === colIndex;
          const hasHint =
            analyzeResults?.has(`${rowIndex},${colIndex}`) ||
            (!gameOver && isValidMove(rowIndex, colIndex) && analyzeResults !== null);

          if (!isThinkingCell && !isHistoryCell && !isRecentAIMove && !hasHint) {
            return null;
          }

          return (
            <CellHtmlOverlay
              key={`overlay-${rowIndex}-${colIndex}`}
              row={rowIndex}
              col={colIndex}
              cellPixelSize={cellPixelSize}
            >
              <div className="absolute inset-0 flex items-center justify-center">
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
                  gameOver={gameOver}
                  isValidMoveCell={isValidMove(rowIndex, colIndex)}
                />
              </div>
            </CellHtmlOverlay>
          );
        })
      )}
    </>
  );
}
