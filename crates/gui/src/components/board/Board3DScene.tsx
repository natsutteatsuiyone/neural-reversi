import { OrthographicCamera } from "@react-three/drei";
import { useThree } from "@react-three/fiber";
import { useRef, useMemo, useEffect } from "react";
import type { OrthographicCamera as ThreeOrthographicCamera } from "three";
import type { AIMoveProgress } from "@/services/types";
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
import { CELL_SIZE, FRAME_WIDTH, createEnvironmentTexture } from "./board3d-utils";

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
  showHintWaitingBar?: boolean;
}

const TOTAL_SIZE = CELL_SIZE * 8 + FRAME_WIDTH * 2 + 1.2;

export function Board3DScene({
  board, lastMove, gameOver, isValidMove, isAITurn, onCellClick,
  aiMoveProgress, lastAIMove, moveHistory, analyzeResults, maxScore, skipAnimation,
  showHintWaitingBar = true,
}: Board3DSceneProps) {
  const size = useThree((s) => s.size);
  const invalidate = useThree((s) => s.invalidate);
  const cameraRef = useRef<ThreeOrthographicCamera>(null);

  // drei's <OrthographicCamera> rewrites l/r/t/b each render — reapply
  // aspect correction here so <Html> overlays don't collapse on resize.
  useEffect(() => {
    const cam = cameraRef.current;
    if (!cam || !size.width || !size.height) return;
    const aspect = size.width / size.height;
    const boardHalf = TOTAL_SIZE / 2;
    cam.left = -(aspect >= 1 ? boardHalf * aspect : boardHalf);
    cam.right = aspect >= 1 ? boardHalf * aspect : boardHalf;
    cam.top = aspect >= 1 ? boardHalf : boardHalf / aspect;
    cam.bottom = -(aspect >= 1 ? boardHalf : boardHalf / aspect);
    cam.updateProjectionMatrix();
    invalidate();
  }, [size.width, size.height, invalidate]);

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

  const envMap = useMemo(() => createEnvironmentTexture(), []);

  // Track previous board state to detect which stones actually changed color
  const prevBoardRef = useRef(board);
  const flipDelays = useMemo(() => {
    const delays = new Map<string, number>();
    for (let row = 0; row < 8; row++) {
      for (let col = 0; col < 8; col++) {
        const cell = board[row][col];
        const prevCell = prevBoardRef.current[row]?.[col];
        if (cell.color && prevCell?.color && cell.color !== prevCell.color && !cell.isNew) {
          delays.set(`${row},${col}`, 0);
        }
      }
    }
    return delays;
  }, [board]);

  // Update after commit so the next render compares against the last committed board,
  // not the in-progress one from the current render.
  useEffect(() => {
    prevBoardRef.current = board;
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

      <primitive attach="environment" object={envMap} />
      <ambientLight intensity={0.25} />
      <directionalLight
        position={[5, 6, 4]}
        intensity={1.1}
        castShadow
        shadow-mapSize={[2048, 2048]}
        shadow-camera-left={-7}
        shadow-camera-right={7}
        shadow-camera-top={7}
        shadow-camera-bottom={-7}
        shadow-camera-near={0.5}
        shadow-camera-far={20}
        shadow-normalBias={0.02}
        shadow-intensity={0.3}
      />
      <directionalLight position={[-5, 6, -2]} intensity={0.35} />
      <directionalLight position={[0, 3, -8]} intensity={0.5} />

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
        isDisabled={isAITurn}
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
                  showWaitingBar={showHintWaitingBar}
                />
              </div>
            </CellHtmlOverlay>
          );
        })
      )}
    </>
  );
}
