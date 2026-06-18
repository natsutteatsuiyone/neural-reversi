import { memo, useCallback, useEffect } from "react";
import { useThree } from "@react-three/fiber";
import type { ThreeEvent } from "@react-three/fiber";
import { CELL_SIZE, BOARD_WORLD_SIZE } from "./board3d-utils";

interface CellInteractionProps {
  onCellClick: (row: number, col: number) => void;
  isValidMove: (row: number, col: number) => boolean;
  isDisabled: () => boolean;
}

export const CellInteraction = memo(function CellInteraction({
  onCellClick,
  isValidMove,
  isDisabled,
}: CellInteractionProps) {
  const gl = useThree((s) => s.gl);

  const cellFromEvent = useCallback((e: ThreeEvent<PointerEvent>) => {
    const { x, z } = e.point;
    const col = Math.floor((x + BOARD_WORLD_SIZE / 2) / CELL_SIZE);
    const row = Math.floor((z + BOARD_WORLD_SIZE / 2) / CELL_SIZE);
    return { row, col };
  }, []);

  const isPlayable = useCallback(
    (row: number, col: number) =>
      !isDisabled() && row >= 0 && row < 8 && col >= 0 && col < 8 && isValidMove(row, col),
    [isDisabled, isValidMove],
  );

  const handleClick = useCallback(
    (e: ThreeEvent<PointerEvent>) => {
      if (isDisabled()) return;
      e.stopPropagation();
      const { row, col } = cellFromEvent(e);
      if (isPlayable(row, col)) {
        onCellClick(row, col);
      }
    },
    [onCellClick, isDisabled, isPlayable, cellFromEvent],
  );

  const clearCursor = useCallback(() => {
    gl.domElement.style.cursor = "";
  }, [gl]);

  const handleMove = useCallback(
    (e: ThreeEvent<PointerEvent>) => {
      const { row, col } = cellFromEvent(e);
      gl.domElement.style.cursor = isPlayable(row, col) ? "pointer" : "";
    },
    [gl, isPlayable, cellFromEvent],
  );

  // Never leave the pointer cursor behind when this interaction layer unmounts.
  useEffect(() => clearCursor, [clearCursor]);

  return (
    <mesh
      position={[0, 0.01, 0]}
      rotation={[-Math.PI / 2, 0, 0]}
      onPointerDown={handleClick}
      onPointerMove={handleMove}
      onPointerOut={clearCursor}
    >
      <planeGeometry args={[BOARD_WORLD_SIZE, BOARD_WORLD_SIZE]} />
      <meshBasicMaterial transparent opacity={0} />
    </mesh>
  );
});
