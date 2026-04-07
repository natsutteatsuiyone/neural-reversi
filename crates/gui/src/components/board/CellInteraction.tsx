import { useCallback } from "react";
import type { ThreeEvent } from "@react-three/fiber";
import { CELL_SIZE, BOARD_WORLD_SIZE } from "./board3d-utils";

interface CellInteractionProps {
  onCellClick: (row: number, col: number) => void;
  isValidMove: (row: number, col: number) => boolean;
  isDisabled: () => boolean;
}

export function CellInteraction({ onCellClick, isValidMove, isDisabled }: CellInteractionProps) {
  const handleClick = useCallback((e: ThreeEvent<PointerEvent>) => {
    if (isDisabled()) return;
    e.stopPropagation();
    const { x, z } = e.point;
    const col = Math.floor((x + BOARD_WORLD_SIZE / 2) / CELL_SIZE);
    const row = Math.floor((z + BOARD_WORLD_SIZE / 2) / CELL_SIZE);
    if (row >= 0 && row < 8 && col >= 0 && col < 8 && isValidMove(row, col)) {
      onCellClick(row, col);
    }
  }, [onCellClick, isValidMove, isDisabled]);

  return (
    <mesh position={[0, 0.01, 0]} rotation={[-Math.PI / 2, 0, 0]} onPointerDown={handleClick}>
      <planeGeometry args={[BOARD_WORLD_SIZE, BOARD_WORLD_SIZE]} />
      <meshBasicMaterial transparent opacity={0} />
    </mesh>
  );
}
