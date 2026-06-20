import { useMemo } from "react";
import {
  BOARD_WORLD_SIZE,
  FRAME_WIDTH,
  FRAME_HEIGHT,
  FRAME_COLOR,
  FRAME_CHAMFER_COLOR,
} from "./board3d-utils";

const CHAMFER_HEIGHT = 0.01;

/** Dark gunmetal border frame with chamfer edge highlights */
export function BoardFrame() {
  const outerSize = BOARD_WORLD_SIZE + FRAME_WIDTH * 2;

  const sides = useMemo(() => {
    const halfBoard = BOARD_WORLD_SIZE / 2;
    const yPos = FRAME_HEIGHT / 2;
    return [
      {
        position: [0, yPos, -(halfBoard + FRAME_WIDTH / 2)] as [number, number, number],
        size: [outerSize, FRAME_HEIGHT, FRAME_WIDTH] as [number, number, number],
      },
      {
        position: [0, yPos, halfBoard + FRAME_WIDTH / 2] as [number, number, number],
        size: [outerSize, FRAME_HEIGHT, FRAME_WIDTH] as [number, number, number],
      },
      {
        position: [-(halfBoard + FRAME_WIDTH / 2), yPos, 0] as [number, number, number],
        size: [FRAME_WIDTH, FRAME_HEIGHT, BOARD_WORLD_SIZE] as [number, number, number],
      },
      {
        position: [halfBoard + FRAME_WIDTH / 2, yPos, 0] as [number, number, number],
        size: [FRAME_WIDTH, FRAME_HEIGHT, BOARD_WORLD_SIZE] as [number, number, number],
      },
    ];
  }, [outerSize]);

  const chamferStrips = useMemo(() => {
    const halfBoard = BOARD_WORLD_SIZE / 2;
    const yPos = FRAME_HEIGHT + CHAMFER_HEIGHT / 2;
    return [
      {
        position: [0, yPos, -(halfBoard + FRAME_WIDTH / 2)] as [number, number, number],
        size: [outerSize, CHAMFER_HEIGHT, FRAME_WIDTH] as [number, number, number],
      },
      {
        position: [0, yPos, halfBoard + FRAME_WIDTH / 2] as [number, number, number],
        size: [outerSize, CHAMFER_HEIGHT, FRAME_WIDTH] as [number, number, number],
      },
      {
        position: [-(halfBoard + FRAME_WIDTH / 2), yPos, 0] as [number, number, number],
        size: [FRAME_WIDTH, CHAMFER_HEIGHT, BOARD_WORLD_SIZE] as [number, number, number],
      },
      {
        position: [halfBoard + FRAME_WIDTH / 2, yPos, 0] as [number, number, number],
        size: [FRAME_WIDTH, CHAMFER_HEIGHT, BOARD_WORLD_SIZE] as [number, number, number],
      },
    ];
  }, [outerSize]);

  return (
    <group>
      {/* Base plane beneath the board */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.01, 0]} receiveShadow>
        <planeGeometry args={[outerSize, outerSize]} />
        <meshStandardMaterial color={FRAME_COLOR} roughness={0.4} metalness={0.7} />
      </mesh>

      {/* Frame sides  Edark gunmetal */}
      {sides.map((side, i) => (
        <mesh key={`side-${i}`} position={side.position} receiveShadow>
          <boxGeometry args={side.size} />
          <meshStandardMaterial color={FRAME_COLOR} roughness={0.4} metalness={0.7} />
        </mesh>
      ))}

      {/* Chamfer highlight strips  Elighter metal on top edge */}
      {chamferStrips.map((strip, i) => (
        <mesh key={`chamfer-${i}`} position={strip.position}>
          <boxGeometry args={strip.size} />
          <meshStandardMaterial color={FRAME_CHAMFER_COLOR} roughness={0.12} metalness={0.9} />
        </mesh>
      ))}
    </group>
  );
}
