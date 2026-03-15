import { useMemo } from "react";
import { BOARD_WORLD_SIZE, FRAME_WIDTH, FRAME_HEIGHT } from "./board3d-utils";

const CHAMFER_HEIGHT = 0.01;

/** Dark gunmetal border frame with chamfer edge highlights */
export function BoardFrame() {
  const outerSize = BOARD_WORLD_SIZE + FRAME_WIDTH * 2;

  const sides = useMemo(() => {
    const halfBoard = BOARD_WORLD_SIZE / 2;
    const yPos = FRAME_HEIGHT / 2;
    return [
      { position: [0, yPos, -(halfBoard + FRAME_WIDTH / 2)] as [number, number, number], size: [outerSize, FRAME_HEIGHT, FRAME_WIDTH] as [number, number, number] },
      { position: [0, yPos, halfBoard + FRAME_WIDTH / 2] as [number, number, number], size: [outerSize, FRAME_HEIGHT, FRAME_WIDTH] as [number, number, number] },
      { position: [-(halfBoard + FRAME_WIDTH / 2), yPos, 0] as [number, number, number], size: [FRAME_WIDTH, FRAME_HEIGHT, BOARD_WORLD_SIZE] as [number, number, number] },
      { position: [halfBoard + FRAME_WIDTH / 2, yPos, 0] as [number, number, number], size: [FRAME_WIDTH, FRAME_HEIGHT, BOARD_WORLD_SIZE] as [number, number, number] },
    ];
  }, []);

  const chamferStrips = useMemo(() => {
    const halfBoard = BOARD_WORLD_SIZE / 2;
    const yPos = FRAME_HEIGHT + CHAMFER_HEIGHT / 2;
    return [
      { position: [0, yPos, -(halfBoard + FRAME_WIDTH / 2)] as [number, number, number], size: [outerSize, CHAMFER_HEIGHT, FRAME_WIDTH] as [number, number, number] },
      { position: [0, yPos, halfBoard + FRAME_WIDTH / 2] as [number, number, number], size: [outerSize, CHAMFER_HEIGHT, FRAME_WIDTH] as [number, number, number] },
      { position: [-(halfBoard + FRAME_WIDTH / 2), yPos, 0] as [number, number, number], size: [FRAME_WIDTH, CHAMFER_HEIGHT, BOARD_WORLD_SIZE] as [number, number, number] },
      { position: [halfBoard + FRAME_WIDTH / 2, yPos, 0] as [number, number, number], size: [FRAME_WIDTH, CHAMFER_HEIGHT, BOARD_WORLD_SIZE] as [number, number, number] },
    ];
  }, []);

  return (
    <group>
      {/* Base plane beneath the board */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.01, 0]}>
        <planeGeometry args={[outerSize, outerSize]} />
        <meshStandardMaterial color="#2d2d38" roughness={0.4} metalness={0.7} />
      </mesh>

      {/* Frame sides — dark gunmetal */}
      {sides.map((side, i) => (
        <mesh key={`side-${i}`} position={side.position}>
          <boxGeometry args={side.size} />
          <meshStandardMaterial color="#2d2d38" roughness={0.4} metalness={0.7} />
        </mesh>
      ))}

      {/* Chamfer highlight strips — lighter metal on top edge */}
      {chamferStrips.map((strip, i) => (
        <mesh key={`chamfer-${i}`} position={strip.position}>
          <boxGeometry args={strip.size} />
          <meshStandardMaterial color="#4a4a55" roughness={0.2} metalness={0.8} />
        </mesh>
      ))}
    </group>
  );
}
