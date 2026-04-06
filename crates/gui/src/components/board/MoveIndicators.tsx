import { cellToWorld } from "./board3d-utils";

interface MoveIndicatorsProps {
  validMoves: { row: number; col: number }[];
  lastMove: { row: number; col: number } | null;
}

const ACCENT_GOLD = "#b8956c";
const VALID_MOVE_DOT_RADIUS = 0.13;
const HIGHLIGHT_INNER_RADIUS = 0.42;
const HIGHLIGHT_OUTER_RADIUS = 0.47;

export function MoveIndicators({ validMoves, lastMove }: MoveIndicatorsProps) {
  return (
    <group>
      {validMoves.map(({ row, col }) => {
        const [x, z] = cellToWorld(row, col);
        return (
          <mesh key={`valid-${row}-${col}`} position={[x, 0.002, z]} rotation={[-Math.PI / 2, 0, 0]}>
            <circleGeometry args={[VALID_MOVE_DOT_RADIUS, 16]} />
            <meshBasicMaterial color="white" transparent opacity={0.3} />
          </mesh>
        );
      })}
      {lastMove && <LastMoveRing row={lastMove.row} col={lastMove.col} />}
    </group>
  );
}

function LastMoveRing({ row, col }: { row: number; col: number }) {
  const [x, z] = cellToWorld(row, col);
  return (
    <mesh position={[x, 0.002, z]} rotation={[-Math.PI / 2, 0, 0]}>
      <ringGeometry args={[HIGHLIGHT_INNER_RADIUS, HIGHLIGHT_OUTER_RADIUS, 32]} />
      <meshBasicMaterial color={ACCENT_GOLD} />
    </mesh>
  );
}
