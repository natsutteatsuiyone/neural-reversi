import { Text } from "@react-three/drei";
import { COLUMN_LABELS, ROW_LABELS } from "@/lib/constants";
import { CELL_SIZE, BOARD_WORLD_SIZE, BOARD_OFFSET, FRAME_WIDTH } from "./board3d-utils";

const LABEL_COLOR = "#a0a0b0";
const FONT_SIZE = 0.3;
const LABEL_OFFSET = FRAME_WIDTH + 0.35;

/** Renders A-H column labels and 1-8 row labels in the 3D scene */
export function BoardLabels() {
  const halfBoard = BOARD_WORLD_SIZE / 2;

  return (
    <group>
      {/* Column labels (A-H) — above the board */}
      {COLUMN_LABELS.map((label, i) => {
        const x = i * CELL_SIZE - BOARD_OFFSET;
        return (
          <Text
            key={`col-${label}`}
            position={[x, 0.01, -(halfBoard + LABEL_OFFSET)]}
            rotation={[-Math.PI / 2, 0, 0]}
            fontSize={FONT_SIZE}
            color={LABEL_COLOR}
            anchorX="center"
            anchorY="middle"
            font={undefined}
          >
            {label.toUpperCase()}
          </Text>
        );
      })}

      {/* Row labels (1-8) — left of the board */}
      {ROW_LABELS.map((label, i) => {
        const z = i * CELL_SIZE - BOARD_OFFSET;
        return (
          <Text
            key={`row-${label}`}
            position={[-(halfBoard + LABEL_OFFSET), 0.01, z]}
            rotation={[-Math.PI / 2, 0, 0]}
            fontSize={FONT_SIZE}
            color={LABEL_COLOR}
            anchorX="center"
            anchorY="middle"
            font={undefined}
          >
            {label}
          </Text>
        );
      })}
    </group>
  );
}
