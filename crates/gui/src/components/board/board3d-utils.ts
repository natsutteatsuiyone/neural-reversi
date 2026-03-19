export const CELL_SIZE = 1;
export const BOARD_OFFSET = 3.5;
export const BOARD_WORLD_SIZE = CELL_SIZE * 8;
export const DISC_RADIUS = 0.4;
export const DISC_HEIGHT = 0.08;
export const FRAME_WIDTH = 0.15;
export const FRAME_HEIGHT = 0.06;
export const GROOVE_WIDTH = 0.03;
export const DISC_COLOR_BLACK = "#2a2a2d";
export const DISC_COLOR_WHITE = "#e8e4df";
export const GROOVE_COLOR = "#1a5238";
export const FLIP_DURATION_S = 0.4;

export function cellToWorld(row: number, col: number): [x: number, z: number] {
  const x = col * CELL_SIZE - BOARD_OFFSET;
  const z = row * CELL_SIZE - BOARD_OFFSET;
  return [x, z];
}
