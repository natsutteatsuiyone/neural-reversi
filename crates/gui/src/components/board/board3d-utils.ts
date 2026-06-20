import * as THREE from "three";

export const CELL_SIZE = 1;
export const BOARD_OFFSET = 3.5;
export const BOARD_WORLD_SIZE = CELL_SIZE * 8;
export const DISC_RADIUS = 0.4;
export const DISC_HEIGHT = 0.08;
export const FRAME_WIDTH = 0.15;
export const FRAME_HEIGHT = 0.06;
export const GROOVE_WIDTH = 0.03;
// Decorative star points (hoshi) at the grid-line intersections 2 and 6.
export const STAR_POINT_RADIUS = 0.06;
export const STAR_POINT_COLOR = "#1a1a1a";
export const DISC_COLOR_BLACK = "#2a2a2d";
export const DISC_COLOR_WHITE = "#e8e4df";
export const GROOVE_COLOR = "#1a5238";
// Felt playing-surface base color (procedural texture seed).
export const FELT_COLOR = "#2d8a5e";
// Dark gunmetal frame and its lighter top-edge chamfer highlight.
export const FRAME_COLOR = "#2d2d38";
export const FRAME_CHAMFER_COLOR = "#4a4a55";
// Column/row label text in the 3D scene.
export const LABEL_COLOR = "#a0a0b0";
// Last-move ring. Mirrors the `--accent-gold` CSS token; kept as a literal
// because three.js materials cannot read Tailwind/CSS custom properties.
export const LAST_MOVE_RING_COLOR = "#b8956c";

export function cellToWorld(row: number, col: number): [x: number, z: number] {
  const x = col * CELL_SIZE - BOARD_OFFSET;
  const z = row * CELL_SIZE - BOARD_OFFSET;
  return [x, z];
}

/** Build a minimal grayscale gradient DataTexture for use as scene.environment.
 * Provides subtle reflection cues on clearcoat discs and metallic frame without
 * requiring an HDRI file. */
export function createEnvironmentTexture(): THREE.DataTexture {
  const width = 4;
  const height = 8;
  const data = new Uint8Array(width * height * 4);
  for (let y = 0; y < height; y++) {
    // Top rows brighter (sky), bottom rows darker (floor)
    const t = y / (height - 1);
    const v = Math.round((0.35 + (1 - t) * 0.55) * 255);
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * 4;
      data[idx] = v;
      data[idx + 1] = v;
      data[idx + 2] = v;
      data[idx + 3] = 255;
    }
  }
  const texture = new THREE.DataTexture(data, width, height);
  texture.mapping = THREE.EquirectangularReflectionMapping;
  texture.colorSpace = THREE.NoColorSpace;
  texture.needsUpdate = true;
  return texture;
}
