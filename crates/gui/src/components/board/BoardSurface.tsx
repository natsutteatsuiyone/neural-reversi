import { useMemo } from "react";
import * as THREE from "three";
import {
  CELL_SIZE,
  BOARD_WORLD_SIZE,
  GROOVE_COLOR,
  GROOVE_WIDTH,
} from "./board3d-utils";

const grooveMaterial = new THREE.MeshStandardMaterial({
  color: GROOVE_COLOR,
  roughness: 0.95,
  metalness: 0,
});

/** Generate a felt-like procedural texture using Canvas API */
function createFeltTexture(): THREE.CanvasTexture {
  const size = 256;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d")!;

  // Base green color
  ctx.fillStyle = "#2d8a5e";
  ctx.fillRect(0, 0, size, size);

  // Add fiber noise
  const imageData = ctx.getImageData(0, 0, size, size);
  const data = imageData.data;
  for (let i = 0; i < data.length; i += 4) {
    const noise = (Math.random() - 0.5) * 25;
    data[i] = Math.max(0, Math.min(255, data[i] + noise));
    data[i + 1] = Math.max(0, Math.min(255, data[i + 1] + noise * 1.2));
    data[i + 2] = Math.max(0, Math.min(255, data[i + 2] + noise * 0.6));
  }
  ctx.putImageData(imageData, 0, 0);

  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(3, 3);
  return texture;
}

/** Generate a procedural normal map for the felt surface (independent noise, tileable) */
function createFeltNormalTexture(): THREE.CanvasTexture {
  const size = 256;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d")!;

  // Build a grayscale height field first
  const height = new Float32Array(size * size);
  for (let i = 0; i < height.length; i++) {
    height[i] = Math.random();
  }

  // Convert height field to tangent-space normals via central differences
  const imageData = ctx.createImageData(size, size);
  const data = imageData.data;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const l = height[y * size + ((x - 1 + size) % size)];
      const r = height[y * size + ((x + 1) % size)];
      const u = height[((y - 1 + size) % size) * size + x];
      const d = height[((y + 1) % size) * size + x];
      const dx = (r - l) * 0.5;
      const dy = (d - u) * 0.5;
      // Tangent-space normal: nx = -dx, ny = +dy (compensates for CanvasTexture flipY=true),
      // nz = 1, normalized, remapped to [0, 255].
      const nx = -dx;
      const ny = dy;
      const nz = 1;
      const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
      const idx = (y * size + x) * 4;
      data[idx] = Math.round(((nx / len) * 0.5 + 0.5) * 255);
      data[idx + 1] = Math.round(((ny / len) * 0.5 + 0.5) * 255);
      data[idx + 2] = Math.round(((nz / len) * 0.5 + 0.5) * 255);
      data[idx + 3] = 255;
    }
  }
  ctx.putImageData(imageData, 0, 0);

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.NoColorSpace;
  texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(3, 3);
  return texture;
}

/** Green felt playing surface with grid grooves */
export function BoardSurface() {
  const feltTexture = useMemo(() => createFeltTexture(), []);
  const feltNormal = useMemo(() => createFeltNormalTexture(), []);

  return (
    <group>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, 0]} receiveShadow>
        <planeGeometry args={[BOARD_WORLD_SIZE, BOARD_WORLD_SIZE]} />
        <meshStandardMaterial
          map={feltTexture}
          normalMap={feltNormal}
          normalScale={new THREE.Vector2(0.3, 0.3)}
          roughness={0.92}
          metalness={0}
        />
      </mesh>
      <GridGrooves />
    </group>
  );
}

/** Renders the 9x9 grid lines as thin dark strips */
function GridGrooves() {
  const halfBoard = BOARD_WORLD_SIZE / 2;

  const lines = useMemo(() => {
    const result: { position: [number, number, number]; size: [number, number] }[] = [];
    for (let i = 0; i <= 8; i++) {
      const x = i * CELL_SIZE - halfBoard;
      result.push({ position: [x, 0.001, 0], size: [GROOVE_WIDTH, BOARD_WORLD_SIZE] });
    }
    for (let i = 0; i <= 8; i++) {
      const z = i * CELL_SIZE - halfBoard;
      result.push({ position: [0, 0.001, z], size: [BOARD_WORLD_SIZE, GROOVE_WIDTH] });
    }
    return result;
  }, []);

  return (
    <group>
      {lines.map((line, i) => (
        <mesh key={i} rotation={[-Math.PI / 2, 0, 0]} position={line.position} material={grooveMaterial}>
          <planeGeometry args={line.size} />
        </mesh>
      ))}
    </group>
  );
}
