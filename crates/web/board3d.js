// Vanilla three.js 3D Reversi board.
//
// Visual design ported from the Tauri GUI (crates/gui Board3DScene + friends):
// orthographic top-down camera, procedural green felt surface with grid grooves,
// dark gunmetal frame, glossy MeshPhysical discs with drop/flip animations, gold
// last-move ring and translucent valid-move dots. Kept framework-free so it can
// be driven imperatively from the existing petite-vue app instead of react-three-fiber.

import * as THREE from "three";

// --- Geometry constants (mirrors gui board3d-utils.ts) ---
const CELL_SIZE = 1;
const BOARD_OFFSET = 3.5;
const BOARD_WORLD_SIZE = CELL_SIZE * 8;
const DISC_RADIUS = 0.4;
const DISC_HEIGHT = 0.08;
const FRAME_WIDTH = 0.15;
const FRAME_HEIGHT = 0.06;
const CHAMFER_HEIGHT = 0.01;
const GROOVE_WIDTH = 0.03;
// Margin around the framed board reserved for the coordinate labels.
const LABEL_MARGIN = 0.6;
const LABEL_OFFSET = FRAME_WIDTH + 0.2;
const LABEL_PLANE_SIZE = 0.5;
const TOTAL_SIZE = CELL_SIZE * 8 + FRAME_WIDTH * 2 + LABEL_MARGIN * 2;

const DISC_COLOR_BLACK = "#2a2a2d";
const DISC_COLOR_WHITE = "#e8e4df";
const GROOVE_COLOR = "#1a5238";
const FRAME_COLOR = "#2d2d38";
const CHAMFER_COLOR = "#4a4a55";
const LABEL_COLOR = "#a0a0b0";
const ACCENT_GOLD = "#b8956c";

// --- Animation constants (mirrors gui Disc3D + timing.ts) ---
const FLIP_DURATION_S = 0.4;
const SEGMENTS = 32;
const DROP_SPRING_STIFFNESS = 300;
const DROP_SPRING_DAMPING = 20;
const FLIP_ARC_HEIGHT = 0.3;
const MAX_FRAME_DELTA = 1 / 30;
const DISC_Y_OFFSET = 0.001;
const DISC_REST_Y = DISC_HEIGHT / 2 + DISC_Y_OFFSET;

const COLUMN_LABELS = ["a", "b", "c", "d", "e", "f", "g", "h"];
const ROW_LABELS = ["1", "2", "3", "4", "5", "6", "7", "8"];

function cellToWorld(row, col) {
  return [col * CELL_SIZE - BOARD_OFFSET, row * CELL_SIZE - BOARD_OFFSET];
}

/** Minimal grayscale gradient environment texture for subtle reflections on the
 * clearcoat discs and metallic frame (avoids shipping an HDRI). */
function createEnvironmentTexture() {
  const width = 4;
  const height = 8;
  const data = new Uint8Array(width * height * 4);
  for (let y = 0; y < height; y++) {
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

/** Procedural felt texture: green base with per-pixel fiber noise. */
function createFeltTexture() {
  const size = 256;
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#2d8a5e";
  ctx.fillRect(0, 0, size, size);
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
  // CanvasTexture defaults to NoColorSpace; without this the sRGB-encoded canvas
  // pixels are read as linear, washing the felt green out to a pale teal.
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(3, 3);
  return texture;
}

/** Procedural tileable normal map giving the felt a soft fibrous relief. */
function createFeltNormalTexture() {
  const size = 256;
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext("2d");
  const height = new Float32Array(size * size);
  for (let i = 0; i < height.length; i++) {
    height[i] = Math.random();
  }
  const imageData = ctx.createImageData(size, size);
  const data = imageData.data;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const l = height[y * size + ((x - 1 + size) % size)];
      const r = height[y * size + ((x + 1) % size)];
      const u = height[((y - 1 + size) % size) * size + x];
      const d = height[((y + 1) % size) * size + x];
      const nx = -(r - l) * 0.5;
      const ny = (d - u) * 0.5;
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

/** A single coordinate-label glyph rendered to a flat plane above the margin. */
function createLabelMesh(char) {
  const px = 128;
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = px;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, px, px);
  ctx.fillStyle = LABEL_COLOR;
  ctx.font = `600 ${px * 0.62}px "Inter", system-ui, sans-serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(char.toUpperCase(), px / 2, px / 2 + px * 0.02);
  const texture = new THREE.CanvasTexture(canvas);
  texture.anisotropy = 4;
  const material = new THREE.MeshBasicMaterial({
    map: texture,
    transparent: true,
    depthWrite: false,
  });
  const mesh = new THREE.Mesh(
    new THREE.PlaneGeometry(LABEL_PLANE_SIZE, LABEL_PLANE_SIZE),
    material,
  );
  mesh.rotation.x = -Math.PI / 2;
  return mesh;
}

function discMaterial(color, clearcoat) {
  return new THREE.MeshPhysicalMaterial({
    color,
    roughness: 0.55,
    metalness: 0,
    clearcoat,
    clearcoatRoughness: 0.15,
  });
}

export function createBoard3D(container, { onCellClick }) {
  // CylinderGeometry material groups are [side, top, bottom].
  const blackMat = discMaterial(DISC_COLOR_BLACK, 0.6);
  const whiteMat = discMaterial(DISC_COLOR_WHITE, 0.4);
  const BLACK_MATERIALS = [blackMat, blackMat, whiteMat];
  const WHITE_MATERIALS = [whiteMat, whiteMat, blackMat];
  const discGeometry = new THREE.CylinderGeometry(
    DISC_RADIUS - 0.02,
    DISC_RADIUS,
    DISC_HEIGHT,
    SEGMENTS,
  );

  const scene = new THREE.Scene();
  scene.environment = createEnvironmentTexture();

  const camera = new THREE.OrthographicCamera(
    -TOTAL_SIZE / 2,
    TOTAL_SIZE / 2,
    TOTAL_SIZE / 2,
    -TOTAL_SIZE / 2,
    0.1,
    100,
  );
  camera.position.set(0, 10, 0);
  camera.rotation.set(-Math.PI / 2, 0, 0);

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  // Match react-three-fiber's <Canvas> defaults (used by the GUI): ACES filmic
  // tone mapping deepens the felt/disc colors that otherwise wash out flat under
  // the bright environment map + directional lights.
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.setClearColor(0x000000, 0);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  container.appendChild(renderer.domElement);
  renderer.domElement.style.display = "block";
  renderer.domElement.style.width = "100%";
  renderer.domElement.style.height = "100%";

  // --- Lights (mirrors gui Board3DScene) ---
  scene.add(new THREE.AmbientLight(0xffffff, 0.25));
  const keyLight = new THREE.DirectionalLight(0xffffff, 1.1);
  keyLight.position.set(5, 6, 4);
  keyLight.castShadow = true;
  keyLight.shadow.mapSize.set(2048, 2048);
  keyLight.shadow.camera.left = -7;
  keyLight.shadow.camera.right = 7;
  keyLight.shadow.camera.top = 7;
  keyLight.shadow.camera.bottom = -7;
  keyLight.shadow.camera.near = 0.5;
  keyLight.shadow.camera.far = 20;
  keyLight.shadow.normalBias = 0.02;
  keyLight.shadow.intensity = 0.3;
  scene.add(keyLight);
  const fillLight = new THREE.DirectionalLight(0xffffff, 0.35);
  fillLight.position.set(-5, 6, -2);
  scene.add(fillLight);
  const rimLight = new THREE.DirectionalLight(0xffffff, 0.5);
  rimLight.position.set(0, 3, -8);
  scene.add(rimLight);

  buildFrame(scene);
  buildSurface(scene);
  buildLabels(scene);

  // --- Move indicators (valid-move dots + gold last-move ring) ---
  const dotsGroup = new THREE.Group();
  scene.add(dotsGroup);
  const dotGeometry = new THREE.CircleGeometry(0.13, 16);
  const dotMaterial = new THREE.MeshBasicMaterial({
    color: "white",
    transparent: true,
    opacity: 0.3,
  });

  const lastMoveRing = new THREE.Mesh(
    new THREE.RingGeometry(0.42, 0.47, 32),
    new THREE.MeshBasicMaterial({ color: ACCENT_GOLD }),
  );
  lastMoveRing.rotation.x = -Math.PI / 2;
  lastMoveRing.position.y = 0.002;
  lastMoveRing.visible = false;
  scene.add(lastMoveRing);

  // --- Hover highlight for the cell under the pointer (legal moves only) ---
  const hoverMesh = new THREE.Mesh(
    new THREE.PlaneGeometry(CELL_SIZE * 0.94, CELL_SIZE * 0.94),
    new THREE.MeshBasicMaterial({
      color: "#2a7d5c",
      transparent: true,
      opacity: 0.55,
      depthWrite: false,
    }),
  );
  hoverMesh.rotation.x = -Math.PI / 2;
  hoverMesh.position.y = 0.0015;
  hoverMesh.visible = false;
  scene.add(hoverMesh);

  // --- Per-cell disc animation state ---
  // Mirrors the GUI Disc3D cap lifecycle: keep the old bi-colored top/bottom
  // materials while rolling 0→π, render that landing frame, then switch to the
  // target material and snap back to rotation 0 in a one-frame settling state.
  // The side material is eased separately so the edge does not pop at landing.
  /**
   * @type {Map<number, {
   *   mesh: THREE.Mesh,
   *   state: string,
   *   progress: number,
   *   dropY: number,
   *   dropVel: number,
   *   targetMaterials: THREE.Material[] | null,
   *   flipSideMaterial: THREE.MeshPhysicalMaterial | null,
   *   sideColorFrom: THREE.Color | null,
   *   sideColorTo: THREE.Color | null,
   *   sideClearcoatFrom: number,
   *   sideClearcoatTo: number
   * }>}
   */
  const discs = new Map();
  let prevBoard = Array.from({ length: 64 }, () => 0);
  let view = { legalMoves: [], showValidMoves: false, lastMove: null };

  function materialsForColor(color) {
    return color === 1 ? BLACK_MATERIALS : WHITE_MATERIALS;
  }

  function disposeFlipSideMaterial(entry) {
    if (entry.flipSideMaterial) {
      entry.flipSideMaterial.dispose();
      entry.flipSideMaterial = null;
    }
    entry.sideColorFrom = null;
    entry.sideColorTo = null;
  }

  function makeDisc(color) {
    const mesh = new THREE.Mesh(discGeometry, materialsForColor(color));
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    return mesh;
  }

  function update(state) {
    const { board, legalMoves, showValidMoves, lastMove, skipAnimation } = state;
    view = { legalMoves: legalMoves || [], showValidMoves: !!showValidMoves, lastMove };

    for (let index = 0; index < 64; index++) {
      const cur = board[index];
      const prev = prevBoard[index];
      const [x, z] = cellToWorld(Math.floor(index / 8), index % 8);
      const entry = discs.get(index);

      if (cur && !prev) {
        // Newly placed stone.
        const mesh = makeDisc(cur);
        const next = {
          mesh,
          state: "idle",
          progress: 0,
          dropY: 0,
          dropVel: 0,
          targetMaterials: null,
          flipSideMaterial: null,
          sideColorFrom: null,
          sideColorTo: null,
          sideClearcoatFrom: 0,
          sideClearcoatTo: 0,
        };
        if (skipAnimation) {
          mesh.position.set(x, DISC_REST_Y, z);
        } else {
          next.state = "dropping";
          next.dropY = 0.5;
          mesh.position.set(x, DISC_REST_Y + next.dropY, z);
        }
        scene.add(mesh);
        discs.set(index, next);
      } else if (cur && prev && cur !== prev && entry) {
        // Flipped stone: roll forward 180° using the same material transition
        // as the GUI.
        if (skipAnimation) {
          disposeFlipSideMaterial(entry);
          entry.mesh.material = materialsForColor(cur);
          entry.mesh.rotation.x = 0;
          entry.mesh.position.y = DISC_REST_Y;
          entry.state = "idle";
          entry.progress = 0;
          entry.dropY = 0;
          entry.dropVel = 0;
          entry.targetMaterials = null;
        } else {
          // Keep the old material during the roll so the old top face flips
          // into the new top color. The side gets a private material that
          // eases toward the target color while the disc is moving, avoiding an
          // edge-color pop when the final material is installed.
          disposeFlipSideMaterial(entry);
          const oldMaterials = materialsForColor(prev);
          const targetMaterials = materialsForColor(cur);
          const flipSideMaterial = oldMaterials[0].clone();
          entry.mesh.material = [flipSideMaterial, oldMaterials[1], oldMaterials[2]];
          entry.mesh.rotation.x = 0;
          entry.mesh.position.y = DISC_REST_Y;
          entry.state = "flipping";
          entry.progress = 0;
          entry.dropY = 0;
          entry.dropVel = 0;
          entry.targetMaterials = targetMaterials;
          entry.flipSideMaterial = flipSideMaterial;
          entry.sideColorFrom = oldMaterials[0].color.clone();
          entry.sideColorTo = targetMaterials[0].color.clone();
          entry.sideClearcoatFrom = oldMaterials[0].clearcoat ?? 0;
          entry.sideClearcoatTo = targetMaterials[0].clearcoat ?? 0;
        }
      } else if (!cur && prev && entry) {
        // Removed stone (undo / replay).
        disposeFlipSideMaterial(entry);
        scene.remove(entry.mesh);
        discs.delete(index);
      }
    }

    prevBoard = board.slice();
    refreshIndicators();
    requestRender();
  }

  function refreshIndicators() {
    dotsGroup.clear();
    if (view.showValidMoves) {
      for (const index of view.legalMoves) {
        const [x, z] = cellToWorld(Math.floor(index / 8), index % 8);
        const dot = new THREE.Mesh(dotGeometry, dotMaterial);
        dot.rotation.x = -Math.PI / 2;
        dot.position.set(x, 0.002, z);
        dotsGroup.add(dot);
      }
    }
    if (typeof view.lastMove === "number") {
      const [x, z] = cellToWorld(Math.floor(view.lastMove / 8), view.lastMove % 8);
      lastMoveRing.position.set(x, 0.002, z);
      lastMoveRing.visible = true;
    } else {
      lastMoveRing.visible = false;
    }
  }

  // --- Animation loop (on-demand: runs only while discs are animating) ---
  let rafId = null;
  let lastTime = 0;
  let idleResolvers = [];

  function requestRender() {
    if (rafId === null) {
      lastTime = performance.now();
      rafId = requestAnimationFrame(tick);
    }
  }

  /** Resolves once no disc is mid-animation (drop/flip settled). Used to hold the
   * AI's move until the player's flip animation has finished. */
  function waitForIdle() {
    return new Promise((resolve) => {
      if (rafId === null) {
        resolve();
      } else {
        idleResolvers.push(resolve);
      }
    });
  }

  function tick(now) {
    const dt = Math.min((now - lastTime) / 1000, MAX_FRAME_DELTA);
    lastTime = now;
    let animating = false;

    for (const entry of discs.values()) {
      const mesh = entry.mesh;
      if (entry.state === "dropping") {
        const springForce = -DROP_SPRING_STIFFNESS * entry.dropY;
        const dampingForce = -DROP_SPRING_DAMPING * entry.dropVel;
        entry.dropVel += (springForce + dampingForce) * dt;
        entry.dropY += entry.dropVel * dt;
        mesh.position.y = DISC_REST_Y + Math.max(0, entry.dropY);
        if (Math.abs(entry.dropY) < 0.001 && Math.abs(entry.dropVel) < 0.01) {
          entry.dropY = 0;
          mesh.position.y = DISC_REST_Y;
          entry.state = "idle";
        } else {
          animating = true;
        }
      } else if (entry.state === "flipping") {
        entry.progress += dt / FLIP_DURATION_S;
        if (entry.progress >= 1) {
          if (entry.flipSideMaterial && entry.sideColorTo) {
            entry.flipSideMaterial.color.copy(entry.sideColorTo);
            entry.flipSideMaterial.clearcoat = entry.sideClearcoatTo;
          }
          // Render one landing frame with the old material at π, matching GUI's
          // pre-settling frame where the bottom face has become the visible top.
          mesh.rotation.x = Math.PI;
          mesh.position.y = DISC_REST_Y;
          entry.state = "settling";
          animating = true;
        } else {
          const t = entry.progress;
          if (entry.flipSideMaterial && entry.sideColorFrom && entry.sideColorTo) {
            entry.flipSideMaterial.color.lerpColors(entry.sideColorFrom, entry.sideColorTo, t);
            entry.flipSideMaterial.clearcoat =
              entry.sideClearcoatFrom + (entry.sideClearcoatTo - entry.sideClearcoatFrom) * t;
          }
          mesh.rotation.x = t * Math.PI;
          mesh.position.y = DISC_REST_Y + Math.sin(t * Math.PI) * FLIP_ARC_HEIGHT;
          animating = true;
        }
      } else if (entry.state === "settling") {
        if (entry.targetMaterials) {
          mesh.material = entry.targetMaterials;
        }
        disposeFlipSideMaterial(entry);
        mesh.rotation.x = 0;
        mesh.position.y = DISC_REST_Y;
        entry.state = "idle";
        entry.progress = 0;
        entry.targetMaterials = null;
      }
    }

    renderer.render(scene, camera);

    if (animating) {
      rafId = requestAnimationFrame(tick);
    } else {
      rafId = null;
      if (idleResolvers.length) {
        const resolvers = idleResolvers;
        idleResolvers = [];
        for (const resolve of resolvers) {
          resolve();
        }
      }
    }
  }

  // --- Pointer interaction (raycast against the board plane) ---
  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  const boardPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
  const hitPoint = new THREE.Vector3();

  function cellFromEvent(event) {
    const rect = renderer.domElement.getBoundingClientRect();
    if (!rect.width || !rect.height) return -1;
    pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(pointer, camera);
    if (!raycaster.ray.intersectPlane(boardPlane, hitPoint)) return -1;
    const col = Math.round(hitPoint.x + BOARD_OFFSET);
    const row = Math.round(hitPoint.z + BOARD_OFFSET);
    if (col < 0 || col > 7 || row < 0 || row > 7) return -1;
    return row * 8 + col;
  }

  function isPlayable(index) {
    return view.showValidMoves && index >= 0 && view.legalMoves.includes(index);
  }

  function onPointerMove(event) {
    const index = cellFromEvent(event);
    if (isPlayable(index)) {
      const [x, z] = cellToWorld(Math.floor(index / 8), index % 8);
      hoverMesh.position.set(x, 0.0015, z);
      if (!hoverMesh.visible) {
        hoverMesh.visible = true;
        requestRender();
      } else {
        hoverMesh.position.set(x, 0.0015, z);
        requestRender();
      }
      container.style.cursor = "pointer";
    } else if (hoverMesh.visible) {
      hoverMesh.visible = false;
      container.style.cursor = "default";
      requestRender();
    } else {
      container.style.cursor = "default";
    }
  }

  function onPointerLeave() {
    if (hoverMesh.visible) {
      hoverMesh.visible = false;
      requestRender();
    }
    container.style.cursor = "default";
  }

  function onClick(event) {
    const index = cellFromEvent(event);
    if (isPlayable(index)) {
      onCellClick(index);
    }
  }

  renderer.domElement.addEventListener("pointermove", onPointerMove);
  renderer.domElement.addEventListener("pointerleave", onPointerLeave);
  renderer.domElement.addEventListener("click", onClick);

  // --- Responsive sizing ---
  function resize() {
    const w = container.clientWidth;
    const h = container.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    const aspect = w / h;
    const half = TOTAL_SIZE / 2;
    camera.left = aspect >= 1 ? -half * aspect : -half;
    camera.right = aspect >= 1 ? half * aspect : half;
    camera.top = aspect >= 1 ? half : half / aspect;
    camera.bottom = aspect >= 1 ? -half : -half / aspect;
    camera.updateProjectionMatrix();
    requestRender();
  }

  const resizeObserver = new ResizeObserver(resize);
  resizeObserver.observe(container);
  resize();

  return { update, waitForIdle };
}

// --- Static scene builders ---

function buildFrame(scene) {
  const outerSize = BOARD_WORLD_SIZE + FRAME_WIDTH * 2;
  const halfBoard = BOARD_WORLD_SIZE / 2;
  const frameMat = new THREE.MeshStandardMaterial({
    color: FRAME_COLOR,
    roughness: 0.4,
    metalness: 0.7,
  });
  const chamferMat = new THREE.MeshStandardMaterial({
    color: CHAMFER_COLOR,
    roughness: 0.12,
    metalness: 0.9,
  });

  // Base plane beneath the board (fills the orthographic frustum).
  const base = new THREE.Mesh(new THREE.PlaneGeometry(outerSize + 2.4, outerSize + 2.4), frameMat);
  base.rotation.x = -Math.PI / 2;
  base.position.y = -0.01;
  base.receiveShadow = true;
  scene.add(base);

  const yPos = FRAME_HEIGHT / 2;
  const sides = [
    {
      pos: [0, yPos, -(halfBoard + FRAME_WIDTH / 2)],
      size: [outerSize, FRAME_HEIGHT, FRAME_WIDTH],
    },
    { pos: [0, yPos, halfBoard + FRAME_WIDTH / 2], size: [outerSize, FRAME_HEIGHT, FRAME_WIDTH] },
    {
      pos: [-(halfBoard + FRAME_WIDTH / 2), yPos, 0],
      size: [FRAME_WIDTH, FRAME_HEIGHT, BOARD_WORLD_SIZE],
    },
    {
      pos: [halfBoard + FRAME_WIDTH / 2, yPos, 0],
      size: [FRAME_WIDTH, FRAME_HEIGHT, BOARD_WORLD_SIZE],
    },
  ];
  for (const side of sides) {
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(...side.size), frameMat);
    mesh.position.set(...side.pos);
    mesh.receiveShadow = true;
    scene.add(mesh);
  }

  const cy = FRAME_HEIGHT + CHAMFER_HEIGHT / 2;
  const chamfers = [
    {
      pos: [0, cy, -(halfBoard + FRAME_WIDTH / 2)],
      size: [outerSize, CHAMFER_HEIGHT, FRAME_WIDTH],
    },
    { pos: [0, cy, halfBoard + FRAME_WIDTH / 2], size: [outerSize, CHAMFER_HEIGHT, FRAME_WIDTH] },
    {
      pos: [-(halfBoard + FRAME_WIDTH / 2), cy, 0],
      size: [FRAME_WIDTH, CHAMFER_HEIGHT, BOARD_WORLD_SIZE],
    },
    {
      pos: [halfBoard + FRAME_WIDTH / 2, cy, 0],
      size: [FRAME_WIDTH, CHAMFER_HEIGHT, BOARD_WORLD_SIZE],
    },
  ];
  for (const strip of chamfers) {
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(...strip.size), chamferMat);
    mesh.position.set(...strip.pos);
    scene.add(mesh);
  }
}

function buildSurface(scene) {
  const feltTexture = createFeltTexture();
  const feltNormal = createFeltNormalTexture();
  const surface = new THREE.Mesh(
    new THREE.PlaneGeometry(BOARD_WORLD_SIZE, BOARD_WORLD_SIZE),
    new THREE.MeshStandardMaterial({
      map: feltTexture,
      normalMap: feltNormal,
      normalScale: new THREE.Vector2(0.3, 0.3),
      roughness: 0.92,
      metalness: 0,
    }),
  );
  surface.rotation.x = -Math.PI / 2;
  surface.receiveShadow = true;
  scene.add(surface);

  const halfBoard = BOARD_WORLD_SIZE / 2;
  const grooveMat = new THREE.MeshStandardMaterial({
    color: GROOVE_COLOR,
    roughness: 0.95,
    metalness: 0,
  });
  for (let i = 0; i <= 8; i++) {
    const x = i * CELL_SIZE - halfBoard;
    const v = new THREE.Mesh(new THREE.PlaneGeometry(GROOVE_WIDTH, BOARD_WORLD_SIZE), grooveMat);
    v.rotation.x = -Math.PI / 2;
    v.position.set(x, 0.001, 0);
    scene.add(v);
    const z = i * CELL_SIZE - halfBoard;
    const h = new THREE.Mesh(new THREE.PlaneGeometry(BOARD_WORLD_SIZE, GROOVE_WIDTH), grooveMat);
    h.rotation.x = -Math.PI / 2;
    h.position.set(0, 0.001, z);
    scene.add(h);
  }
}

function buildLabels(scene) {
  const halfBoard = BOARD_WORLD_SIZE / 2;
  COLUMN_LABELS.forEach((label, i) => {
    const mesh = createLabelMesh(label);
    mesh.position.set(i * CELL_SIZE - BOARD_OFFSET, 0.01, -(halfBoard + LABEL_OFFSET));
    scene.add(mesh);
  });
  ROW_LABELS.forEach((label, i) => {
    const mesh = createLabelMesh(label);
    mesh.position.set(-(halfBoard + LABEL_OFFSET), 0.01, i * CELL_SIZE - BOARD_OFFSET);
    scene.add(mesh);
  });
}
