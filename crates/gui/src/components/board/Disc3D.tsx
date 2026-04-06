import { useRef, useState, useEffect } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";
import type { Player } from "@/types";
import {
  DISC_RADIUS,
  DISC_HEIGHT,
  DISC_COLOR_BLACK,
  DISC_COLOR_WHITE,
  FLIP_DURATION_S,
  cellToWorld,
} from "./board3d-utils";

interface Disc3DProps {
  row: number;
  col: number;
  color: Player;
  isNew?: boolean;
  flipDelay?: number;
  skipAnimation?: boolean;
}

const SEGMENTS = 32;
const DROP_SPRING_STIFFNESS = 300;
const DROP_SPRING_DAMPING = 20;
const FLIP_ARC_HEIGHT = 0.3;
// After long AI thinking, frameloop="demand" produces a huge first delta.
// Clamping prevents Euler integration instability in the spring simulation.
const MAX_FRAME_DELTA = 1 / 30;
// Tiny offset above y=0 to prevent Z-fighting between disc bottom and board surface.
const DISC_Y_OFFSET = 0.001;

const sharedGeometry = new THREE.CylinderGeometry(
  DISC_RADIUS - 0.02, DISC_RADIUS, DISC_HEIGHT, SEGMENTS,
);

function discMaterial(color: string, clearcoat: number) {
  return new THREE.MeshPhysicalMaterial({ color, roughness: 0.55, metalness: 0, clearcoat, clearcoatRoughness: 0.15 });
}

const BLACK_MATERIALS = [
  discMaterial(DISC_COLOR_BLACK, 0.6),
  discMaterial(DISC_COLOR_BLACK, 0.6),
  discMaterial(DISC_COLOR_WHITE, 0.4),
];

const WHITE_MATERIALS = [
  discMaterial(DISC_COLOR_WHITE, 0.4),
  discMaterial(DISC_COLOR_WHITE, 0.4),
  discMaterial(DISC_COLOR_BLACK, 0.6),
];

// "settling" is a one-frame transitional state after the flip animation completes.
// It allows React to re-render with the new displayColor (updating the material prop)
// before resetting rotation.x to 0, preventing a one-frame flicker of the old color
// at upright orientation.
type AnimState = "idle" | "dropping" | "flipping" | "flip-waiting" | "settling";

export function Disc3D({ row, col, color, isNew, flipDelay = 0, skipAnimation }: Disc3DProps) {
  const meshRef = useRef<THREE.Mesh>(null);
  const { invalidate } = useThree();
  const [x, z] = cellToWorld(row, col);

  const targetColorRef = useRef(color);
  const [displayColor, setDisplayColor] = useState(color);
  const materials = displayColor === "black" ? BLACK_MATERIALS : WHITE_MATERIALS;

  const [animState, setAnimState] = useState<AnimState>(
    isNew && !skipAnimation ? "dropping" : "idle"
  );
  const animProgress = useRef(0);
  const flipWaitTime = useRef(0);
  const prevColor = useRef<Player>(color);
  const dropY = useRef(isNew && !skipAnimation ? 0.5 : 0);
  const dropVelocity = useRef(0);

  useEffect(() => {
    targetColorRef.current = color;
    if (prevColor.current !== color && !isNew && !skipAnimation) {
      // Keep old materials during flip so the animation shows:
      // old color (top) → flip → new color (bottom of old piece)
      if (flipDelay > 0) {
        setAnimState("flip-waiting");
        flipWaitTime.current = 0;
      } else {
        setAnimState("flipping");
      }
      animProgress.current = 0;
    } else {
      setDisplayColor(color);
    }
    prevColor.current = color;
  }, [color, isNew, flipDelay, skipAnimation]);

  useFrame((_, rawDelta) => {
    if (!meshRef.current) return;
    const mesh = meshRef.current;
    const dt = Math.min(rawDelta, MAX_FRAME_DELTA);
    const restY = DISC_HEIGHT / 2 + DISC_Y_OFFSET;

    if (animState === "dropping") {
      const displacement = dropY.current;
      const springForce = -DROP_SPRING_STIFFNESS * displacement;
      const dampingForce = -DROP_SPRING_DAMPING * dropVelocity.current;
      dropVelocity.current += (springForce + dampingForce) * dt;
      dropY.current += dropVelocity.current * dt;
      mesh.position.y = restY + Math.max(0, dropY.current);
      if (Math.abs(dropY.current) < 0.001 && Math.abs(dropVelocity.current) < 0.01) {
        dropY.current = 0;
        mesh.position.y = restY;
        mesh.rotation.x = 0;
        setAnimState("idle");
      }
      invalidate();
    } else if (animState === "flip-waiting") {
      flipWaitTime.current += dt;
      if (flipWaitTime.current >= flipDelay) {
        setAnimState("flipping");
        animProgress.current = 0;
      }
      invalidate();
    } else if (animState === "flipping") {
      animProgress.current += dt / FLIP_DURATION_S;
      if (animProgress.current >= 1) {
        // Don't reset rotation yet — transition to "settling" first so that
        // React re-renders with the new material before rotation snaps to 0.
        // The disc at rotation π looks identical to rotation 0 (top/bottom share
        // the same color), so this intermediate state is visually seamless.
        mesh.rotation.x = Math.PI;
        mesh.position.y = restY;
        setDisplayColor(targetColorRef.current);
        setAnimState("settling");
      } else {
        const t = animProgress.current;
        mesh.rotation.x = t * Math.PI;
        mesh.position.y = restY + Math.sin(t * Math.PI) * FLIP_ARC_HEIGHT;
      }
      invalidate();
    } else if (animState === "settling") {
      // React has re-rendered with the new displayColor, so the material prop
      // now matches the target. Safe to reset rotation without flicker.
      mesh.rotation.x = 0;
      mesh.position.y = restY;
      setAnimState("idle");
      invalidate();
    }
  });

  return (
    <mesh
      ref={meshRef}
      position={[x, DISC_HEIGHT / 2 + DISC_Y_OFFSET, z]}
      material={materials}
      geometry={sharedGeometry}
      castShadow
      receiveShadow
    />
  );
}
