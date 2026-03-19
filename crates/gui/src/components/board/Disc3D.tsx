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
}

const SEGMENTS = 32;
const DROP_SPRING_STIFFNESS = 300;
const DROP_SPRING_DAMPING = 20;
const FLIP_ARC_HEIGHT = 0.3;

const ROUGHNESS = 0.7;
const METALNESS = 0.05;

const sharedGeometry = new THREE.CylinderGeometry(
  DISC_RADIUS - 0.02, DISC_RADIUS, DISC_HEIGHT, SEGMENTS,
);

const BLACK_MATERIALS = [
  new THREE.MeshStandardMaterial({ color: DISC_COLOR_BLACK, roughness: ROUGHNESS, metalness: METALNESS }),
  new THREE.MeshStandardMaterial({ color: DISC_COLOR_BLACK, roughness: ROUGHNESS, metalness: METALNESS }),
  new THREE.MeshStandardMaterial({ color: DISC_COLOR_WHITE, roughness: ROUGHNESS, metalness: METALNESS }),
];

const WHITE_MATERIALS = [
  new THREE.MeshStandardMaterial({ color: DISC_COLOR_WHITE, roughness: ROUGHNESS, metalness: METALNESS }),
  new THREE.MeshStandardMaterial({ color: DISC_COLOR_WHITE, roughness: ROUGHNESS, metalness: METALNESS }),
  new THREE.MeshStandardMaterial({ color: DISC_COLOR_BLACK, roughness: ROUGHNESS, metalness: METALNESS }),
];

export function Disc3D({ row, col, color, isNew, flipDelay = 0 }: Disc3DProps) {
  const meshRef = useRef<THREE.Mesh>(null);
  const { invalidate } = useThree();
  const [x, z] = cellToWorld(row, col);

  const targetColorRef = useRef(color);
  const [displayColor, setDisplayColor] = useState(color);
  const materials = displayColor === "black" ? BLACK_MATERIALS : WHITE_MATERIALS;

  const [animState, setAnimState] = useState<"idle" | "dropping" | "flipping" | "flip-waiting">(
    isNew ? "dropping" : "idle"
  );
  const animProgress = useRef(0);
  const flipWaitTime = useRef(0);
  const prevColor = useRef<Player>(color);
  const dropY = useRef(isNew ? 0.5 : 0);
  const dropVelocity = useRef(0);

  useEffect(() => {
    targetColorRef.current = color;
    if (prevColor.current !== color && !isNew) {
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
  }, [color, isNew, flipDelay]);

  useFrame((_, delta) => {
    if (!meshRef.current) return;
    const mesh = meshRef.current;

    if (animState === "dropping") {
      const displacement = dropY.current;
      const springForce = -DROP_SPRING_STIFFNESS * displacement;
      const dampingForce = -DROP_SPRING_DAMPING * dropVelocity.current;
      dropVelocity.current += (springForce + dampingForce) * delta;
      dropY.current += dropVelocity.current * delta;
      mesh.position.y = DISC_HEIGHT / 2 + Math.max(0, dropY.current);
      if (Math.abs(dropY.current) < 0.001 && Math.abs(dropVelocity.current) < 0.01) {
        dropY.current = 0;
        mesh.position.y = DISC_HEIGHT / 2;
        mesh.rotation.x = 0;
        setAnimState("idle");
      }
      invalidate();
    } else if (animState === "flip-waiting") {
      flipWaitTime.current += delta;
      if (flipWaitTime.current >= flipDelay) {
        setAnimState("flipping");
        animProgress.current = 0;
      }
      invalidate();
    } else if (animState === "flipping") {
      animProgress.current += delta / FLIP_DURATION_S;
      if (animProgress.current >= 1) {
        mesh.rotation.x = 0;
        mesh.position.y = DISC_HEIGHT / 2;
        setAnimState("idle");
        setDisplayColor(targetColorRef.current);
      } else {
        const t = animProgress.current;
        mesh.rotation.x = t * Math.PI;
        mesh.position.y = DISC_HEIGHT / 2 + Math.sin(t * Math.PI) * FLIP_ARC_HEIGHT;
      }
      invalidate();
    }
  });

  return (
    <mesh ref={meshRef} position={[x, DISC_HEIGHT / 2, z]} material={materials} geometry={sharedGeometry} />
  );
}
