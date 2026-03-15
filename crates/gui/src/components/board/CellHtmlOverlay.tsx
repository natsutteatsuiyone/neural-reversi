import { Html } from "@react-three/drei";
import type { ReactNode } from "react";
import { cellToWorld } from "./board3d-utils";

interface CellHtmlOverlayProps {
  row: number;
  col: number;
  cellPixelSize: number;
  children: ReactNode;
  zIndex?: number;
}

export function CellHtmlOverlay({
  row, col, cellPixelSize, children, zIndex = 10,
}: CellHtmlOverlayProps) {
  const [x, z] = cellToWorld(row, col);

  return (
    <Html
      position={[x, 0.02, z]}
      center
      style={{
        width: `${cellPixelSize}px`,
        height: `${cellPixelSize}px`,
        position: "relative",
        pointerEvents: "none",
        zIndex,
      }}
      zIndexRange={[zIndex, zIndex]}
    >
      {children}
    </Html>
  );
}
