import { Canvas } from "@react-three/fiber";
import { Suspense } from "react";
import { Board3DScene } from "./Board3DScene";
import { useBoardScene } from "./use-board-scene";

export function Board() {
  const scene = useBoardScene();

  return (
    <div className="h-full w-full">
      <Canvas frameloop="demand" shadows resize={{ debounce: { scroll: 50, resize: 100 } }}>
        <Suspense fallback={null}>
          <Board3DScene {...scene} />
        </Suspense>
      </Canvas>
    </div>
  );
}
