import { useCallback } from "react";
import { Header } from "./Header";
import { Board } from "@/components/board/Board";
import { Sidebar } from "./Sidebar";
import {
  AIAnalysisPanelContent,
  AIAnalysisPanelHeader,
} from "@/components/ai/AIAnalysisPanel";
import { NewGameModal } from "@/components/game/NewGameModal";
import { AboutModal } from "@/components/game/AboutModal";
import { SolverModal, SolverPanel } from "@/components/solver";
import { Toaster } from "@/components/ui/sonner";
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from "@/components/ui/resizable";
import { useMediaQuery } from "@/hooks/use-media-query";
import { useReversiStore } from "@/stores/use-reversi-store";

type Layout = { [panelId: string]: number };

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

function BoardArea() {
  return (
    <main className="flex h-full min-h-0 min-w-0 items-center justify-center p-3 sm:p-4 lg:p-6">
      <Board />
    </main>
  );
}

function RightColumn() {
  const isSolverActive = useReversiStore((state) => state.isSolverActive);
  return isSolverActive ? <SolverPanel /> : <Sidebar />;
}

function DesktopLayout() {
  const rightPanelSize = useReversiStore((state) => state.rightPanelSize);
  const bottomPanelSize = useReversiStore((state) => state.bottomPanelSize);
  const setRightPanelSize = useReversiStore((state) => state.setRightPanelSize);
  const setBottomPanelSize = useReversiStore((state) => state.setBottomPanelSize);
  const isAnalysisOpen = useReversiStore((state) => state.aiAnalysisPanelOpen);
  const isSolverActive = useReversiStore((state) => state.isSolverActive);

  const showBottomPanel = isAnalysisOpen && !isSolverActive;

  const handleHorizontalLayout = useCallback(
    (layout: Layout) => {
      const next = layout.sidebar;
      if (typeof next === "number") {
        setRightPanelSize(next);
      }
    },
    [setRightPanelSize],
  );

  const handleVerticalLayout = useCallback(
    (layout: Layout) => {
      const next = layout["bottom-analysis"];
      if (typeof next === "number") {
        setBottomPanelSize(next);
      }
    },
    [setBottomPanelSize],
  );

  const sidebarSize = clamp(rightPanelSize, 18, 45);
  const boardSize = 100 - sidebarSize;
  const bottomSize = clamp(bottomPanelSize, 18, 70);
  const topSize = 100 - bottomSize;

  return (
    <ResizablePanelGroup
      direction="vertical"
      onLayoutChanged={handleVerticalLayout}
      defaultLayout={
        showBottomPanel
          ? { "top-area": topSize, "bottom-analysis": bottomSize }
          : { "top-area": 100 }
      }
      className="flex-1 min-h-0"
      key={showBottomPanel ? "v-open" : "v-closed"}
    >
      <ResizablePanel id="top-area" minSize="30">
        <ResizablePanelGroup
          direction="horizontal"
          onLayoutChanged={handleHorizontalLayout}
          defaultLayout={{ board: boardSize, sidebar: sidebarSize }}
          className="h-full"
        >
          <ResizablePanel id="board" minSize="40">
            <BoardArea />
          </ResizablePanel>
          <ResizableHandle withHandle />
          <ResizablePanel id="sidebar" minSize="18" maxSize="45">
            <RightColumn />
          </ResizablePanel>
        </ResizablePanelGroup>
      </ResizablePanel>
      {showBottomPanel && (
        <>
          <ResizableHandle withHandle />
          <ResizablePanel id="bottom-analysis" minSize="18" maxSize="70">
            <AIAnalysisPanelContent />
          </ResizablePanel>
        </>
      )}
    </ResizablePanelGroup>
  );
}

function MobileLayout() {
  const isAnalysisOpen = useReversiStore((state) => state.aiAnalysisPanelOpen);
  const isSolverActive = useReversiStore((state) => state.isSolverActive);
  const showBottomPanel = isAnalysisOpen && !isSolverActive;

  return (
    <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
      <main className="flex min-h-0 min-w-0 flex-1 items-center justify-center p-3 sm:p-4">
        <Board />
      </main>
      <div className="flex max-h-[45%] min-h-0 shrink-0 basis-72 flex-col border-t border-white/10 sm:basis-80">
        <RightColumn />
      </div>
      {showBottomPanel && (
        <div className="flex max-h-[55%] min-h-0 shrink-0 flex-col overflow-hidden border-t border-white/10">
          <AIAnalysisPanelContent />
        </div>
      )}
    </div>
  );
}

export function GameLayout() {
  const isDesktop = useMediaQuery("(min-width: 1024px)");
  const isSolverActive = useReversiStore((state) => state.isSolverActive);

  return (
    <div className="flex h-dvh min-h-0 flex-col overflow-hidden bg-background">
      <NewGameModal />
      <SolverModal />
      <AboutModal />
      <Header />

      {isDesktop ? <DesktopLayout /> : <MobileLayout />}

      {!isSolverActive && <AIAnalysisPanelHeader />}

      <Toaster position="top-center" />
    </div>
  );
}
