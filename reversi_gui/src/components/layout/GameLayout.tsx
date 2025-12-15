import { Header } from "./Header";
import { Board } from "@/components/board/Board";
import { Sidebar } from "./Sidebar";
import { AIAnalysisPanel } from "@/components/ai/AIAnalysisPanel";
import { NewGameModal } from "@/components/game/NewGameModal";
import { Toaster } from "@/components/ui/sonner";

export function GameLayout() {
  return (
    <div className="h-screen flex flex-col bg-background overflow-hidden">
      <NewGameModal />
      <Header />
      
      <div className="flex-1 flex overflow-hidden">
        {/* Main Board Area */}
        <main className="flex-1 flex items-center justify-center p-6">
          <Board />
        </main>

        {/* Sidebar */}
        <Sidebar />
      </div>

      {/* AI Analysis Panel (Collapsible) */}
      <AIAnalysisPanel />
      
      <Toaster position="top-center" />
    </div>
  );
}
