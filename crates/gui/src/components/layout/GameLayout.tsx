import { Header } from "./Header";
import { Board } from "@/components/board/Board";
import { Sidebar } from "./Sidebar";
import { AIAnalysisPanel } from "@/components/ai/AIAnalysisPanel";
import { NewGameModal } from "@/components/game/NewGameModal";
import { Toaster } from "@/components/ui/sonner";

export function GameLayout() {
  return (
    <div className="flex h-dvh min-h-0 flex-col overflow-hidden bg-background">
      <NewGameModal />
      <Header />
      
      <div className="flex min-h-0 flex-1 flex-col overflow-hidden lg:flex-row">
        {/* Main Board Area */}
        <main className="flex min-h-0 min-w-0 flex-1 items-center justify-center p-3 sm:p-4 lg:p-6">
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