import { ChevronDown, ChevronUp, Activity, BarChart3 } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";
import { useReversiStore } from "@/stores/use-reversi-store";
import { AIThinkingLog } from "./AIThinkingLog";
import { EvaluationChart } from "./EvaluationChart";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";

export function AIAnalysisPanel() {
  const gameMode = useReversiStore((state) => state.gameMode);
  const isAIThinking = useReversiStore((state) => state.isAIThinking);
  const aiThinkingHistory = useReversiStore((state) => state.aiThinkingHistory);
  const isOpen = useReversiStore((state) => state.aiAnalysisPanelOpen);
  const setIsOpen = useReversiStore((state) => state.setAIAnalysisPanelOpen);

  // Only show for AI games
  if (gameMode !== "ai-black" && gameMode !== "ai-white") {
    return null;
  }

  const latestEntry = aiThinkingHistory[aiThinkingHistory.length - 1];

  return (
    <div className="border-t border-white/10 bg-background-secondary shrink-0">
      {/* Trigger */}
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className={cn(
          "w-full flex items-center justify-between px-4 py-3 hover:bg-white/5 transition-colors cursor-pointer",
          isOpen && "border-b border-white/10"
        )}
      >
        <div className="flex items-center gap-3">
          <Activity className={cn(
            "w-4 h-4",
            isAIThinking ? "text-accent-blue animate-pulse" : "text-foreground-muted"
          )} />
          <span className="text-sm font-medium text-foreground">AI Analysis</span>
          {isAIThinking && (
            <span className="text-xs bg-accent-blue/20 text-accent-blue px-2 py-0.5 rounded-full">
              Thinking...
            </span>
          )}
          {!isAIThinking && latestEntry && (
            <span className="text-xs text-primary font-mono font-semibold">
              Best: {latestEntry.bestMove} ({latestEntry.score > 0 ? "+" : ""}{latestEntry.score}) {latestEntry.acc === 100 ? latestEntry.depth : `${latestEntry.depth}@${latestEntry.acc}%`}
            </span>
          )}
        </div>
        {isOpen ? (
          <ChevronDown className="w-4 h-4 text-foreground-muted" />
        ) : (
          <ChevronUp className="w-4 h-4 text-foreground-muted" />
        )}
      </button>

      {/* Content */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="p-4 pb-6">
              <Tabs defaultValue="log" className="h-[240px]">
                <TabsList className="bg-white/10">
                  <TabsTrigger 
                    value="log" 
                    className="gap-1.5 data-[state=active]:bg-white/15 data-[state=active]:text-foreground text-foreground-secondary"
                  >
                    <Activity className="w-3.5 h-3.5" />
                    Thinking Log
                  </TabsTrigger>
                  <TabsTrigger 
                    value="chart" 
                    className="gap-1.5 data-[state=active]:bg-white/15 data-[state=active]:text-foreground text-foreground-secondary"
                  >
                    <BarChart3 className="w-3.5 h-3.5" />
                    Evaluation
                  </TabsTrigger>
                </TabsList>
                <TabsContent value="log" className="h-[200px] mt-2">
                  <AIThinkingLog />
                </TabsContent>
                <TabsContent value="chart" className="h-[200px] mt-2">
                  <EvaluationChart />
                </TabsContent>
              </Tabs>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
