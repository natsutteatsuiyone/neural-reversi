import { useState, useCallback, useMemo } from "react";
import { ChevronDown, ChevronUp, Activity, BarChart3, Search, Square } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";
import { useReversiStore } from "@/stores/use-reversi-store";
import { ANALYSIS_LEVELS } from "@/types";
import { AIThinkingLog } from "./AIThinkingLog";
import { EvaluationChart } from "./EvaluationChart";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useTranslation } from "react-i18next";

export function AIAnalysisPanel() {
  const { t } = useTranslation();
  const gameMode = useReversiStore((state) => state.gameMode);
  const isAIThinking = useReversiStore((state) => state.isAIThinking);
  const aiThinkingHistory = useReversiStore((state) => state.aiThinkingHistory);
  const isOpen = useReversiStore((state) => state.aiAnalysisPanelOpen);
  const setIsOpen = useReversiStore((state) => state.setAIAnalysisPanelOpen);
  const isGameAnalyzing = useReversiStore((state) => state.isGameAnalyzing);
  const analyzeGame = useReversiStore((state) => state.analyzeGame);
  const abortGameAnalysis = useReversiStore((state) => state.abortGameAnalysis);
  const moveHistory = useReversiStore((state) => state.moveHistory);
  const gameAnalysisResult = useReversiStore((state) => state.gameAnalysisResult);
  const gameAnalysisLevel = useReversiStore((state) => state.gameAnalysisLevel);
  const setGameAnalysisLevel = useReversiStore((state) => state.setGameAnalysisLevel);
  const [activeTab, setActiveTab] = useState("log");

  const handleAnalyzeGame = useCallback(() => {
    setActiveTab("chart");
    analyzeGame();
  }, [analyzeGame]);

  const totalMoves = useMemo(() => {
    let count = 0;
    for (const m of moveHistory.allMoves) {
      if (m.row >= 0) count++;
    }
    return count;
  }, [moveHistory.allMoves]);

  // Only show for AI games
  if (gameMode !== "ai-black" && gameMode !== "ai-white") {
    return null;
  }

  const latestEntry = aiThinkingHistory[aiThinkingHistory.length - 1];
  const canAnalyze = totalMoves > 0 && !isAIThinking && !isGameAnalyzing;

  return (
    <div className="border-t border-white/10 bg-background-secondary shrink-0">
      {/* Trigger */}
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className={cn(
          "flex w-full items-center justify-between gap-3 px-4 py-3 transition-colors cursor-pointer hover:bg-white/5",
          isOpen && "border-b border-white/10"
        )}
      >
        <div className="flex min-w-0 flex-wrap items-center gap-2 sm:gap-3">
          <Activity className={cn(
            "w-4 h-4",
            isAIThinking ? "text-accent-blue animate-pulse" : "text-foreground-muted"
          )} />
          <span className="text-sm font-medium text-foreground">{t('analysis.title')}</span>
          {isAIThinking && (
            <span className="text-xs bg-accent-blue/20 text-accent-blue px-2 py-0.5 rounded-full">
              {t('ai.thinking')}
            </span>
          )}
          {isGameAnalyzing && (
            <span className="text-xs bg-amber-500/20 text-amber-400 px-2 py-0.5 rounded-full">
              {t('analysis.analyzing', {
                current: gameAnalysisResult?.length ?? 0,
                total: totalMoves,
              })}
            </span>
          )}
          {!isAIThinking && !isGameAnalyzing && latestEntry && (
            <span className="max-w-full truncate text-xs font-mono font-semibold text-primary">
              {t('analysis.best')} {latestEntry.bestMove} ({latestEntry.score > 0 ? "+" : ""}{latestEntry.score}) {latestEntry.acc === 100 ? latestEntry.depth : `${latestEntry.depth}@${latestEntry.acc}%`}
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
              <Tabs value={activeTab} onValueChange={setActiveTab} className="h-[240px]">
                <div className="flex items-center justify-between">
                  <TabsList className="bg-white/10">
                    <TabsTrigger
                      value="log"
                      className="gap-1.5 data-[state=active]:bg-white/15 data-[state=active]:text-foreground text-foreground-secondary"
                    >
                      <Activity className="w-3.5 h-3.5" />
                      {t('analysis.thinkingLog')}
                    </TabsTrigger>
                    <TabsTrigger
                      value="chart"
                      className="gap-1.5 data-[state=active]:bg-white/15 data-[state=active]:text-foreground text-foreground-secondary"
                    >
                      <BarChart3 className="w-3.5 h-3.5" />
                      {t('analysis.evaluation')}
                    </TabsTrigger>
                  </TabsList>
                  <div className="flex items-center gap-2">
                    <div className="flex items-center gap-1.5 text-xs text-foreground-muted">
                      {t('analysis.analysisLevel')}
                      <Select
                        value={gameAnalysisLevel.toString()}
                        onValueChange={(v) => setGameAnalysisLevel(Number(v))}
                        disabled={isGameAnalyzing}
                      >
                        <SelectTrigger size="sm" className="h-6 min-w-[3rem] px-2 py-0 text-xs text-white/90 border-white/15 bg-white/10">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent className="border-white/20">
                          {ANALYSIS_LEVELS.map((level) => (
                            <SelectItem key={level} value={level.toString()} className="text-xs text-white/90 focus:bg-white/10 focus:text-white">
                              {level}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                    {isGameAnalyzing ? (
                      <button
                        type="button"
                        onClick={abortGameAnalysis}
                        className="flex items-center gap-1.5 px-3 py-1 text-xs font-medium rounded-md bg-red-500/20 text-red-400 hover:bg-red-500/30 transition-colors cursor-pointer"
                      >
                        <Square className="w-3 h-3" />
                        {t('analysis.abort')}
                      </button>
                    ) : (
                      <button
                        type="button"
                        onClick={handleAnalyzeGame}
                        disabled={!canAnalyze}
                        className="flex items-center gap-1.5 px-3 py-1 text-xs font-medium rounded-md bg-accent-blue/20 text-accent-blue hover:bg-accent-blue/30 transition-colors cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
                      >
                        <Search className="w-3 h-3" />
                        {t('analysis.analyze')}
                      </button>
                    )}
                  </div>
                </div>
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
