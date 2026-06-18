import { useCallback, useState } from "react";
import { ChevronDown, ChevronUp, Activity, BarChart3, Search, Square } from "lucide-react";
import { cn } from "@/lib/utils";
import { formatScore, scoreToneClass, formatDepth } from "@/lib/score-format";
import { useReversiStore } from "@/stores/use-reversi-store";
import { ANALYSIS_LEVELS } from "@/domain/game/types";
import { AIThinkingLog } from "./AIThinkingLog";
import { EvaluationChart } from "./EvaluationChart";
import { Button } from "@/components/ui/button";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useTranslation } from "react-i18next";

export function AIAnalysisPanelHeader() {
  const { t } = useTranslation();
  const isAIThinking = useReversiStore((state) => state.isAIThinking);
  const aiThinkingHistory = useReversiStore((state) => state.aiThinkingHistory);
  const isOpen = useReversiStore((state) => state.aiAnalysisPanelOpen);
  const setIsOpen = useReversiStore((state) => state.setAIAnalysisPanelOpen);
  const isGameAnalyzing = useReversiStore((state) => state.isGameAnalyzing);
  const gameAnalysisResult = useReversiStore((state) => state.gameAnalysisResult);
  const moveHistory = useReversiStore((state) => state.moveHistory);

  const totalMoves = moveHistory.playedMoveCount;

  const latestEntry = aiThinkingHistory[aiThinkingHistory.length - 1];

  return (
    <button
      type="button"
      aria-expanded={isOpen}
      onClick={() => setIsOpen(!isOpen)}
      className={cn(
        "flex w-full shrink-0 items-center justify-between gap-3 border-t border-card-border bg-background-secondary px-4 py-2.5 transition-colors cursor-pointer hover:bg-white/5",
      )}
    >
      <div className="flex min-w-0 flex-wrap items-center gap-2 sm:gap-3">
        <Activity
          className={cn(
            "w-4 h-4",
            isAIThinking ? "text-accent-blue animate-pulse" : "text-foreground-muted",
          )}
        />
        <span className="text-sm font-medium text-foreground">{t("analysis.title")}</span>
        {isAIThinking && (
          <span className="rounded-full bg-accent-blue/20 px-2 py-0.5 text-xs text-accent-blue">
            {t("ai.thinking")}
          </span>
        )}
        {isGameAnalyzing && (
          <span className="rounded-full bg-accent-amber/20 px-2 py-0.5 text-xs text-accent-amber">
            {t("analysis.analyzing", {
              current: gameAnalysisResult?.length ?? 0,
              total: totalMoves,
            })}
          </span>
        )}
        {!isAIThinking && !isGameAnalyzing && latestEntry && (
          <div className="flex items-center gap-2 text-xs font-mono">
            <span className="text-foreground-muted">{t("analysis.best")}</span>
            <span className="font-semibold text-primary">{latestEntry.bestMove}</span>
            <span className="mx-1 h-3 border-l border-card-border" />
            <span className={cn("font-semibold", scoreToneClass(latestEntry.score))}>
              {formatScore(latestEntry.score, "raw")}
            </span>
            <span className="mx-1 h-3 border-l border-card-border" />
            <span className="text-foreground-muted">
              {formatDepth(latestEntry.depth, latestEntry.acc)}
            </span>
          </div>
        )}
      </div>
      {isOpen ? (
        <ChevronDown className="w-4 h-4 text-foreground-muted" />
      ) : (
        <ChevronUp className="w-4 h-4 text-foreground-muted" />
      )}
    </button>
  );
}

export function AIAnalysisPanelContent() {
  const { t } = useTranslation();
  const isAIThinking = useReversiStore((state) => state.isAIThinking);
  const isGameAnalyzing = useReversiStore((state) => state.isGameAnalyzing);
  const analyzeGame = useReversiStore((state) => state.analyzeGame);
  const abortGameAnalysis = useReversiStore((state) => state.abortGameAnalysis);
  const moveHistory = useReversiStore((state) => state.moveHistory);
  const gameAnalysisLevel = useReversiStore((state) => state.gameAnalysisLevel);
  const setGameAnalysisLevel = useReversiStore((state) => state.setGameAnalysisLevel);
  const [activeTab, setActiveTab] = useState("log");

  const handleAnalyzeGame = useCallback(() => {
    setActiveTab("chart");
    analyzeGame();
  }, [analyzeGame]);

  const totalMoves = moveHistory.playedMoveCount;

  const canAnalyze = totalMoves > 0 && !isAIThinking && !isGameAnalyzing;

  return (
    <div className="flex h-full min-h-0 flex-col bg-background-secondary">
      <div className="flex h-full min-h-0 flex-col p-4">
        <Tabs
          value={activeTab}
          onValueChange={setActiveTab}
          className="flex h-full min-h-0 flex-col"
        >
          <div className="flex items-center justify-between">
            <TabsList>
              <TabsTrigger value="log">
                <Activity className="w-3.5 h-3.5" />
                {t("analysis.thinkingLog")}
              </TabsTrigger>
              <TabsTrigger value="chart">
                <BarChart3 className="w-3.5 h-3.5" />
                {t("analysis.evaluation")}
              </TabsTrigger>
            </TabsList>
            <div className="flex items-center gap-2">
              <div className="flex items-center gap-1.5 text-xs text-foreground-muted">
                {t("analysis.analysisLevel")}
                <Select
                  value={gameAnalysisLevel.toString()}
                  onValueChange={(v) => setGameAnalysisLevel(Number(v))}
                  disabled={isGameAnalyzing}
                >
                  <SelectTrigger
                    size="sm"
                    className="h-6 min-w-[3rem] border-card-border bg-white/5 px-2 py-0 text-xs text-foreground-secondary"
                  >
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="border-card-border">
                    {ANALYSIS_LEVELS.map((level) => (
                      <SelectItem
                        key={level}
                        value={level.toString()}
                        className="text-xs text-foreground-secondary focus:bg-white/10 focus:text-foreground"
                      >
                        {level}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              {isGameAnalyzing ? (
                <Button
                  variant="soft"
                  size="sm"
                  onClick={abortGameAnalysis}
                  className="gap-1.5 h-7 px-3 text-xs bg-destructive/20 text-destructive hover:bg-destructive/30 hover:shadow-sm"
                >
                  <Square className="w-3 h-3" />
                  {t("analysis.abort")}
                </Button>
              ) : (
                <Button
                  variant="soft"
                  size="sm"
                  onClick={handleAnalyzeGame}
                  disabled={!canAnalyze}
                  className="gap-1.5 h-7 px-3 text-xs bg-accent-blue/20 text-accent-blue hover:bg-accent-blue/30 hover:shadow-sm"
                >
                  <Search className="w-3 h-3" />
                  {t("analysis.analyze")}
                </Button>
              )}
            </div>
          </div>
          <TabsContent value="log" className="flex-1 min-h-0 mt-2">
            <AIThinkingLog />
          </TabsContent>
          <TabsContent value="chart" className="flex-1 min-h-0 mt-2">
            <EvaluationChart />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
