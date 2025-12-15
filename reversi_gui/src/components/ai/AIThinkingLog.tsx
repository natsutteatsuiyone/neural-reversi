import { useRef, useLayoutEffect } from "react";
import { useReversiStore } from "@/stores/use-reversi-store";
import { cn } from "@/lib/utils";

function formatScore(score: number): string {
  return score > 0 ? `+${score}` : String(score);
}

function formatDepth(depth: number, acc: number): string {
  return acc === 100 ? `${depth}` : `${depth}@${acc}%`;
}

function formatNps(nps: number): string {
  if (nps <= 0) return "-";
  if (nps >= 1_000_000_000) return `${(nps / 1_000_000_000).toFixed(1)}G`;
  if (nps >= 1_000_000) return `${(nps / 1_000_000).toFixed(1)}M`;
  if (nps >= 1_000) return `${(nps / 1_000).toFixed(1)}K`;
  return nps.toFixed(0);
}

function formatNodes(nodes: number): string {
  if (nodes >= 1_000_000_000) return `${(nodes / 1_000_000_000).toFixed(2)}G`;
  if (nodes >= 1_000_000) return `${(nodes / 1_000_000).toFixed(2)}M`;
  if (nodes >= 1_000) return `${(nodes / 1_000).toFixed(1)}K`;
  return nodes.toString();
}

export function AIThinkingLog() {
  const scrollRef = useRef<HTMLDivElement>(null);
  const aiThinkingHistory = useReversiStore((state) => state.aiThinkingHistory);
  const isAIThinking = useReversiStore((state) => state.isAIThinking);

  const prevHistoryLengthRef = useRef(aiThinkingHistory.length);

  useLayoutEffect(() => {
    if (prevHistoryLengthRef.current !== aiThinkingHistory.length) {
      if (scrollRef.current) {
        scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
      }
      prevHistoryLengthRef.current = aiThinkingHistory.length;
    }
  });

  const hasData = aiThinkingHistory.length > 0;
  const latestEntry = aiThinkingHistory[aiThinkingHistory.length - 1];

  return (
    <div className="h-full flex flex-col bg-white/5 rounded-lg overflow-hidden border border-white/10">
      {/* Summary Stats */}
      {hasData && (
        <div className="flex gap-4 px-3 py-2 text-xs border-b border-white/10 bg-white/5 shrink-0">
          <div className="flex items-center gap-1.5">
            <span className="text-foreground-muted">Best:</span>
            <span className="font-mono font-semibold text-foreground">{latestEntry.bestMove}</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="text-foreground-muted">Nodes:</span>
            <span className="font-mono text-foreground">{formatNodes(latestEntry.nodes)}</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="text-foreground-muted">NPS:</span>
            <span className="font-mono text-foreground">{formatNps(latestEntry.nps)}</span>
          </div>
        </div>
      )}

      {/* Log Table */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto scrollbar-thin">
        <table className="w-full text-xs">
          <thead className="sticky top-0 bg-background-secondary border-b border-white/10">
            <tr>
              <th className="text-left px-3 py-2 font-medium text-foreground-muted w-16">Depth</th>
              <th className="text-left px-3 py-2 font-medium text-foreground-muted w-16">Score</th>
              <th className="text-left px-3 py-2 font-medium text-foreground-muted">PV Line</th>
            </tr>
          </thead>
          <tbody>
            {hasData ? (
              aiThinkingHistory.map((entry, index) => (
                <tr
                  key={`${entry.depth}-${entry.acc}-${index}`}
                  className={cn(
                    "border-b border-white/5",
                    index === aiThinkingHistory.length - 1 && "bg-primary/10"
                  )}
                >
                  <td className="px-3 py-1.5 font-mono text-foreground">
                    {formatDepth(entry.depth, entry.acc)}
                  </td>
                  <td className={cn(
                    "px-3 py-1.5 font-mono font-semibold",
                    entry.score > 0 ? "text-primary" : entry.score < 0 ? "text-destructive" : "text-foreground"
                  )}>
                    {formatScore(entry.score)}
                  </td>
                  <td className="px-3 py-1.5 font-mono text-foreground-muted truncate max-w-[200px]">
                    {entry.pvLine}
                  </td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan={3} className="text-center py-8 text-foreground-muted">
                  {isAIThinking ? "Thinking..." : "Waiting for AI turn"}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
