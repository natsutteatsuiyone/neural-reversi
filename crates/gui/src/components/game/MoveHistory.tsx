import { useRef, useState, useLayoutEffect, useEffect } from "react";
import { Bot, Check, Copy, List, RotateCcw, RotateCw } from "lucide-react";
import { cn } from "@/lib/utils";
import { formatScore } from "@/lib/score-format";
import { useReversiStore } from "@/stores/use-reversi-store";
import { isGameSearchActive } from "@/stores/engine-activity";
import { Stone } from "@/components/board/Stone";
import { Button } from "@/components/ui/button";
import { useTranslation } from "react-i18next";

export function MoveHistory() {
  const { t } = useTranslation();
  const scrollRef = useRef<HTMLDivElement>(null);
  const moveHistory = useReversiStore((state) => state.moveHistory);
  const moves = moveHistory.allMoves;
  const gameStatus = useReversiStore((state) => state.gameStatus);
  const gameSearchActive = useReversiStore((state) => isGameSearchActive(state.engineActivity));
  const undoMove = useReversiStore((state) => state.undoMove);
  const redoMove = useReversiStore((state) => state.redoMove);
  const goToMove = useReversiStore((state) => state.goToMove);
  const currentIndex = moveHistory.length;
  const canNavigate = gameStatus !== "waiting" && !gameSearchActive;

  const [copied, setCopied] = useState(false);
  const copyTimerRef = useRef<ReturnType<typeof setTimeout>>(undefined);

  useEffect(() => () => clearTimeout(copyTimerRef.current), []);

  const canUndo = moveHistory.canUndo && canNavigate;
  const canRedo = moveHistory.canRedo && canNavigate;

  const copyTranscript = () => {
    const transcript = moveHistory.currentMoves
      .filter((m) => m.row >= 0)
      .map((m) => m.notation.toLowerCase())
      .join("");
    clearTimeout(copyTimerRef.current);
    navigator.clipboard.writeText(transcript).then(
      () => {
        setCopied(true);
        copyTimerRef.current = setTimeout(() => setCopied(false), 1500);
      },
      () => {},
    );
  };

  useLayoutEffect(() => {
    const container = scrollRef.current;
    if (!container || currentIndex === 0) return;
    // At the latest move, pin the list to the bottom so the newest row is fully visible.
    if (currentIndex === moves.length) {
      container.scrollTop = container.scrollHeight;
      return;
    }
    const activeEl = container.children[0]?.children[currentIndex - 1] as HTMLElement | undefined;
    activeEl?.scrollIntoView({ block: "nearest" });
  }, [currentIndex, moves.length]);

  return (
    <div className="relative h-full">
      <div className="absolute right-2 top-2 z-10 flex items-center gap-1 rounded-md bg-background-secondary/90 backdrop-blur-sm">
        <Button
          variant="ghost"
          size="icon-sm"
          onClick={copyTranscript}
          disabled={currentIndex === 0}
          aria-label={t("history.copy")}
          className="text-foreground-secondary hover:text-foreground hover:bg-white/10 hover:shadow-sm"
        >
          {copied ? <Check className="w-4 h-4 text-primary" /> : <Copy className="w-4 h-4" />}
        </Button>
        <Button
          variant="ghost"
          size="icon-sm"
          onClick={undoMove}
          disabled={!canUndo}
          aria-label={t("history.undo")}
          className="text-foreground-secondary hover:text-foreground hover:bg-white/10 hover:shadow-sm"
        >
          <RotateCcw className="w-4 h-4" />
        </Button>
        <Button
          variant="ghost"
          size="icon-sm"
          onClick={redoMove}
          disabled={!canRedo}
          aria-label={t("history.redo")}
          className="text-foreground-secondary hover:text-foreground hover:bg-white/10 hover:shadow-sm"
        >
          <RotateCw className="w-4 h-4" />
        </Button>
      </div>

      <div
        ref={scrollRef}
        aria-label={t("history.title")}
        className="h-full overflow-y-auto p-2 scrollbar-thin"
      >
        {moves.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full gap-2">
            <List className="w-5 h-5 text-foreground-muted/50" />
            <p className="text-sm text-foreground-muted">{t("history.noMoves")}</p>
          </div>
        ) : (
          <div className="space-y-0.5">
            {moves.map((move, index) => (
              <div
                key={move.id}
                onClick={canNavigate ? () => goToMove(index + 1) : undefined}
                className={cn(
                  "grid grid-cols-[32px_1fr] gap-1 text-sm rounded-md transition-colors",
                  canNavigate ? "cursor-pointer hover:bg-white/8" : "cursor-default",
                  index === currentIndex - 1 ? "bg-primary/10" : "bg-transparent",
                  index >= currentIndex && "opacity-40",
                )}
              >
                <div className="text-foreground-muted font-mono text-xs flex items-center justify-center tabular-nums">
                  {index + 1}.
                </div>
                <div className="flex items-center gap-1.5 px-2 py-1 rounded">
                  <Stone color={move.player} size="sm" />
                  <span className="font-medium text-foreground">{move.notation}</span>
                  {move.isAI && (
                    <>
                      <Bot className="w-3 h-3 text-accent-blue" />
                      {move.score !== undefined && (
                        <span className="text-xs font-mono text-foreground-muted tabular-nums">
                          {formatScore(move.score, "raw")}
                        </span>
                      )}
                    </>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
