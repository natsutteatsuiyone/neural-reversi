import { useRef, useLayoutEffect } from "react";
import { Bot, RotateCcw, RotateCw } from "lucide-react";
import { cn } from "@/lib/utils";
import { useReversiStore } from "@/stores/use-reversi-store";
import { Stone } from "@/components/board/Stone";
import { Button } from "@/components/ui/button";
import { useTranslation } from "react-i18next";

export function MoveHistory() {
  const { t } = useTranslation();
  const scrollRef = useRef<HTMLDivElement>(null);
  const moves = useReversiStore((state) => state.moves);
  const allMoves = useReversiStore((state) => state.allMoves);
  const gameStatus = useReversiStore((state) => state.gameStatus);
  const isAIThinking = useReversiStore((state) => state.isAIThinking);
  const isAnalyzing = useReversiStore((state) => state.isAnalyzing);
  const undoMove = useReversiStore((state) => state.undoMove);
  const redoMove = useReversiStore((state) => state.redoMove);
  const prevMovesLengthRef = useRef(moves.length);

  const canUndo = moves.length > 0 && gameStatus === "playing" && !isAIThinking && !isAnalyzing;
  const canRedo = moves.length < allMoves.length && gameStatus === "playing" && !isAIThinking && !isAnalyzing;

  useLayoutEffect(() => {
    if (prevMovesLengthRef.current !== moves.length) {
      if (scrollRef.current) {
        scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
      }
      prevMovesLengthRef.current = moves.length;
    }
  });

  return (
    <div className="h-full flex flex-col">
      <div className="px-4 py-2 border-b border-white/10 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-foreground">{t('history.title')}</h3>
        <div className="flex items-center gap-1">
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={undoMove}
            disabled={!canUndo}
            aria-label={t('history.undo')}
            className="text-foreground-secondary hover:text-foreground hover:bg-white/10"
          >
            <RotateCcw className="w-4 h-4" />
          </Button>
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={redoMove}
            disabled={!canRedo}
            aria-label={t('history.redo')}
            className="text-foreground-secondary hover:text-foreground hover:bg-white/10"
          >
            <RotateCw className="w-4 h-4" />
          </Button>
        </div>
      </div>

      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto p-2 scrollbar-thin"
      >
        {moves.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <p className="text-sm text-foreground-muted">{t('history.noMoves')}</p>
          </div>
        ) : (
          <div className="space-y-0.5">
            {moves.map((move, index) => (
              <div
                key={move.id}
                className={cn(
                  "grid grid-cols-[28px_1fr] gap-1 text-sm rounded-md",
                  index === moves.length - 1 && "bg-white/10"
                )}
              >
                {/* Move number */}
                <div className="text-foreground-muted font-mono text-xs flex items-center justify-center">
                  {index + 1}.
                </div>

                {/* Move details */}
                <div
                  className={cn(
                    "flex items-center gap-1.5 px-2 py-1 rounded",
                    index === moves.length - 1 && "bg-primary/20"
                  )}
                >
                  <Stone color={move.player} size="sm" />
                  <span className="font-medium text-foreground">{move.notation}</span>
                  {move.isAI && (
                    <>
                      <Bot className="w-3 h-3 text-accent-blue" />
                      {move.score !== undefined && (
                        <span className="text-xs font-mono text-foreground-muted">
                          {move.score > 0 ? "+" : ""}
                          {move.score}
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
