"use client"

import { cn } from "@/lib/utils"
import { useRef, useLayoutEffect } from "react"
import { Bot, ChevronLeft, ChevronRight } from "lucide-react"
import { useReversiStore } from "@/stores/use-reversi-store"
import { Button } from "@/components/ui/button"

export function MoveHistory() {
  const scrollRef = useRef<HTMLDivElement>(null)
  const moves = useReversiStore((state) => state.moves);
  const allMoves = useReversiStore((state) => state.allMoves);
  const gameStatus = useReversiStore((state) => state.gameStatus);
  const isAIThinking = useReversiStore((state) => state.isAIThinking);
  const isAnalyzing = useReversiStore((state) => state.isAnalyzing);
  const undoMove = useReversiStore((state) => state.undoMove);
  const redoMove = useReversiStore((state) => state.redoMove);

  const prevMovesLengthRef = useRef(moves.length);

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

      <div className="flex justify-between items-center mb-2">
        <h2 className="text-lg font-medium text-white/90">Move History</h2>
        <div className="flex gap-1">
          <Button
            variant="ghost"
            size="icon"
            onClick={undoMove}
            disabled={moves.length === 0 || gameStatus !== "playing" || isAIThinking || isAnalyzing}
            className="h-8 w-8 text-white/90 hover:bg-white/10 cursor-pointer"
            aria-label="Undo Move"
          >
            <ChevronLeft className="h-4 w-4" />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            onClick={redoMove}
            disabled={moves.length >= allMoves.length || gameStatus !== "playing" || isAIThinking || isAnalyzing}
            className="h-8 w-8 text-white/90 hover:bg-white/10 cursor-pointer"
            aria-label="Redo Move"
          >
            <ChevronRight className="h-4 w-4" />
          </Button>
        </div>
      </div>

      <div
        ref={scrollRef}
        className="flex-1 bg-white/10 rounded-lg p-2 overflow-y-auto
          scrollbar-thin
          scrollbar-track-white/5
          scrollbar-thumb-white/25 hover:scrollbar-thumb-white/40
          scrollbar-track-rounded-full scrollbar-thumb-rounded-full
          [&::-webkit-scrollbar]:w-2
          [&::-webkit-scrollbar-thumb]:bg-white/25
          [&::-webkit-scrollbar-thumb]:hover:bg-white/40
          [&::-webkit-scrollbar-thumb]:rounded-full
          [&::-webkit-scrollbar-track]:bg-white/5
          [&::-webkit-scrollbar-track]:rounded-full
          [&::-webkit-scrollbar-thumb]:transition-colors
          [&::-webkit-scrollbar-thumb]:duration-150"
      >
        {moves.length === 0 ? (
          <p className="text-sm text-center text-white/50 p-3">No moves yet</p>
        ) : (
          <div className="space-y-0.5">
            {moves.map((move, index) => (
              <div
                key={move.id}
                className={cn(
                  "flex items-center gap-2 px-2 py-1 rounded text-sm",
                  index === moves.length - 1 && "bg-white/10",
                )}
              >
                <span className="text-white/60 w-6 font-medium text-xs">{index + 1}.</span>
                <div
                  className={cn(
                    "w-4 h-4 rounded-full shrink-0",
                    move.player === "black"
                      ? "bg-gradient-to-br from-neutral-600 to-black"
                      : "bg-gradient-to-br from-white to-neutral-200 border border-white/20",
                  )}
                />
                <div className="flex items-center gap-2">
                  <span className="font-medium text-white/90">{move.notation}</span>
                  {move.isAI && (
                    <>
                      <Bot className="w-3.5 h-3.5 text-emerald-300" />
                      <span className="text-xs font-mono text-emerald-200/70">
                        {move.score !== undefined && (move.score > 0 ? "+" : "") + move.score}
                      </span>
                    </>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

