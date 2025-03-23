"use client"

import { cn } from "@/lib/utils"
import { useEffect, useRef } from "react"
import { Bot } from "lucide-react"
import { useReversiStore } from "@/stores/use-reversi-store"

export function MoveHistory() {
  const scrollRef = useRef<HTMLDivElement>(null)
  const moves = useReversiStore((state) => state.moves);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [moves])

  return (
    <div className="h-full flex flex-col">
      <h2 className="text-lg font-medium text-white/90 mb-2">Move History</h2>
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

