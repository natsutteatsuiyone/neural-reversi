import { cn } from "@/lib/utils"
import type { Player } from "@/types"

interface GamePieceProps {
  color: Player
  isNew?: boolean
}

export function GamePiece({ color, isNew }: GamePieceProps) {
  return (
    <div
      className={cn(
        "w-[85%] h-[85%] rounded-full transition-colors",
        isNew && "animate-scale-in",
        color === "black"
          ? "bg-gradient-to-br from-neutral-600 to-black shadow-[inset_-2px_-2px_8px_rgba(255,255,255,0.1),inset_2px_2px_8px_rgba(0,0,0,0.4),0_2px_4px_rgba(0,0,0,0.4)]"
          : "bg-gradient-to-br from-white to-neutral-200 border border-neutral-300 shadow-[inset_-2px_-2px_8px_rgba(0,0,0,0.05),inset_2px_2px_8px_rgba(255,255,255,0.8),0_2px_4px_rgba(0,0,0,0.2)]",
      )}
    />
  )
}

