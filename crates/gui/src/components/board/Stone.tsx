import { cn } from "@/lib/utils";
import type { Player } from "@/types";

interface StoneProps {
  color: Player;
  size?: "sm" | "md" | "lg";
}

export function Stone({ color, size = "md" }: StoneProps) {
  const sizeClasses = {
    sm: "w-4 h-4",
    md: "w-[80%] h-[80%]",
    lg: "w-10 h-10",
  };

  return (
    <div
      className={cn(
        "rounded-full",
        sizeClasses[size],
        color === "black"
          ? "bg-gradient-to-br from-stone-black-from to-stone-black-to stone-shadow-black ring-1 ring-white/20"
          : "bg-gradient-to-br from-stone-white-from to-stone-white-to stone-shadow-white border border-gray-200"
      )}
    />
  );
}
