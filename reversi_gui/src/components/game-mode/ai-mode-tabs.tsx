import { Button } from "@/components/ui/button";
import { Zap, Timer } from "lucide-react";
import { cn } from "@/lib/utils";
import type { AIMode } from "@/types";

interface AIModeTabsProps {
    aiMode: AIMode;
    setAIMode: (mode: AIMode) => void;
    disabled?: boolean;
}

export function AIModeTabs({ aiMode, setAIMode, disabled }: AIModeTabsProps) {
    return (
        <div className="grid grid-cols-2 gap-2 p-1 bg-black/20 rounded-lg">
            <Button
                variant={aiMode === "game-time" ? "secondary" : "ghost"}
                size="sm"
                onClick={() => setAIMode("game-time")}
                disabled={disabled}
                className={cn(
                    "h-8 text-xs font-medium px-0",
                    aiMode === "game-time"
                        ? "bg-white/10 text-white hover:bg-white/20"
                        : "text-white/60 hover:text-white hover:bg-white/5"
                )}
            >
                <Timer className="w-3 h-3 mr-1.5" />
                Time
            </Button>
            <Button
                variant={aiMode === "level" ? "secondary" : "ghost"}
                size="sm"
                onClick={() => setAIMode("level")}
                disabled={disabled}
                className={cn(
                    "h-8 text-xs font-medium px-0",
                    aiMode === "level"
                        ? "bg-white/10 text-white hover:bg-white/20"
                        : "text-white/60 hover:text-white hover:bg-white/5"
                )}
            >
                <Zap className="w-3 h-3 mr-1.5" />
                Level
            </Button>
        </div>
    );
}
