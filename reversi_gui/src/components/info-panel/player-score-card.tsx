import { cn } from "@/lib/utils";
import type { AIMoveProgress, AIMoveResult } from "@/lib/ai";
import { AIInfo } from "./ai-info";

const DISC_CLASS: Record<"black" | "white", string> = {
    black: "w-7 h-7 rounded-full bg-gradient-to-br from-neutral-600 to-black shadow-md",
    white: "w-7 h-7 rounded-full bg-gradient-to-br from-white to-neutral-200 border border-white/20 shadow-md",
};

interface PlayerScoreCardProps {
    color: "black" | "white";
    score: number;
    isCurrent: boolean;
    isAIControlled: boolean;
    aiLevel: number;
    isThinking: boolean;
    lastAIMove: AIMoveResult | null;
    aiMoveProgress: AIMoveProgress | null;
    onAbort: () => Promise<void>;
    aiMode: string;
    aiRemainingTime: number;
}

export function PlayerScoreCard({
    color,
    score,
    isCurrent,
    isAIControlled,
    aiLevel,
    isThinking,
    lastAIMove,
    aiMoveProgress,
    onAbort,
    aiMode,
    aiRemainingTime,
}: PlayerScoreCardProps) {
    return (
        <div
            className={cn("rounded-lg h-14", isCurrent ? "bg-white/20" : "bg-white/10")}
        >
            <div className="h-full px-3 flex items-center gap-3">
                <div className="shrink-0">
                    <div className={DISC_CLASS[color]} />
                </div>
                <div className="flex-1 min-w-0 flex items-center gap-3">
                    <div className="flex items-center gap-2">
                        <span className="text-xl font-bold text-white/90">{score}</span>
                        <span className="text-sm text-white/70">
                            {isAIControlled ? (
                                <div className="flex flex-col">
                                    <span>AI</span>
                                    {aiMode === "level" && <span>Lv.{aiLevel}</span>}
                                </div>
                            ) : (
                                <div>Player</div>
                            )}
                        </span>
                    </div>
                    {isAIControlled && (
                        <AIInfo
                            thinking={isThinking}
                            lastMove={isThinking ? null : lastAIMove}
                            aiMoveProgress={aiMoveProgress}
                            onAbort={onAbort}
                            aiMode={aiMode}
                            aiRemainingTime={aiRemainingTime}
                        />
                    )}
                </div>
                <div className="shrink-0">
                    <div
                        className={cn(
                            "w-2 h-2 rounded-full transition-colors",
                            isCurrent ? "bg-emerald-400" : "bg-transparent"
                        )}
                    />
                </div>
            </div >
        </div >
    );
}
