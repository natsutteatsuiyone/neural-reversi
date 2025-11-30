import { Bot, CircleX } from "lucide-react";
import { cn } from "@/lib/utils";
import { getNotation } from "@/lib/game-logic";
import type { AIMoveProgress, AIMoveResult } from "@/lib/ai";

interface AIInfoProps {
    thinking: boolean;
    lastMove: AIMoveResult | null;
    aiMoveProgress: AIMoveProgress | null;
    onAbort: () => Promise<void>;
    aiMode: string;
    aiRemainingTime: number;
}

function formatScore(score: number): string {
    return score > 0 ? `+${score}` : String(score);
}

function formatDepth(depth: number, acc: number): string {
    return acc === 100 ? `${depth}` : `${depth}@${acc}%`;
}

export function AIInfo({ thinking, lastMove, aiMoveProgress, onAbort, aiMode, aiRemainingTime }: AIInfoProps) {
    if (!thinking && !lastMove && aiMode !== "game-time") {
        return null;
    }

    const showProgress = thinking && aiMoveProgress;
    const showLastMove = !thinking && lastMove;

    let moveLabel = "";
    let scoreLabel = "";
    let depthLabel = "";

    if (showProgress) {
        moveLabel = aiMoveProgress!.bestMove;
        scoreLabel = formatScore(aiMoveProgress!.score);
        depthLabel = formatDepth(aiMoveProgress!.depth, aiMoveProgress!.acc);
    } else if (showLastMove) {
        const move = lastMove!;
        moveLabel = getNotation(move.row, move.col);
        scoreLabel = formatScore(move.score);
        depthLabel = formatDepth(move.depth, move.acc);
    }

    const formatTime = (ms: number) => {
        const totalSeconds = Math.max(0, Math.floor(ms / 1000));
        const minutes = Math.floor(totalSeconds / 60);
        const seconds = totalSeconds % 60;
        return `${minutes}:${seconds.toString().padStart(2, "0")}`;
    };

    return (
        <div className="flex items-center grow">
            <div className="flex items-center gap-2 grow">
                <Bot
                    className={cn(
                        "w-4 h-4",
                        thinking ? "text-emerald-300 animate-pulse" : "text-emerald-300"
                    )}
                />
                {aiMode === "game-time" && (
                    <div className="text-sm text-emerald-200/90 font-mono font-bold mr-2">
                        {formatTime(aiRemainingTime)}
                    </div>
                )}
                {moveLabel && (
                    <div className="text-sm text-emerald-200/70 font-mono">
                        <div>
                            <span>{moveLabel}</span>
                            <span className="ml-1">({scoreLabel})</span>
                        </div>
                        <div>{depthLabel}</div>
                    </div>
                )}
            </div>
            {thinking && (
                <div className="text-sm text-emerald-200/70 flex flex-col items-center">
                    <button type="button" onClick={() => void onAbort()}>
                        <CircleX className="w-5 h-5 text-amber-200 hover:text-amber-100 cursor-pointer" />
                    </button>
                </div>
            )}
        </div>
    );
}
