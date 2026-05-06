import type { AIMode, Board, Player } from "@/domain/game/types";
import type { AIService, AIMoveProgress, AIMoveResult } from "./types";

type SearchTimer = ReturnType<typeof setInterval>;

export interface AIMoveSearchProgress {
  progress: AIMoveProgress;
  nps: number;
}

interface AIMoveSearchOperationOptions {
  ai: Pick<AIService, "getAIMove">;
  board: Board;
  player: Player;
  level: number;
  mode: AIMode;
  timeLimitSeconds: number;
  remainingTimeMs: number;
  getRemainingTime: () => number;
  onStart: (startTime: number) => void;
  onTimerChange: (timer: SearchTimer | null) => void;
  onRemainingTime: (remainingTime: number) => void;
  onProgress: (entry: AIMoveSearchProgress) => void;
  onFinish: () => void;
}

function isSameProgress(a: AIMoveProgress | null, b: AIMoveProgress): boolean {
  return Boolean(
    a &&
    a.depth === b.depth &&
    a.score === b.score &&
    a.nodes === b.nodes &&
    a.pvLine === b.pvLine,
  );
}

/**
 * Owns AI move search lifecycle details that are independent of Zustand:
 * progress dedupe, NPS calculation, game-time countdown, and timer cleanup.
 */
export async function runAIMoveSearch({
  ai,
  board,
  player,
  level,
  mode,
  timeLimitSeconds,
  remainingTimeMs,
  getRemainingTime,
  onStart,
  onTimerChange,
  onRemainingTime,
  onProgress,
  onFinish,
}: AIMoveSearchOperationOptions): Promise<AIMoveResult> {
  const startTime = Date.now();
  const isGameTime = mode === "game-time";
  let timer: SearchTimer | null = null;
  let managingRemainingTime = isGameTime;
  let lastWrittenRemainingTime = remainingTimeMs;
  let lastProgress: AIMoveProgress | null = null;

  const clearTimer = () => {
    if (!timer) return;
    clearInterval(timer);
    timer = null;
    onTimerChange(null);
  };

  const applyManagedRemainingTime = (remainingTime: number) => {
    if (!managingRemainingTime) return;
    if (remainingTime === lastWrittenRemainingTime) return;
    if (getRemainingTime() !== lastWrittenRemainingTime) {
      managingRemainingTime = false;
      clearTimer();
      return;
    }
    lastWrittenRemainingTime = remainingTime;
    onRemainingTime(remainingTime);
  };

  onStart(startTime);

  if (isGameTime) {
    timer = setInterval(() => {
      const elapsed = Date.now() - startTime;
      const remaining = Math.max(0, remainingTimeMs - elapsed);
      applyManagedRemainingTime(remaining);
      if (remaining === 0) clearTimer();
    }, 100);
    onTimerChange(timer);
  }

  try {
    const aiMove = await ai.getAIMove(
      board,
      player,
      level,
      mode === "time" ? timeLimitSeconds * 1000 : undefined,
      isGameTime ? remainingTimeMs : undefined,
      (progress) => {
        if (isSameProgress(lastProgress, progress)) return;
        lastProgress = progress;

        const elapsedMs = Date.now() - startTime;
        const nps = elapsedMs > 0 ? (progress.nodes / elapsedMs) * 1000 : 0;
        onProgress({ progress, nps });
      },
    );

    clearTimer();
    if (aiMove && isGameTime) {
      applyManagedRemainingTime(Math.max(0, remainingTimeMs - aiMove.timeTaken));
    }
    return aiMove;
  } finally {
    clearTimer();
    onFinish();
  }
}
