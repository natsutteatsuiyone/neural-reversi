import type { ReversiState } from "./slices/types";
import { isGameSearchActive } from "./engine-activity";
import { FLIP_DURATION_MS, PASS_NOTIFICATION_DURATION_MS } from "@/lib/timing";

/**
 * Automation (CONTEXT.md → Automation): decides and schedules auto-play —
 * the AI moves itself on its turn, Hint Mode auto-analyses on a human turn —
 * including the post-move animation / pass-notification delay before the next
 * auto-step.
 *
 * The pending timer and the "deferred while game analysis runs" flag are
 * private to this closure; they are never part of the store's public state.
 * Callers only say *what happened*, never *how* it is scheduled.
 */
export interface Automation {
  /** Re-evaluate now: play the AI move, or run hint analysis, if it is time. */
  trigger(): void;
  /** Drop any pending step (timer + deferred flag). */
  cancel(): void;
  /** A move was just applied: schedule the next auto-step. The caller reports
   *  only *what happened* — Automation owns *how long* to wait (CONTEXT.md →
   *  Automation): after a pass notice, after the flip animation a human is
   *  watching, or immediately. */
  afterMove(opts: { passed: boolean; flipped: boolean }): void;
  /** Resume a step deferred behind a game-analysis run, once it has ended. */
  resumeIfQueued(): void;
  /** Mark a step as deferred so the next `resumeIfQueued` runs it. */
  queueResume(): void;
}

export function createAutomation(read: () => ReversiState): Automation {
  let timer: ReturnType<typeof setTimeout> | null = null;
  let resumePending = false;

  function cancel(): void {
    if (timer) {
      clearTimeout(timer);
      timer = null;
    }
    resumePending = false;
  }

  function trigger(): void {
    const state = read();
    if (state.gameStatus !== "playing") return;
    if (state.paused) return;
    if (isGameSearchActive(state.engineActivity)) return;

    if (state.isAITurn()) {
      void state.makeAIMove();
      return;
    }
    if (state.isHintMode) {
      void state.analyzeBoard();
    }
  }

  function scheduleAfter(delayMs: number): void {
    cancel();
    timer = setTimeout(() => {
      timer = null;
      // A game analysis started during the delay: defer this step and let
      // `resumeIfQueued` run it once analysis ends (CONTEXT.md → Automation).
      if (read().isGameAnalyzing) {
        resumePending = true;
        return;
      }
      resumePending = false;
      trigger();
    }, delayMs);
  }

  function afterMove({ passed, flipped }: { passed: boolean; flipped: boolean }): void {
    if (passed) {
      scheduleAfter(PASS_NOTIFICATION_DURATION_MS);
      return;
    }
    if (flipped) {
      scheduleAfter(FLIP_DURATION_MS);
      return;
    }
    trigger();
  }

  function resumeIfQueued(): void {
    if (!resumePending || read().isGameAnalyzing) return;
    resumePending = false;
    trigger();
  }

  function queueResume(): void {
    resumePending = true;
  }

  return { trigger, cancel, afterMove, resumeIfQueued, queueResume };
}
