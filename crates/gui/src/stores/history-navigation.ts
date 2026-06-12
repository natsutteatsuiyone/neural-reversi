import type { ReversiState } from "./slices/types";
import { isGameSearchActive } from "@/stores/engine-activity";
import {
  createGoToMovePatch,
  createHistoryNavigationPatch,
  type GameHistoryPatch,
  type HistoryNavigationDirection,
  type HistoryNavigationState,
} from "@/domain/game/game-session";

type Get = () => ReversiState;
type Set = (
  partial: Partial<ReversiState> | ((state: ReversiState) => Partial<ReversiState>),
) => void;

/**
 * History navigation orchestration: the pure position math lives in
 * `domain/game/game-session`; this is the single place that adapts the store
 * state into its input, applies the view clears, and runs the
 * post-navigation finalize (resume-pause + hint re-analyse). Keeping it here
 * means undo/redo/goToMove are one-line delegations and the navigation rules
 * have one home.
 */
function toNavigationState(state: ReversiState): HistoryNavigationState {
  return {
    gameStatus: state.gameStatus,
    moveHistory: state.moveHistory,
    historyStartBoard: state.historyStartBoard,
    historyStartPlayer: state.historyStartPlayer,
    gameTimeLimitMs: state.gameTimeLimit * 1000,
  };
}

function withClears(patch: GameHistoryPatch): Partial<ReversiState> {
  return { ...patch, analyzeResults: null, showPassNotification: null };
}

/**
 * The single entry rule every history step runs first: navigation is blocked
 * while an in-game Engine Search is active (CONTEXT.md → Engine Activity) —
 * undo/redo/goToMove would change the position out from under the running
 * search — and otherwise the pending auto-play step is cancelled. Concentrating
 * it here (instead of three slice methods each hand-coding a guard, with
 * `goToMove` using the canonical predicate and undo/redo using the narrower
 * `isGameAnalyzing`) keeps the slice methods one-line delegations and the
 * navigation rules one home (ADR-0003). Returns whether navigation may proceed.
 */
function beginNavigation(get: Get): boolean {
  if (isGameSearchActive(get().engineActivity)) return false;
  get().cancelAutomation();
  return true;
}

function finalize(get: Get, set: Set): void {
  const state = get();
  if (state.gameStatus !== "playing") return;

  const canResumeAI = state.isAITurn() && state.validMoves.length > 0;
  set({ paused: canResumeAI });
  if (state.isHintMode && state.validMoves.length > 0) {
    void state.analyzeBoard();
  }
}

export function navigateHistory(get: Get, set: Set, direction: HistoryNavigationDirection): void {
  if (!beginNavigation(get)) return;
  const patch = createHistoryNavigationPatch(toNavigationState(get()), direction);
  if (patch) set(withClears(patch));
  // finalize runs even when navigation was a no-op, matching the prior
  // undo/redo behaviour.
  finalize(get, set);
}

export function goToHistoryMove(get: Get, set: Set, position: number): void {
  if (!beginNavigation(get)) return;
  const patch = createGoToMovePatch(toNavigationState(get()), position);
  if (!patch) return;

  set(withClears(patch));
  if (!patch.gameOver && patch.gameStatus === "playing") {
    finalize(get, set);
  }
}
