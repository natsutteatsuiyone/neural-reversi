export type Player = "black" | "white";

type Cell = {
  color: Player | null;
  isNew?: boolean;
};

export type Board = Cell[][];

export type MoveRecord = {
  id: number;
  player: Player;
  row: number;
  col: number;
  notation: string;
  score?: number;
  isAI?: boolean;
  remainingTime?: number;
};

export type GameMode = "ai-black" | "ai-white" | "pvp";
export type GameStatus = "waiting" | "playing" | "finished";
export type AIMode = "level" | "game-time";

/**
 * What the engine reports about a candidate move while searching a position.
 * A domain concept (score/depth/accuracy/PV of a move) — the transport layer
 * carries it but does not own it. `services/types` re-exports this so existing
 * `@/services/types` importers are unaffected.
 */
export type AIMoveProgress = {
  bestMove: string;
  row: number;
  col: number;
  score: number;
  depth: number;
  targetDepth: number;
  acc: number;
  nodes: number;
  pvLine: string;
  isEndgame: boolean;
};

/** What the engine reports about one already-played move during game analysis. */
export type GameAnalysisProgress = {
  moveIndex: number;
  pvLine: string;
  bestMove: string;
  bestScore: number;
  playedScore: number;
  scoreLoss: number;
  depth: number;
};

export const ANALYSIS_LEVELS = [1, 4, 8, 12, 16, 20, 24, 28, 30] as const;
