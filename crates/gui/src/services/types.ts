import type { UnlistenFn } from "@tauri-apps/api/event";
import type { AIMode, Board, GameMode, Player } from "@/types";
import type { Language } from "@/i18n";

export type AIMoveResult = {
  row: number;
  col: number;
  score: number;
  depth: number;
  acc: number;
  timeTaken: number;
} | null;

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

export type GameAnalysisProgress = {
  moveIndex: number;
  bestMove: string;
  bestScore: number;
  playedScore: number;
  scoreLoss: number;
  depth: number;
};

export interface AIService {
  checkReady(): Promise<void>;

  getAIMove(
    board: Board,
    player: Player,
    level: number,
    timeLimit: number | undefined,
    remainingTime: number | undefined,
    callback: (progress: AIMoveProgress) => void,
  ): Promise<AIMoveResult>;

  analyze(
    board: Board,
    player: Player,
    level: number,
    callback: (progress: AIMoveProgress) => void,
  ): Promise<void>;

  analyzeGame(
    board: Board,
    player: Player,
    moves: string[],
    level: number,
    callback: (progress: GameAnalysisProgress) => void,
  ): Promise<void>;

  initialize(): Promise<void>;
  resizeTT(hashSize: number): Promise<void>;
  abortSearch(): Promise<void>;
  abortGameAnalysis(): Promise<void>;
}

export interface AppSettings {
  gameMode: GameMode;
  aiLevel: number;
  aiMode: AIMode;
  timeLimit: number;
  gameTimeLimit: number;
  hintLevel: number;
  gameAnalysisLevel: number;
  hashSize: number;
  aiAnalysisPanelOpen: boolean;
  rightPanelSize: number;
  bottomPanelSize: number;
  language: Language | null;
  solverTargetSelectivity: SolverSelectivity;
  solverMode: SolverMode;
}

export const DEFAULT_SETTINGS: AppSettings = {
  gameMode: "ai-white",
  aiLevel: 21,
  aiMode: "game-time",
  timeLimit: 1,
  gameTimeLimit: 60,
  hintLevel: 21,
  gameAnalysisLevel: 20,
  hashSize: 512,
  aiAnalysisPanelOpen: false,
  rightPanelSize: 25,
  bottomPanelSize: 30,
  language: null,
  solverTargetSelectivity: 100,
  solverMode: "multiPv",
};

export interface SettingsService {
  loadSettings(): Promise<AppSettings>;
  saveSetting<K extends keyof AppSettings>(key: K, value: AppSettings[K]): Promise<boolean>;
}

export const SOLVER_SELECTIVITIES = [73, 95, 99, 100] as const;
export type SolverSelectivity = typeof SOLVER_SELECTIVITIES[number];

/**
 * Maps UI selectivity percentages to the backend `u8` expected by
 * `solver_search_command` (matches `reversi_core::probcut::Selectivity` discriminants).
 */
export const SOLVER_SELECTIVITY_TO_U8: Record<SolverSelectivity, 0 | 2 | 4 | 5> = {
  73: 0,   // Level1
  95: 2,   // Level3
  99: 4,   // Level5
  100: 5,  // None
};

export const SOLVER_MODES = ["bestOnly", "multiPv"] as const;
export type SolverMode = typeof SOLVER_MODES[number];

/**
 * Progress payload for solver searches. Mirrors {@link AIMoveProgress} but
 * carries `runId` so the store can drop events emitted by a superseded run
 * (e.g. one that was aborted before its queued events drained on the JS side).
 */
export type SolverProgressPayload = AIMoveProgress & {
  runId: number;
};

/**
 * A single candidate move in solver results, keyed by "row,col" in the store.
 */
export interface SolverCandidate {
  move: string;
  row: number;
  col: number;
  score: number;
  depth: number;
  targetDepth: number;
  acc: number;
  /** Space-separated move notation list, as received from the backend. */
  pvLine: string;
  /** True while this iteration deepens by selectivity rather than by depth. */
  isEndgame: boolean;
  isComplete: boolean;
}

export interface SolverService {
  /**
   * Kicks off a solver search on the given position. Returns once the command
   * has been dispatched; results arrive asynchronously via the progress
   * listener.
   */
  startSearch(
    board: Board,
    player: Player,
    targetSelectivity: SolverSelectivity,
    mode: SolverMode,
    runId: number,
  ): Promise<void>;
  /** Aborts any in-flight search (shared with the main AI search mutex). */
  abort(): Promise<void>;
  /**
   * Subscribes to `solver-progress` events. Returns an unsubscribe function
   * that the caller MUST invoke when the listener is no longer needed.
   */
  onProgress(callback: (payload: SolverProgressPayload) => void): Promise<UnlistenFn>;
}

export interface Services {
  ai: AIService;
  settings: SettingsService;
  solver: SolverService;
}
