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
  language: Language | null;
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
  language: null,
};

export interface SettingsService {
  loadSettings(): Promise<AppSettings>;
  saveSetting<K extends keyof AppSettings>(key: K, value: AppSettings[K]): Promise<boolean>;
}

export interface Services {
  ai: AIService;
  settings: SettingsService;
}
