import type { AIMoveProgress, AIMoveResult } from "@/services/types";
import type { MoveHistory } from "@/lib/move-history";
import type { AIMode, Board, GameMode, Player } from "@/types";
import type { Move } from "@/lib/store-helpers";
import type { Language } from "@/i18n";
import type { AppSettings } from "@/services/types";

export interface GameSlice {
    board: Board;
    historyStartBoard: Board;
    historyStartPlayer: Player;
    moveHistory: MoveHistory;
    currentPlayer: "black" | "white";
    gameOver: boolean;
    gameStatus: "waiting" | "playing" | "finished";
    isPass: boolean;
    lastMove: Move | null;
    validMoves: [number, number][];
    skipAnimation: boolean;
    paused: boolean;
    automationTimer: ReturnType<typeof setTimeout> | null;
    getScores: () => { black: number; white: number };
    isAITurn: () => boolean;
    isValidMove: (row: number, col: number) => boolean;
    makeMove: (move: Move) => Promise<void>;
    makePass: () => void;
    undoMove: () => void;
    redoMove: () => void;
    goToMove: (position: number) => void;
    resumeAI: () => void;
    resetGame: () => Promise<void>;
    startGame: () => Promise<boolean>;
    setGameStatus: (status: "waiting" | "playing" | "finished") => void;
}

export interface AIThinkingEntry extends AIMoveProgress {
    nps: number;
}

export interface AISlice {
    aiLevel: number;
    aiMoveProgress: AIMoveProgress | null;
    aiThinkingHistory: AIThinkingEntry[];
    aiSearchStartTime: number | null;
    isAIThinking: boolean;
    lastAIMove: AIMoveResult | null;
    aiMode: AIMode;
    aiRemainingTime: number;
    searchTimer: ReturnType<typeof setInterval> | null;
    makeAIMove: () => Promise<void>;
    abortAIMove: () => Promise<void>;
    setAILevelChange: (level: number) => void;
    setAIMode: (mode: AIMode) => void;
    clearAiThinkingHistory: () => void;
}

export interface MoveAnalysis {
    moveIndex: number;
    player: "black" | "white";
    playedMove: string;
    playedScore: number;
    bestMove: string;
    bestScore: number;
    scoreLoss: number;
    depth: number;
}

export interface UISlice {
    showPassNotification: "black" | "white" | null;
    isAnalyzing: boolean;
    analyzeResults: Map<string, AIMoveProgress> | null;
    isNewGameModalOpen: boolean;
    isHintMode: boolean;
    isGameAnalyzing: boolean;
    gameAnalysisResult: MoveAnalysis[] | null;
    hidePassNotification: () => void;
    analyzeBoard: () => Promise<void>;
    setNewGameModalOpen: (open: boolean) => void;
    setHintMode: (enabled: boolean) => void;
    analyzeGame: () => Promise<void>;
    abortGameAnalysis: () => Promise<void>;
}

export interface SettingsSlice {
    gameMode: GameMode;
    timeLimit: number;
    gameTimeLimit: number;
    hintLevel: number;
    gameAnalysisLevel: number;
    hashSize: number;
    aiAnalysisPanelOpen: boolean;
    language: Language | null;
    hydrateSettings: (settings: AppSettings) => void;
    setGameMode: (mode: GameMode) => void;
    setTimeLimit: (limit: number) => void;
    setGameTimeLimit: (limit: number) => void;
    setHintLevel: (level: number) => void;
    setGameAnalysisLevel: (level: number) => void;
    setHashSize: (size: number) => void;
    setAIAnalysisPanelOpen: (open: boolean) => void;
    setLanguagePreference: (language: Language | null) => Promise<boolean>;
}

export type SetupTab = "manual" | "transcript" | "boardString";

export interface SetupSlice {
    setupBoard: Board;
    setupCurrentPlayer: Player;
    setupTab: SetupTab;
    transcriptInput: string;
    boardStringInput: string;
    setupError: string | null;
    resetSetup: () => void;
    setSetupTab: (tab: SetupTab) => void;
    setSetupCurrentPlayer: (player: Player) => void;
    setSetupBoard: (board: Board) => void;
    setSetupCellColor: (row: number, col: number) => void;
    setTranscriptInput: (input: string) => void;
    setBoardStringInput: (input: string) => void;
    clearSetupBoard: () => void;
    resetSetupToInitial: () => void;
    startFromSetup: () => Promise<boolean>;
}

export type ReversiState = GameSlice & AISlice & UISlice & SettingsSlice & SetupSlice;
