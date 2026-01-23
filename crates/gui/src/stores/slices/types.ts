import type { AIMoveProgress, AIMoveResult } from "@/lib/ai";
import type { AIMode, Board, GameMode, MoveRecord } from "@/types";
import type { Move } from "@/lib/store-helpers";

export interface GameSlice {
    board: Board;
    moves: MoveRecord[];
    allMoves: MoveRecord[];
    currentPlayer: "black" | "white";
    gameOver: boolean;
    gameStatus: "waiting" | "playing" | "finished";
    isPass: boolean;
    lastMove: Move | null;
    validMoves: [number, number][];
    getScores: () => { black: number; white: number };
    isAITurn: () => boolean;
    isValidMove: (row: number, col: number) => boolean;
    makeMove: (move: Move) => Promise<void>;
    makePass: () => void;
    undoMove: () => void;
    redoMove: () => void;
    resetGame: () => Promise<void>;
    startGame: () => Promise<void>;
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

export interface UISlice {
    showPassNotification: "black" | "white" | null;
    isAnalyzing: boolean;
    analyzeResults: Map<string, AIMoveProgress> | null;
    isNewGameModalOpen: boolean;
    isHintMode: boolean;
    hidePassNotification: () => void;
    analyzeBoard: () => Promise<void>;
    setNewGameModalOpen: (open: boolean) => void;
    setHintMode: (enabled: boolean) => void;
}

export interface SettingsSlice {
    gameMode: GameMode;
    timeLimit: number;
    gameTimeLimit: number;
    hintLevel: number;
    aiAnalysisPanelOpen: boolean;
    setGameMode: (mode: GameMode) => void;
    setTimeLimit: (limit: number) => void;
    setGameTimeLimit: (limit: number) => void;
    setHintLevel: (level: number) => void;
    setAIAnalysisPanelOpen: (open: boolean) => void;
}

export type ReversiState = GameSlice & AISlice & UISlice & SettingsSlice;
