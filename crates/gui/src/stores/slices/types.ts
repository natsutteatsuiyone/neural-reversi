import type {
    AIMoveProgress,
    AIMoveResult,
    SolverCandidate,
    SolverMode,
    SolverProgressPayload,
    SolverSelectivity,
} from "@/services/types";
import type { MoveHistory } from "@/domain/game/move-history";
import type { SolverHistoryEntry as DomainSolverHistoryEntry } from "@/domain/solver/solver-session";
import type { AIMode, Board, GameMode, Player } from "@/domain/game/types";
import type { Move } from "@/domain/game/store-helpers";
import type { MoveAnalysis } from "@/domain/game/game-analysis";
import type { Language } from "@/i18n";
import type { AppSettings } from "@/services/types";

export type NewGameSettings = Pick<AppSettings, "gameMode" | "aiLevel" | "aiMode" | "gameTimeLimit">;

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
    automationResumePending: boolean;
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
    startGame: (settings?: NewGameSettings) => Promise<boolean>;
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
    checkAIReady: () => Promise<boolean>;
    makeAIMove: () => Promise<void>;
    abortAIMove: () => Promise<void>;
    setAILevelChange: (level: number) => void;
    setAIMode: (mode: AIMode) => void;
    clearAiThinkingHistory: () => void;
}

export type { MoveAnalysis };

export interface UISlice {
    showPassNotification: "black" | "white" | null;
    isAnalyzing: boolean;
    hintAnalysisAbortPending: boolean;
    analyzeResults: Map<string, AIMoveProgress> | null;
    isNewGameModalOpen: boolean;
    newGameModalSession: number;
    isAboutModalOpen: boolean;
    isHintMode: boolean;
    isGameAnalyzing: boolean;
    gameAnalysisResult: MoveAnalysis[] | null;
    hidePassNotification: () => void;
    analyzeBoard: () => Promise<void>;
    openNewGameModal: () => void;
    closeNewGameModal: () => void;
    openAboutModal: () => void;
    closeAboutModal: () => void;
    setHintMode: (enabled: boolean) => void;
    analyzeGame: () => Promise<void>;
    abortGameAnalysis: () => Promise<void>;
    /**
     * Aborts the current hint analysis and restarts `analyzeBoard` after the
     * abort resolves if hint mode is still active. Intended for slice-internal
     * coordination, not for user-facing commands.
     */
    restartHintAnalysisAfterAbort: () => void;
}

export interface SettingsSlice {
    gameMode: GameMode;
    timeLimit: number;
    gameTimeLimit: number;
    hintLevel: number;
    gameAnalysisLevel: number;
    hashSize: number;
    aiAnalysisPanelOpen: boolean;
    rightPanelSize: number;
    bottomPanelSize: number;
    language: Language | null;
    hydrateSettings: (settings: AppSettings) => void;
    setGameMode: (mode: GameMode) => void;
    setTimeLimit: (limit: number) => void;
    setGameTimeLimit: (limit: number) => void;
    setHintLevel: (level: number) => void;
    setGameAnalysisLevel: (level: number) => void;
    setHashSize: (size: number) => void;
    setAIAnalysisPanelOpen: (open: boolean) => void;
    setRightPanelSize: (size: number) => void;
    setBottomPanelSize: (size: number) => void;
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
    startFromSetup: (settings?: NewGameSettings) => Promise<boolean>;
}

export type SolverHistoryEntry = DomainSolverHistoryEntry;

export interface SolverConfig {
    selectivity: SolverSelectivity;
    mode: SolverMode;
}

export interface SolverSlice {
    isSolverActive: boolean;
    isSolverModalOpen: boolean;
    solverRootBoard: Board | null;
    solverRootPlayer: Player | null;
    solverHistory: SolverHistoryEntry[];
    solverCurrentBoard: Board | null;
    solverCurrentPlayer: Player | null;
    targetSelectivity: SolverSelectivity;
    solverMode: SolverMode;
    solverCandidates: Map<string, SolverCandidate>;
    isSolverSearching: boolean;
    /**
     * True only while the current search has been explicitly paused by the
     * user via `stopSolverSearch`. Cleared when any new search launches
     * (navigation, selectivity change, resume, solver exit). Lets the UI
     * offer Resume after a manual stop without offering it after a search
     * completes naturally.
     */
    isSolverStopped: boolean;
    openSolverModal: () => void;
    closeSolverModal: () => void;
    subscribeSolverProgress: () => Promise<() => void>;
    startSolver: (board: Board, player: Player, config?: SolverConfig) => Promise<boolean>;
    startSolverFromSetup: (config?: SolverConfig) => Promise<boolean>;
    exitSolver: () => Promise<void>;
    advanceSolver: (row: number, col: number) => Promise<void>;
    undoSolver: () => Promise<void>;
    setTargetSelectivity: (sel: SolverSelectivity) => Promise<void>;
    setSolverMode: (mode: SolverMode) => Promise<void>;
    stopSolverSearch: () => Promise<void>;
    resumeSolverSearch: () => Promise<void>;
    applySolverProgress: (payload: SolverProgressPayload) => void;
}

export type ReversiState = GameSlice & AISlice & UISlice & SettingsSlice & SetupSlice & SolverSlice;
