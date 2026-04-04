import { StateCreator } from "zustand";
import {
    analyze,
    analyzeGame as analyzeGameApi,
    abortGameAnalysis as abortGameAnalysisApi,
    type AIMoveProgress,
} from "@/lib/ai";
import type { ReversiState, UISlice, MoveAnalysis } from "./types";
import { triggerAutomation } from "./game-slice";
import { getNotation } from "@/lib/game-logic";

export const createUISlice: StateCreator<
    ReversiState,
    [],
    [],
    UISlice
> = (set, get) => ({
    showPassNotification: null,
    isAnalyzing: false,
    analyzeResults: null,
    isNewGameModalOpen: false,
    isHintMode: false,
    isGameAnalyzing: false,
    gameAnalysisResult: null,

    setNewGameModalOpen: (open) => set({ isNewGameModalOpen: open }),

    setHintMode: (enabled) => {
        set({ isHintMode: enabled, analyzeResults: null });
        if (enabled) {
            get().analyzeBoard();
        } else {
            get().abortAIMove();
        }
    },

    hidePassNotification: () => {
        set({ showPassNotification: null });
        const { makePass } = get();
        makePass();
        triggerAutomation(get);
    },

    analyzeBoard: async () => {
        const { isHintMode, gameStatus, isAIThinking, isAITurn, isAnalyzing } = get();

        // Analyze only if Hint Mode is ON, game is playing, not AI thinking, not AI's turn, and not already analyzing
        if (!isHintMode || gameStatus !== "playing" || isAIThinking || isAITurn() || isAnalyzing) {
            return;
        }

        const board = get().board;
        const player = get().currentPlayer;
        const results = new Map<string, AIMoveProgress>();

        set({ analyzeResults: null, isAnalyzing: true });

        try {
            await analyze(board, player, get().hintLevel, (ev) => {
                // Check if hint mode was disabled or analysis was cancelled
                if (!get().isHintMode || !get().isAnalyzing) return;

                if (ev.payload.row !== undefined && ev.payload.col !== undefined) {
                    const key = `${ev.payload.row},${ev.payload.col}`;
                    results.set(key, ev.payload);
                    set({ analyzeResults: new Map(results) });
                }
            });
        } finally {
            set({ isAnalyzing: false });
        }
    },

    analyzeGame: async () => {
        const { isGameAnalyzing, isAIThinking, moveHistory, historyStartBoard, historyStartPlayer } = get();
        if (isGameAnalyzing || isAIThinking) return;

        const allMoves = moveHistory.allMoves;
        if (allMoves.length === 0) return;

        const moves: string[] = allMoves.map((m) =>
            m.row < 0 ? "--" : getNotation(m.row, m.col)
        );

        const level = get().gameAnalysisLevel;
        const analysisResults: MoveAnalysis[] = [];
        set({
            isGameAnalyzing: true,
            gameAnalysisResult: null,
        });

        try {
            await analyzeGameApi(
                historyStartBoard,
                historyStartPlayer,
                moves,
                level,
                (ev) => {
                    if (!get().isGameAnalyzing) return;
                    const p = ev.payload;

                    const move = allMoves[p.moveIndex];
                    analysisResults.push({
                        moveIndex: p.moveIndex,
                        player: move.player,
                        playedMove: move.notation,
                        playedScore: p.playedScore,
                        bestMove: p.bestMove,
                        bestScore: p.bestScore,
                        scoreLoss: p.scoreLoss,
                        depth: p.depth,
                    });

                    set({ gameAnalysisResult: [...analysisResults] });
                }
            );
        } catch (error) {
            console.error("Game analysis failed:", error);
        } finally {
            set({ isGameAnalyzing: false });
        }
    },

    abortGameAnalysis: async () => {
        set({ isGameAnalyzing: false });
        await abortGameAnalysisApi();
    },
});
