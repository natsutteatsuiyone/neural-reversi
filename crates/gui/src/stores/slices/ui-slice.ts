import { StateCreator } from "zustand";
import type { AIMoveProgress } from "@/services/types";
import type { Services } from "@/services/types";
import type { ReversiState, UISlice, MoveAnalysis } from "./types";
import { getNotation } from "@/lib/game-logic";

export function createUISlice(services: Services): StateCreator<
    ReversiState,
    [],
    [],
    UISlice
> {
  return (set, get) => ({
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
            void get().analyzeBoard();
        } else {
            void get().abortAIMove();
        }
    },

    hidePassNotification: () => set({ showPassNotification: null }),

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
            await services.ai.analyze(board, player, get().hintLevel, (progress) => {
                const s = get();
                if (!s.isHintMode || !s.isAnalyzing) return;

                if (progress.row !== undefined && progress.col !== undefined) {
                    const key = `${progress.row},${progress.col}`;
                    results.set(key, progress);
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
            await services.ai.analyzeGame(
                historyStartBoard,
                historyStartPlayer,
                moves,
                level,
                (p) => {
                    if (!get().isGameAnalyzing) return;

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
        await services.ai.abortGameAnalysis();
    },
  });
}
