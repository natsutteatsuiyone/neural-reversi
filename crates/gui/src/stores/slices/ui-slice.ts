import { StateCreator } from "zustand";
import type { AIMoveProgress, GameAnalysisProgress } from "@/services/types";
import type { Services } from "@/services/types";
import { SearchOperation } from "@/services/search-operation";
import type { ReversiState, UISlice, MoveAnalysis } from "./types";
import { getNotation } from "@/domain/game/game-logic";

export function createUISlice(services: Services): StateCreator<
    ReversiState,
    [],
    [],
    UISlice
> {
  return (set, get) => {
    const hintSearch = new SearchOperation();
    const gameAnalysisSearch = new SearchOperation();

    return ({
    showPassNotification: null,
    isAnalyzing: false,
    hintAnalysisAbortPending: false,
    analyzeResults: null,
    isNewGameModalOpen: false,
    newGameModalSession: 0,
    isAboutModalOpen: false,
    isHintMode: false,
    isGameAnalyzing: false,
    gameAnalysisResult: null,

    openNewGameModal: () => {
        get().resetSetup();
        set((state) => ({
            isNewGameModalOpen: true,
            newGameModalSession: state.newGameModalSession + 1,
        }));
    },

    closeNewGameModal: () => set({ isNewGameModalOpen: false }),

    openAboutModal: () => set({ isAboutModalOpen: true }),

    closeAboutModal: () => set({ isAboutModalOpen: false }),

    setHintMode: (enabled) => {
        if (enabled) {
            set({ isHintMode: true, analyzeResults: null });
            const { isAnalyzing, hintAnalysisAbortPending } = get();
            if (!isAnalyzing && !hintAnalysisAbortPending) {
                void get().analyzeBoard();
            }
        } else {
            const state = get();
            const shouldAbortHintAnalysis = state.isAnalyzing && !state.isAIThinking;
            set({ isHintMode: false, analyzeResults: null });
            if (shouldAbortHintAnalysis) {
                get().restartHintAnalysisAfterAbort();
            } else {
                // Invalidate any in-flight analyzeBoard so its finally clause
                // won't clear `isAnalyzing` belonging to a future run.
                hintSearch.invalidate();
            }
        }
    },

    restartHintAnalysisAfterAbort: () => {
        void (async () => {
            await hintSearch.abortLatest({
                commitAbort: () => set({
                    hintAnalysisAbortPending: true,
                }),
                abort: () => services.ai.abortSearch(),
                onError: (error) => {
                    console.error("Hint abort failed:", error);
                },
                onCurrentFinally: () => {
                    const currentState = get();
                    set({
                        isAnalyzing: false,
                        hintAnalysisAbortPending: false,
                    });
                    if (currentState.isHintMode) {
                        void currentState.analyzeBoard();
                    }
                },
            });
        })();
    },

    hidePassNotification: () => set({ showPassNotification: null }),

    analyzeBoard: async () => {
        const { isHintMode, gameStatus, isAIThinking, isAITurn, isAnalyzing, hintAnalysisAbortPending } = get();

        // Analyze only if Hint Mode is ON, game is playing, not AI thinking, not AI's turn, and not already analyzing
        if (
            !isHintMode ||
            gameStatus !== "playing" ||
            isAIThinking ||
            isAITurn() ||
            isAnalyzing ||
            hintAnalysisAbortPending
        ) {
            return;
        }

        const board = get().board;
        const player = get().currentPlayer;
        const results = new Map<string, AIMoveProgress>();

        await hintSearch.startCurrent<[AIMoveProgress]>({
            commitStart: () => set({
                analyzeResults: null,
                isAnalyzing: true,
            }),
            start: (acceptProgress) => services.ai.analyze(board, player, get().hintLevel, acceptProgress),
            onProgress: (progress) => {
                const s = get();
                if (!s.isHintMode || !s.isAnalyzing) return;

                if (progress.row !== undefined && progress.col !== undefined) {
                    const key = `${progress.row},${progress.col}`;
                    const existing = results.get(key);
                    // Drop no-op re-emits so downstream selectors don't see a
                    // fresh Map reference each tick (mirrors solver-slice).
                    if (
                        existing &&
                        existing.score === progress.score &&
                        existing.depth === progress.depth &&
                        existing.targetDepth === progress.targetDepth &&
                        existing.acc === progress.acc &&
                        existing.isEndgame === progress.isEndgame &&
                        existing.pvLine === progress.pvLine
                    ) {
                        return;
                    }
                    results.set(key, progress);
                    set({ analyzeResults: new Map(results) });
                }
            },
            onError: (error) => {
                throw error;
            },
            onCurrentFinally: () => set({ isAnalyzing: false }),
        });
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

        await gameAnalysisSearch.startCurrent<[GameAnalysisProgress]>({
            commitStart: () => set({
                isGameAnalyzing: true,
                gameAnalysisResult: null,
            }),
            start: (acceptProgress) => services.ai.analyzeGame(
                historyStartBoard,
                historyStartPlayer,
                moves,
                level,
                acceptProgress,
            ),
            onProgress: (p) => {
                const state = get();
                if (!state.isGameAnalyzing) return;

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
            },
            onError: (error) => {
                console.error("Game analysis failed:", error);
            },
            onCurrentFinally: () => set({ isGameAnalyzing: false }),
        });
    },

    abortGameAnalysis: async () => {
        await gameAnalysisSearch.abortLatest({
            commitAbort: () => set({
                isGameAnalyzing: false,
            }),
            abort: () => services.ai.abortGameAnalysis(),
            onCurrentFinally: () => undefined,
        });
    },
    });
  };
}
