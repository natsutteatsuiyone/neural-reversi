import { StateCreator } from "zustand";
import type { AIMoveProgress, GameAnalysisProgress } from "@/services/types";
import type { Services } from "@/services/types";
import type { EngineSearch } from "@/domain/engine/engine-search";
import type { ReversiState, UISlice } from "./types";
import {
    appendGameAnalysisProgress,
    applyHintAnalysisProgress,
    createGameAnalysisMoveList,
    type MoveAnalysis,
} from "@/domain/game/game-analysis";
import { resumeQueuedAutomation } from "./game-slice";

export function createUISlice(services: Services, engineSearch: EngineSearch): StateCreator<
    ReversiState,
    [],
    [],
    UISlice
> {
  return (set, get) => {
    // Bumped by every restartHintAnalysisAfterAbort. A hint abort's
    // guaranteed-once teardown clears `hintAnalysisAbortPending` only if no
    // newer hint abort has claimed since — so a stale, superseded abort
    // cannot drop the guard that the newer pending abort still owns
    // (mirrors the solver's solverClaimGeneration).
    let hintAbortGeneration = 0;
    // Bumped synchronously by every analyzeBoard call (in call order). A hint
    // run's guaranteed-once onTeardown clears `isAnalyzing` only if no newer
    // analyzeBoard has claimed since — so a stale run superseded before it
    // installed (queued behind a slow game-analysis abort while a newer
    // hint run already installed and owns `isAnalyzing`) cannot clobber the
    // newer run's flag (mirrors the solver's solverClaimGeneration; S15).
    let hintAnalysisGeneration = 0;
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
            }
            // No engine action otherwise: an in-flight hint analysis only occurs in the
            // shouldAbortHintAnalysis branch, and EngineSearch's generation/exactly-once
            // teardown already prevents a stale analyzeBoard from clobbering a future
            // run's isAnalyzing. Do NOT call engineSearch.abort() here — it would
            // supersede and abort a possibly in-flight AI move.
        }
    },

    restartHintAnalysisAfterAbort: () => {
        // Set the abort-pending guard SYNCHRONOUSLY (before engineSearch.abort),
        // exactly as the old hintSearch.abortLatest commitAbort did via the
        // synchronous invalidate(). EngineSearch's onAbort would run only after
        // `await supersede()`, i.e. asynchronously — that would let a same-tick
        // setHintLevel slip past the `hintAnalysisAbortPending` dedup guard in
        // settings-slice and issue a redundant backend abort.
        set({ hintAnalysisAbortPending: true });
        const generation = ++hintAbortGeneration;
        void engineSearch.abort({
            abort: () => services.ai.abortSearch(),
            onError: (error) => console.error("Hint abort failed:", error),
            onSettled: () => {
                const currentState = get();
                set({ isAnalyzing: false, hintAnalysisAbortPending: false });
                if (currentState.isHintMode) void currentState.analyzeBoard();
            },
            // Guaranteed-once: a superseding start skips onSettled, so clear
            // the synchronous abort-pending breadcrumb here too — but only if
            // no newer hint abort has claimed since, or this stale abort would
            // drop the guard the newer pending abort still owns.
            onTeardown: () => {
                if (generation === hintAbortGeneration) {
                    set({ hintAnalysisAbortPending: false });
                }
            },
        });
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
        let results = new Map<string, AIMoveProgress>();
        const generation = ++hintAnalysisGeneration;

        await engineSearch.start<AIMoveProgress, void>({
            // onStart/run run AFTER the (possibly slow) supersede, so Hint Mode
            // may have been turned off while this start was queued. Recheck it:
            // setHintMode(false) cannot cancel a not-yet-started hint search
            // (isAnalyzing is still false then), so the search itself must bail.
            // onStart and run are invoked in the same synchronous turn (no user
            // input can interleave between the two isHintMode reads).
            onStart: () => { if (get().isHintMode) set({ analyzeResults: null, isAnalyzing: true }); },
            run: (accept) =>
                get().isHintMode ? services.ai.analyze(board, player, get().hintLevel, accept) : Promise.resolve(),
            abort: () => services.ai.abortSearch(),
            onProgress: (progress) => {
                const s = get();
                if (!s.isHintMode || !s.isAnalyzing) return;

                const nextResults = applyHintAnalysisProgress(results, progress);
                if (!nextResults) return;

                results = nextResults;
                set({ analyzeResults: results });
            },
            onError: (error) => console.error("Hint analysis failed:", error),
            // Guard against clobbering a newer hint run: a run superseded
            // before it installed (queued behind a slow game-analysis abort)
            // never set `isAnalyzing` itself, and a newer analyzeBoard may
            // already own it. Only the latest generation may clear it. (S15.)
            onTeardown: () => { if (generation === hintAnalysisGeneration) set({ isAnalyzing: false }); },
        });
    },

    analyzeGame: async () => {
        const { isGameAnalyzing, isAIThinking, moveHistory, historyStartBoard, historyStartPlayer } = get();
        if (isGameAnalyzing || isAIThinking) return;

        const allMoves = moveHistory.allMoves;
        if (allMoves.length === 0) return;

        const moves = createGameAnalysisMoveList(allMoves);
        const level = get().gameAnalysisLevel;
        let analysisResults: MoveAnalysis[] = [];

        await engineSearch.start<GameAnalysisProgress, void>({
            // Commit the pending flag SYNCHRONOUSLY (before the possibly slow
            // supersede), not in onStart. moves/historyStart* were captured
            // above; the move/navigation guards key off isGameAnalyzing, so
            // committing it here locks the board for the queued window and the
            // backend cannot analyze a history the user mutated meanwhile.
            // onTeardown clears it exactly once (incl. superseded).
            onClaim: () => set({ isGameAnalyzing: true, gameAnalysisResult: null }),
            run: (accept) =>
                services.ai.analyzeGame(historyStartBoard, historyStartPlayer, moves, level, accept),
            abort: () => services.ai.abortGameAnalysis(),
            onProgress: (p) => {
                const state = get();
                if (!state.isGameAnalyzing) return;

                analysisResults = appendGameAnalysisProgress(analysisResults, allMoves, p);
                set({ gameAnalysisResult: analysisResults });
            },
            onError: (error) => console.error("Game analysis failed:", error),
            onTeardown: () => { set({ isGameAnalyzing: false }); resumeQueuedAutomation(get, set); },
        });
    },

    abortGameAnalysis: async () => {
        await engineSearch.abort({
            onAbort: () => set({ isGameAnalyzing: false }),
            abort: () => services.ai.abortGameAnalysis(),
            onSettled: () => resumeQueuedAutomation(get, set),
        });
    },
    });
  };
}
