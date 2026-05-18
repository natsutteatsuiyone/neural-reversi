import type { AIService } from "@/services/types";
import type { EngineSearch } from "@/domain/engine/engine-search";
import type { AIMoveProgress, Board, Player } from "@/domain/game/types";
import {
  applyHintAnalysisProgress,
  type HintAnalysisResults,
} from "@/domain/game/game-analysis";

/**
 * Everything the Hint Analysis feature reads. `analyzeBoard` and
 * `restartHintAnalysisAfterAbort` are the store-bound actions: the session
 * re-enters analysis through them (never a private method) so the existing
 * store-action call graph — and the tests that spy on it — is preserved.
 */
export interface HintAnalysisSessionState {
  isHintMode: boolean;
  /** Projection of the Engine Activity (CONTEXT.md → Engine Activity). */
  isAnalyzing: boolean;
  isAIThinking: boolean;
  hintAnalysisAbortPending: boolean;
  gameStatus: "waiting" | "playing" | "finished";
  board: Board;
  currentPlayer: Player;
  hintLevel: number;
  analyzeResults: HintAnalysisResults | null;
  isAITurn: () => boolean;
  analyzeBoard: () => Promise<void>;
  restartHintAnalysisAfterAbort: () => void;
}

export type HintAnalysisSessionPatch = Partial<{
  isHintMode: boolean;
  isAnalyzing: boolean;
  hintAnalysisAbortPending: boolean;
  analyzeResults: HintAnalysisResults | null;
}>;

export type HintAnalysisSessionCommit = (
  partial:
    | HintAnalysisSessionPatch
    | ((state: HintAnalysisSessionState) => HintAnalysisSessionPatch),
) => void;

interface HintAnalysisSessionOptions {
  ai: AIService;
  read: () => HintAnalysisSessionState;
  commit: HintAnalysisSessionCommit;
  engineSearch: EngineSearch;
}

/**
 * Owns the Hint Analysis lifecycle: the Hint Mode toggle, the position
 * analysis run, the abort-then-restart protocol, and the abort-pending
 * dedupe guard's generation counter. The store slices are thin delegates
 * (mirroring how `SolverSession` backs the Solver slice).
 *
 * `isAnalyzing` is a view of the Engine Activity (CONTEXT.md → Engine
 * Activity), so a hint run no longer needs a generation counter for it.
 * `hintAnalysisAbortPending` is a separate feature breadcrumb (read by the
 * settings level-change path to dedupe backend aborts) that the activity
 * does NOT own, so its own generation guard lives here: a hint abort's
 * guaranteed-once teardown clears `hintAnalysisAbortPending` only if no
 * newer hint abort has claimed since, or a stale superseded abort would
 * drop the guard the newer pending abort still owns.
 */
export class HintAnalysisSession {
  private readonly ai: AIService;
  private readonly read: () => HintAnalysisSessionState;
  private readonly commit: HintAnalysisSessionCommit;
  private readonly engineSearch: EngineSearch;
  private abortGeneration = 0;

  constructor({ ai, read, commit, engineSearch }: HintAnalysisSessionOptions) {
    this.ai = ai;
    this.read = read;
    this.commit = commit;
    this.engineSearch = engineSearch;
  }

  /**
   * A hint analysis is in flight and stale, so it must be
   * aborted-then-restarted (not just left to supersede). `isAnalyzing` is
   * the Engine Activity projection for the `hint` kind (CONTEXT.md → Engine
   * Activity); `!isAIThinking` excludes an AI move that is using the shared
   * engine — restarting then would abort the AI move. The single home for
   * "is there a stale hint run to abort?", shared by the Hint Mode toggle
   * and the level-change path.
   */
  private staleHintInFlight(state: HintAnalysisSessionState): boolean {
    return state.isAnalyzing && !state.isAIThinking;
  }

  setMode(enabled: boolean): void {
    if (enabled) {
      this.commit({ isHintMode: true, analyzeResults: null });
      const { isAnalyzing, hintAnalysisAbortPending } = this.read();
      if (!isAnalyzing && !hintAnalysisAbortPending) {
        void this.read().analyzeBoard();
      }
    } else {
      const state = this.read();
      const shouldAbortHintAnalysis = this.staleHintInFlight(state);
      this.commit({ isHintMode: false, analyzeResults: null });
      if (shouldAbortHintAnalysis) {
        this.read().restartHintAnalysisAfterAbort();
      }
      // No engine action otherwise: an in-flight hint analysis only occurs in the
      // shouldAbortHintAnalysis branch, and EngineSearch's generation/exactly-once
      // teardown already prevents a stale analyzeBoard from clobbering a future
      // run's isAnalyzing. Do NOT call engineSearch.abort() here — it would
      // supersede and abort a possibly in-flight AI move.
    }
  }

  restartAfterAbort(): void {
    // Set the abort-pending guard SYNCHRONOUSLY (before engineSearch.abort),
    // exactly as the old hintSearch.abortLatest commitAbort did via the
    // synchronous invalidate(). EngineSearch's onAbort would run only after
    // `await supersede()`, i.e. asynchronously — that would let a same-tick
    // setHintLevel slip past the `hintAnalysisAbortPending` dedup guard in
    // the settings level-change path and issue a redundant backend abort.
    this.commit({ hintAnalysisAbortPending: true });
    const generation = ++this.abortGeneration;
    void this.engineSearch.abort({
      abort: () => this.ai.abortSearch(),
      onError: (error) => console.error("Hint abort failed:", error),
      onSettled: () => {
        const currentState = this.read();
        this.commit({ isAnalyzing: false, hintAnalysisAbortPending: false });
        if (currentState.isHintMode) void currentState.analyzeBoard();
      },
      // Guaranteed-once: a superseding start skips onSettled, so clear
      // the synchronous abort-pending breadcrumb here too — but only if
      // no newer hint abort has claimed since, or this stale abort would
      // drop the guard the newer pending abort still owns.
      onTeardown: () => {
        if (generation === this.abortGeneration) {
          this.commit({ hintAnalysisAbortPending: false });
        }
      },
    });
  }

  async analyze(): Promise<void> {
    const { isHintMode, gameStatus, isAIThinking, isAITurn, isAnalyzing, hintAnalysisAbortPending } =
      this.read();

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

    const board = this.read().board;
    const player = this.read().currentPlayer;
    let results: HintAnalysisResults = new Map<string, AIMoveProgress>();

    await this.engineSearch.start<AIMoveProgress, void>({
      // `isAnalyzing` is stamped at claim and cleared on teardown by the
      // Engine Activity owner — no generation counter here.
      kind: "hint",
      // onStart/run run AFTER the (possibly slow) supersede, so Hint Mode
      // may have been turned off while this start was queued. Recheck it:
      // the search itself must bail (feature-specific re-validation that
      // the activity does not own). onStart and run are invoked in the
      // same synchronous turn (no user input can interleave).
      onStart: () => {
        if (this.read().isHintMode) this.commit({ analyzeResults: null });
      },
      run: (accept) =>
        this.read().isHintMode
          ? this.ai.analyze(board, player, this.read().hintLevel, accept)
          : Promise.resolve(),
      abort: () => this.ai.abortSearch(),
      onProgress: (progress) => {
        const s = this.read();
        if (!s.isHintMode || !s.isAnalyzing) return;

        const nextResults = applyHintAnalysisProgress(results, progress);
        if (!nextResults) return;

        results = nextResults;
        this.commit({ analyzeResults: results });
      },
      onError: (error) => console.error("Hint analysis failed:", error),
    });
  }

  /**
   * A user changed the hint level. The in-flight (or next) analysis must
   * re-target the new level. Dedupe against an already-pending abort so a
   * rapid level change cannot issue a redundant backend abort. Re-enters
   * through the store-bound actions, exactly as the prior settings-slice
   * coordination did.
   */
  onLevelChanged(): void {
    const state = this.read();
    if (!state.isHintMode || state.hintAnalysisAbortPending) return;

    if (this.staleHintInFlight(state)) {
      state.restartHintAnalysisAfterAbort();
    } else {
      void state.analyzeBoard();
    }
  }
}
