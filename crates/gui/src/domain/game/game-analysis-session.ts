import type { AIService, GameAnalysisProgress } from "@/services/types";
import type { EngineSearch } from "@/domain/engine/engine-search";
import type { Board, Player } from "@/domain/game/types";
import type { MoveHistory } from "@/domain/game/move-history";
import {
  appendGameAnalysisProgress,
  createGameAnalysisMoveList,
  type MoveAnalysis,
} from "@/domain/game/game-analysis";

/**
 * Everything the Game Analysis feature reads. `isGameAnalyzing` and
 * `isAIThinking` are projections of the Engine Activity (CONTEXT.md →
 * Engine Activity); `resumeQueuedAutomation` is the store-bound automation
 * hook a finished/superseded run releases (CONTEXT.md → Automation).
 */
export interface GameAnalysisSessionState {
  isGameAnalyzing: boolean;
  isAIThinking: boolean;
  moveHistory: MoveHistory;
  historyStartBoard: Board;
  historyStartPlayer: Player;
  gameAnalysisLevel: number;
  resumeQueuedAutomation: () => void;
}

export type GameAnalysisSessionPatch = Partial<{
  gameAnalysisResult: MoveAnalysis[] | null;
}>;

export type GameAnalysisSessionCommit = (partial: GameAnalysisSessionPatch) => void;

interface GameAnalysisSessionOptions {
  ai: AIService;
  read: () => GameAnalysisSessionState;
  commit: GameAnalysisSessionCommit;
  engineSearch: EngineSearch;
}

/**
 * Owns the Game Analysis lifecycle: starting the whole-game pass, aborting
 * it, accumulating per-move results, and releasing queued automation on
 * teardown. The UI slice is a thin delegate, mirroring how
 * `HintAnalysisSession` / `SolverSession` back their slices and giving all
 * four Engine Searches the same shape.
 *
 * Game analysis stays a "JS-only search" filtered by the current-run
 * wrapper (`engineSearch.start`), with **no** round-tripped run id — see
 * `docs/adr/0001`. This class does not change that; it only relocates the
 * choreography that was inline in the store.
 */
export class GameAnalysisSession {
  private readonly ai: AIService;
  private readonly read: () => GameAnalysisSessionState;
  private readonly commit: GameAnalysisSessionCommit;
  private readonly engineSearch: EngineSearch;

  constructor({ ai, read, commit, engineSearch }: GameAnalysisSessionOptions) {
    this.ai = ai;
    this.read = read;
    this.commit = commit;
    this.engineSearch = engineSearch;
  }

  async analyze(): Promise<void> {
    const { isGameAnalyzing, isAIThinking, moveHistory, historyStartBoard, historyStartPlayer } =
      this.read();
    if (isGameAnalyzing || isAIThinking) return;

    const allMoves = moveHistory.allMoves;
    if (allMoves.length === 0) return;

    const moves = createGameAnalysisMoveList(allMoves);
    const level = this.read().gameAnalysisLevel;
    let analysisResults: MoveAnalysis[] = [];

    await this.engineSearch.start<GameAnalysisProgress, void>({
      kind: "game-analysis",
      // `isGameAnalyzing` is the Engine Activity, stamped SYNCHRONOUSLY
      // at claim by its owner (before the possibly slow supersede). The
      // move/navigation guards key off it, so the board is locked for
      // the queued window and the backend cannot analyze a history the
      // user mutated meanwhile; it returns to idle on teardown exactly
      // once (incl. superseded). Only `gameAnalysisResult` is feature
      // payload cleared here.
      onClaim: () => this.commit({ gameAnalysisResult: null }),
      run: (accept) =>
        this.ai.analyzeGame(historyStartBoard, historyStartPlayer, moves, level, accept),
      abort: () => this.ai.abortGameAnalysis(),
      onProgress: (p) => {
        if (!this.read().isGameAnalyzing) return;

        analysisResults = appendGameAnalysisProgress(analysisResults, allMoves, p);
        this.commit({ gameAnalysisResult: analysisResults });
      },
      onError: (error) => console.error("Game analysis failed:", error),
      onTeardown: () => {
        this.read().resumeQueuedAutomation();
      },
    });
  }

  async abort(): Promise<void> {
    await this.engineSearch.abort({
      // `isGameAnalyzing` returns to idle via the Engine Activity owner
      // (abort stamps idle synchronously at claim).
      abort: () => this.ai.abortGameAnalysis(),
      onSettled: () => this.read().resumeQueuedAutomation(),
    });
  }
}
