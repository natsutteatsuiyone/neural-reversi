import { StateCreator } from "zustand";
import type { GameAnalysisSession } from "@/domain/game/game-analysis-session";
import type { HintAnalysisSession } from "@/domain/game/hint-analysis-session";
import type { ReversiState, UISlice } from "./types";

export function createUISlice(
  hintSession: HintAnalysisSession,
  gameAnalysisSession: GameAnalysisSession,
): StateCreator<ReversiState, [], [], UISlice> {
  return (set, get) => {
    return {
      showPassNotification: null,
      isAnalyzing: false,
      hintAnalysisAbortPending: false,
      analyzeResults: null,
      isNewGameModalOpen: false,
      isAboutModalOpen: false,
      isHintMode: false,
      isGameAnalyzing: false,
      gameAnalysisResult: null,

      openNewGameModal: () => {
        get().resetSetup();
        set({ isNewGameModalOpen: true });
      },

      closeNewGameModal: () => set({ isNewGameModalOpen: false }),

      openAboutModal: () => set({ isAboutModalOpen: true }),

      closeAboutModal: () => set({ isAboutModalOpen: false }),

      // Hint Analysis / Game Analysis lifecycles live in their sessions
      // (CONTEXT.md → Engine Activity); these are thin delegates, mirroring
      // how the Solver slice delegates to SolverSession.
      setHintMode: (enabled) => hintSession.setMode(enabled),

      restartHintAnalysisAfterAbort: () => hintSession.restartAfterAbort(),

      hidePassNotification: () => set({ showPassNotification: null }),

      analyzeBoard: () => hintSession.analyze(),

      analyzeGame: () => gameAnalysisSession.analyze(),

      abortGameAnalysis: () => gameAnalysisSession.abort(),
    };
  };
}
