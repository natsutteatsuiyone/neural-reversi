import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import type { Board, Player } from "@/domain/game/types";
import { serializeBoardForAI } from "./board-serialization";
import { TAURI_COMMAND, TAURI_EVENT } from "./tauri-contract";
import type { SolverMode, SolverProgressPayload, SolverService, SolverSelectivity } from "./types";
import { SOLVER_SELECTIVITIES } from "./types";

export class TauriSolverService implements SolverService {
  async startSearch(
    board: Board,
    player: Player,
    targetSelectivity: SolverSelectivity,
    mode: SolverMode,
    runId: number,
  ): Promise<void> {
    const boardString = serializeBoardForAI(board, player);
    try {
      await invoke(TAURI_COMMAND.solverSearch, {
        boardString,
        targetSelectivity: SOLVER_SELECTIVITIES.indexOf(targetSelectivity),
        multiPv: mode === "multiPv",
        runId,
      });
    } catch (error) {
      console.error("Failed to start solver search:", error);
      throw error;
    }
  }

  async abort(): Promise<void> {
    try {
      await invoke(TAURI_COMMAND.abortAiSearch);
    } catch (error) {
      console.error("Failed to abort solver search:", error);
    }
  }

  async onProgress(callback: (payload: SolverProgressPayload) => void): Promise<UnlistenFn> {
    return listen<SolverProgressPayload>(TAURI_EVENT.solverProgress, (event) =>
      callback(event.payload),
    );
  }
}
