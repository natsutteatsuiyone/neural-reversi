import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import type { Board, Player } from "@/types";
import { serializeBoardForAI } from "./board-serialization";
import type { SolverProgressPayload, SolverService, SolverSelectivity } from "./types";
import { SOLVER_SELECTIVITY_TO_U8 } from "./types";

export class TauriSolverService implements SolverService {
  async startSearch(
    board: Board,
    player: Player,
    targetSelectivity: SolverSelectivity,
    runId: number,
  ): Promise<void> {
    const boardString = serializeBoardForAI(board, player);
    const targetSelectivityU8 = SOLVER_SELECTIVITY_TO_U8[targetSelectivity];
    try {
      await invoke("solver_search_command", {
        boardString,
        targetSelectivity: targetSelectivityU8,
        runId,
      });
    } catch (error) {
      console.error("Failed to start solver search:", error);
      throw error;
    }
  }

  async abort(): Promise<void> {
    try {
      await invoke("abort_ai_search_command");
    } catch (error) {
      console.error("Failed to abort solver search:", error);
    }
  }

  async onProgress(callback: (payload: SolverProgressPayload) => void): Promise<UnlistenFn> {
    return listen<SolverProgressPayload>("solver-progress", (event) => callback(event.payload));
  }
}
