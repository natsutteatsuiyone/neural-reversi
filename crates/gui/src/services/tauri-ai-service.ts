import { invoke } from "@tauri-apps/api/core";
import { type Event, listen } from "@tauri-apps/api/event";
import type { Board, Player } from "@/types";
import { getValidMoves } from "@/lib/game-logic";
import { serializeBoardForAI } from "./board-serialization";
import type { AIService, AIMoveResult, AIMoveProgress, GameAnalysisProgress } from "./types";

async function withEventListener<T, R>(
  event: string,
  callback: (payload: T) => void,
  fn: () => Promise<R>,
): Promise<R> {
  const unlisten = await listen<T>(event, (ev: Event<T>) => callback(ev.payload));
  try {
    return await fn();
  } finally {
    unlisten();
  }
}

export class TauriAIService implements AIService {
  async checkReady(): Promise<void> {
    try {
      await invoke("check_ai_ready_command");
    } catch (error) {
      console.error("AI backend is not ready:", error);
      throw error;
    }
  }

  async getAIMove(
    board: Board,
    player: Player,
    level: number,
    timeLimit: number | undefined,
    remainingTime: number | undefined,
    callback: (progress: AIMoveProgress) => void,
  ): Promise<AIMoveResult> {
    const validMoves = getValidMoves(board, player);
    if (validMoves.length === 0) return null;

    const boardString = serializeBoardForAI(board, player);

    try {
      return await withEventListener<AIMoveProgress, AIMoveResult>(
        "ai-move-progress",
        callback,
        () => invoke<AIMoveResult>("ai_move_command", {
          boardString,
          level,
          timeLimit,
          remainingTime,
        }),
      );
    } catch {
      return null;
    }
  }

  async initialize(): Promise<void> {
    try {
      await invoke("init_ai_command");
    } catch (error) {
      console.error("Failed to initialize search:", error);
      throw error;
    }
  }

  async resizeTT(hashSize: number): Promise<void> {
    try {
      await invoke("resize_tt_command", { hashSize });
    } catch (error) {
      console.error("Failed to resize TT:", error);
    }
  }

  async abortSearch(): Promise<void> {
    try {
      await invoke("abort_ai_search_command");
    } catch (error) {
      console.error("Failed to abort search:", error);
    }
  }

  async analyze(
    board: Board,
    player: Player,
    level: number,
    callback: (progress: AIMoveProgress) => void,
  ): Promise<void> {
    const validMoves = getValidMoves(board, player);
    if (validMoves.length === 0) return;

    const boardString = serializeBoardForAI(board, player);

    await withEventListener<AIMoveProgress, void>(
      "ai-move-progress",
      callback,
      () => invoke("analyze_command", { boardString, level }),
    );
  }

  async analyzeGame(
    board: Board,
    player: Player,
    moves: string[],
    level: number,
    callback: (progress: GameAnalysisProgress) => void,
  ): Promise<void> {
    const boardString = serializeBoardForAI(board, player);

    await withEventListener<GameAnalysisProgress, void>(
      "game-analysis-progress",
      callback,
      () => invoke("analyze_game_command", { boardString, moves, level }),
    );
  }

  async abortGameAnalysis(): Promise<void> {
    try {
      await invoke("abort_game_analysis_command");
    } catch (error) {
      console.error("Failed to abort game analysis:", error);
    }
  }
}
