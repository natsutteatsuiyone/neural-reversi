import { invoke } from "@tauri-apps/api/core";
import { type Event, listen } from "@tauri-apps/api/event";
import type { Board, Player } from "@/domain/game/types";
import { getValidMoves } from "@/domain/game/game-logic";
import { serializeBoardForAI } from "./board-serialization";
import { TAURI_COMMAND, TAURI_EVENT } from "./tauri-contract";
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
      await invoke(TAURI_COMMAND.checkAiReady);
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

    return await withEventListener<AIMoveProgress, AIMoveResult>(
      TAURI_EVENT.aiMoveProgress,
      callback,
      () =>
        invoke<AIMoveResult>(TAURI_COMMAND.aiMove, {
          boardString,
          level,
          timeLimit,
          remainingTime,
        }),
    );
  }

  async initialize(): Promise<void> {
    try {
      await invoke(TAURI_COMMAND.initAi);
    } catch (error) {
      console.error("Failed to initialize search:", error);
      throw error;
    }
  }

  async resizeTT(hashSize: number): Promise<void> {
    try {
      await invoke(TAURI_COMMAND.resizeTt, { hashSize });
    } catch (error) {
      console.error("Failed to resize TT:", error);
    }
  }

  async abortSearch(): Promise<void> {
    try {
      await invoke(TAURI_COMMAND.abortAiSearch);
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

    await withEventListener<AIMoveProgress, void>(TAURI_EVENT.aiMoveProgress, callback, () =>
      invoke(TAURI_COMMAND.analyze, { boardString, level }),
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
      TAURI_EVENT.gameAnalysisProgress,
      callback,
      () => invoke(TAURI_COMMAND.analyzeGame, { boardString, moves, level }),
    );
  }

  async abortGameAnalysis(): Promise<void> {
    try {
      await invoke(TAURI_COMMAND.abortGameAnalysis);
    } catch (error) {
      console.error("Failed to abort game analysis:", error);
    }
  }
}
