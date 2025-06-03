import { type AIMoveProgress, type AIMoveResult, abortAISearch, analyze, getAIMove, initializeAI } from "@/lib/ai";
import {
  calculateScores,
  getValidMoves,
  initializeBoard,
  opponentPlayer as nextPlayer,
} from "@/lib/game-logic";
import {
  type Move,
  applyMove,
  checkGameOver,
  createMoveRecord,
  createPassMove,
  getUndoMoves,
  reconstructBoardFromMoves,
} from "@/lib/store-helpers";
import type { Board, GameMode, MoveRecord } from "@/types";
import { create } from "zustand";

interface ReversiState {
  board: Board;
  moves: MoveRecord[];
  currentPlayer: "black" | "white";
  gameOver: boolean;
  gameMode: GameMode;
  gameStatus: "waiting" | "playing" | "finished";
  isPass: boolean;
  lastAIMove: AIMoveResult | null;
  lastMove: Move | null;
  validMoves: [number, number][];
  aiLevel: number;
  aiAccuracy: number;
  aiMoveProgress: AIMoveProgress | null;
  showPassNotification: boolean;
  isAIThinking: boolean;
  isAnalyzing: boolean;
  analyzeResults: Map<string, AIMoveProgress> | null;

  getScores: () => { black: number; white: number };
  isAITurn: () => boolean;
  isValidMove: (row: number, col: number) => boolean;
  makeAIMove: () => Promise<void>;
  abortAIMove: () => Promise<void>;
  makeMove: (move: Move) => void;
  makePass: () => void;
  undoMove: () => void;
  resetGame: () => Promise<void>;
  startGame: () => Promise<void>;
  setAILevelChange: (level: number) => void;
  setAIAccuracyChange: (accuracy: number) => void;
  setGameMode: (mode: GameMode) => void;
  setGameStatus: (status: "waiting" | "playing" | "finished") => void;
  hidePassNotification: () => void;
  analyzeBoard: () => Promise<void>;
}

export const useReversiStore = create<ReversiState>((set, get) => ({
  board: initializeBoard(),
  moves: [],
  currentPlayer: "black",
  gameOver: false,
  gameMode: "ai-white",
  gameStatus: "waiting",
  isPass: false,
  lastAIMove: null,
  lastMove: null,
  validMoves: [],
  aiLevel: 10,
  aiAccuracy: 1,
  aiMoveProgress: null,
  showPassNotification: false,
  isAIThinking: false,
  isAnalyzing: false,
  analyzeResults: null,

  getScores: () => {
    return calculateScores(get().board);
  },

  analyzeBoard: async () => {
    if (get().gameMode !== "analyze" || get().gameStatus !== "playing") {
      return;
    }

    await abortAISearch();

    const board = get().board;
    const player = get().currentPlayer;
    const results = new Map<string, AIMoveProgress>();

    set({ analyzeResults: null, isAnalyzing: true });

    try {
      await analyze(board, player, get().aiLevel, get().aiAccuracy, (ev) => {
        if (ev.payload.row !== undefined && ev.payload.col !== undefined) {
          const key = `${ev.payload.row},${ev.payload.col}`;
          results.set(key, ev.payload);
          set({ analyzeResults: new Map(results) });
        }
      });
    } finally {
      set({ isAnalyzing: false });
    }
  },

  isAITurn: () => {
    const { gameMode, gameOver, currentPlayer } = get();
    return (
      (!gameOver && gameMode === "ai-black" && currentPlayer === "black") ||
      (gameMode === "ai-white" && currentPlayer === "white")
    );
  },

  isValidMove: (row, col) => {
    const { validMoves, gameStatus } = get();
    if (gameStatus !== "playing") {
      return false;
    }
    return validMoves.some((move) => move[0] === row && move[1] === col);
  },

  makeAIMove: async () => {
    set({ isAIThinking: true });
    const player = get().currentPlayer;
    const board = get().board;
    const aiMove = await getAIMove(board, player, get().aiLevel, get().aiAccuracy, (ev) => {
      set({ aiMoveProgress: ev.payload });
    });
    set({ aiMoveProgress: null, isAIThinking: false });
    if (aiMove) {
      const move = {
        row: aiMove.row,
        col: aiMove.col,
        score: aiMove.score,
        isAI: true,
      };
      get().makeMove(move);
      set({
        lastAIMove: aiMove,
      });
    }
  },

  abortAIMove: async () => {
    if (get().isAIThinking || get().isAnalyzing) {
      await abortAISearch();
      set({ isAIThinking: false, isAnalyzing: false, aiMoveProgress: null });
    }
  },

  makeMove: (move: Move) => {
    set((state) => {
      const currentPlayer = state.currentPlayer;
      const newBoard = applyMove(state.board, move, currentPlayer);
      const newMoveRecord = createMoveRecord(state.moves.length, currentPlayer, move);
      const nextPlayerTurn = nextPlayer(currentPlayer);

      return {
        board: newBoard,
        moves: [...state.moves, newMoveRecord],
        currentPlayer: nextPlayerTurn,
        isPass: false,
        lastMove: move,
        validMoves: getValidMoves(newBoard, nextPlayerTurn),
        analyzeResults: null,
      };
    });

    const state = get();
    const { gameOver, shouldPass } = checkGameOver(state.board, state.currentPlayer);

    if (gameOver) {
      set({ gameOver: true, gameStatus: "finished" });
    } else if (shouldPass) {
      set({ showPassNotification: true });
    } else if (state.isAITurn()) {
      void state.makeAIMove();
    } else if (state.gameMode === "analyze") {
      void state.analyzeBoard();
    }
  },

  makePass: () => {
    set((state) => {
      const currentPlayer = state.currentPlayer;
      const passMove = createPassMove(state.moves.length, currentPlayer);
      const nextPlayerTurn = nextPlayer(currentPlayer);

      return {
        board: state.board.map((row) => row.map((cell) => ({ ...cell }))),
        moves: [...state.moves, passMove],
        currentPlayer: nextPlayerTurn,
        validMoves: getValidMoves(state.board, nextPlayerTurn),
        isPass: true,
      };
    });
  },

  resetGame: async () => {
    if (get().isAIThinking || get().isAnalyzing) {
      await get().abortAIMove();
    }

    set({
      board: initializeBoard(),
      moves: [],
      currentPlayer: "black",
      gameOver: false,
      gameStatus: "waiting",
      isPass: false,
      lastMove: null,
      lastAIMove: null,
      showPassNotification: false,
      isAIThinking: false,
      isAnalyzing: false,
      analyzeResults: null,
    });
  },

  startGame: async () => {
    await initializeAI();

    set((state) => {
      const validMoves = getValidMoves(state.board, state.currentPlayer);
      return {
        board: initializeBoard(),
        gameStatus: "playing",
        validMoves,
      };
    });

    if (get().gameMode === "analyze") {
      void get().analyzeBoard();
    } else if (get().gameMode === "ai-black") {
      void get().makeAIMove();
    }
  },

  setAILevelChange: (level) => set({ aiLevel: level }),

  setAIAccuracyChange: (accuracy) => set({ aiAccuracy: accuracy }),

  setGameMode: (mode) => {
    set({
      gameMode: mode,
      analyzeResults: null
    });

    if (mode === "analyze" && get().gameStatus === "playing") {
      void get().analyzeBoard();
    }
  },

  setGameStatus: (status) => set({ gameStatus: status }),

  hidePassNotification: () => {
    set({ showPassNotification: false });
    get().makePass();
    if (get().isAITurn()) {
      void get().makeAIMove();
    } else if (get().gameMode === "analyze") {
      void get().analyzeBoard();
    }
  },

  undoMove: () => {
    set((state) => {
      if (state.gameStatus !== "playing" || state.moves.length === 0) {
        return state;
      }

      const newMoves = getUndoMoves(state.moves, state.gameMode);
      const { board, currentPlayer } = reconstructBoardFromMoves(newMoves);
      const validMoves = getValidMoves(board, currentPlayer);

      const lastMove = newMoves.length > 0
        ? {
            row: newMoves[newMoves.length - 1].row,
            col: newMoves[newMoves.length - 1].col,
            isAI: !!newMoves[newMoves.length - 1].isAI,
            score: newMoves[newMoves.length - 1].score,
          }
        : null;

      return {
        board,
        moves: newMoves,
        currentPlayer,
        lastMove,
        validMoves,
        isPass: false,
        analyzeResults: null,
      };
    });

    if (get().gameMode === "analyze" && get().gameStatus === "playing") {
      void get().analyzeBoard();
    }
  },
}));
