import { type AIMoveProgress, type AIMoveResult, abortAISearch, analyze, getAIMove, initializeAI } from "@/lib/ai";
import {
  calculateScores,
  getFlippedDiscs,
  getNotation,
  getValidMoves,
  initializeBoard,
  opponentPlayer as nextPlayer,
} from "@/lib/game-logic";
import type { Board, GameMode, MoveRecord } from "@/types";
import { create } from "zustand";

interface Move {
  row: number;
  col: number;
  isAI: boolean;
  score?: number;
}

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
  resetGame: () => void;
  startGame: () => void;
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
      const newBoard = state.board.map((row) =>
        row.map((cell) => ({ ...cell }))
      );
      const currentPlayer = get().currentPlayer;
      const flipped = getFlippedDiscs(
        state.board,
        move.row,
        move.col,
        currentPlayer
      );

      newBoard[move.row][move.col] = {
        color: currentPlayer,
        isNew: true,
      };

      for (const [r, c] of flipped) {
        newBoard[r][c] = { color: currentPlayer };
      }

      return {
        board: newBoard,
        moves: [
          ...state.moves,
          {
            id: state.moves.length,
            player: currentPlayer,
            row: move.row,
            col: move.col,
            notation: getNotation(move.row, move.col),
            score: move.score,
            isAI: move.isAI,
          },
        ],
        currentPlayer: nextPlayer(state.currentPlayer),
        isPass: false,
        lastMove: move,
        validMoves: getValidMoves(newBoard, nextPlayer(currentPlayer)),
        analyzeResults: null,
      };
    });

    const { validMoves } = get();
    if (validMoves.length === 0) {
      const otherPlayer = nextPlayer(get().currentPlayer);
      const otherMoves = getValidMoves(get().board, otherPlayer);
      if (otherMoves.length === 0) {
        set(() => {
          return {
            gameOver: true,
            gameStatus: "finished",
          };
        });
      } else {
        set({ showPassNotification : true });
      }
    } else {
      if (get().isAITurn()) {
        void get().makeAIMove();
      } else if (get().gameMode === "analyze") {
        void get().analyzeBoard();
      }
    }
  },

  makePass: () => {
    set((state) => {
      const currentPlayer = state.currentPlayer;
      const passMove = {
        id: state.moves.length,
        player: currentPlayer,
        row: -1,
        col: -1,
        notation: "Pass",
      };

      const validMoves = getValidMoves(state.board, nextPlayer(currentPlayer));

      return {
        board: state.board.map((row) => row.map((cell) => ({ ...cell }))),
        moves: [...state.moves, passMove],
        currentPlayer: nextPlayer(currentPlayer),
        validMoves,
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

      const newMoves = [...state.moves];

      if ((state.gameMode === "ai-black" || state.gameMode === "ai-white") && newMoves.length >= 1) {
        const lastMove = newMoves[newMoves.length - 1];

        if (lastMove.isAI) {
          newMoves.pop();

          if (newMoves.length > 0 && !newMoves[newMoves.length - 1].isAI) {
            newMoves.pop();
          }
        } else {
          newMoves.pop();
        }
      }
      else {
        newMoves.pop();
      }

      const newBoard = initializeBoard();

      for (const move of newMoves) {
        if (move.row >= 0 && move.col >= 0) {
          const flipped = getFlippedDiscs(
            newBoard,
            move.row,
            move.col,
            move.player
          );

          newBoard[move.row][move.col] = {
            color: move.player,
          };

          for (const [r, c] of flipped) {
            newBoard[r][c] = { color: move.player };
          }
        }
      }

      let currentPlayer: "black" | "white" = "black";
      if (newMoves.length > 0) {
        currentPlayer = nextPlayer(newMoves[newMoves.length - 1].player);
      }

      const validMoves = getValidMoves(newBoard, currentPlayer);

      return {
        board: newBoard,
        moves: newMoves,
        currentPlayer,
        lastMove: newMoves.length > 0 ? {
          row: newMoves[newMoves.length - 1].row,
          col: newMoves[newMoves.length - 1].col,
          isAI: !!newMoves[newMoves.length - 1].isAI, // boolean型に確実に変換
          score: newMoves[newMoves.length - 1].score || undefined
        } : null,
        validMoves,
        isPass: false,
        analyzeResults: null,
      };
    });

    // Analyzeモードの場合は盤面を分析
    if (get().gameMode === "analyze" && get().gameStatus === "playing") {
      void get().analyzeBoard();
    }
  },
}));
