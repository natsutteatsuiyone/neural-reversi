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
  cloneBoard,
  createMoveRecord,
  createPassMove,
  getUndoMoves,
  getRedoMoves,
  reconstructBoardFromMoves,
} from "@/lib/store-helpers";
import type { Board, GameMode, MoveRecord } from "@/types";
import { create } from "zustand";

interface ReversiState {
  board: Board;
  moves: MoveRecord[];
  allMoves: MoveRecord[]; // Store all moves for redo functionality
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
  redoMove: () => void;
  resetGame: () => Promise<void>;
  startGame: () => Promise<void>;
  setAILevelChange: (level: number) => void;
  setAIAccuracyChange: (accuracy: number) => void;
  setGameMode: (mode: GameMode) => void;
  setGameStatus: (status: "waiting" | "playing" | "finished") => void;
  hidePassNotification: () => void;
  analyzeBoard: () => Promise<void>;
}

function toLastMove(moves: MoveRecord[]): Move | null {
  const last = moves.length > 0 ? moves[moves.length - 1] : undefined;
  if (!last || last.row < 0 || last.col < 0) {
    return null;
  }

  return {
    row: last.row,
    col: last.col,
    isAI: Boolean(last.isAI),
    score: last.score,
  };
}

function deriveStateFromMoves(moves: MoveRecord[]): {
  board: Board;
  currentPlayer: "black" | "white";
  validMoves: [number, number][];
  lastMove: Move | null;
} {
  const { board, currentPlayer } = reconstructBoardFromMoves(moves);

  return {
    board,
    currentPlayer,
    validMoves: getValidMoves(board, currentPlayer),
    lastMove: toLastMove(moves),
  };
}

function triggerAutomation(state: ReversiState): void {
  if (state.gameStatus !== "playing") {
    return;
  }

  if (state.isAITurn()) {
    void state.makeAIMove();
    return;
  }

  if (state.gameMode === "analyze") {
    void state.analyzeBoard();
  }
}
export const useReversiStore = create<ReversiState>((set, get) => ({
  board: initializeBoard(),
  moves: [],
  allMoves: [],
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
        allMoves: [...state.moves, newMoveRecord], // Update allMoves when making a new move
        currentPlayer: nextPlayerTurn,
        isPass: false,
        lastMove: move,
        validMoves: getValidMoves(newBoard, nextPlayerTurn),
        analyzeResults: null,
      };
    });

    const updatedState = get();
    const { gameOver, shouldPass } = checkGameOver(updatedState.board, updatedState.currentPlayer);

    if (gameOver) {
      set({ gameOver: true, gameStatus: "finished" });
      return;
    }

    if (shouldPass) {
      set({ showPassNotification: true });
      return;
    }

    triggerAutomation(updatedState);
  },

  makePass: () => {
    set((state) => {
      const currentPlayer = state.currentPlayer;
      const passMove = createPassMove(state.moves.length, currentPlayer);
      const nextPlayerTurn = nextPlayer(currentPlayer);
      const boardClone = cloneBoard(state.board);

      return {
        board: boardClone,
        moves: [...state.moves, passMove],
        allMoves: [...state.moves, passMove], // Update allMoves when passing
        currentPlayer: nextPlayerTurn,
        validMoves: getValidMoves(boardClone, nextPlayerTurn),
        isPass: true,
        analyzeResults: null,
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
      allMoves: [],
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
      validMoves: [],
      aiMoveProgress: null,
    });
  },

  startGame: async () => {
    await initializeAI();

    set(() => {
      const board = initializeBoard();
      const currentPlayer = "black";
      return {
        board,
        moves: [],
        allMoves: [],
        currentPlayer,
        gameStatus: "playing",
        gameOver: false,
        isPass: false,
        lastMove: null,
        lastAIMove: null,
        validMoves: getValidMoves(board, currentPlayer),
        showPassNotification: false,
        analyzeResults: null,
      };
    });

    triggerAutomation(get());
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
    const { makePass } = get();
    makePass();
    triggerAutomation(get());
  },

  undoMove: () => {
    set((state) => {
      if (state.gameStatus !== "playing" || state.moves.length === 0) {
        return state;
      }

      const newMoves = getUndoMoves(state.moves, state.gameMode);
      const derived = deriveStateFromMoves(newMoves);

      return {
        ...derived,
        moves: newMoves,
        isPass: false,
        analyzeResults: null,
        gameOver: false,
      };
    });

    const state = get();
    if (state.gameMode === "analyze" && state.gameStatus === "playing") {
      void state.analyzeBoard();
    }
  },

  redoMove: () => {
    set((state) => {
      if (state.gameStatus !== "playing" || state.moves.length >= state.allMoves.length) {
        return state;
      }

      const newMoves = getRedoMoves(state.moves, state.allMoves, state.gameMode);
      const derived = deriveStateFromMoves(newMoves);
      const { gameOver } = checkGameOver(derived.board, derived.currentPlayer);

      return {
        ...derived,
        moves: newMoves,
        isPass: false,
        analyzeResults: null,
        gameOver,
      };
    });

    const state = get();
    if (state.gameMode === "analyze" && state.gameStatus === "playing") {
      void state.analyzeBoard();
    }
  },
}));
