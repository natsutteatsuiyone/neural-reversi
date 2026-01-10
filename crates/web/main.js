import { createApp, reactive } from "https://unpkg.com/petite-vue?module";

const FILES = "abcdefgh";
const LEVEL_MIN = 1;
const LEVEL_MAX = 15;
const DEFAULT_LEVEL = 1;
const boardIndices = Array.from({ length: 64 }, (_, idx) => idx);
const PASS_TOAST_DURATION_MS = 1400;
const TOAST_FADE_DURATION_MS = 250;
let toastSequenceId = 0;
let forcedPassBlockId = 0;

// --- Web Worker ---
const worker = new Worker(new URL("./reversi-worker.js", import.meta.url), {
  type: "module",
});

const locales = {
  ja: {
    loading: "読み込み中…",
    colorLabel: "あなたの石",
    colorOptions: {
      black: "黒 (先手)",
      white: "白 (後手)",
    },
    levelLabel: "レベル",
    buttons: {
      newGame: "新しい対局",
      startGame: "ゲーム開始",
      undo: "一手戻す",
    },
    moves: {},
    moveLog: {
      aiEvalLabel: "AI評価値",
      noEval: "--",
    },
    externalLinks: {
      desktop: {
        label: "デスクトップ版",
        aria: "デスクトップ版 (Releases)",
      },
    },
    modal: {
      title: "ゲーム設定",
      description: "対局を始める前に設定を選んでください。",
    },
    scoreboard: {
      black: "黒",
      white: "白",
      empty: "残りマス",
    },
    boardAria: "リバーシ盤面",
    cellAria: (notation) => `マス ${notation}`,
    colors: {
      black: "黒",
      white: "白",
    },
    searchStatus: {
      idle: "探索状況: 待機中",
      preparing: "探索状況: 探索準備中…",
      movePlaceholder: "—",
      format: ({ depth, scoreText, nodeText, moveText, selectivity }) =>
        `探索状況: 深さ${depth} / 評価${scoreText} / ノード${nodeText} / 手 ${moveText} / 選択度${selectivity}`,
    },
    gameOver: {
      winTitle: "あなたの勝ち！",
      loseTitle: "AIの勝ち",
      drawTitle: "引き分け",
    },
    messages: {
      gameOver: (black, white, result) => {
        let text = `ゲーム終了：黒 ${black} - 白 ${white}。`;
        if (result === "human") {
          text += "あなたの勝ちです！";
        } else if (result === "ai") {
          text += "AIの勝ちです。";
        } else {
          text += "引き分けでした。";
        }
        return text;
      },
      passHuman: "パスしました。AIの番です。",
      passAi: "AIはパスしました。あなたの番です。",
      aiThinking: "AIが考え中です…",
      humanTurn: (colorName) =>
        `あなたの番です（${colorName}）。`,
      aiTurn: (colorName) => `AIの番です（${colorName}）。`,
      confirmNewGame: "現在の対局を終了して新しい対局を始めますか？",
    },
  },
  en: {
    loading: "Loading…",
    colorLabel: "Your color",
    colorOptions: {
      black: "Black\u00a0(First)",
      white: "White\u00a0(Second)",
    },
    levelLabel: "Level",
    buttons: {
      newGame: "New Game",
      startGame: "Start Game",
      undo: "Undo",
    },
    moves: {},
    moveLog: {
      aiEvalLabel: "AI evaluation",
      noEval: "--",
    },
    externalLinks: {
      desktop: {
        label: "Desktop App",
        aria: "Desktop App (Releases)",
      },
    },
    modal: {
      title: "Game Settings",
      description: "Choose your preferences before starting the match.",
    },
    scoreboard: {
      black: "Black",
      white: "White",
      empty: "Remaining",
    },
    boardAria: "Reversi board",
    cellAria: (notation) => `Square ${notation}`,
    colors: {
      black: "Black",
      white: "White",
    },
    searchStatus: {
      idle: "Search: Idle",
      preparing: "Search: Preparing…",
      movePlaceholder: "—",
      format: ({ depth, scoreText, nodeText, moveText, selectivity }) =>
        `Search: Depth ${depth} / Score ${scoreText} / Nodes ${nodeText} / Move ${moveText} / Selectivity ${selectivity}`,
    },
    gameOver: {
      winTitle: "You Win!",
      loseTitle: "AI Wins",
      drawTitle: "Draw",
    },
    messages: {
      gameOver: (black, white, result) => {
        let text = `Game over: Black ${black} - White ${white}.`;
        if (result === "human") {
          text += " You win!";
        } else if (result === "ai") {
          text += " AI wins.";
        } else {
          text += " It's a draw.";
        }
        return text;
      },
      passHuman: "You passed. AI's turn.",
      passAi: "AI passed. Your turn.",
      aiThinking: "AI is thinking…",
      humanTurn: (colorName) =>
        `Your turn (${colorName}).`,
      aiTurn: (colorName) => `AI's turn (${colorName}).`,
      confirmNewGame: "End current game and start a new one?",
    },
  },
};

const defaultLocale = detectPreferredLocale();

const state = reactive({
  board: Array(64).fill(0),
  legalMoves: [],
  humanIsBlack: true,
  humanColor: 1,
  aiColor: 2,
  lastHumanMove: null,
  lastAiMove: null,
  aiThinking: false,
  passNotice: null,
  initialLoading: true,
  level: DEFAULT_LEVEL,
  selectedLevel: String(DEFAULT_LEVEL),
  selectedColor: "black",
  pendingLevel: String(DEFAULT_LEVEL),
  pendingColor: "black",
  locale: defaultLocale,
  scoreBlack: 2,
  scoreWhite: 2,
  emptyCount: 60,
  message: locales[defaultLocale].loading,
  messageKind: "info",
  searchProgress: null,
  searchStatusText: locales[defaultLocale].searchStatus.idle,
  isHumanTurn: false,
  showSettingsModal: true,
  // New state properties
  isGameOver: false,
  currentPlayer: 1,
  gameResult: null,
  gameOverModalDismissed: false,
  // Move history (Single Source of Truth for both display and undo)
  moveHistory: [],
  toastMessage: null,
  toastVisible: false,
  delayAiUntilToast: false,
});

const view = {
  state,
  boardIndices,
  get localeTexts() {
    return currentLocale();
  },
  get moveLog() {
    return state.moveHistory.map((move, index) => ({
      id: `${index + 1}-${move.player}-${toNotation(move.index)}`,
      color: move.player,
      notation: toNotation(move.index),
      turn: index + 1,
      isAiMove: move.player === state.aiColor,
      evaluation: move.evaluation,
    }));
  },
  toNotation,
  cellClasses,
  isCellDisabled,
  handleCellClick,
  handleNewGame,
  handleModalColorChange,
  handleModalLevelChange,
  handleStartGame,
  handleCloseGameOverModal,
  handleCloseSettingsModal,
  formatEvaluation,
  handleUndo,
};

createApp(view).mount();

updateDocumentLang();
updateDesktopLinkLocale();

// --- Worker Communication ---
worker.onmessage = async (event) => {
  const { type, payload } = event.data;

  switch (type) {
    case "initialized":
      state.initialLoading = false;
      syncStateFromGame(payload);
      ensureHumanPassIfNeeded();
      break;
    case "state_updated":
      syncStateFromGame(payload);
      runAiTurn();
      ensureHumanPassIfNeeded();
      break;
    case "ai_moved": {
      state.aiThinking = false;
      const nextState = payload.gameState;
      const aiMoveOccurred = payload.move !== undefined;
      if (aiMoveOccurred) {
        state.lastAiMove = payload.move;
        state.lastHumanMove = null;
        const evaluation = state.searchProgress?.score;
        logMove(state.aiColor, payload.move, { evaluation });
      } else if (!nextState.isGameOver && nextState.currentPlayer === state.humanColor) {
        state.passNotice = "ai";
        state.lastAiMove = null;
        void showPassToast("ai");
      }

      const humanForcedPass =
        aiMoveOccurred &&
        !nextState.isGameOver &&
        nextState.currentPlayer === state.aiColor;

      if (humanForcedPass) {
        state.passNotice = "human";
        state.lastHumanMove = null;
        const blockId = ++forcedPassBlockId;
        state.delayAiUntilToast = true;
        void showPassToast("human").finally(() => {
          if (forcedPassBlockId !== blockId) {
            return;
          }
          state.delayAiUntilToast = false;
          runAiTurn();
        });
      }

      workerApi.getState();
      break;
    }
    case "search_progress":
      handleSearchProgress(payload);
      break;
    case "replay_completed":
      syncStateFromGame(payload);
      break;
  }
};

const workerApi = {
  init(humanIsBlack, level) {
    worker.postMessage({ type: "init", payload: { humanIsBlack, level } });
  },
  humanMove(index) {
    worker.postMessage({ type: "human_move", payload: { index } });
  },
  aiMove() {
    worker.postMessage({ type: "ai_move" });
  },
  pass() {
    worker.postMessage({ type: "pass" });
  },
  reset(humanIsBlack, level) {
    worker.postMessage({ type: "reset", payload: { humanIsBlack, level } });
  },
  setLevel(level) {
    worker.postMessage({ type: "set_level", payload: { level } });
  },
  getState() {
    worker.postMessage({ type: "get_state" });
  },
  replayMoves(humanIsBlack, level, moves) {
    worker.postMessage({ type: "replay_moves", payload: { humanIsBlack, level, moves } });
  },
};

// --- Application Logic ---

void (function bootstrap() {
  try {
    workerApi.init(state.humanIsBlack, state.level);
  } catch (e) {
    console.error("Error bootstrapping application:", e);
    state.initialLoading = false;
    state.message = "Error loading application. Please refresh.";
  }
})();

function logMove(color, index, options = {}) {
  if (!Number.isInteger(index) || index < 0 || index >= 64) {
    return;
  }
  const isAiMove = color === state.aiColor;
  const evaluation =
    isAiMove && Number.isFinite(options.evaluation)
      ? Number(options.evaluation)
      : null;
  state.moveHistory.push({
    player: color,
    index,
    evaluation,
  });
  scrollMoveLogToBottom();
}

function scrollMoveLogToBottom() {
  requestAnimationFrame(() => {
    const container = document.getElementById("move-log-scroll");
    if (container) {
      container.scrollTop = container.scrollHeight;
    }
  });
}

function toNotation(index) {
  const file = FILES[index % 8];
  const rank = Math.floor(index / 8) + 1;
  return `${file}${rank}`;
}

function cellClasses(index) {
  return {
    black: state.board[index] === 1,
    white: state.board[index] === 2,
    legal: state.isHumanTurn && state.legalMoves.includes(index),
    "last-human": state.lastHumanMove === index,
    "last-ai": state.lastAiMove === index,
  };
}

function isCellDisabled(index) {
  return !(state.isHumanTurn && state.legalMoves.includes(index));
}

async function handleCellClick(index) {
  if (state.showSettingsModal) {
    return;
  }
  if (
    state.aiThinking ||
    state.isGameOver ||
    state.currentPlayer !== state.humanColor
  ) {
    return;
  }

  if (!Number.isInteger(index)) {
    return;
  }

  state.lastHumanMove = index;
  state.lastAiMove = null;
  state.passNotice = null;
  logMove(state.humanColor, index);
  workerApi.humanMove(index);
}

function handleModalColorChange(event) {
  const nextValue = event?.target?.value === "white" ? "white" : "black";
  state.pendingColor = nextValue;
}

function handleModalLevelChange(event) {
  const nextLevel = clampLevelValue(event?.target?.value);
  state.pendingLevel = String(nextLevel);
}

async function handleStartGame() {
  if (state.initialLoading) {
    return;
  }
  const colorChoice = state.pendingColor === "white" ? "white" : "black";
  const levelChoice = clampLevelValue(state.pendingLevel);
  state.humanIsBlack = colorChoice === "black";
  state.level = levelChoice;
  state.selectedLevel = String(levelChoice);
  state.selectedColor = colorChoice;
  state.showSettingsModal = false;
  await resetGame();
}

async function handleNewGame() {
  if (state.aiThinking) {
    return;
  }

  clearToast();
  state.pendingColor = state.selectedColor;
  state.pendingLevel = state.selectedLevel;
  state.showSettingsModal = true;
}

function handleCloseSettingsModal() {
  state.showSettingsModal = false;
}

function handleCloseGameOverModal() {
  state.gameOverModalDismissed = true;
}

async function resetGame() {
  workerApi.reset(state.humanIsBlack, state.level);
  state.lastHumanMove = null;
  state.lastAiMove = null;
  state.passNotice = null;
  clearToast();
  state.moveHistory = [];
  state.gameResult = null;
  state.gameOverModalDismissed = false;
  state.pendingColor = state.selectedColor;
  state.pendingLevel = state.selectedLevel;
  scrollMoveLogToBottom();
}

async function runAiTurn() {
  if (
    state.isGameOver ||
    state.currentPlayer !== state.aiColor ||
    state.aiThinking ||
    state.delayAiUntilToast
  ) {
    return;
  }

  state.aiThinking = true;
  state.searchProgress = null;
  updateSearchStatusText();
  await waitForNextFrame();
  workerApi.aiMove();
}

async function ensureHumanPassIfNeeded() {
  if (state.showSettingsModal || state.isGameOver || state.aiThinking) {
    return;
  }

  const shouldPass =
    state.currentPlayer === state.humanColor &&
    state.legalMoves.length === 0;

  if (!shouldPass) {
    return;
  }

  state.passNotice = "human";
  state.lastHumanMove = null;
  state.lastAiMove = null;
  await showPassToast("human");
  workerApi.pass();
}

function syncStateFromGame(gameState) {
  if (!gameState) {
    return;
  }

  state.board = gameState.board;
  state.legalMoves = gameState.legalMoves;
  state.isGameOver = gameState.isGameOver;
  state.currentPlayer = gameState.currentPlayer;
  state.humanColor = gameState.humanColor;
  state.aiColor = gameState.aiColor;
  state.humanIsBlack = gameState.humanColor === 1;

  state.isHumanTurn = !state.isGameOver && state.currentPlayer === state.humanColor;
  state.selectedColor = state.humanIsBlack ? "black" : "white";
  state.selectedLevel = String(state.level);
  state.pendingColor = state.selectedColor;
  state.pendingLevel = state.selectedLevel;

  const scores = gameState.score;
  state.scoreBlack = scores[0];
  state.scoreWhite = scores[1];
  state.emptyCount = gameState.emptyCount;

  const { text, kind } = computeMessage();
  state.message = text;
  state.messageKind = kind;

  updateSearchStatusText();
}

function computeMessage() {
  const locale = currentLocale();
  const colors = locale.colors;

  if (state.isGameOver) {
    const black = state.scoreBlack;
    const white = state.scoreWhite;
    let result = "draw";
    if (black > white) {
      result = state.humanIsBlack ? "human" : "ai";
    } else if (white > black) {
      result = state.humanIsBlack ? "ai" : "human";
    }
    state.gameResult = result;
    state.passNotice = null;
    return { text: "", kind: "info" };
  }

  if (state.passNotice === "human") {
    state.passNotice = null;
    return { text: locale.messages.passHuman, kind: "info" };
  }

  if (state.passNotice === "ai") {
    state.passNotice = null;
    return { text: locale.messages.passAi, kind: "info" };
  }

  if (state.aiThinking) {
    return { text: locale.messages.aiThinking, kind: "info" };
  }

  if (state.isHumanTurn) {
    return { text: "", kind: "info" };
  }

  const aiColorName = state.humanIsBlack ? colors.white : colors.black;
  return { text: locale.messages.aiTurn(aiColorName), kind: "info" };
}

function clampLevelValue(value) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    return DEFAULT_LEVEL;
  }
  const rounded = Math.round(parsed);
  if (rounded < LEVEL_MIN) {
    return LEVEL_MIN;
  }
  if (rounded > LEVEL_MAX) {
    return LEVEL_MAX;
  }
  return rounded;
}

function waitForNextFrame() {
  return new Promise((resolve) => {
    requestAnimationFrame(() => {
      setTimeout(resolve, 0);
    });
  });
}

function handleSearchProgress(update) {
  state.searchProgress = {
    depth: Number(update?.depth ?? 0),
    score: Number(update?.score ?? 0),
    nodes: Number(update?.nodes ?? 0),
    selectivity: Number(update?.selectivity ?? 0),
    bestMoveIndex:
      typeof update?.bestMoveIndex === "number" ? update.bestMoveIndex : null,
  };
  updateSearchStatusText();
}

function updateSearchStatusText() {
  const locale = currentLocale();
  if (!state.aiThinking) {
    state.searchStatusText = locale.searchStatus.idle;
    return;
  }

  if (!state.searchProgress) {
    state.searchStatusText = locale.searchStatus.preparing;
    return;
  }

  const { depth, score, nodes, bestMoveIndex, selectivity } = state.searchProgress;
  const moveText = Number.isFinite(bestMoveIndex)
    ? toNotation(bestMoveIndex)
    : locale.searchStatus.movePlaceholder;
  const scoreText = Number.isFinite(score) ? score.toFixed(2) : "--";
  const nodeText = formatNumber(nodes);

  state.searchStatusText = locale.searchStatus.format({
    depth,
    scoreText,
    nodeText,
    moveText,
    selectivity,
  });
}

function formatNumber(value) {
  if (!Number.isFinite(value)) {
    return "--";
  }
  const locale = state.locale === "en" ? "en-US" : "ja-JP";
  return value.toLocaleString(locale);
}

function formatEvaluation(value) {
  const locale = currentLocale();
  if (!Number.isFinite(value)) {
    return locale.moveLog?.noEval ?? "--";
  }
  const normalized = Math.abs(value) < 0.005 ? 0 : value;
  const fixed = normalized === 0 ? "0.00" : (Math.floor(normalized * 100) / 100).toFixed(2);
  if (normalized > 0) {
    return `+${fixed}`;
  }
  if (normalized < 0) {
    return fixed;
  }
  return fixed;
}

function detectPreferredLocale() {
  const fallback = "ja";
  const tryMatch = (code) => {
    if (!code) {
      return null;
    }
    const lowered = String(code).toLowerCase();
    if (locales[lowered]) {
      return lowered;
    }
    const primary = lowered.split("-")[0];
    if (locales[primary]) {
      return primary;
    }
    return null;
  };

  const candidates = [];
  if (typeof navigator !== "undefined") {
    if (Array.isArray(navigator.languages)) {
      candidates.push(...navigator.languages);
    }
    if (navigator.language) {
      candidates.push(navigator.language);
    }
  }

  for (const candidate of candidates) {
    const match = tryMatch(candidate);
    if (match) {
      return match;
    }
  }

  return fallback;
}

function currentLocale() {
  return locales[state.locale] ?? locales.ja;
}

function updateDocumentLang() {
  document.documentElement.lang = state.locale;
}

function updateDesktopLinkLocale() {
  const locale = currentLocale();
  const labelTarget = document.querySelector('[data-locale-text="desktop-link"]');
  const ariaTarget = document.querySelector('[data-locale-aria="desktop-link"]');

  if (labelTarget && locale?.externalLinks?.desktop?.label) {
    labelTarget.textContent = locale.externalLinks.desktop.label;
  }

  if (ariaTarget && locale?.externalLinks?.desktop?.aria) {
    ariaTarget.setAttribute("aria-label", locale.externalLinks.desktop.aria);
  }
}

function clearToast() {
  toastSequenceId += 1;
  forcedPassBlockId += 1;
  state.toastVisible = false;
  state.toastMessage = null;
  state.delayAiUntilToast = false;
}

async function showPassToast(kind) {
  const locale = currentLocale();
  const message =
    kind === "ai"
      ? locale.messages.passAi
      : locale.messages.passHuman;
  return showToast(message, { duration: PASS_TOAST_DURATION_MS });
}

async function showToast(message, options = {}) {
  if (!message) {
    return;
  }

  toastSequenceId += 1;
  const currentSeq = toastSequenceId;
  const duration = Number.isFinite(options.duration)
    ? options.duration
    : PASS_TOAST_DURATION_MS;

  state.toastMessage = message;
  state.toastVisible = true;

  await waitMs(duration);

  if (currentSeq !== toastSequenceId) {
    return;
  }

  state.toastVisible = false;

  await waitMs(TOAST_FADE_DURATION_MS);

  if (currentSeq !== toastSequenceId) {
    return;
  }

  state.toastMessage = null;
}

function waitMs(ms = 0) {
  return new Promise((resolve) => {
    setTimeout(resolve, Math.max(0, ms));
  });
}

async function handleUndo() {
  if (state.aiThinking || state.moveHistory.length === 0 || state.showSettingsModal) {
    return;
  }

  // Remove the last move(s) from history
  // If the last move was by AI, remove both AI and human moves (2 moves)
  // If the last move was by human, remove only that move
  const lastMove = state.moveHistory[state.moveHistory.length - 1];
  const movesToUndo = lastMove.player === state.aiColor ? 2 : 1;

  // Remove moves from history
  for (let i = 0; i < movesToUndo && state.moveHistory.length > 0; i++) {
    state.moveHistory.pop();
  }

  // Clear visual state
  state.lastHumanMove = null;
  state.lastAiMove = null;
  state.passNotice = null;
  state.gameResult = null;
  state.gameOverModalDismissed = false;

  // Replay moves to restore game state
  // Convert reactive array to plain array for Worker
  const plainMoves = state.moveHistory.map(move => ({ player: move.player, index: move.index }));
  workerApi.replayMoves(state.humanIsBlack, state.level, plainMoves);
}

// ESC key handler for modals
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    // Close settings modal if open
    if (state.showSettingsModal && !state.initialLoading) {
      handleCloseSettingsModal();
    }
    // Close game over modal if open
    else if (state.isGameOver && !state.gameOverModalDismissed) {
      handleCloseGameOverModal();
    }
  }
});
