import {
  accessibleNamePredicate,
  displayedButton,
  displayedText,
  exactTextPredicate,
  textXpath,
  waitForDisplayed,
  waitForGone,
  xpathLiteral,
} from "../support/dom.js";

const NEW_GAME = ["New Game", "新規ゲーム"];
const SOLVER = ["Solver", "ソルバー"];
const START_GAME = ["Start Game", "開始"];
const START_SOLVER = ["Start Solver", "ソルバー開始"];
const EXIT_SOLVER = ["Exit Solver", "ソルバー終了"];
const BOARD_SETUP = ["Board Setup", "盤面設定"];
const CLEAR_SETUP = ["Clear", "クリア"];
const INITIAL_POSITION = ["Initial", "初期配置"];
const VS_AI = ["vs AI", "AI対戦"];
const VS_HUMAN = ["vs Human", "対人戦"];
const YOU_PLAY = ["You play", "あなたの色"];
const AI_MODE = ["AI Mode", "AIモード"];
const TIMED = ["Timed", "持ち時間"];
const AI_LEVEL = ["AI Level", "AIレベル"];
const TIME_PER_GAME = ["Time per game", "持ち時間"];
const EASY = ["Easy", "易しい"];
const HARD = ["Hard", "難しい"];
const MENU = ["Menu", "メニュー"];
const ABOUT = ["About", "このアプリについて"];
const LICENSE = ["License", "ライセンス"];
const THIRD_PARTY = ["Third-Party", "サードパーティ"];
const HINT_LEVEL = ["Hint Level", "ヒントレベル"];
const HASH_SIZE = ["Hash Size", "ハッシュサイズ"];
const LANGUAGE = ["Language", "言語"];
const ENGLISH = ["English"];
const JAPANESE = ["日本語"];
const TRANSCRIPT = ["Transcript", "棋譜入力"];
const BOARD_STRING = ["Board String", "ボード文字列"];
const INVALID_LENGTH = [
  "Transcript length must be even (2 chars per move)",
  "棋譜の長さが不正です（1手2文字）",
];
const INVALID_BOARD_LENGTH = [
  "Board string must be exactly 64 characters",
  "ボード文字列は64文字である必要があります",
];
const NEED_BOTH_COLORS = [
  "Board must have at least one black and one white stone",
  "黒と白の石が少なくとも1つずつ必要です",
];
const NO_MOVES = ["No moves yet", "まだ着手がありません"];
const UNDO = ["Undo", "戻る", "一手戻す"];
const REDO = ["Redo", "進む"];
const COPY = ["Copy transcript", "棋譜をコピー"];
const STOP = ["Stop", "停止"];
const RESUME = ["Resume", "再開"];
const AI_ANALYSIS = ["AI Analysis", "AI分析"];
const THINKING_LOG = ["Thinking Log", "思考ログ"];
const EVALUATION = ["Evaluation", "評価値"];
const ANALYZE = ["Analyze", "解析"];
const SELECTIVITY = ["Selectivity", "精度"];
const MODE = ["Mode", "モード"];
const BEST_MOVE_ONLY = ["Best move only", "最善手のみ"];
const ALL_MOVES = ["All moves (MultiPV)", "全手スコア"];
const BLACK_WINS = ["Black wins!", "黒の勝ちです！"];
const PASS = ["Pass", "パス"];
const WHITE_PASS = ["White: Pass", "白：パス"];

const INITIAL_BOARD_STRING = "---------------------------OX------XO---------------------------";
const QUICK_SOLVER_BOARD_STRING = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX-OOOX";
const PASS_TURN_BOARD_STRING =
  "-OX-----" + "--------" + "--------" + "--------" + "--------" + "--------" + "----O---" + "----X---";

async function waitForWindowWidth(predicate, timeoutMsg) {
  await browser.waitUntil(
    async () => {
      const width = await browser.execute(() => window.innerWidth);
      return predicate(width);
    },
    { timeout: 5000, timeoutMsg },
  );
}

async function displayedDialogTitle(labels) {
  return await waitForDisplayed(
    `//*[@data-slot = 'dialog-title' and (${exactTextPredicate(labels)})]`,
    `Expected a visible dialog title matching one of: ${labels.join(", ")}`,
  );
}

async function displayedDialogButton(labels, timeout) {
  return await waitForDisplayed(
    `//*[@data-slot = 'dialog-content']//button[${accessibleNamePredicate(labels)}]`,
    `Expected a visible dialog button named one of: ${labels.join(", ")}`,
    timeout,
  );
}

async function waitForDialogToClose(labels) {
  const xpath = `//*[@data-slot = 'dialog-title' and (${exactTextPredicate(labels)})]`;
  await waitForGone(xpath, `Expected dialog to close: ${labels.join(", ")}`);
}

async function displayedAboutTitle() {
  return await waitForDisplayed(
    "//*[@data-slot = 'dialog-title' and contains(normalize-space(.), 'Neural Reversi')]",
    "Expected the About dialog title to contain Neural Reversi.",
  );
}

async function waitForAboutDialogToClose() {
  await waitForGone(
    "//*[@data-slot = 'dialog-title' and contains(normalize-space(.), 'Neural Reversi')]",
    "Expected the About dialog to close.",
  );
}

async function fieldByPlaceholder(labels) {
  return await waitForDisplayed(
    `//input[${labels.map((label) => `@placeholder = ${xpathLiteral(label)}`).join(" or ")}] | ` +
      `//textarea[${labels.map((label) => `@placeholder = ${xpathLiteral(label)}`).join(" or ")}]`,
    `Expected an input with placeholder matching one of: ${labels.join(", ")}`,
  );
}

async function menuItem(labels) {
  return await waitForDisplayed(
    `//*[@data-slot = 'dropdown-menu-item' or @data-slot = 'dropdown-menu-sub-trigger' or @data-slot = 'dropdown-menu-radio-item'][${accessibleNamePredicate(labels)}]`,
    `Expected a visible menu item named one of: ${labels.join(", ")}`,
  );
}

async function openHeaderMenu() {
  const menuButton = await displayedButton(MENU);
  await menuButton.click();
}

async function openAnalysisPanel() {
  const analysisHeader = await waitForDisplayed(
    `//button[${accessibleNamePredicate(AI_ANALYSIS)}]`,
    "Expected the AI Analysis header button.",
  );
  if ((await analysisHeader.getAttribute("aria-expanded")) !== "true") {
    await analysisHeader.click();
  }
  await displayedText(THINKING_LOG);
  return analysisHeader;
}

async function closeOpenMenus() {
  await browser.keys("Escape");
  await browser.keys("Escape");
  await waitForGone(
    "//*[@data-slot = 'dropdown-menu-content' or @data-slot = 'dropdown-menu-sub-content']",
    "Expected header menu popups to close.",
    3000,
  );
}

async function openSubmenu(labels) {
  const item = await menuItem(labels);
  await item.moveTo();
  await item.click();
  return item;
}

async function expectMenuItemChecked(labels) {
  const item = await menuItem(labels);
  await browser.waitUntil(
    async () =>
      (await item.getAttribute("aria-checked")) === "true" ||
      (await item.getAttribute("data-checked")) !== null,
    {
      timeout: 3000,
      timeoutMsg: `Expected menu item to be checked: ${labels.join(", ")}`,
    },
  );
}

async function expectSwitchChecked(switchElement, checked) {
  await browser.waitUntil(
    async () => {
      const dataChecked = await switchElement.getAttribute("data-checked");
      const ariaChecked = await switchElement.getAttribute("aria-checked");
      return checked
        ? dataChecked !== null || ariaChecked === "true"
        : dataChecked === null && ariaChecked !== "true";
    },
    {
      timeout: 3000,
      timeoutMsg: `Expected switch checked state to become ${checked}.`,
    },
  );
}

async function radioLabel(labels) {
  return await waitForDisplayed(
    `//label[${labels.map((label) => `normalize-space(.) = ${xpathLiteral(label)}`).join(" or ")}]`,
    `Expected a visible radio label matching one of: ${labels.join(", ")}`,
  );
}

async function expectRadioChecked(labels) {
  const radio = await waitForDisplayed(
    `//label[${labels.map((label) => `normalize-space(.) = ${xpathLiteral(label)}`).join(" or ")}]//*[@data-slot = 'radio-group-item']`,
    `Expected a visible radio item matching one of: ${labels.join(", ")}`,
  );
  await browser.waitUntil(
    async () =>
      (await radio.getAttribute("aria-checked")) === "true" ||
      (await radio.getAttribute("data-checked")) !== null,
    {
      timeout: 3000,
      timeoutMsg: `Expected radio item to be checked: ${labels.join(", ")}`,
    },
  );
}

async function expectCopySuccessIndicator() {
  await waitForDisplayed(
    `//button[${accessibleNamePredicate(COPY)}]//*[local-name() = 'svg' and contains(concat(' ', normalize-space(@class), ' '), ' text-primary ')]`,
    "Expected Copy transcript to show the post-write success indicator.",
    3000,
  );
}

async function isUnavailable(button) {
  const disabled = await button.getAttribute("disabled");
  const ariaDisabled = await button.getAttribute("aria-disabled");
  const dataDisabled = await button.getAttribute("data-disabled");

  return (
    Boolean(disabled) ||
    ariaDisabled === "true" ||
    dataDisabled !== null ||
    !(await button.isEnabled())
  );
}

async function expectUnavailable(button) {
  await browser.waitUntil(async () => await isUnavailable(button), {
    timeout: 3000,
    timeoutMsg: "Expected button to become unavailable.",
  });
  expect(await isUnavailable(button)).toBe(true);
}

async function expectAvailable(button) {
  await browser.waitUntil(async () => !(await isUnavailable(button)), {
    timeout: 3000,
    timeoutMsg: "Expected button to become available.",
  });
  expect(await isUnavailable(button)).toBe(false);
}

async function clickBoardCell(row, col) {
  const canvas = await $("canvas");
  await canvas.waitForDisplayed({
    timeout: 10000,
    timeoutMsg: "Expected board canvas before clicking a cell.",
  });

  const point = await browser.execute(
    (target, r, c) => {
      const rect = target.getBoundingClientRect();
      const totalSize = 9.5;
      const cellPx = rect.width / rect.height >= 1 ? rect.height / totalSize : rect.width / totalSize;
      return {
        x: Math.round((c - 3.5) * cellPx),
        y: Math.round((r - 3.5) * cellPx),
      };
    },
    canvas,
    row,
    col,
  );

  await browser.action("pointer").move({ origin: canvas, x: point.x, y: point.y }).down().up().perform();
}

describe("User behavior coverage", () => {
  beforeEach(async () => {
    await browser.setWindowSize(1200, 900);
    await waitForWindowWidth((width) => width >= 1024, "Expected desktop viewport.");
  });

  it("exposes the header menu and About license views accessibly", async () => {
    await openHeaderMenu();

    await menuItem(["Hint Level", "ヒントレベル"]);
    await menuItem(["Hash Size", "ハッシュサイズ"]);
    await menuItem(["Language", "言語"]);

    const aboutItem = await menuItem(ABOUT);
    await aboutItem.click();

    await displayedAboutTitle();
    await displayedText(LICENSE);
    await displayedText(THIRD_PARTY);

    const thirdPartyTab = await displayedButton(THIRD_PARTY);
    await thirdPartyTab.click();
    await displayedText(THIRD_PARTY);

    const closeButton = await displayedButton(["Close"]);
    await closeButton.click();
    await waitForAboutDialogToClose();
  });

  it("changes language and hash size from the header menu", async () => {
    await openHeaderMenu();
    await openSubmenu(HINT_LEVEL);
    const hintLevel1Item = await menuItem(["Lv.1"]);
    await hintLevel1Item.click();
    await expectMenuItemChecked(["Lv.1"]);

    await openSubmenu(HASH_SIZE);
    const hash128Item = await menuItem(["128 MB"]);
    await hash128Item.click();
    await expectMenuItemChecked(["128 MB"]);

    await openSubmenu(LANGUAGE);
    const englishItem = await menuItem(ENGLISH);
    await englishItem.click();
    await displayedButton(["New Game"]);

    await openSubmenu(LANGUAGE);
    const japaneseItem = await menuItem(JAPANESE);
    await japaneseItem.click();
    await displayedButton(["新規ゲーム"]);
    await closeOpenMenus();
  });

  it("opens the analysis panel and keeps Analyze disabled before any move", async () => {
    const analysisHeader = await openAnalysisPanel();
    await displayedText(EVALUATION);
    await expectUnavailable(await displayedButton(ANALYZE));

    await analysisHeader.click();
    await waitForGone(textXpath(THINKING_LOG), "Expected AI Analysis panel content to close.");
    await analysisHeader.click();
    await displayedText(THINKING_LOG);
  });

  it("uses resizable desktop panels and stacked mobile panels", async () => {
    const boardCanvas = await $("canvas");
    await boardCanvas.waitForDisplayed({
      timeout: 10000,
      timeoutMsg: "Expected board canvas in desktop layout.",
    });
    await waitForDisplayed(
      "//*[@data-separator and @aria-orientation = 'vertical']",
      "Expected a desktop sidebar resize separator.",
    );

    await openAnalysisPanel();
    await waitForDisplayed(
      "//*[@data-separator and @aria-orientation = 'horizontal']",
      "Expected a desktop bottom-panel resize separator after opening analysis.",
    );

    await browser.setWindowSize(760, 900);
    await waitForWindowWidth((width) => width < 1024, "Expected mobile viewport.");
    await boardCanvas.waitForDisplayed({
      timeout: 10000,
      timeoutMsg: "Expected board canvas to remain visible in mobile layout.",
    });
    await waitForDisplayed(
      `//*[${accessibleNamePredicate(["Move History", "棋譜"])}]`,
      "Expected move history panel to remain available in mobile layout.",
    );
    await displayedText(THINKING_LOG);
    await waitForGone(
      "//*[@data-separator]",
      "Expected mobile layout to remove desktop resizable separators.",
    );
  });

  it("configures a New Game AI matchup and starts it from the modal", async () => {
    const newGameButton = await displayedButton(NEW_GAME);
    await newGameButton.click();
    await displayedDialogTitle(NEW_GAME);

    const aiTab = await displayedDialogButton(VS_AI);
    await aiTab.click();
    await displayedText(YOU_PLAY);
    await displayedText(AI_MODE);

    const timedTab = await displayedDialogButton(TIMED);
    await timedTab.click();
    await displayedText(TIME_PER_GAME);
    await displayedText(["30s"]);

    const levelTab = await displayedDialogButton(AI_LEVEL);
    await levelTab.click();
    await displayedText(EASY);
    await displayedText(HARD);

    await timedTab.click();
    await displayedText(TIME_PER_GAME);

    const whitePlayer = await displayedDialogButton(["White", "白"]);
    await whitePlayer.click();
    const blackPlayer = await displayedDialogButton(["Black", "黒"]);
    await blackPlayer.click();

    const startGameButton = await displayedDialogButton(START_GAME);
    await startGameButton.click();
    await waitForDialogToClose(NEW_GAME);
    await displayedText(NO_MOVES);
    await displayedText(["1:00"]);
  });

  it("validates manual setup boards and clears errors after reset", async () => {
    const newGameButton = await displayedButton(NEW_GAME);
    await newGameButton.click();
    await displayedDialogTitle(NEW_GAME);

    const pvpTab = await displayedButton(VS_HUMAN);
    await pvpTab.click();

    const boardSetupButton = await displayedButton(BOARD_SETUP);
    await boardSetupButton.click();
    await displayedDialogTitle(BOARD_SETUP);

    const clearButton = await displayedDialogButton(CLEAR_SETUP);
    await clearButton.click();

    const startGameButton = await displayedDialogButton(START_GAME);
    await startGameButton.click();
    await displayedText(NEED_BOTH_COLORS);
    await expectUnavailable(await displayedDialogButton(START_GAME));

    const initialButton = await displayedDialogButton(INITIAL_POSITION);
    await initialButton.click();
    await waitForGone(
      textXpath(NEED_BOTH_COLORS),
      "Expected initial setup reset to clear the validation error.",
    );
    await expectAvailable(await displayedDialogButton(START_GAME));

    const cancelButton = await displayedDialogButton(["Cancel", "キャンセル"]);
    await cancelButton.click();
    await waitForDialogToClose(BOARD_SETUP);
  });

  it("validates setup inputs, starts a PvP setup game, and supports board/history undo", async () => {
    const newGameButton = await displayedButton(NEW_GAME);
    await newGameButton.click();
    await displayedDialogTitle(NEW_GAME);

    const pvpTab = await displayedButton(VS_HUMAN);
    await pvpTab.click();

    const boardSetupButton = await displayedButton(BOARD_SETUP);
    await boardSetupButton.click();
    await displayedDialogTitle(BOARD_SETUP);

    const transcriptTab = await displayedButton(TRANSCRIPT);
    await transcriptTab.click();
    const transcriptInput = await fieldByPlaceholder(["e.g. F5D6C3D3C4", "例: F5D6C3D3C4"]);
    await transcriptInput.setValue("f");
    await displayedText(INVALID_LENGTH);
    await expectUnavailable(await displayedDialogButton(START_GAME));

    await transcriptInput.setValue("f5");
    await waitForGone(textXpath(INVALID_LENGTH), "Expected valid transcript to clear the error.");

    const boardStringTab = await displayedButton(BOARD_STRING);
    await boardStringTab.click();
    const boardStringInput = await fieldByPlaceholder([
      "64 chars: X=black, O=white, -=empty",
      "64文字: X=黒, O=白, -=空",
    ]);
    await boardStringInput.setValue("x");
    await displayedText(INVALID_BOARD_LENGTH);
    await expectUnavailable(await displayedDialogButton(START_GAME));

    await boardStringInput.setValue(INITIAL_BOARD_STRING);
    await waitForGone(
      textXpath(INVALID_BOARD_LENGTH),
      "Expected valid board string to clear the error.",
    );
    const blackTurnButton = await displayedDialogButton(["Black", "黒"]);
    await blackTurnButton.click();

    const startGameButton = await displayedDialogButton(START_GAME);
    await startGameButton.click();
    await waitForDialogToClose(BOARD_SETUP);

    await waitForDisplayed(textXpath(NO_MOVES), "Expected a fresh move history after starting.");
    await expectUnavailable(await displayedButton(COPY));

    await clickBoardCell(3, 3);
    await displayedText(NO_MOVES);
    await expectUnavailable(await displayedButton(COPY));

    const hintSwitch = await $('[data-slot="switch"]');
    await hintSwitch.waitForDisplayed({
      timeout: 10000,
      timeoutMsg: "Expected the hint switch to be visible.",
    });
    await expectSwitchChecked(hintSwitch, false);
    await hintSwitch.click();
    await expectSwitchChecked(hintSwitch, true);
    await waitForDisplayed(
      "//*[@data-board-hint = 'waiting' or @data-board-hint = 'score']",
      "Expected hint mode to render waiting or scored board hints.",
      15000,
    );
    await hintSwitch.click();
    await expectSwitchChecked(hintSwitch, false);

    await clickBoardCell(2, 3);
    await displayedText(["d3"]);
    await waitForGone(textXpath(NO_MOVES), "Expected the first move to replace the empty history.");
    await expectAvailable(await displayedButton(ANALYZE));

    const undoButton = await displayedButton(UNDO);
    await expectAvailable(undoButton);
    await undoButton.click();
    await expectUnavailable(undoButton);
    await expectUnavailable(await displayedButton(COPY));
    await expectAvailable(await displayedButton(REDO));
    const d3Move = await displayedText(["d3"]);
    await d3Move.click();
    await expectAvailable(await displayedButton(COPY));
    await expectAvailable(undoButton);

    await browser.keys("ArrowLeft");
    await expectUnavailable(await displayedButton(COPY));
    await expectAvailable(await displayedButton(REDO));

    await browser.keys("ArrowRight");
    await expectAvailable(await displayedButton(COPY));

    const copyButton = await displayedButton(COPY);
    await copyButton.click();
    await expectCopySuccessIndicator();
  });

  it("starts a near-final PvP setup and reports game over after the final move", async () => {
    const newGameButton = await displayedButton(NEW_GAME);
    await newGameButton.click();
    await displayedDialogTitle(NEW_GAME);

    const pvpTab = await displayedButton(VS_HUMAN);
    await pvpTab.click();

    const boardSetupButton = await displayedButton(BOARD_SETUP);
    await boardSetupButton.click();
    await displayedDialogTitle(BOARD_SETUP);

    const boardStringTab = await displayedButton(BOARD_STRING);
    await boardStringTab.click();
    const boardStringInput = await fieldByPlaceholder([
      "64 chars: X=black, O=white, -=empty",
      "64文字: X=黒, O=白, -=空",
    ]);
    await boardStringInput.setValue(QUICK_SOLVER_BOARD_STRING);
    await waitForGone(
      textXpath(INVALID_BOARD_LENGTH),
      "Expected valid near-final board string to clear length validation.",
    );
    const blackTurnButton = await displayedDialogButton(["Black", "黒"]);
    await blackTurnButton.click();

    const startGameButton = await displayedDialogButton(START_GAME);
    await startGameButton.click();
    await waitForDialogToClose(BOARD_SETUP);

    await clickBoardCell(7, 3);
    await displayedText(["d8"]);
    await displayedText(BLACK_WINS, 10000);

    await openAnalysisPanel();
    const analyzeButton = await displayedButton(ANALYZE);
    await expectAvailable(analyzeButton);
    await analyzeButton.click();
    await waitForDisplayed(
      "//*[@data-evaluation-chart = 'analyzed' and @data-analysis-points != '0']",
      "Expected whole-game analysis to populate the evaluation chart.",
      15000,
    );
    const undoButton = await displayedButton(UNDO);
    await undoButton.click();
    await expectUnavailable(await displayedButton(COPY));
    const chartPoint = await waitForDisplayed(
      "//*[@data-evaluation-chart = 'analyzed']//*[local-name() = 'circle']",
      "Expected the evaluation chart to render a clickable move point.",
    );
    await chartPoint.click();
    await expectAvailable(await displayedButton(COPY));
  });

  it("records and announces an automatic pass turn after a forced-pass move", async () => {
    const newGameButton = await displayedButton(NEW_GAME);
    await newGameButton.click();
    await displayedDialogTitle(NEW_GAME);

    const pvpTab = await displayedButton(VS_HUMAN);
    await pvpTab.click();

    const boardSetupButton = await displayedButton(BOARD_SETUP);
    await boardSetupButton.click();
    await displayedDialogTitle(BOARD_SETUP);

    const boardStringTab = await displayedButton(BOARD_STRING);
    await boardStringTab.click();
    const boardStringInput = await fieldByPlaceholder([
      "64 chars: X=black, O=white, -=empty",
      "64文字: X=黒, O=白, -=空",
    ]);
    await boardStringInput.setValue(PASS_TURN_BOARD_STRING);
    await waitForGone(
      textXpath(INVALID_BOARD_LENGTH),
      "Expected valid pass-turn board string to clear length validation.",
    );
    const blackTurnButton = await displayedDialogButton(["Black", "黒"]);
    await blackTurnButton.click();

    const startGameButton = await displayedDialogButton(START_GAME);
    await startGameButton.click();
    await waitForDialogToClose(BOARD_SETUP);

    await clickBoardCell(0, 0);
    await displayedText(["a1"]);
    await displayedText(PASS);
    await displayedText(WHITE_PASS, 10000);
  });

  it("validates solver setup without starting a long solver search", async () => {
    const solverButton = await displayedButton(SOLVER);
    await solverButton.click();
    await displayedDialogTitle(SOLVER);
    await displayedText(SELECTIVITY);
    await displayedText(MODE);

    const selectivity95 = await radioLabel(["95%"]);
    await selectivity95.click();
    await expectRadioChecked(["95%"]);
    const bestMoveOnly = await radioLabel(BEST_MOVE_ONLY);
    await bestMoveOnly.click();
    await expectRadioChecked(BEST_MOVE_ONLY);
    const allMoves = await radioLabel(ALL_MOVES);
    await allMoves.click();
    await expectRadioChecked(ALL_MOVES);

    const boardStringTab = await displayedButton(BOARD_STRING);
    await boardStringTab.click();
    const boardStringInput = await fieldByPlaceholder([
      "64 chars: X=black, O=white, -=empty",
      "64文字: X=黒, O=白, -=空",
    ]);
    await boardStringInput.setValue("x");
    await displayedText(INVALID_BOARD_LENGTH);
    await expectUnavailable(await displayedDialogButton(START_SOLVER));

    const cancelButton = await displayedDialogButton(["Cancel", "キャンセル"]);
    await cancelButton.click();
    await waitForDialogToClose(SOLVER);
  });

  it("stops and resumes an active solver search", async () => {
    const solverButton = await displayedButton(SOLVER);
    await solverButton.click();
    await displayedDialogTitle(SOLVER);

    const selectivity73 = await radioLabel(["73%"]);
    await selectivity73.click();
    await expectRadioChecked(["73%"]);
    const bestMoveOnly = await radioLabel(BEST_MOVE_ONLY);
    await bestMoveOnly.click();
    await expectRadioChecked(BEST_MOVE_ONLY);

    const startSolverButton = await displayedDialogButton(START_SOLVER);
    await startSolverButton.click();
    await waitForDialogToClose(SOLVER);

    const stopButton = await displayedButton(STOP, 10000);
    await stopButton.click();
    const resumeButton = await displayedButton(RESUME, 10000);
    await resumeButton.click();
    const resumedStopButton = await displayedButton(STOP, 10000);
    await resumedStopButton.click();
    await displayedButton(RESUME, 10000);

    const exitSolverButton = await displayedButton(EXIT_SOLVER);
    await exitSolverButton.click();
    await displayedText(NO_MOVES);
  });

  it("starts a fast solver session, navigates a candidate, undoes, and exits", async () => {
    const solverButton = await displayedButton(SOLVER);
    await solverButton.click();
    await displayedDialogTitle(SOLVER);

    const boardStringTab = await displayedButton(BOARD_STRING);
    await boardStringTab.click();
    const boardStringInput = await fieldByPlaceholder([
      "64 chars: X=black, O=white, -=empty",
      "64文字: X=黒, O=白, -=空",
    ]);
    await boardStringInput.setValue(QUICK_SOLVER_BOARD_STRING);
    await waitForGone(
      textXpath(INVALID_BOARD_LENGTH),
      "Expected valid quick solver board string to clear length validation.",
    );
    const blackTurnButton = await displayedDialogButton(["Black", "黒"]);
    await blackTurnButton.click();

    const startSolverButton = await displayedDialogButton(START_SOLVER);
    await startSolverButton.click();
    await waitForDialogToClose(SOLVER);

    await displayedText(["Candidates", "候補手"]);
    const d8Candidate = await displayedText(["d8"], 15000);
    await d8Candidate.click();

    const solverUndoButton = await displayedButton(UNDO);
    await expectAvailable(solverUndoButton);
    await displayedText(["No legal moves. Use Undo to step back.", "合法手がありません。一手戻すを押してください。"]);
    await solverUndoButton.click();
    await displayedText(["d8"], 15000);

    const activeSelectivity95 = await radioLabel(["95%"]);
    await activeSelectivity95.click();
    await expectRadioChecked(["95%"]);
    const activeBestMoveOnly = await radioLabel(BEST_MOVE_ONLY);
    await activeBestMoveOnly.click();
    await expectRadioChecked(BEST_MOVE_ONLY);
    await displayedText(["d8"], 15000);

    const exitSolverButton = await displayedButton(EXIT_SOLVER);
    await exitSolverButton.click();
    await displayedText(NO_MOVES);
  });
});
