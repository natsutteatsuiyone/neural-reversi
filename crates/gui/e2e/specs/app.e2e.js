const NEW_GAME = ["New Game", "新規ゲーム"];
const SOLVER = ["Solver", "ソルバー"];
const START_GAME = ["Start Game", "開始"];
const BOARD_SETUP = ["Board Setup", "盤面設定"];
const VS_AI = ["vs AI", "AI対戦"];
const VS_HUMAN = ["vs Human", "対人戦"];
const CANCEL = ["Cancel", "キャンセル"];
const MOVE_HISTORY = ["Move History", "棋譜"];
const START_SOLVER = ["Start Solver", "ソルバー開始"];
const SELECTIVITY = ["Selectivity", "精度"];
const MODE = ["Mode", "モード"];

function xpathLiteral(value) {
  if (!value.includes("'")) return `'${value}'`;
  if (!value.includes("\"")) return `"${value}"`;

  return `concat(${value
    .split("'")
    .map((part) => `'${part}'`)
    .join(', "\"\'\"", ')})`;
}

function exactTextPredicate(labels) {
  return labels.map((label) => `normalize-space(.) = ${xpathLiteral(label)}`).join(" or ");
}

function accessibleNamePredicate(labels) {
  return labels
    .map(
      (label) =>
        `@aria-label = ${xpathLiteral(label)} or normalize-space(.) = ${xpathLiteral(label)}`,
    )
    .join(" or ");
}

async function firstDisplayed(xpath) {
  const elements = await $$(xpath);

  for (const element of elements) {
    if (await element.isDisplayed()) return element;
  }

  return null;
}

async function waitForDisplayed(xpath, timeoutMsg) {
  await browser.waitUntil(async () => Boolean(await firstDisplayed(xpath)), {
    timeout: 10000,
    timeoutMsg,
  });

  return await firstDisplayed(xpath);
}

async function displayedButton(labels) {
  return await waitForDisplayed(
    `//button[${accessibleNamePredicate(labels)}]`,
    `Expected a visible button named one of: ${labels.join(", ")}`,
  );
}

async function displayedText(labels) {
  return await waitForDisplayed(
    `//*[${exactTextPredicate(labels)}]`,
    `Expected visible text matching one of: ${labels.join(", ")}`,
  );
}

async function displayedDialogTitle(labels) {
  return await waitForDisplayed(
    `//*[@data-slot = 'dialog-title' and (${exactTextPredicate(labels)})]`,
    `Expected a visible dialog title matching one of: ${labels.join(", ")}`,
  );
}

async function waitForDialogToClose(labels) {
  const xpath = `//*[@data-slot = 'dialog-title' and (${exactTextPredicate(labels)})]`;
  await browser.waitUntil(async () => !(await firstDisplayed(xpath)), {
    timeout: 10000,
    timeoutMsg: `Expected dialog to close: ${labels.join(", ")}`,
  });
}

describe("Neural Reversi desktop app", () => {
  it("starts on the main game screen", async () => {
    await displayedButton(NEW_GAME);
    await displayedButton(SOLVER);
    await displayedText(MOVE_HISTORY);

    const boardCanvas = await $("canvas");
    await boardCanvas.waitForDisplayed({
      timeout: 10000,
      timeoutMsg: "Expected the Reversi board canvas to be visible.",
    });
  });

  it("opens the new game flow", async () => {
    const newGameButton = await displayedButton(NEW_GAME);
    await newGameButton.click();

    await displayedDialogTitle(NEW_GAME);
    await displayedText(VS_AI);
    await displayedText(VS_HUMAN);
    await displayedButton(START_GAME);

    const boardSetupButton = await displayedButton(BOARD_SETUP);
    await boardSetupButton.click();

    await displayedDialogTitle(BOARD_SETUP);
    await displayedText(["Manual", "手動配置"]);
    await displayedText(["Transcript", "棋譜入力"]);
    await displayedText(["Board String", "ボード文字列"]);

    const cancelButton = await displayedButton(CANCEL);
    await cancelButton.click();
    await waitForDialogToClose(BOARD_SETUP);
  });

  it("opens the solver setup flow", async () => {
    const solverButton = await displayedButton(SOLVER);
    await solverButton.click();

    await displayedDialogTitle(SOLVER);
    await displayedText(SELECTIVITY);
    await displayedText(MODE);
    await displayedButton(START_SOLVER);

    const cancelButton = await displayedButton(["Cancel", "キャンセル"]);
    await cancelButton.click();
    await waitForDialogToClose(SOLVER);
  });
});
