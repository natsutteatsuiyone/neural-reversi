import {
  buttonXpath,
  displayedButton,
  firstDisplayed,
  setDesktopViewport,
  textXpath,
  waitForDisplayed,
  waitForGone,
} from "../support/dom.js";

// The e2e build runs under a dedicated Tauri identifier with an isolated data
// dir, into which the launcher seeds `gameMode: "ai-black"` (plus level 1 for a
// fast move) before the app starts (see `wdio.conf.js`), so the launch
// auto-start is the AI's turn first.
const START = ["Start", "開始"];
const STOP = ["Stop", "停止"];
const NO_MOVES = ["No moves yet", "まだ着手がありません"];
const AI_ANALYSIS = ["AI Analysis", "AI分析"];

describe("Launch auto-start", () => {
  beforeEach(async () => {
    await setDesktopViewport();
  });

  it("auto-starts a paused game with a Start button and no auto-move when the AI plays first", async () => {
    // The board renders, i.e. a game is live without the user opening New Game.
    const boardCanvas = await $("canvas");
    await boardCanvas.waitForDisplayed({
      timeout: 15000,
      timeoutMsg: "Expected the Reversi board canvas to be visible on launch.",
    });

    // AI is black and moves first, so the game starts paused: the AI card shows
    // a Start button — labelled Start (not Resume) because no move has been
    // played yet.
    await displayedButton(START, 15000);

    // The pause held: the AI did not auto-play (the move list is still empty and
    // the AI is not thinking, so there is no Stop button).
    await waitForDisplayed(textXpath(NO_MOVES), "Expected an empty move list on launch.");
    const stopButton = await firstDisplayed(buttonXpath(STOP));
    expect(stopButton).toBe(null);
  });

  it("plays the AI's first move only after the Start button is pressed", async () => {
    const startButton = await displayedButton(START, 15000);
    await startButton.click();

    // Pressing Start resumes the AI: the Start button goes away and the AI makes
    // its first move, so the empty-move-list placeholder disappears.
    await waitForGone(
      buttonXpath(START),
      "Expected the Start button to disappear after the AI resumes.",
    );
    await waitForGone(
      textXpath(NO_MOVES),
      "Expected the AI to play a move after Start was pressed.",
      30000,
    );

    const analysisButton = await displayedButton(AI_ANALYSIS, 10000);
    if ((await analysisButton.getAttribute("aria-expanded")) !== "true") {
      await analysisButton.click();
    }
    await waitForDisplayed(
      "//tbody/tr[count(td) = 3]",
      "Expected the thinking log to keep at least one AI search row after the move.",
      10000,
    );
  });
});
