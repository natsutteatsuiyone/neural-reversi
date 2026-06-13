import { expect, test } from "@playwright/test";

async function waitForAppReady(page) {
  await page.goto("/");
  await expect(page.locator("#loading-overlay")).toBeHidden({
    timeout: 60_000,
  });
}

async function installFakeWorker(page) {
  await page.addInitScript(() => {
    const initialGameState = {
      board: Array(64).fill(0),
      legalMoves: [19, 26, 37, 44],
      isGameOver: false,
      currentPlayer: 1,
      humanColor: 1,
      aiColor: 2,
      score: [2, 2],
      emptyCount: 60,
    };
    initialGameState.board[27] = 2;
    initialGameState.board[28] = 1;
    initialGameState.board[35] = 1;
    initialGameState.board[36] = 2;

    window.__fakeWorkers = [];
    window.__initialGameState = initialGameState;
    window.Worker = class FakeWorker {
      constructor() {
        this.messages = [];
        this.onmessage = null;
        window.__fakeWorkers.push(this);
      }

      postMessage(message) {
        this.messages.push(message);
        if (message.type === "init") {
          this.emit({
            type: "initialized",
            payload: initialGameState,
            generation: message.generation,
          });
        }
        if (message.type === "reset") {
          this.emit({
            type: "state_updated",
            payload: initialGameState,
            generation: message.generation,
          });
        }
      }

      emit(message) {
        setTimeout(() => {
          this.onmessage?.({ data: message });
        }, 0);
      }
    };
  });
}

test("loads the game shell", async ({ page }) => {
  await waitForAppReady(page);

  await expect(page).toHaveTitle("Neural Reversi");
  await expect(page.getByRole("heading", { name: "Neural Reversi" })).toBeVisible();
  await expect(page.getByRole("dialog", { name: "ゲーム設定" })).toBeVisible();
  await expect(page.locator("#board-3d canvas")).toBeVisible();
  await expect(page.locator("#score-black")).toHaveText("2");
  await expect(page.locator("#score-white")).toHaveText("2");
  await expect(page.locator("#empty-count")).toHaveText("60");
});

test("starts a game and reopens settings", async ({ page }) => {
  await waitForAppReady(page);

  const settingsDialog = page.getByRole("dialog", { name: "ゲーム設定" });
  await expect(settingsDialog).toBeVisible();

  await settingsDialog.getByRole("button", { name: "ゲーム開始" }).click();
  await expect(settingsDialog).toBeHidden();
  await expect(page.locator("#new-game")).toBeEnabled();

  await page.locator("#new-game").click();
  await expect(settingsDialog).toBeVisible();
});

test("drops stale worker responses and does not start AI while settings are open", async ({
  page,
}) => {
  await installFakeWorker(page);
  await waitForAppReady(page);

  const settingsDialog = page.getByRole("dialog", { name: "ゲーム設定" });
  await settingsDialog.getByRole("button", { name: "ゲーム開始" }).click();
  await expect(settingsDialog).toBeHidden();

  await page.locator("#new-game").click();
  await expect(settingsDialog).toBeVisible();

  const resetGeneration = await page.evaluate(() => {
    const worker = window.__fakeWorkers[0];
    const resets = worker.messages.filter((message) => message.type === "reset");
    return resets[resets.length - 1].generation;
  });

  await page.evaluate((generation) => {
    const worker = window.__fakeWorkers[0];
    worker.emit({
      type: "state_updated",
      payload: {
        ...window.__initialGameState,
        currentPlayer: 2,
      },
      generation,
    });
  }, resetGeneration);
  await page.waitForTimeout(250);

  await expect(page.locator(".thinking-badge")).toHaveCount(0);
  let aiMoveRequests = await page.evaluate(
    () => window.__fakeWorkers[0].messages.filter((message) => message.type === "ai_move").length,
  );
  expect(aiMoveRequests).toBe(0);

  await page.evaluate((staleGeneration) => {
    const worker = window.__fakeWorkers[0];
    worker.emit({
      type: "ai_moved",
      payload: {
        move: 19,
        gameState: window.__initialGameState,
      },
      generation: staleGeneration,
    });
  }, resetGeneration - 1);
  await page.waitForTimeout(250);

  await expect(page.locator("#move-log-scroll li")).toHaveCount(0);
  await expect(page.locator("#empty-count")).toHaveText("60");

  await page.keyboard.press("Escape");
  await expect(settingsDialog).toBeHidden();
  await expect
    .poll(() =>
      page.evaluate(
        () =>
          window.__fakeWorkers[0].messages.filter((message) => message.type === "ai_move").length,
      ),
    )
    .toBe(1);
});

test("auto-hint toggle drives hints on every human turn", async ({ page }) => {
  await installFakeWorker(page);
  await waitForAppReady(page);

  const settingsDialog = page.getByRole("dialog", { name: "ゲーム設定" });
  await settingsDialog.getByRole("button", { name: "ゲーム開始" }).click();
  await expect(settingsDialog).toBeHidden();

  const hintToggle = page.locator("#hint-toggle");
  await expect(hintToggle).toBeVisible();
  await expect(hintToggle).not.toBeChecked();

  const hintBadge = page.locator(".thinking-badge", { hasText: "ヒントを計算中…" });
  const completeHint = async (generation) => {
    await page.evaluate((gen) => {
      const worker = window.__fakeWorkers[0];
      worker.emit({
        type: "hint_progress",
        payload: {
          depth: 4,
          score: 1.5,
          probcut: 73,
          nodes: 1000,
          bestMove: "d3",
          bestMoveIndex: 19,
        },
        generation: gen,
      });
      worker.emit({
        type: "hint_completed",
        payload: {
          hints: [
            { move: 19, score: 1.5 },
            { move: 26, score: -0.5 },
          ],
        },
        generation: gen,
      });
    }, generation);
  };
  const hintRequestCount = () =>
    page.evaluate(
      () => window.__fakeWorkers[0].messages.filter((message) => message.type === "hint").length,
    );

  await page.locator(".hint-toggle").click();
  await expect(hintToggle).toBeChecked();
  await expect(hintBadge).toBeVisible();
  const hintRequests = await page.evaluate(() =>
    window.__fakeWorkers[0].messages.filter((message) => message.type === "hint"),
  );
  expect(hintRequests).toHaveLength(1);
  const hintGeneration = hintRequests[0].generation;

  await completeHint(hintGeneration);
  await expect(hintBadge).toHaveCount(0);

  await page.evaluate((generation) => {
    const worker = window.__fakeWorkers[0];
    worker.emit({
      type: "state_updated",
      payload: window.__initialGameState,
      generation,
    });
  }, hintGeneration);
  await expect.poll(hintRequestCount).toBe(2);
  await expect(hintBadge).toBeVisible();

  await completeHint(hintGeneration);
  await expect(hintBadge).toHaveCount(0);

  await page.locator(".hint-toggle").click();
  await expect(hintToggle).not.toBeChecked();
  await page.evaluate((generation) => {
    const worker = window.__fakeWorkers[0];
    worker.emit({
      type: "state_updated",
      payload: window.__initialGameState,
      generation,
    });
  }, hintGeneration);
  await page.waitForTimeout(250);
  expect(await hintRequestCount()).toBe(2);

  await page.locator(".hint-toggle").click();
  await expect(hintToggle).toBeChecked();
  await expect(hintBadge).toBeVisible();
  await expect.poll(hintRequestCount).toBe(3);

  await page.locator(".hint-toggle").click();
  await expect(hintToggle).not.toBeChecked();
  await expect(hintBadge).toHaveCount(0);

  await completeHint(hintGeneration);
  await page.waitForTimeout(250);
  await expect(hintBadge).toHaveCount(0);
});
