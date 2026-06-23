// Shared WebDriver DOM helpers for the Tauri E2E specs. Locators match either
// the English or Japanese label so a spec passes regardless of the persisted
// language. Imported by both `specs/app.e2e.js` and `specs/launch-autostart.e2e.js`.

export function xpathLiteral(value) {
  if (!value.includes("'")) return `'${value}'`;
  if (!value.includes('"')) return `"${value}"`;

  // Both quote kinds present: splice the apostrophes back in as `"'"` terms.
  return `concat(${value
    .split("'")
    .map((part) => `'${part}'`)
    .join(`, "'", `)})`;
}

export function exactTextPredicate(labels) {
  return labels.map((label) => `normalize-space(.) = ${xpathLiteral(label)}`).join(" or ");
}

export function accessibleNamePredicate(labels) {
  return labels
    .map(
      (label) =>
        `@aria-label = ${xpathLiteral(label)} or normalize-space(.) = ${xpathLiteral(label)}`,
    )
    .join(" or ");
}

export async function firstDisplayed(xpath) {
  const elements = await $$(xpath);

  for (const element of elements) {
    if (await element.isDisplayed()) return element;
  }

  return null;
}

export async function waitForDisplayed(xpath, timeoutMsg, timeout = 10000) {
  let found = null;
  await browser.waitUntil(
    async () => {
      found = await firstDisplayed(xpath);
      return Boolean(found);
    },
    { timeout, timeoutMsg },
  );

  return found;
}

export async function waitForGone(xpath, timeoutMsg, timeout = 10000) {
  await browser.waitUntil(async () => !(await firstDisplayed(xpath)), {
    timeout,
    timeoutMsg,
  });
}

export function buttonXpath(labels) {
  return `//button[${accessibleNamePredicate(labels)}]`;
}

export function textXpath(labels) {
  return `//*[${exactTextPredicate(labels)}]`;
}

export async function displayedButton(labels, timeout) {
  return await waitForDisplayed(
    buttonXpath(labels),
    `Expected a visible button named one of: ${labels.join(", ")}`,
    timeout,
  );
}

export async function displayedText(labels, timeout) {
  return await waitForDisplayed(
    `//*[${exactTextPredicate(labels)}]`,
    `Expected visible text matching one of: ${labels.join(", ")}`,
    timeout,
  );
}

export async function setDesktopViewport() {
  await browser.setWindowSize(1200, 900);
}
