import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { PlayerCard } from "../PlayerCard";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string) =>
      ({
        "ai.thinking": "Thinking...",
        "game.resume": "Resume",
        "game.start": "Start",
        "game.stop": "Stop",
        "player.ai": "AI",
        "player.level": "Lv.",
        "player.you": "You",
      })[key] ?? key,
  }),
}));

const baseProps = {
  color: "black" as const,
  score: 2,
  isCurrent: true,
  isAIControlled: true,
  aiLevel: 7,
  isThinking: false,
  aiMode: "level",
  aiRemainingTime: 65_000,
};

function renderCard(props: Partial<Parameters<typeof PlayerCard>[0]> = {}): string {
  return renderToStaticMarkup(createElement(PlayerCard, { ...baseProps, ...props }));
}

describe("PlayerCard", () => {
  it("labels a non-thinking level AI player", () => {
    expect(renderCard()).toContain("AI Lv.7");
  });

  it("shows the thinking label and Stop action during an AI search", () => {
    const markup = renderCard({ isThinking: true, onStop: vi.fn(), aiMode: "game-time" });

    expect(markup).toContain("Thinking...");
    expect(markup).toContain("Stop");
    expect(markup).toContain("1:05");
  });

  it("shows Resume instead of the timer when an AI turn is paused", () => {
    const markup = renderCard({ onResume: vi.fn(), aiMode: "game-time" });

    expect(markup).toContain("Resume");
    expect(markup).not.toContain("1:05");
  });

  it("uses Start for the initial paused AI turn", () => {
    expect(renderCard({ onResume: vi.fn(), resumeIsStart: true })).toContain("Start");
  });

  it("uses the supplied human player label", () => {
    expect(
      renderCard({
        isAIControlled: false,
        playerLabel: "Guest",
      }),
    ).toContain("Guest");
  });
});
