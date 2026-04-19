import { beforeEach, describe, expect, it, vi } from "vitest";

const { invokeMock, listenMock } = vi.hoisted(() => ({
  invokeMock: vi.fn(),
  listenMock: vi.fn(),
}));

vi.mock("@tauri-apps/api/core", () => ({
  invoke: invokeMock,
}));

vi.mock("@tauri-apps/api/event", () => ({
  listen: listenMock,
}));

import { TauriAIService } from "./tauri-ai-service";

describe("TauriAIService", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    listenMock.mockResolvedValue(vi.fn());
  });

  it("swallows abortSearch failures and logs them", async () => {
    const service = new TauriAIService();
    const error = new Error("abort failed");
    const consoleErrorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    invokeMock.mockRejectedValueOnce(error);

    await expect(service.abortSearch()).resolves.toBeUndefined();

    expect(invokeMock).toHaveBeenCalledWith("abort_ai_search_command");
    expect(consoleErrorSpy).toHaveBeenCalledWith("Failed to abort search:", error);

    consoleErrorSpy.mockRestore();
  });

  it("swallows abortGameAnalysis failures and logs them", async () => {
    const service = new TauriAIService();
    const error = new Error("abort failed");
    const consoleErrorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    invokeMock.mockRejectedValueOnce(error);

    await expect(service.abortGameAnalysis()).resolves.toBeUndefined();

    expect(invokeMock).toHaveBeenCalledWith("abort_game_analysis_command");
    expect(consoleErrorSpy).toHaveBeenCalledWith("Failed to abort game analysis:", error);

    consoleErrorSpy.mockRestore();
  });
});
