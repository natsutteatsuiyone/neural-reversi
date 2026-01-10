import { load, type Store } from "@tauri-apps/plugin-store";

export interface AppSettings {
    gameMode: string;
    aiLevel: number;
    aiMode: string;
    timeLimit: number;
    gameTimeLimit: number;
    hintLevel: number;
    aiAnalysisPanelOpen: boolean;
}

const DEFAULT_SETTINGS: AppSettings = {
    gameMode: "ai-white",
    aiLevel: 21,
    aiMode: "game-time",
    timeLimit: 1,
    gameTimeLimit: 60,
    hintLevel: 21,
    aiAnalysisPanelOpen: false,
};

let store: Store | null = null;

async function getStore(): Promise<Store> {
    if (!store) {
        store = await load("settings.json", { autoSave: true, defaults: {} });
    }
    return store;
}

export async function loadSettings(): Promise<AppSettings> {
    try {
        const s = await getStore();
        const gameMode = await s.get<string>("gameMode") ?? DEFAULT_SETTINGS.gameMode;
        const aiLevel = await s.get<number>("aiLevel") ?? DEFAULT_SETTINGS.aiLevel;
        const aiMode = await s.get<string>("aiMode") ?? DEFAULT_SETTINGS.aiMode;
        const timeLimit = await s.get<number>("timeLimit") ?? DEFAULT_SETTINGS.timeLimit;
        const gameTimeLimit = await s.get<number>("gameTimeLimit") ?? DEFAULT_SETTINGS.gameTimeLimit;
        const hintLevel = await s.get<number>("hintLevel") ?? DEFAULT_SETTINGS.hintLevel;
        const aiAnalysisPanelOpen = await s.get<boolean>("aiAnalysisPanelOpen") ?? DEFAULT_SETTINGS.aiAnalysisPanelOpen;

        return { gameMode, aiLevel, aiMode, timeLimit, gameTimeLimit, hintLevel, aiAnalysisPanelOpen };
    } catch (error) {
        console.error("Failed to load settings:", error);
        return DEFAULT_SETTINGS;
    }
}

export async function saveSettings(settings: Partial<AppSettings>): Promise<void> {
    try {
        const s = await getStore();
        for (const [key, value] of Object.entries(settings)) {
            await s.set(key, value);
        }
    } catch (error) {
        console.error("Failed to save settings:", error);
    }
}

export async function saveSetting<K extends keyof AppSettings>(
    key: K,
    value: AppSettings[K]
): Promise<void> {
    try {
        const s = await getStore();
        await s.set(key, value);
    } catch (error) {
        console.error(`Failed to save setting ${key}:`, error);
    }
}
