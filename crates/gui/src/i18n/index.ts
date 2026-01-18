import i18n from "i18next";
import { initReactI18next } from "react-i18next";
import en from "./locales/en.json";
import ja from "./locales/ja.json";

export type Language = "en" | "ja";

export function isLanguage(value: string | null): value is Language {
  return value === "en" || value === "ja";
}

export function detectLanguage(): Language {
  const browserLang = navigator.language;
  return browserLang.startsWith("ja") ? "ja" : "en";
}

export function resolveLanguage(savedLanguage: string | null): Language {
  return isLanguage(savedLanguage) ? savedLanguage : detectLanguage();
}

export async function initI18n(savedLanguage: string | null): Promise<void> {
  const language = resolveLanguage(savedLanguage);

  await i18n.use(initReactI18next).init({
    resources: {
      en: { translation: en },
      ja: { translation: ja },
    },
    lng: language,
    fallbackLng: "en",
    interpolation: {
      escapeValue: false,
    },
  });
}

export async function changeLanguage(language: string | null): Promise<Language> {
  const resolved = resolveLanguage(language);
  await i18n.changeLanguage(resolved);
  return resolved;
}

export { i18n };
