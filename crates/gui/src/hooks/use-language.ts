import { useTranslation } from "react-i18next";
import { useReversiStore } from "@/stores/use-reversi-store";
import { changeLanguage, resolveLanguage, isLanguage } from "@/i18n";

export function useLanguage() {
  const { i18n } = useTranslation();
  const savedLanguage = useReversiStore((state) => state.language);
  const setLanguagePreference = useReversiStore((state) => state.setLanguagePreference);
  const language = savedLanguage ?? "auto";

  const setLanguage = async (value: string) => {
    const newSavedLanguage = value === "auto" ? null :
      (isLanguage(value) ? value : null);

    try {
      const resolved = await changeLanguage(newSavedLanguage);
      const saved = await setLanguagePreference(newSavedLanguage);
      if (!saved) {
        console.warn("Language preference could not be saved");
      }
      return resolved;
    } catch (error) {
      console.error("Failed to change language:", error);
      if (isLanguage(i18n.language)) {
        return i18n.language;
      }
      return resolveLanguage(savedLanguage);
    }
  };

  return { language, setLanguage };
}
