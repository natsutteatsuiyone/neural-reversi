import { useState, useEffect } from "react";
import { useTranslation } from "react-i18next";
import { saveSetting } from "@/lib/settings-store";
import { changeLanguage, resolveLanguage, isLanguage, type Language } from "@/i18n";

export function useLanguage() {
  const { i18n } = useTranslation();
  const [savedLanguage, setSavedLanguage] = useState<Language | null>(null);

  // Determine display value: "auto" if null, otherwise the saved language
  const language = savedLanguage ?? "auto";

  useEffect(() => {
    // Initialize from current i18n language
    // If the current language matches the detected language, assume "auto"
    const detected = resolveLanguage(null);
    if (i18n.language === detected) {
      setSavedLanguage(null);
    } else if (isLanguage(i18n.language)) {
      setSavedLanguage(i18n.language);
    }
  }, [i18n.language]);

  const setLanguage = async (value: string) => {
    const newSavedLanguage = value === "auto" ? null :
      (isLanguage(value) ? value : null);

    try {
      const resolved = await changeLanguage(newSavedLanguage);
      const saved = await saveSetting("language", newSavedLanguage);
      if (!saved) {
        console.warn("Language preference could not be saved");
      }
      setSavedLanguage(newSavedLanguage);
      return resolved;
    } catch (error) {
      console.error("Failed to change language:", error);
      return resolveLanguage(savedLanguage);
    }
  };

  return { language, setLanguage };
}
