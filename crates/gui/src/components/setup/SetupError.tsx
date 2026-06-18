import { useTranslation } from "react-i18next";
import { useReversiStore } from "@/stores/use-reversi-store";

/**
 * The setup error has exactly one source (the setup slice) and one
 * interpretation (the `illegalMove:` / i18n-key mapping below). This widget
 * owns both behind a zero-prop interface, so the three setup tabs render
 * `<SetupError />` without each re-reading `state.setupError` and threading
 * it in — the error's source/display has one home.
 */
export function SetupError() {
  const { t } = useTranslation();
  const error = useReversiStore((state) => state.setupError);

  if (!error) return null;

  const message = error.startsWith("illegalMove:")
    ? t("setup.error.illegalMove", { move: error.split(":")[1] })
    : t(`setup.error.${error}`);

  return <p className="text-sm text-destructive">{message}</p>;
}
