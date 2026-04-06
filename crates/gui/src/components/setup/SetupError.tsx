import { useTranslation } from "react-i18next";

interface SetupErrorProps {
  error: string | null;
}

export function SetupError({ error }: SetupErrorProps) {
  const { t } = useTranslation();

  if (!error) return null;

  const message = error.startsWith("illegalMove:")
    ? t("setup.error.illegalMove", { move: error.split(":")[1] })
    : t(`setup.error.${error}`);

  return <p className="text-sm text-red-400">{message}</p>;
}
