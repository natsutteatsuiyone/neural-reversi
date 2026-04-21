import { useCallback, useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { useTranslation } from "react-i18next";
import { Info } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useReversiStore } from "@/stores/use-reversi-store";

type TabValue = "license" | "thirdParty";

function AboutModalContent() {
  const { t } = useTranslation();
  const [version, setVersion] = useState<string | null>(null);
  const [licenseText, setLicenseText] = useState<string | null>(null);
  const [thirdPartyText, setThirdPartyText] = useState<string | null>(null);
  const [tab, setTab] = useState<TabValue>("license");

  useEffect(() => {
    void invoke<string>("get_app_version").then(setVersion).catch(() => {
      setVersion("");
    });
    void invoke<string>("get_license_text")
      .then(setLicenseText)
      .catch((err) => {
        console.error("Failed to load license text:", err);
        setLicenseText(t("about.loadFailed"));
      });
    void invoke<string>("get_third_party_licenses_text")
      .then(setThirdPartyText)
      .catch((err) => {
        console.error("Failed to load third-party licenses:", err);
        setThirdPartyText(t("about.loadFailed"));
      });
  }, [t]);

  const handleTabChange = (next: string) => {
    setTab(next as TabValue);
  };

  return (
    <DialogContent
      aria-describedby={undefined}
      className="bg-card border-white/10 sm:max-w-2xl"
    >
      <DialogHeader>
        <DialogTitle className="flex items-baseline gap-2 text-xl text-foreground">
          <Info className="w-5 h-5 self-center text-accent-blue" />
          <span>Neural Reversi</span>
          {version && (
            <span className="text-base font-normal text-foreground-muted">v{version}</span>
          )}
        </DialogTitle>
      </DialogHeader>

      <Tabs value={tab} onValueChange={handleTabChange} className="py-2">
        <TabsList className="w-full">
          <TabsTrigger value="license" className="flex-1">
            {t("about.tabs.license")}
          </TabsTrigger>
          <TabsTrigger value="thirdParty" className="flex-1">
            {t("about.tabs.thirdParty")}
          </TabsTrigger>
        </TabsList>

        <TabsContent value="license" className="mt-4">
          <LicenseViewer text={licenseText} placeholder={t("about.loading")} />
        </TabsContent>

        <TabsContent value="thirdParty" className="mt-4">
          <LicenseViewer text={thirdPartyText} placeholder={t("about.loading")} />
        </TabsContent>
      </Tabs>
    </DialogContent>
  );
}

function LicenseViewer({ text, placeholder }: { text: string | null; placeholder: string }) {
  return (
    <pre className="h-[60vh] overflow-auto whitespace-pre-wrap break-words rounded-md border border-white/10 bg-background-secondary p-3 font-mono text-xs text-foreground-secondary">
      {text ?? placeholder}
    </pre>
  );
}

export function AboutModal() {
  const isOpen = useReversiStore((state) => state.isAboutModalOpen);
  const closeAboutModal = useReversiStore((state) => state.closeAboutModal);

  const handleOpenChange = useCallback(
    (open: boolean) => {
      if (!open) closeAboutModal();
    },
    [closeAboutModal],
  );

  return (
    <Dialog open={isOpen} onOpenChange={handleOpenChange}>
      {isOpen && <AboutModalContent />}
    </Dialog>
  );
}
