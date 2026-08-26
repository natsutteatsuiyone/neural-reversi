import React from "react";
import ReactDOM from "react-dom/client";
import App from "@/app/App";
import { initI18n } from "@/i18n";
import { hydrateReversiStore, useReversiStore } from "@/stores/use-reversi-store";

async function bootstrap() {
  try {
    await hydrateReversiStore(useReversiStore);
    await initI18n(useReversiStore.getState().language);

    ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
      <React.StrictMode>
        <App />
      </React.StrictMode>,
    );
  } catch (error) {
    console.error("Application failed to initialize:", error);
    const root = document.getElementById("root");
    if (root) {
      root.innerHTML = `
        <div style="padding: 20px; font-family: system-ui; color: #333;">
          <h1>Failed to start application</h1>
          <p>Please restart. If the problem persists, reset settings.</p>
        </div>
      `;
    }
  }
}

bootstrap();
