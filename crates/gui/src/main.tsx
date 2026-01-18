import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { loadSettings } from "./lib/settings-store";
import { initI18n } from "./i18n";

async function bootstrap() {
  try {
    const settings = await loadSettings();
    await initI18n(settings.language);

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
