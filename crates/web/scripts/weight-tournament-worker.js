import { readFileSync } from "fs";
import { importPreferredWasmModule } from "../wasm-loader.js";
import { playMatch } from "./weight-tournament.js";

let WeightMatchRunner = null;
let openings = [];
const weightCache = new Map();

function serializeError(error) {
  return error instanceof Error ? (error.stack ?? error.message) : String(error);
}

function loadWeight(weight) {
  const cached = weightCache.get(weight.path);
  if (cached !== undefined) {
    return cached;
  }

  const loaded = {
    ...weight,
    bytes: new Uint8Array(readFileSync(weight.path)),
  };
  weightCache.set(weight.path, loaded);
  return loaded;
}

self.onmessage = async (event) => {
  const message = event.data;

  try {
    if (message.type === "init") {
      openings = message.openings;
      const { module, relaxedSimd } = await importPreferredWasmModule({
        relaxedPath: "./pkg-node-relaxed/web.js",
        fallbackPath: "./pkg-node/web.js",
      });
      WeightMatchRunner = module.WeightMatchRunner;
      if (!WeightMatchRunner) {
        throw new Error("WeightMatchRunner is missing; rebuild with `bun run build:wasm:node`");
      }

      self.postMessage({ relaxedSimd, type: "ready" });
      return;
    }

    if (message.type === "match") {
      if (!WeightMatchRunner) {
        throw new Error("worker received a match before initialization");
      }

      self.postMessage({
        id: message.id,
        result: playMatch(
          WeightMatchRunner,
          loadWeight(message.engine1),
          loadWeight(message.engine2),
          openings,
        ),
        type: "result",
      });
    }
  } catch (error) {
    self.postMessage({
      error: serializeError(error),
      id: message.id,
      type: "error",
    });
  }
};
