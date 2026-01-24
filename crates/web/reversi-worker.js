// reversi_web/reversi-worker.js
// trunk builds the WASM module and places it in the dist directory
// We dynamically import it when needed
let Game;
let initModule;

let game;
let searchStartTime;

self.onmessage = async (event) => {
  const { type, payload } = event.data;

  if (type === "init") {
    try {
      // Dynamically import the WASM module built by wasm-pack
      const module = await import('/dist/pkg/web.js');
      initModule = module.default;
      Game = module.Game;

      // Initialize the WASM module
      await initModule();

      game = new Game(payload.humanIsBlack);
      game.set_level(payload.level);
      game.set_progress_callback((progress) => {
        const elapsed = performance.now() - searchStartTime;
        const nps = elapsed > 0 ? progress.nodes / (elapsed / 1000) : 0;
        console.log("Search progress:", progress, "NPS:", nps.toFixed(0));
        self.postMessage({ type: "search_progress", payload: progress });
      });
      self.postMessage({ type: "initialized", payload: getGameState() });
    } catch (error) {
      console.error("Failed to initialize WASM module:", error);
      self.postMessage({ type: "error", payload: { message: error.message } });
    }
    return;
  }

  if (!game) {
    console.error("Worker not initialized yet.");
    return;
  }

  switch (type) {
    case "human_move": {
      const moved = game.human_move(payload.index);
      if (moved) {
        self.postMessage({ type: "state_updated", payload: getGameState() });
      }
      break;
    }
    case "ai_move": {
      searchStartTime = performance.now();
      const move = game.ai_move();
      self.postMessage({
        type: "ai_moved",
        payload: { move, gameState: getGameState() },
      });
      break;
    }
    case "pass": {
      const passed = game.pass();
      if (passed) {
        self.postMessage({ type: "state_updated", payload: getGameState() });
      }
      break;
    }
    case "reset": {
      game.reset(payload.humanIsBlack);
      game.set_level(payload.level);
      self.postMessage({ type: "state_updated", payload: getGameState() });
      break;
    }
    case "set_level": {
      game.set_level(payload.level);
      break;
    }
    case "get_state": {
      self.postMessage({ type: "state_updated", payload: getGameState() });
      break;
    }
    case "replay_moves": {
      // Reset and replay a sequence of moves
      game.reset(payload.humanIsBlack);
      game.set_level(payload.level);

      for (const move of payload.moves) {
        // Use make_move_unchecked to replay moves regardless of whose turn it is
        game.make_move_unchecked(move.index);
      }

      self.postMessage({ type: "replay_completed", payload: getGameState() });
      break;
    }
  }
};

function getGameState() {
  if (!game) return null;
  return {
    board: Array.from(game.board()),
    legalMoves: Array.from(game.legal_moves()),
    isGameOver: game.is_game_over(),
    currentPlayer: game.current_player(),
    humanColor: game.human_color(),
    aiColor: game.ai_color(),
    score: game.score(),
    emptyCount: game.empty_count(),
  };
}
