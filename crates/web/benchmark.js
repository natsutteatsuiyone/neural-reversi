import init, { BenchmarkRunner } from './dist/pkg/reversi_web.js';

// Constants
const UI_UPDATE_DELAY = 100;
const STATUS_HIDE_DELAY_SUCCESS = 2000;
const STATUS_HIDE_DELAY_ERROR = 3000;

// State
let benchmarkRunner = null;
let isRunning = false;

// DOM elements
const elements = {
  loading: document.getElementById('loading'),
  controls: document.getElementById('benchmark-controls'),
  status: document.getElementById('status'),
  results: document.getElementById('results'),
  resultsContainer: document.getElementById('results-container'),
  inputs: {
    iterations: document.getElementById('iterations'),
    searchDepth: document.getElementById('search-depth'),
    searchIterations: document.getElementById('search-iterations'),
    endgameIterations: document.getElementById('endgame-iterations'),
    perftDepth: document.getElementById('perft-depth'),
  },
  buttons: {
    runAll: document.getElementById('run-all'),
    moveGen: document.getElementById('run-move-gen'),
    eval: document.getElementById('run-eval'),
    search: document.getElementById('run-search'),
    endgame: document.getElementById('run-endgame'),
    perft: document.getElementById('run-perft'),
  },
};

// Benchmark definitions
const benchmarks = {
  moveGen: {
    name: 'Move Generation Benchmark',
    buttonId: 'moveGen',
    run: (runner, iterations) => runner.bench_move_generation(iterations),
  },
  eval: {
    name: 'Evaluation Benchmark',
    buttonId: 'eval',
    run: (runner, iterations) => runner.bench_evaluation(iterations),
  },
  search: {
    name: 'Search Benchmark',
    buttonId: 'search',
    run: (runner, iterations, depth) => runner.bench_search(depth, iterations),
    usesSearchParams: true,
  },
  endgame: {
    name: 'Endgame Search Benchmark',
    buttonId: 'endgame',
    run: (runner, iterations) => runner.bench_endgame(iterations),
    usesEndgameParams: true,
  },
  perft: {
    name: 'Perft Benchmark',
    buttonId: 'perft',
    run: (runner, depth) => runner.bench_perft(depth, 1),
    usesPerftParams: true,
  },
};

// Initialize WASM module
async function initializeWasm() {
  try {
    await init();
    benchmarkRunner = new BenchmarkRunner();

    elements.loading.classList.add('hidden');
    elements.controls.classList.remove('hidden');

    showStatus('Initialization complete', 'success');
    setTimeout(() => hideStatus(), STATUS_HIDE_DELAY_SUCCESS);
  } catch (error) {
    console.error('Failed to initialize WASM:', error);
    elements.loading.innerHTML = `
      <p style="color: #e74c3c;">Failed to initialize WASM module</p>
      <p style="color: #666; font-size: 0.9rem; margin-top: 1rem;">Error: ${error.message}</p>
    `;
  }
}

function showStatus(message, type = 'running') {
  elements.status.textContent = message;
  elements.status.className = `status ${type}`;
  elements.status.classList.remove('hidden');
}

function hideStatus() {
  elements.status.classList.add('hidden');
}

function formatNumber(num) {
  if (num >= 1000000) {
    return (num / 1000000).toFixed(2) + 'M';
  } else if (num >= 1000) {
    return (num / 1000).toFixed(2) + 'K';
  }
  return num.toFixed(2);
}

function displayResult(result) {
  const resultItem = document.createElement('div');
  resultItem.className = 'result-item';

  resultItem.innerHTML = `
    <div class="result-header">
      <div class="result-name">${result.name}</div>
      <div class="result-badge">${result.iterations.toLocaleString()} iterations</div>
    </div>
    <div class="result-stats">
      <div class="stat">
        <div class="stat-label">Total Time</div>
        <div class="stat-value">${result.total_time_ms.toFixed(2)} ms</div>
      </div>
      <div class="stat">
        <div class="stat-label">Average Time</div>
        <div class="stat-value">${result.avg_time_us.toFixed(2)} μs</div>
      </div>
      <div class="stat">
        <div class="stat-label">Throughput</div>
        <div class="stat-value">${formatNumber(result.ops_per_sec)} ops/s</div>
      </div>
    </div>
  `;

  elements.resultsContainer.appendChild(resultItem);
  elements.results.classList.remove('hidden');
}

function clearResults() {
  elements.resultsContainer.innerHTML = '';
}

function setButtonsEnabled(enabled) {
  Object.values(elements.buttons).forEach(btn => btn.disabled = !enabled);
}

async function runBenchmark(name, benchmarkFn) {
  if (isRunning) return;

  isRunning = true;
  setButtonsEnabled(false);
  showStatus(`Running ${name}...`, 'running');

  try {
    // Use setTimeout to allow UI to update
    await new Promise(resolve => setTimeout(resolve, UI_UPDATE_DELAY));

    const result = await benchmarkFn();
    displayResult(result);

    showStatus(`${name} completed`, 'success');
    setTimeout(() => hideStatus(), STATUS_HIDE_DELAY_SUCCESS);
  } catch (error) {
    console.error(`Benchmark failed:`, error);
    showStatus(`Error: ${error.message}`, 'error');
    setTimeout(() => hideStatus(), STATUS_HIDE_DELAY_ERROR);
  } finally {
    isRunning = false;
    setButtonsEnabled(true);
  }
}

function getInputValues() {
  return {
    iterations: parseInt(elements.inputs.iterations.value, 10),
    searchDepth: parseInt(elements.inputs.searchDepth.value, 10),
    searchIterations: parseInt(elements.inputs.searchIterations.value, 10),
    endgameIterations: parseInt(elements.inputs.endgameIterations.value, 10),
    perftDepth: parseInt(elements.inputs.perftDepth.value, 10),
  };
}

async function runAllBenchmarks() {
  if (isRunning) return;

  isRunning = true;
  setButtonsEnabled(false);
  clearResults();

  const { iterations, searchDepth, searchIterations, endgameIterations, perftDepth } = getInputValues();
  const benchmarkOrder = ['moveGen', 'eval', 'search', 'endgame', 'perft'];

  try {
    for (const key of benchmarkOrder) {
      const benchmark = benchmarks[key];
      const displayName = benchmark.name.replace(' Benchmark', '').toLowerCase();

      showStatus(`Running ${displayName} benchmark...`, 'running');
      await new Promise(resolve => setTimeout(resolve, UI_UPDATE_DELAY));

      let result;
      if (benchmark.usesSearchParams) {
        result = benchmark.run(benchmarkRunner, searchIterations, searchDepth);
      } else if (benchmark.usesEndgameParams) {
        result = benchmark.run(benchmarkRunner, endgameIterations);
      } else if (benchmark.usesPerftParams) {
        result = benchmark.run(benchmarkRunner, perftDepth);
      } else {
        result = benchmark.run(benchmarkRunner, iterations);
      }

      displayResult(result);
    }

    showStatus('All benchmarks completed!', 'success');
    setTimeout(() => hideStatus(), STATUS_HIDE_DELAY_ERROR);
  } catch (error) {
    console.error('Benchmarks failed:', error);
    showStatus(`Error: ${error.message}`, 'error');
    setTimeout(() => hideStatus(), STATUS_HIDE_DELAY_ERROR);
  } finally {
    isRunning = false;
    setButtonsEnabled(true);
  }
}

// Setup event listeners
function setupEventListeners() {
  // Run all benchmarks
  elements.buttons.runAll.addEventListener('click', () => {
    clearResults();
    runAllBenchmarks();
  });

  // Setup individual benchmark buttons
  Object.entries(benchmarks).forEach(([key, benchmark]) => {
    elements.buttons[benchmark.buttonId].addEventListener('click', () => {
      clearResults();
      const { iterations, searchDepth, searchIterations, endgameIterations, perftDepth } = getInputValues();

      const benchmarkFn = () => {
        if (benchmark.usesSearchParams) {
          return benchmark.run(benchmarkRunner, searchIterations, searchDepth);
        } else if (benchmark.usesEndgameParams) {
          return benchmark.run(benchmarkRunner, endgameIterations);
        } else if (benchmark.usesPerftParams) {
          return benchmark.run(benchmarkRunner, perftDepth);
        } else {
          return benchmark.run(benchmarkRunner, iterations);
        }
      };

      runBenchmark(benchmark.name, benchmarkFn);
    });
  });
}

// Initialize when page loads
setupEventListeners();
initializeWasm();
