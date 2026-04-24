//! Test cases and OBF (Othello Board Format) file loading.
//!
//! Per-line parsing lives in [`reversi_core::obf`]; this module wraps the
//! parsed [`ObfPosition`] in a [`TestCase`] (line number + invariant that
//! test data is present), and handles file/directory/preset discovery.

use std::fmt;
use std::path::{Path, PathBuf};

use reversi_core::{board::Board, disc::Disc, obf::ObfPosition, square::Square};

/// A single test case: an [`ObfPosition`] tagged with its source line number.
#[derive(Debug, Clone)]
pub struct TestCase {
    pub line_number: usize,
    position: ObfPosition,
    expected_score: i32,
}

impl fmt::Display for TestCase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let side = match self.side_to_move() {
            Disc::Black => "Black",
            Disc::White => "White",
            Disc::Empty => "Empty",
        };
        write!(
            f,
            "#{} ({side} to move, score: {})",
            self.line_number, self.expected_score
        )
    }
}

impl TestCase {
    /// Wrap an [`ObfPosition`] as a test case. Errors if the position carries
    /// no test data (no scored moves and no `PS:` score).
    pub(crate) fn new(line_number: usize, position: ObfPosition) -> Result<Self, String> {
        let expected_score = position
            .expected_score()
            .ok_or_else(|| "No valid moves found".to_string())?;
        Ok(Self {
            line_number,
            position,
            expected_score,
        })
    }

    pub fn is_pass(&self) -> bool {
        self.position.is_pass()
    }

    pub fn board(&self) -> Board {
        self.position.board
    }

    pub fn side_to_move(&self) -> Disc {
        self.position.side_to_move
    }

    pub fn expected_score(&self) -> i32 {
        self.expected_score
    }

    pub fn rank_of(&self, sq: Square) -> Option<usize> {
        self.position.rank_of(sq)
    }

    pub fn expected_score_for_move(&self, sq: Square) -> Option<i32> {
        self.position.score_of(sq)
    }

    /// Comma-separated list of best moves (those tied with the highest score).
    ///
    /// Uses `Debug` formatting (uppercase, e.g. `E6`) to match the PV
    /// output produced elsewhere in `evaltest`.
    pub fn best_moves_str(&self) -> String {
        self.position
            .best_moves()
            .map(|sq| format!("{sq:?}"))
            .collect::<Vec<_>>()
            .join(",")
    }

    /// Expected-move column string: `"PS"` for pass positions, otherwise
    /// [`best_moves_str`](Self::best_moves_str).
    pub fn expected_moves_str(&self) -> String {
        if self.is_pass() {
            "PS".to_string()
        } else {
            self.best_moves_str()
        }
    }
}

/// A named collection of test cases loaded from a single OBF file.
pub struct ProblemSet {
    pub name: String,
    pub cases: Vec<TestCase>,
}

/// Load `.obf` files matching the given specifiers. Each specifier is either
/// a preset name (`fforum`, `hard-20`, ...) or a file path ending in `.obf`.
pub fn load_problems(specifiers: &[String], problem_dir: &Path) -> Vec<ProblemSet> {
    specifiers
        .iter()
        .flat_map(|spec| resolve_specifier(spec, problem_dir))
        .filter_map(|path| load_problem_set(&path))
        .collect()
}

/// Load every `.obf` file in `problem_dir`, sorted by filename.
pub fn load_all_problems(problem_dir: &Path) -> Vec<ProblemSet> {
    let read_dir = match std::fs::read_dir(problem_dir) {
        Ok(rd) => rd,
        Err(e) => {
            eprintln!(
                "Warning: Cannot read problem directory {}: {e}",
                problem_dir.display()
            );
            return Vec::new();
        }
    };
    let mut entries: Vec<_> = read_dir
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            (path.extension().and_then(|e| e.to_str()) == Some("obf")).then_some(path)
        })
        .collect();
    entries.sort();

    entries.iter().filter_map(|p| load_problem_set(p)).collect()
}

fn load_problem_set(path: &Path) -> Option<ProblemSet> {
    match parse_obf_file(path) {
        Ok(cases) => Some(ProblemSet {
            name: path
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned(),
            cases,
        }),
        Err(e) => {
            eprintln!("Warning: Failed to load {}: {e}", path.display());
            None
        }
    }
}

fn resolve_specifier(spec: &str, problem_dir: &Path) -> Vec<PathBuf> {
    if spec.ends_with(".obf") {
        let direct = Path::new(spec);
        if direct.exists() {
            return vec![direct.to_path_buf()];
        }
        let in_dir = problem_dir.join(spec);
        if in_dir.exists() {
            return vec![in_dir];
        }
        eprintln!("Warning: OBF file not found: {spec}");
        return vec![];
    }

    if spec == "fforum" {
        return [
            "fforum-1-19.obf",
            "fforum-20-39.obf",
            "fforum-40-59.obf",
            "fforum-60-79.obf",
        ]
        .iter()
        .map(|name| problem_dir.join(name))
        .filter(|p| p.exists())
        .collect();
    }

    let path = problem_dir.join(format!("{spec}.obf"));
    if path.exists() {
        vec![path]
    } else {
        eprintln!("Warning: Problem set not found: {spec}");
        vec![]
    }
}

fn parse_obf_file(path: &Path) -> Result<Vec<TestCase>, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Cannot read {}: {e}", path.display()))?;

    let mut cases = Vec::new();
    for (line_idx, line) in content.lines().enumerate() {
        let line_number = line_idx + 1;
        let report = |e: String| format!("{}:{line_number}: {e}", path.display());
        let Some(pos) = ObfPosition::parse(line).map_err(report)? else {
            continue;
        };
        cases.push(TestCase::new(line_number, pos).map_err(report)?);
    }
    Ok(cases)
}

/// Find the problem directory by searching, in order:
/// 1. `<exe-dir>/problem`
/// 2. `./problem`
/// 3. `$EVALTEST_PROBLEM_DIR`
pub fn find_problem_dir() -> Option<PathBuf> {
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent().map(|p| p.join("problem"))
        && dir.is_dir()
    {
        return Some(dir);
    }

    let cwd = PathBuf::from("problem");
    if cwd.is_dir() {
        return Some(cwd);
    }

    if let Ok(dir) = std::env::var("EVALTEST_PROBLEM_DIR") {
        let path = PathBuf::from(dir);
        if path.is_dir() {
            return Some(path);
        }
    }

    None
}
