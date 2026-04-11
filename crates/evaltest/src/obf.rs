//! OBF (Othello Board Format) file parser
//!
//! Parses `.obf` files where each line has the format:
//! `<board64> <side>; <move1>:<score1>; <move2>:<score2>; ...`

use reversi_core::disc::Disc;
use std::path::Path;

use crate::test_case::TestCase;

/// A named collection of test cases loaded from a single OBF file
pub struct ProblemSet {
    pub name: String,
    pub cases: Vec<TestCase>,
}

/// A single move with its score from the OBF file
struct MoveScore {
    move_name: String,
    score: i32,
}

/// Parse all `.obf` files matching the given problem specifiers.
///
/// Each specifier can be:
/// - A preset name: "fforum", "hard-20", "hard-25", "hard-30"
/// - A file path ending in ".obf"
///
/// Returns problem sets in the order specified.
pub fn load_problems(specifiers: &[String], problem_dir: &Path) -> Vec<ProblemSet> {
    let mut result = Vec::new();
    for spec in specifiers {
        let paths = resolve_specifier(spec, problem_dir);
        for path in paths {
            let name = path
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            match parse_obf_file(&path) {
                Ok(cases) => result.push(ProblemSet { name, cases }),
                Err(e) => eprintln!("Warning: Failed to load {}: {e}", path.display()),
            }
        }
    }
    result
}

/// Load all `.obf` files from the problem directory
pub fn load_all_problems(problem_dir: &Path) -> Vec<ProblemSet> {
    let mut entries: Vec<_> = std::fs::read_dir(problem_dir)
        .unwrap_or_else(|e| {
            panic!(
                "Cannot read problem directory {}: {e}",
                problem_dir.display()
            )
        })
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("obf") {
                Some(path)
            } else {
                None
            }
        })
        .collect();
    entries.sort();

    let mut result = Vec::new();
    for path in entries {
        let name = path
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        match parse_obf_file(&path) {
            Ok(cases) => result.push(ProblemSet { name, cases }),
            Err(e) => eprintln!("Warning: Failed to load {}: {e}", path.display()),
        }
    }
    result
}

/// Resolve a specifier to one or more file paths
fn resolve_specifier(spec: &str, problem_dir: &Path) -> Vec<std::path::PathBuf> {
    // Direct file path
    if spec.ends_with(".obf") {
        let path = Path::new(spec);
        if path.exists() {
            return vec![path.to_path_buf()];
        }
        let in_dir = problem_dir.join(spec);
        if in_dir.exists() {
            return vec![in_dir];
        }
        eprintln!("Warning: OBF file not found: {spec}");
        return vec![];
    }

    // Preset name
    match spec {
        "fforum" => {
            let mut paths: Vec<_> = [
                "fforum-1-19.obf",
                "fforum-20-39.obf",
                "fforum-40-59.obf",
                "fforum-60-79.obf",
            ]
            .iter()
            .filter_map(|name| {
                let p = problem_dir.join(name);
                if p.exists() { Some(p) } else { None }
            })
            .collect();
            paths.sort();
            paths
        }
        preset => {
            let filename = format!("{preset}.obf");
            let path = problem_dir.join(&filename);
            if path.exists() {
                vec![path]
            } else {
                eprintln!("Warning: Problem set not found: {spec}");
                vec![]
            }
        }
    }
}

/// Parse a single OBF file into test cases
fn parse_obf_file(path: &Path) -> Result<Vec<TestCase>, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Cannot read {}: {e}", path.display()))?;

    let mut cases = Vec::new();
    for (line_idx, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let case = parse_obf_line(line, line_idx + 1)
            .map_err(|e| format!("{}:{}: {e}", path.display(), line_idx + 1))?;
        cases.push(case);
    }
    Ok(cases)
}

/// Parse a single OBF line into a TestCase
fn parse_obf_line(line: &str, line_number: usize) -> Result<TestCase, String> {
    // Split on ';' to get segments: ["<board> <side>", " <move1>:<score1>", ...]
    let segments: Vec<&str> = line.split(';').collect();
    if segments.len() < 2 {
        return Err("Expected at least board and one move".to_string());
    }

    // Parse board and side
    let header = segments[0].trim();
    let (board_str, side_to_move) = parse_board_header(header)?;

    // Parse move-score pairs
    let mut move_scores = Vec::new();
    let mut pass_score = None;
    for segment in &segments[1..] {
        let segment = segment.trim();
        if segment.is_empty() {
            continue;
        }
        if let Some(score) = parse_pass_score(segment) {
            pass_score = Some(score);
            continue;
        }
        if let Some(ms) = parse_move_score(segment)? {
            move_scores.push(ms);
        }
    }

    if let Some(score) = pass_score {
        return Ok(TestCase::new(
            line_number,
            board_str,
            side_to_move,
            score,
            vec![],
            vec![],
            vec![],
            vec![],
        ));
    }

    if move_scores.is_empty() {
        return Err("No valid moves found".to_string());
    }

    // Group moves by score rank
    let expected_score = move_scores[0].score;
    let (best, second_best, third_best) = group_moves_by_score(&move_scores);
    let all_move_scores: Vec<(String, i32)> = move_scores
        .iter()
        .map(|ms| (ms.move_name.clone(), ms.score))
        .collect();

    Ok(TestCase::new(
        line_number,
        board_str,
        side_to_move,
        expected_score,
        best,
        second_best,
        third_best,
        all_move_scores,
    ))
}

/// Parse the board header: "<64-char board> <X|O>"
fn parse_board_header(header: &str) -> Result<(String, Disc), String> {
    if !header.is_ascii() {
        return Err(format!(
            "Board header contains non-ASCII characters: '{header}'"
        ));
    }
    if header.len() < 66 {
        return Err(format!("Board header too short: '{header}'"));
    }
    let board_str = &header[..64];
    let side_char = header[64..].trim();
    let side = match side_char {
        "X" => Disc::Black,
        "O" => Disc::White,
        _ => return Err(format!("Invalid side to move: '{side_char}'")),
    };
    Ok((board_str.to_string(), side))
}

/// Parse a score string: strips leading '+' and parses as i32
fn parse_score(s: &str) -> Result<i32, String> {
    s.trim()
        .trim_start_matches('+')
        .parse()
        .map_err(|e| format!("Invalid score '{s}': {e}"))
}

/// Parse a "PS:score" segment for pass positions, returning the score if matched
fn parse_pass_score(segment: &str) -> Option<i32> {
    let (key, value) = segment.split_once(':')?;
    if key.trim().eq_ignore_ascii_case("PS") {
        parse_score(value).ok()
    } else {
        None
    }
}

/// Parse a single "move:score" segment, returning None for sentinel entries
fn parse_move_score(segment: &str) -> Result<Option<MoveScore>, String> {
    let (move_name, score_str) = segment
        .split_once(':')
        .ok_or_else(|| format!("Invalid move:score format: '{segment}'"))?;
    let move_name = move_name.trim();

    // Skip sentinel entries like "--:-127"
    if move_name == "--" {
        return Ok(None);
    }

    let score = parse_score(score_str)?;

    Ok(Some(MoveScore {
        move_name: move_name.to_string(),
        score,
    }))
}

/// Group moves by score into best/second-best/third-best.
/// Moves with the same score share the same rank.
///
/// Assumes `move_scores` is sorted by descending score (as per OBF format).
fn group_moves_by_score(move_scores: &[MoveScore]) -> (Vec<String>, Vec<String>, Vec<String>) {
    let mut best = Vec::new();
    let mut second_best = Vec::new();
    let mut third_best = Vec::new();

    if move_scores.is_empty() {
        return (best, second_best, third_best);
    }

    let mut current_rank = 0;
    let mut current_score = move_scores[0].score;

    for ms in move_scores {
        if ms.score != current_score {
            current_rank += 1;
            current_score = ms.score;
        }
        match current_rank {
            0 => best.push(ms.move_name.clone()),
            1 => second_best.push(ms.move_name.clone()),
            2 => third_best.push(ms.move_name.clone()),
            _ => {} // ignore lower-ranked moves
        }
    }

    (best, second_best, third_best)
}

/// Find the problem directory by searching standard locations
pub fn find_problem_dir() -> Option<std::path::PathBuf> {
    // 1. Next to the executable
    if let Ok(exe) = std::env::current_exe() {
        let dir = exe.parent().map(|p| p.join("problem"));
        if let Some(ref d) = dir
            && d.is_dir()
        {
            return dir;
        }
    }

    // 2. Current working directory
    let cwd = std::path::PathBuf::from("problem");
    if cwd.is_dir() {
        return Some(cwd);
    }

    // 3. Environment variable
    if let Ok(dir) = std::env::var("EVALTEST_PROBLEM_DIR") {
        let path = std::path::PathBuf::from(dir);
        if path.is_dir() {
            return Some(path);
        }
    }

    None
}
