use std::collections::BTreeSet;
use std::fs;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

const INITIAL_POSITION: &str =
    "---------------------------OX------XO--------------------------- X\n";
static NEXT_FIXTURE_ID: AtomicUsize = AtomicUsize::new(0);

#[test]
fn solve_all_moves_prints_every_legal_root_move() {
    let stdout = run_solve_all_moves(INITIAL_POSITION);
    let rows = move_rows(&stdout);
    let root_moves: BTreeSet<&str> = rows.iter().map(|(_, square)| *square).collect();
    assert_eq!(
        rows.len(),
        root_moves.len(),
        "duplicate root move\n{stdout}"
    );

    assert_eq!(
        root_moves,
        BTreeSet::from(["c4", "d3", "e6", "f5"]),
        "all-moves output should include every legal opening move\n{stdout}"
    );

    // A1 is White's only move and loses every remaining disc. Exercise the
    // output parser with a real negative score, not a synthetic string.
    let losing = format!("-XO{} O\n", "X".repeat(61));
    let stdout = run_solve_all_moves(&losing);
    assert_eq!(move_rows(&stdout), vec![(-58, "A1")]);
}

#[test]
fn solve_all_moves_prints_a_separate_table_for_each_position() {
    let positions = format!("{INITIAL_POSITION}{INITIAL_POSITION}");
    let stdout = run_solve_all_moves(&positions);
    let table_count = stdout
        .lines()
        .filter(|line| line.contains("Principal Variation"))
        .count();

    assert_eq!(
        table_count, 2,
        "all-moves output should print one table per input position\n{stdout}"
    );
}

#[test]
fn solve_all_moves_prints_metadata_outside_the_move_table() {
    let stdout = run_solve_all_moves(INITIAL_POSITION);
    let table_header = stdout
        .lines()
        .find(|line| line.contains("Principal Variation"))
        .expect("table header");

    assert!(
        !table_header.contains("Time")
            && !table_header.contains("Nodes")
            && !table_header.contains("N/s"),
        "move table should not contain search metrics columns\n{stdout}"
    );
    assert!(
        stdout.lines().any(|line| {
            line.trim_start().starts_with("Time: ")
                && line.contains("Nodes: ")
                && line.contains("N/s: ")
        }),
        "search metrics should be printed outside the move table\n{stdout}"
    );
    assert!(
        !table_header.contains('#') && !table_header.contains("Depth"),
        "move table should not contain columns repeated for every PV row\n{stdout}"
    );
    assert!(
        stdout
            .lines()
            .any(|line| line.starts_with("Position #1  Depth: ")),
        "position number and depth should be printed outside the move table\n{stdout}"
    );
}

fn run_solve_all_moves(positions: &str) -> String {
    let mut path = std::env::temp_dir();
    let fixture_id = NEXT_FIXTURE_ID.fetch_add(1, Ordering::Relaxed);
    path.push(format!(
        "neural-reversi-solve-all-moves-{}-{fixture_id}.obf",
        std::process::id()
    ));
    fs::write(&path, positions).expect("write position fixture");

    let output = Command::new(env!("CARGO_BIN_EXE_cli"))
        .args([
            "solve",
            "--hash-size",
            "1",
            "--threads",
            "1",
            "--level",
            "1",
            "--all-moves",
        ])
        .arg(&path)
        .output()
        .expect("run cli solve");

    let _ = fs::remove_file(&path);

    assert!(
        output.status.success(),
        "solve failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    String::from_utf8(output.stdout).expect("stdout utf8")
}

fn move_rows(stdout: &str) -> Vec<(i32, &str)> {
    let mut rows = Vec::new();
    for row in stdout
        .lines()
        .filter(|line| line.trim_start().starts_with('|'))
    {
        let columns: Vec<_> = row.split('|').map(str::trim).collect();
        assert_eq!(columns.len(), 4, "unexpected table row: {row}");
        let score = columns[1];
        if score == "Score" || score.chars().all(|ch| matches!(ch, '-' | ':')) {
            continue;
        }
        let score = score
            .parse()
            .unwrap_or_else(|_| panic!("invalid score row: {row}"));
        let square = columns[2]
            .split_whitespace()
            .next()
            .expect("PV must contain a move");
        rows.push((score, square));
    }
    rows
}
