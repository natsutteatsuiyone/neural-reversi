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
    let root_moves: BTreeSet<&str> = stdout
        .lines()
        .filter_map(first_pv_token)
        .filter(|token| matches!(*token, "c4" | "d3" | "e6" | "f5"))
        .collect();

    assert_eq!(
        root_moves,
        BTreeSet::from(["c4", "d3", "e6", "f5"]),
        "all-moves output should include every legal opening move\n{stdout}"
    );
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
fn solve_all_moves_prints_search_metrics_outside_the_move_table() {
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
}

#[test]
fn solve_all_moves_prints_position_and_depth_outside_the_move_table() {
    let stdout = run_solve_all_moves(INITIAL_POSITION);
    let table_header = stdout
        .lines()
        .find(|line| line.contains("Principal Variation"))
        .expect("table header");

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

fn first_pv_token(row: &str) -> Option<&str> {
    let mut columns = row.split('|');
    columns.next()?;
    let score = columns.next()?.trim();
    if score == "Score" || score.starts_with('-') {
        return None;
    }
    let pv = columns.next()?.trim();
    pv.split_whitespace().next()
}
