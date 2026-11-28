//! Runs `cli solve` on inputs it cannot fully use and checks how it fails.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// FFO #1.
const POSITION: &str = "--XXXXX--OOOXX-O-OOOXXOX-OXOXOXXOXXXOXXX--XOXOXX-XXXOOO--OOOOO-- X;\n";

fn temp_file(name: &str, contents: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "neural-reversi-cli-invalid-input-{}-{name}",
        std::process::id()
    ));
    fs::write(&path, contents).unwrap();
    path
}

fn solve(positions: &Path, extra: &[&Path]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_cli"))
        .args(["solve", "--exact", "--threads", "1", "--hash-size", "1"])
        .arg(positions)
        .args(extra)
        .output()
        .unwrap()
}

#[test]
fn weight_file_of_the_wrong_format_is_an_error_not_a_panic() {
    let positions = temp_file("weights.obf", POSITION);
    let weights = temp_file("weights.zst", "not a network");

    let output = solve(&positions, &[Path::new("--eval-file"), &weights]);
    let _ = fs::remove_file(&positions);
    let _ = fs::remove_file(&weights);

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "{stderr}");
    assert!(stderr.contains("weights.zst"), "{stderr}");
    assert!(!stderr.contains("panicked"), "{stderr}");
}

#[test]
fn solve_fails_when_it_skips_a_line_or_finds_no_position() {
    let mixed = temp_file("mixed.obf", &format!("{POSITION}--XXXXX X;\n"));
    let comments = temp_file("comments.obf", "% no position here\n\n");

    let skipped = solve(&mixed, &[]);
    let none = solve(&comments, &[]);
    let _ = fs::remove_file(&mixed);
    let _ = fs::remove_file(&comments);

    // The valid line is still solved.
    let stdout = String::from_utf8_lossy(&skipped.stdout);
    let stderr = String::from_utf8_lossy(&skipped.stderr);
    assert!(stdout.contains("|  +18  |"), "{stdout}");
    assert!(stderr.contains("Error parsing line 2"), "{stderr}");
    assert!(stderr.contains("skipped 1 invalid line"), "{stderr}");
    assert_eq!(skipped.status.code(), Some(1));

    let stderr = String::from_utf8_lossy(&none.stderr);
    assert!(stderr.contains("no positions"), "{stderr}");
    assert_eq!(none.status.code(), Some(1));
}
