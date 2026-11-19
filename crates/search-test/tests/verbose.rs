use std::path::Path;
use std::process::Command;

#[test]
fn initial_position_prints_total_without_iteration_rows() {
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let output = Command::new(env!("CARGO_BIN_EXE_search-test"))
        .arg("--problem-dir")
        .arg(&fixtures)
        .args([
            "--problem",
            "initial",
            "--time",
            "1",
            "--threads",
            "1",
            "--hash-size",
            "1",
            "--verbose",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    let rows: Vec<_> = stdout
        .lines()
        .filter(|line| line.starts_with('|'))
        .collect();
    assert_eq!(rows.len(), 3, "{stdout}");
    assert!(rows[2].starts_with("| total "), "{stdout}");
}
