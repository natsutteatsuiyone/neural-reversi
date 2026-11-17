//! Runs `cli solve` on a small position file and checks what it prints.

use std::fs;
use std::process::Command;

/// FFO #1, then White must pass, then neither side can move.
const POSITIONS: &str = "\
--XXXXX--OOOXX-O-OOOXXOX-OXOXOXXOXXXOXXX--XOXOXX-XXXOOO--OOOOO-- X;
XO------------------------------------------------------X------X O;
X------X------------------------------------------------X------X X;
";

fn cells(line: &str) -> Vec<&str> {
    line.trim_matches('|').split('|').map(str::trim).collect()
}

#[test]
fn solve_prints_the_thread_count_and_an_aligned_table() {
    let file = std::env::temp_dir().join(format!(
        "neural-reversi-cli-solve-output-{}.obf",
        std::process::id()
    ));
    fs::write(&file, POSITIONS).unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_cli"))
        .args(["solve", "--exact", "--threads", "2", "--hash-size", "16"])
        .arg(&file)
        .output()
        .unwrap();
    let _ = fs::remove_file(&file);

    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(stdout.contains("- Threads:   2\n"), "{stdout}");

    let table: Vec<&str> = stdout.lines().filter(|l| l.starts_with('|')).collect();
    let [header, separator, ffo1, pass, game_over, total] = table[..] else {
        panic!("expected 6 table lines:\n{stdout}");
    };

    let bars = |line: &str| -> Vec<usize> { line.match_indices('|').map(|(i, _)| i).collect() };
    for line in &table[1..] {
        assert_eq!(bars(line), bars(header), "{stdout}");
    }
    assert!(
        cells(separator)
            .iter()
            .all(|cell| cell.chars().all(|c| c == '-' || c == ':')),
        "{separator}"
    );

    assert_eq!(cells(ffo1)[..3], ["1", "14", "+18"]);
    assert!(cells(ffo1)[6].starts_with("g8 "), "{ffo1}");

    // The score is White's, and the line starts with White's pass.
    assert_eq!(cells(pass)[..3], ["2", "60", "-64"]);
    assert_eq!(cells(pass)[6], "PS c1", "{pass}");

    assert_eq!(cells(game_over)[..3], ["3", "END", "+64"]);
    assert_eq!(cells(game_over)[6], "--");

    assert_eq!(cells(total)[0], "Total");
}
