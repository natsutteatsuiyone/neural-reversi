use std::process::Command;

#[test]
fn missing_engine_reports_io_error_without_debug_formatting() {
    let missing_engine = std::env::temp_dir().join(format!(
        "neural-reversi-missing-engine-{}",
        std::process::id()
    ));
    assert!(!missing_engine.exists());
    let engine_command = format!("\"{}\"", missing_engine.display());
    let output = Command::new(env!("CARGO_BIN_EXE_match-runner"))
        .args(["--engine1", &engine_command, "--engine2", &engine_command])
        .args(["--ggs-random-discs", "4", "--pairs", "1", "--seed", "1"])
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.starts_with("Error: IO error: "), "{stderr}");
    assert!(!stderr.contains("Os {"), "{stderr}");
}
