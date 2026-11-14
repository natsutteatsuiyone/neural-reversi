use std::fs;
use std::process::Command;

use reversi_core::eval::EVAL_FILE_NAME;
use reversi_core::probcut::VERIFY_DEPTH;

#[test]
fn empty_fit_exits_with_an_input_error_without_panicking() {
    let path = std::env::temp_dir().join(format!(
        "neural-reversi-endgame-fit-cli-{}.csv",
        std::process::id()
    ));
    fs::write(
        &path,
        format!(
            "# main_weights={EVAL_FILE_NAME}\n\
             game,empties,s0,s{VERIFY_DEPTH},exact,beta,selectivity,cut_node,engine_cut,replayed_cut\n"
        ),
    )
    .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_datagen"))
        .arg("fit-endgame-probcut")
        .arg("--input")
        .arg(&path)
        .output()
        .unwrap();
    fs::remove_file(path).unwrap();
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        stderr.contains("no samples in the fitting folds"),
        "{stderr}"
    );
    assert!(!stderr.contains("panicked"), "{stderr}");
}
