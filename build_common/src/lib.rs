use std::env;
use std::fs;
use std::path::Path;
use std::path::PathBuf;

use reversi_core::eval::constants::EVAL_FILE_NAME;
use reversi_core::eval::constants::EVAL_SM_FILE_NAME;

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Env(env::VarError),
    Path(String),
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self { Error::Io(e) }
}
impl From<env::VarError> for Error {
    fn from(e: env::VarError) -> Self { Error::Env(e) }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Io(e) => write!(f, "IO error: {}", e),
            Error::Env(e) => write!(f, "Environment variable error: {}", e),
            Error::Path(s) => write!(f, "Path error: {}", s),
        }
    }
}

impl std::error::Error for Error {}

/// Gets the target profile directory.
///
/// # Arguments
///
/// * `out_dir` - The output directory path.
/// * `profile` - The profile name (e.g., "debug", "release").
///
/// # Returns
///
/// * `Ok(PathBuf)` if the target profile directory was found.
/// * `Err(Error)` if there was an error during the process.
fn get_target_profile_dir(out_dir: &Path, profile: &str) -> Result<PathBuf, Error> {
    let mut dir = out_dir;
    while let Some(parent) = dir.parent() {
        if dir.file_name().and_then(|s| s.to_str()) == Some("target") {
            return Ok(dir.join(profile));
        }
        dir = parent;
    }
    Err(Error::Path(out_dir.to_string_lossy().into_owned()))
}

/// Copies specified files to the `target/{profile}/` directory.
///
/// # Arguments
///
/// * `files_to_copy` - A slice of tuples where each tuple contains the relative source path and the destination filename.
///
/// # Returns
///
/// * `Ok(())` if the files were copied successfully.
/// * `Err(Error)` if there was an error during the process.
fn copy_files_to_target_profile_dir(
    files_to_copy: &[(&str, &str)],
) -> Result<(), Error> {
    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    let profile = env::var("PROFILE")?;
    let target_dir = get_target_profile_dir(&out_dir, &profile)?;
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);

    for (relative_src_path, dest_name) in files_to_copy {
        let src_path = manifest_dir.join(relative_src_path);
        let dest_path = target_dir.join(dest_name);

        if src_path.exists() {
            // Ensure destination directory exists
            if let Some(parent_dir) = dest_path.parent() {
                fs::create_dir_all(parent_dir)?;
            }

            match fs::copy(&src_path, &dest_path) {
                Ok(_) => {
                    println!(
                        "cargo:note=Copied {} to {}",
                        src_path.display(),
                        dest_path.display()
                    );
                }
                Err(e) => {
                    eprintln!(
                        "cargo:warning=Failed to copy {} to {}: {}",
                        src_path.display(),
                        dest_path.display(),
                        e
                    );
                }
            }

            match src_path.canonicalize() {
                Ok(canonical_src_path) => {
                    println!("cargo:rerun-if-changed={}", canonical_src_path.display());
                }
                Err(e) => {
                     eprintln!(
                        "cargo:warning=Failed to canonicalize source path {} for rerun-if-changed: {}",
                        src_path.display(),
                        e
                    );
                    // Fallback to non-canonicalized path
                    println!("cargo:rerun-if-changed={}", src_path.display());
                }
            }
        } else {
            println!(
                "cargo:warning=Source file {} (resolved to {}) does not exist, skipping copy.",
                relative_src_path,
                src_path.display()
            );
        }
    }
    Ok(())
}

/// Copies weight files to the target profile directory.
///
/// # Arguments
/// * `weights_directory` - The directory containing the weight files.
///
/// # Returns
///
/// * `Ok(())` if the files were copied successfully.
/// * `Err(Error)` if there was an error during the process.
pub fn copy_weight_files(weights_directory: &str) -> Result<(), Error> {
    let eval_path = format!("{}/{}", weights_directory, EVAL_FILE_NAME);
    let eval_sm_path = format!("{}/{}", weights_directory, EVAL_SM_FILE_NAME);

    let files_to_copy = [
        (eval_path.as_str(), EVAL_FILE_NAME),
        (eval_sm_path.as_str(), EVAL_SM_FILE_NAME),
    ];

    copy_files_to_target_profile_dir(&files_to_copy)
}
