#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
#![cfg_attr(
    target_env = "msvc",
    allow(
        linker_messages,
        reason = "localized MSVC import-library creation output is informational"
    )
)]

fn main() {
    gui::run();
}
