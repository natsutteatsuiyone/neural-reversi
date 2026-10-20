#![cfg_attr(
    all(test, target_env = "msvc"),
    allow(
        linker_messages,
        reason = "localized MSVC import-library creation output is informational"
    )
)]

rust_i18n::i18n!("locales", fallback = "en");

mod engine;
mod game;
mod session;
mod settings;
mod ui;

pub use ui::run;

#[cfg(test)]
mod tests {
    use std::{fs, path::PathBuf};

    #[test]
    fn every_tr_key_has_english_and_japanese_text() {
        let backend = crate::_rust_i18n_backend();
        let mut dirs = vec![PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/src"))];
        let mut keys = 0;
        while let Some(dir) = dirs.pop() {
            for entry in fs::read_dir(dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    dirs.push(path);
                    continue;
                }
                let source = fs::read_to_string(&path).unwrap();
                // Split so this file does not match itself.
                for (index, call) in source.match_indices(concat!("tr", "!(")) {
                    if source[..index].ends_with(|c: char| c.is_alphanumeric() || c == '_') {
                        continue;
                    }
                    let rest = source[index + call.len()..]
                        .trim_start()
                        .strip_prefix('"')
                        .unwrap_or_else(|| panic!("{}: tr! key must be a literal", path.display()));
                    let key = &rest[..rest.find('"').unwrap()];
                    for locale in ["en", "ja"] {
                        assert!(
                            backend.translate(locale, key).is_some(),
                            "{}: `{key}` has no {locale} text",
                            path.display()
                        );
                    }
                    keys += 1;
                }
            }
        }
        assert!(keys > 0);
    }
}
