//! Native preferences and one-time import of both historical Tauri store shapes.
use std::{
    fs,
    io::ErrorKind,
    path::{Path, PathBuf},
};

use directories::BaseDirs;
use reversi_core::disc::Disc;
use reversi_core::level::ENDGAME_STEPS;
use serde::{Deserialize, Serialize};
use serde_json::Value;

static SYSTEM_JAPANESE: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    sys_locale::get_locale().is_some_and(|locale| {
        locale
            .split(['-', '_', '.', '@'])
            .next()
            .is_some_and(|language| language.eq_ignore_ascii_case("ja"))
    })
});

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GameMode {
    #[serde(rename = "ai-black")]
    AiBlack,
    #[serde(rename = "ai-white")]
    AiWhite,
    #[serde(rename = "pvp")]
    Human,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Language {
    #[serde(rename = "auto")]
    Auto,
    #[serde(rename = "en")]
    English,
    #[serde(rename = "ja")]
    Japanese,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Settings {
    pub game_mode: GameMode,
    pub ai_level: usize,
    pub timed: bool,
    /// Total AI clock per game, in seconds.
    pub game_time_limit: u64,
    pub hint_level: usize,
    pub hints: bool,
    pub analysis_level: usize,
    /// Transposition table size in MiB.
    pub hash_size: usize,
    pub analysis_open: bool,
    pub language: Language,
    /// Index into `Level::ENDGAME_SELECTIVITY` (73%, 86%, 95%, 99%, 100%).
    pub solver_selectivity: usize,
    pub solver_multi_pv: bool,
    /// Layout proportions, in percent.
    pub right_panel_size: f32,
    pub bottom_panel_size: f32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            game_mode: GameMode::AiBlack,
            ai_level: 21,
            timed: true,
            game_time_limit: 60,
            hint_level: 21,
            hints: false,
            analysis_level: 20,
            hash_size: 512,
            analysis_open: false,
            language: Language::Auto,
            solver_selectivity: ENDGAME_STEPS - 1,
            solver_multi_pv: true,
            right_panel_size: 25.0,
            bottom_panel_size: 30.0,
        }
    }
}

impl Settings {
    pub fn load() -> Result<Self, String> {
        Self::load_from(&Self::directory()?)
    }

    pub fn save(&self) -> Result<(), String> {
        self.save_to(&Self::directory()?)
    }

    pub fn ai_color(&self) -> Option<Disc> {
        match self.game_mode {
            GameMode::AiBlack => Some(Disc::Black),
            GameMode::AiWhite => Some(Disc::White),
            GameMode::Human => None,
        }
    }

    pub fn locale(&self) -> &'static str {
        match self.language {
            Language::Japanese => "ja",
            Language::English => "en",
            Language::Auto if *SYSTEM_JAPANESE => "ja",
            Language::Auto => "en",
        }
    }

    fn directory() -> Result<PathBuf, String> {
        BaseDirs::new()
            .map(|dirs| dirs.data_dir().join("neural-reversi"))
            .ok_or_else(|| "Cannot locate the application settings directory".into())
    }

    fn load_from(directory: &Path) -> Result<Self, String> {
        let native = directory.join("native-settings.json");
        if let Some(text) = read_optional(&native)? {
            return Self::decode(&text).map_err(|error| format!("{}: {error}", native.display()));
        }
        let legacy = directory.join("settings.json");
        if let Some(text) = read_optional(&legacy)? {
            let settings =
                Self::decode(&text).map_err(|error| format!("{}: {error}", legacy.display()))?;
            settings.save_to(directory)?;
            return Ok(settings);
        }
        Ok(Self::default())
    }

    fn save_to(&self, directory: &Path) -> Result<(), String> {
        let text = serde_json::to_vec_pretty(self).map_err(|error| error.to_string())?;
        fs::create_dir_all(directory)
            .map_err(|error| format!("Cannot create {}: {error}", directory.display()))?;
        let destination = directory.join("native-settings.json");
        let temporary = directory.join("native-settings.json.tmp");
        fs::write(&temporary, text)
            .map_err(|error| format!("Cannot write {}: {error}", temporary.display()))?;
        fs::rename(&temporary, &destination)
            .map_err(|error| format!("Cannot save {}: {error}", destination.display()))
    }

    fn decode(text: &str) -> Result<Self, String> {
        let mut value: Value = serde_json::from_str(text)
            .map_err(|error| format!("Invalid settings JSON: {error}"))?;
        if let Some(stored) = value.get("neural-reversi-settings") {
            value = match stored {
                Value::String(text) => serde_json::from_str(text)
                    .map_err(|error| format!("Invalid persisted settings: {error}"))?,
                Value::Object(_) => stored.clone(),
                _ => return Err("Persisted settings must be a JSON object or string".into()),
            };
        }
        if value.get("state").is_some() {
            if let Some(version) = value.get("version")
                && !version.as_u64().is_some_and(|version| version <= 2)
            {
                return Err(format!("Unsupported settings version: {version}"));
            }
            value = value.get_mut("state").unwrap().take();
        }
        let fields = value
            .as_object_mut()
            .ok_or("Settings must be a JSON object")?;
        // Normalize old names before deserialization; a current field wins if both exist.
        for (old, new) in [
            ("isHintMode", "hints"),
            ("gameAnalysisLevel", "analysisLevel"),
            ("aiAnalysisPanelOpen", "analysisOpen"),
        ] {
            if let Some(old_value) = fields.remove(old) {
                fields.entry(new).or_insert(old_value);
            }
        }
        if let Some(mode) = fields.remove("aiMode")
            && !fields.contains_key("timed")
        {
            let timed = match mode.as_str() {
                Some("game-time") => true,
                Some("level") => false,
                _ => return Err(format!("Invalid AI mode: {mode}")),
            };
            fields.insert("timed".into(), Value::Bool(timed));
        }
        if let Some(mode) = fields.remove("solverMode")
            && !fields.contains_key("solverMultiPv")
        {
            let multi_pv = match mode.as_str() {
                Some("multiPv") => true,
                Some("bestOnly") => false,
                _ => return Err(format!("Invalid solver mode: {mode}")),
            };
            fields.insert("solverMultiPv".into(), Value::Bool(multi_pv));
        }
        let selectivity = fields
            .remove("targetSelectivity")
            .or_else(|| fields.remove("solverTargetSelectivity"));
        if !fields.contains_key("solverSelectivity")
            && let Some(selectivity) = selectivity
        {
            let index = match selectivity.as_u64() {
                Some(73) => 0,
                Some(86) => 1,
                Some(95) => 2,
                Some(99) => 3,
                Some(100) => 4,
                _ => return Err(format!("Invalid solver selectivity: {selectivity}")),
            };
            fields.insert("solverSelectivity".into(), Value::from(index));
        }
        if fields.get("language").is_some_and(Value::is_null) {
            fields.insert("language".into(), Value::String("auto".into()));
        }
        let mut settings: Self =
            serde_json::from_value(value).map_err(|error| format!("Invalid settings: {error}"))?;
        settings.bound_values();
        Ok(settings)
    }

    fn bound_values(&mut self) {
        self.ai_level = self.ai_level.min(30);
        self.hint_level = self.hint_level.min(30);
        self.analysis_level = self.analysis_level.min(30);
        self.game_time_limit = self.game_time_limit.clamp(1, 86_400);
        self.hash_size = self.hash_size.clamp(1, 16_384);
        self.solver_selectivity = self.solver_selectivity.min(ENDGAME_STEPS - 1);
        self.right_panel_size = self.right_panel_size.clamp(18.0, 45.0);
        self.bottom_panel_size = self.bottom_panel_size.clamp(18.0, 70.0);
    }
}

fn read_optional(path: &Path) -> Result<Option<String>, String> {
    match fs::read_to_string(path) {
        Ok(text) => Ok(Some(text)),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(None),
        Err(error) => Err(format!("Cannot read {}: {error}", path.display())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn imports_zustand_and_flat_tauri_preferences_with_real_field_mapping() {
        let old = json!({
            "gameMode":"pvp", "aiLevel":8, "aiMode":"level", "gameTimeLimit":180,
            "hintLevel":12, "isHintMode":true, "gameAnalysisLevel":15, "hashSize":256,
            "aiAnalysisPanelOpen":true, "language":null, "rightPanelSize":32.5,
            "bottomPanelSize":48, "targetSelectivity":95, "solverMode":"bestOnly"
        });
        let envelope =
            json!({"neural-reversi-settings":json!({"state":old,"version":2}).to_string()});
        let settings = Settings::decode(&envelope.to_string()).unwrap();
        assert_eq!(settings.game_mode, GameMode::Human);
        assert!(!settings.timed);
        assert!(settings.hints);
        assert_eq!(settings.analysis_level, 15);
        assert!(settings.analysis_open);
        assert_eq!(settings.language, Language::Auto);
        assert_eq!(settings.solver_selectivity, 2);
        assert!(!settings.solver_multi_pv);
        assert_eq!(settings.right_panel_size, 32.5);
        assert_eq!(
            settings,
            Settings::decode(&serde_json::to_string(&settings).unwrap()).unwrap()
        );
        let flat = Settings::decode(
            r#"{"solverTargetSelectivity":99,"aiMode":"game-time","language":"ja"}"#,
        )
        .unwrap();
        assert_eq!(flat.solver_selectivity, 3);
        assert!(flat.timed);
        assert_eq!(flat.locale(), "ja");
        for key in ["targetSelectivity", "solverTargetSelectivity"] {
            let settings = Settings::decode(&json!({ key: 86 }).to_string()).unwrap();
            assert_eq!(settings.solver_selectivity, 1);
        }
    }

    #[test]
    fn bounds_resources_and_reports_invalid_data_without_hiding_it() {
        let settings = Settings::decode(r#"{"aiLevel":300,"hintLevel":90,"analysisLevel":90,"hashSize":999999,"gameTimeLimit":0,"solverSelectivity":100,"rightPanelSize":-50,"bottomPanelSize":90}"#).unwrap();
        assert_eq!(
            (
                settings.ai_level,
                settings.hint_level,
                settings.analysis_level
            ),
            (30, 30, 30)
        );
        assert_eq!(settings.hash_size, 16_384);
        assert_eq!(settings.game_time_limit, 1);
        assert_eq!(settings.solver_selectivity, 4);
        assert_eq!(
            (settings.right_panel_size, settings.bottom_panel_size),
            (18.0, 70.0)
        );
        assert!(Settings::decode("{").is_err());
        assert!(Settings::decode(r#"{"state":{},"version":3}"#).is_err());
        assert!(Settings::decode(r#"{"aiMode":"invalid"}"#).is_err());
        assert!(Settings::decode(r#"{"hashSize":"huge"}"#).is_err());
    }

    #[test]
    fn migrates_without_overwriting_legacy_then_prefers_native_settings() {
        let directory = std::env::temp_dir().join(format!(
            "neural-reversi-settings-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&directory).unwrap();
        let legacy = r#"{"aiLevel":9,"language":"ja","solverTargetSelectivity":73}"#;
        fs::write(directory.join("settings.json"), legacy).unwrap();
        let mut settings = Settings::load_from(&directory).unwrap();
        assert_eq!(settings.ai_level, 9);
        assert_eq!(settings.solver_selectivity, 0);
        assert_eq!(
            fs::read_to_string(directory.join("settings.json")).unwrap(),
            legacy
        );
        settings.ai_level = 19;
        settings.save_to(&directory).unwrap();
        assert_eq!(Settings::load_from(&directory).unwrap().ai_level, 19);
        fs::write(directory.join("native-settings.json"), "invalid").unwrap();
        assert!(Settings::load_from(&directory).is_err());
        fs::remove_dir_all(directory).unwrap();
    }
}
