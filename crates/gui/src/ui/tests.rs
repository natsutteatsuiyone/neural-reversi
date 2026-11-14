use std::time::Duration;

use nsh_core::prelude::*;
use nsh_engine::{EngineConfig, Environment, Idiom, Os, Root};
use nsh_testing::Headless;

use super::Ui;
use crate::settings::{GameMode, Language, Settings};

#[component]
fn TestApp() -> impl IntoElement {
    let settings = Settings {
        game_mode: GameMode::Human,
        language: Language::English,
        hash_size: 64,
        ai_level: 4,
        timed: false,
        hint_level: 1,
        hints: false,
        analysis_level: 4,
        analysis_open: false,
        ..Settings::default()
    };
    Ui::with_settings(settings, None, |_| Ok(())).render()
}

#[test]
fn headless_play_setup_solver_and_history_preserve_user_flows() -> Result<(), String> {
    let config = EngineConfig {
        viewport: size(px(1100.), px(920.)),
        environment: Environment {
            os: Os::Windows,
            idiom: Idiom::Desktop,
            reduce_motion: true,
            ..Environment::default()
        },
        ..EngineConfig::new(Root::of::<TestApp>())
    };
    let mut app = Headless::start(config).map_err(|error| error.to_string())?;
    app.wait_for_within("role:button[New Game]", Duration::from_secs(60))?;
    app.hover("role:button[Settings]")?;
    app.wait_for("text:Settings")?;
    app.hover("role:button[New Game]")?;
    app.wait_for_gone("text:Settings", Duration::from_secs(2))?;
    assert_eq!(app.screen().count("label:Board >> role:cell")?, 64);
    assert_eq!(
        app.screen()
            .count("label:/^[A-H][1-8] Empty Legal move$/")?,
        4
    );

    app.press("role:button[New Game]")?;
    app.press("role:tab[vs Human]")?;
    assert!(app.screen().selected("role:tab[vs Human]")?);
    app.press("role:button[Start Game]")?;
    app.wait_for_gone("role:button[Start Game]", Duration::from_secs(2))?;
    app.press("id:board-cell-19")?;
    app.wait_for("label:/^D3 Black/")?;
    app.wait_for("label:/^D4 Black/")?;
    assert_eq!(app.screen().count("label:/^[A-H][1-8] Black/")?, 4);
    assert_eq!(app.screen().count("label:/^[A-H][1-8] White/")?, 1);
    app.press("role:button[Copy transcript through this move]")?;
    assert_eq!(app.screen().clipboard(), Some("d3"));

    app.press("role:button[Previous move]")?;
    app.wait_for("label:D3 Empty Legal move")?;
    app.press("role:button[Next move]")?;
    app.wait_for("label:/^D3 Black/")?;
    app.press("role:button[Previous move]")?;
    app.press("id:board-cell-44")?;
    app.wait_for("label:/^E6 Black/")?;
    assert!(!app.screen().enabled("role:button[Next move]")?);
    app.press("role:button[Copy transcript through this move]")?;
    assert_eq!(app.screen().clipboard(), Some("e6"));

    app.press("role:button[New Game]")?;
    app.press("role:button[Board Setup]")?;
    app.press("role:tab[Transcript]")?;
    app.type_text("label:Transcript", "z9")?;
    app.wait_for("text:Invalid move notation found")?;
    assert!(!app.screen().enabled("role:button[Start Game]")?);
    app.clear("label:Transcript")?;
    app.type_text("label:Transcript", "D3C3")?;
    app.wait_for_gone("text:Invalid move notation found", Duration::from_secs(2))?;
    assert!(app.screen().enabled("role:button[Start Game]")?);
    app.press("role:button[Cancel]")?;
    app.wait_for("label:/^E6 Black/")?;
    app.press("role:button[Copy transcript through this move]")?;
    assert_eq!(app.screen().clipboard(), Some("e6"));

    app.press("role:button[New Game]")?;
    app.press("role:button[Board Setup]")?;
    app.press("role:tab[Transcript]")?;
    app.type_text("label:Transcript", "D3C3")?;
    app.press("role:button[Start Game]")?;
    app.wait_for_gone("role:button[Start Game]", Duration::from_secs(2))?;
    app.wait_for("label:/^D3 Black/")?;
    app.wait_for("label:/^C3 White/")?;
    assert!(
        !app.screen()
            .enabled("role:button[Copy transcript through this move]")?
    );
    app.press("id:board-cell-26")?;
    app.wait_for("label:/^C4 Black/")?;
    app.press("role:button[Copy transcript through this move]")?;
    assert_eq!(app.screen().clipboard(), Some("c4"));

    app.press("role:button[Solver]")?;
    app.press("role:tab[Board String]")?;
    let position = format!("-OXOOOOO{}", "O".repeat(56));
    app.type_text("label:Board String", &position)?;
    app.press("role:radio[Black]")?;
    app.press("role:radio[100%]")?;
    app.press("role:radio[Best move only]")?;
    app.press("role:button[Start Solver]")?;
    app.wait_for_within("label:/^[aA]1 -58/", Duration::from_secs(10))?;
    app.press("id:board-cell-0")?;
    app.wait_for("label:/^A1 Black/")?;
    app.press("role:button[Undo]")?;
    app.wait_for("label:A1 Empty Legal move")?;
    app.press("role:button[Exit Solver]")?;
    app.wait_for("label:/^D3 Black/")?;
    app.wait_for("label:/^C4 Black/")?;
    app.press("role:button[Copy transcript through this move]")?;
    assert_eq!(app.screen().clipboard(), Some("c4"));

    app.press("role:button[New Game]")?;
    app.press_key("escape")?;
    app.wait_for_gone("role:button[Start Game]", Duration::from_secs(2))?;
    app.press_key("enter")?;
    app.wait_for("role:button[Start Game]")?;
    app.press("role:button[Cancel]")?;
    app.press("role:button[Settings]")?;
    app.press("text:Language")?;
    app.press("text:日本語")?;
    app.wait_for("role:button[新規ゲーム]")?;
    app.wait_for("text:ヒント")?;
    app.wait_for("role:button[AI分析]")?;
    app.press_key("enter")?;
    app.wait_for("text:言語")?;
    app.press("text:言語")?;
    app.press("text:English")?;
    app.wait_for("text:Hint")?;
    app.wait_for("role:button[AI Analysis]")?;
    app.set_viewport(size(px(480.), px(620.)))?;
    app.wait_for("role:button[New Game]")?;
    app.press("role:button[New Game]")?;
    app.wait_for("role:button[Start Game]")?;
    app.press("role:button[Cancel]")?;
    app.wait_for("label:/^C4 Black/")?;
    Ok(())
}
