use std::sync::{Mutex, MutexGuard, mpsc};
use std::time::{Duration, Instant, SystemTime};

use nsh_core::prelude::*;
use nsh_core::{Checked, MountItem, ViewKind};
use nsh_engine::{Clock, EngineConfig, Environment, Idiom, Os, Root};
use nsh_testing::{Headless, LayoutNode};

use super::Ui;
use crate::settings::{GameMode, Language, Settings};

static LOCALE_LOCK: Mutex<()> = Mutex::new(());

fn lock_locale() -> MutexGuard<'static, ()> {
    LOCALE_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

#[component]
fn TestApp() -> impl IntoElement {
    let settings = Settings {
        game_mode: GameMode::Human,
        language: Language::English,
        hash_size: 16,
        ai_level: 4,
        timed: false,
        hint_level: 1,
        hints: false,
        analysis_level: 4,
        analysis_open: false,
        ..Settings::default()
    };
    let ui = Ui::with_settings(settings, None, |_| Ok(()));
    view()
        .relative()
        .size_full()
        .flex_col()
        .child(ui.render())
        .child(when(
            move || ui.read(|app| app.session.is_ready() && app.session.activity().is_none()),
            || {
                view()
                    .absolute()
                    .bottom_0()
                    .right_0()
                    .size(px(1.))
                    .pointer_events(PointerEvents::None)
                    .test_id("test-engine-idle")
            },
        ))
}

fn headless_config() -> EngineConfig {
    headless_config_with_motion(Root::of::<TestApp>(), true)
}

fn headless_config_with_motion(root: Root, reduce_motion: bool) -> EngineConfig {
    EngineConfig {
        viewport: size(px(1100.), px(920.)),
        clock: Clock::manual(SystemTime::UNIX_EPOCH),
        environment: Environment {
            os: Os::Windows,
            idiom: Idiom::Desktop,
            reduce_motion,
            ..Environment::default()
        },
        ..EngineConfig::new(root)
    }
}

fn wait_for_engine(app: &mut Headless, selector: &str, timeout: Duration) -> Result<(), String> {
    let deadline = Instant::now() + timeout;
    loop {
        app.run_for(Duration::from_millis(40))?;
        if app.screen().matches(selector)? {
            return Ok(());
        }
        if Instant::now() >= deadline {
            return app.wait_for_within(selector, Duration::ZERO);
        }
        std::thread::sleep(Duration::from_millis(1));
    }
}

#[test]
fn headless_play_setup_solver_and_history_preserve_user_flows() -> Result<(), String> {
    let _locale = lock_locale();
    let mut app = Headless::start(headless_config()).map_err(|error| error.to_string())?;
    wait_for_engine(&mut app, "role:button[New Game]", Duration::from_secs(60))?;
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
    wait_for_engine(&mut app, "id:test-engine-idle", Duration::from_secs(5))?;
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
    wait_for_engine(&mut app, "id:test-engine-idle", Duration::from_secs(5))?;
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
    wait_for_engine(&mut app, "label:/^[aA]1 -58/", Duration::from_secs(10))?;
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

#[test]
fn headless_new_game_vs_ai_tab_preserves_white_selection() -> Result<(), String> {
    let _locale = lock_locale();
    let mut app = Headless::start(headless_config()).map_err(|error| error.to_string())?;
    wait_for_engine(&mut app, "role:button[New Game]", Duration::from_secs(60))?;
    app.press("role:button[New Game]")?;
    app.press("role:tab[vs AI]")?;
    app.press("role:radio[White]")?;
    assert_eq!(
        app.screen()
            .accessibility_state("role:radio[White]")?
            .checked,
        Some(Checked::True)
    );
    app.press("role:tab[vs AI]")?;
    assert_eq!(
        app.screen()
            .accessibility_state("role:radio[White]")?
            .checked,
        Some(Checked::True)
    );
    Ok(())
}

#[test]
fn headless_analysis_expanded_state_and_about_tabs_work() -> Result<(), String> {
    let _locale = lock_locale();
    let mut app = Headless::start(headless_config()).map_err(|error| error.to_string())?;
    wait_for_engine(
        &mut app,
        "role:button[AI Analysis]",
        Duration::from_secs(60),
    )?;
    assert_eq!(
        app.screen()
            .accessibility_state("role:button[AI Analysis]")?
            .expanded,
        Some(false)
    );
    app.press("role:button[AI Analysis]")?;
    app.wait_for("role:tab[Evaluation]")?;
    assert_eq!(
        app.screen()
            .accessibility_state("role:button[AI Analysis]")?
            .expanded,
        Some(true)
    );
    app.press("role:tab[Evaluation]")?;
    app.press("role:button[AI Analysis]")?;
    app.wait_for_gone("role:tab[Evaluation]", Duration::from_secs(2))?;
    assert_eq!(
        app.screen()
            .accessibility_state("role:button[AI Analysis]")?
            .expanded,
        Some(false)
    );

    app.press("role:button[Settings]")?;
    app.wait_for("text:About")?;
    app.press("text:About")?;
    app.wait_for("role:tab[Third-Party]")?;
    app.press("role:tab[Third-Party]")?;
    assert!(app.screen().selected("role:tab[Third-Party]")?);
    app.press_key("escape")?;
    app.wait_for_gone("role:tab[Third-Party]", Duration::from_secs(2))?;
    Ok(())
}

fn disc_image(app: &Headless, cell_selector: &str) -> Result<LayoutNode, String> {
    let cell_id = app.screen().id(cell_selector)?;
    let nodes = app.screen().layout_nodes();
    let disc = nodes
        .iter()
        .find(|node| node.parent == cell_id)
        .ok_or_else(|| format!("missing disc in `{cell_selector}`"))?;
    nodes
        .iter()
        .find(|node| node.parent == disc.id && node.kind == ViewKind::Image)
        .cloned()
        .ok_or_else(|| format!("missing disc image in `{cell_selector}`"))
}

#[component]
fn FlipModalTestApp() -> impl IntoElement {
    let settings = Settings {
        game_mode: GameMode::Human,
        language: Language::English,
        hash_size: 16,
        ..Settings::default()
    };
    let ui = Ui::with_settings(settings, None, |_| Ok(()));
    let modal_cycled = signal(false);
    let played = computed(move || ui.read(|app| app.session.game().history.len() == 1));
    effect(move || {
        if played.get() {
            set_timeout(Duration::from_millis(120), move || {
                ui.action(super::Action::NewGame);
                set_timeout(Duration::from_millis(80), move || {
                    ui.action(super::Action::Close);
                    modal_cycled.set(true);
                });
            });
        }
    });
    view()
        .relative()
        .size_full()
        .flex_col()
        .child(ui.render())
        .child(when(
            move || modal_cycled.get(),
            || {
                view()
                    .absolute()
                    .bottom_0()
                    .right_0()
                    .size(px(1.))
                    .pointer_events(PointerEvents::None)
                    .test_id("test-modal-cycled")
            },
        ))
}

#[test]
fn headless_flip_animation_completes_after_opening_and_cancelling_modal() -> Result<(), String> {
    let _locale = lock_locale();
    let mut app = Headless::start(headless_config_with_motion(
        Root::of::<FlipModalTestApp>(),
        false,
    ))
    .map_err(|error| error.to_string())?;
    wait_for_engine(&mut app, "role:button[New Game]", Duration::from_secs(60))?;

    let (images, updates) = mpsc::channel();
    app.observe(move |_, batch| {
        for item in &batch.items {
            if let MountItem::Image { id, source, .. } = item {
                images.send((*id, source.clone())).unwrap();
            }
        }
    });
    app.press("id:board-cell-19")?;
    app.wait_for("label:/^D4 Black/")?;
    app.wait_for("id:test-modal-cycled")?;
    app.run_for(Duration::from_millis(450))?;
    let d4 = disc_image(&app, "id:board-cell-27")?;
    let finished = d4.bounds.size;
    let last_image = updates.try_iter().filter(|(id, _)| *id == d4.id).last();
    assert!(
        last_image
            .and_then(|(_, source)| source?.data)
            .is_some_and(|data| data.bytes() == include_bytes!("../../assets/board/disc-black.png")),
        "D4 did not show the black disc after modal open/cancel"
    );
    assert!(
        (finished.height.0 - finished.width.0).abs() < 0.01,
        "expected D4 flip to complete after modal open/cancel, got {finished:?}"
    );
    Ok(())
}

#[component]
fn UnmountableBoardApp() -> impl IntoElement {
    let mounted = signal(true);
    let settings = Settings {
        game_mode: GameMode::Human,
        language: Language::English,
        hash_size: 16,
        ..Settings::default()
    };
    let ui = Ui::with_settings(settings, None, |_| Ok(()));
    view()
        .size_full()
        .flex_col()
        .child(
            nsh_base::button()
                .accessibility_label(Some(SharedString::from("Unmount Board")))
                .on_press(move || mounted.set(false))
                .child(text("Unmount Board")),
        )
        .child(when(
            move || mounted.get(),
            move || super::board::render_board(ui, computed(|| 400_f32)),
        ))
}

#[test]
fn headless_unmounting_board_does_not_panic() -> Result<(), Box<dyn std::error::Error>> {
    const CHILD: &str = "NEURAL_REVERSI_BOARD_UNMOUNT_TEST_CHILD";
    // An empty libtest filter match also exits with 0.
    const UNMOUNTED: i32 = 42;
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe()?)
            .args([
                "--exact",
                "ui::tests::headless_unmounting_board_does_not_panic",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .output()?;
        assert_eq!(
            output.status.code(),
            Some(UNMOUNTED),
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return Ok(());
    }

    // NSH catches cleanup panics on its engine thread.
    std::panic::set_hook(Box::new(|info| {
        eprintln!("{info}");
        std::process::exit(1);
    }));
    let mut app = Headless::start(headless_config_with_motion(
        Root::of::<UnmountableBoardApp>(),
        false,
    ))?;
    app.wait_for("id:board-grid")?;
    app.press("role:button[Unmount Board]")?;
    app.wait_for_gone("id:board-grid", Duration::from_secs(2))?;
    app.run_for(Duration::from_millis(50))?;
    drop(app);
    std::process::exit(UNMOUNTED);
}
