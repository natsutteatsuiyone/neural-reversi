use std::borrow::Cow;

use gpui_kit::{AssetSource, SharedString, Styled, Svg, px, rgb, svg};

/// Embeds `assets/icons/<icon>` as `<icon>` and `assets/<file>` as `<file>`.
macro_rules! assets {
    (icons: [$($icon:literal),* $(,)?], files: [$($file:literal),* $(,)?] $(,)?) => {
        &[
            $(($icon, include_bytes!(concat!("../../assets/icons/", $icon)))),*,
            $(($file, include_bytes!(concat!("../../assets/", $file)))),*
        ]
    };
}

const ASSETS: &[(&str, &[u8])] = assets! {
    icons: [
        "play.svg",
        "calculator.svg",
        "lightbulb.svg",
        "menu.svg",
        "bot.svg",
        "user.svg",
        "clock.svg",
        "timer.svg",
        "stop-circle.svg",
        "arrow-left.svg",
        "loader-circle.svg",
        "chevron-left.svg",
        "chevron-right.svg",
        "chevron-first.svg",
        "chevron-last.svg",
        "copy.svg",
        "chart.svg",
        "activity.svg",
        "search.svg",
        "x.svg",
        "globe.svg",
        "hard-drive.svg",
        "info.svg",
        "trophy.svg",
        "check.svg",
        "chevron-down.svg",
        "chevron-up.svg",
        "square.svg",
        "grip.svg",
        "list.svg",
        "users.svg",
        "zap.svg",
    ],
    files: [
        "board/felt.png",
        "board/disc-black.png",
        "board/disc-white.png",
        "stones/black-16.png",
        "stones/white-16.png",
        "stones/black-21_6.png",
        "stones/white-21_6.png",
        "stones/black-40.png",
        "stones/white-40.png",
    ],
};

pub(super) struct GuiAssets;

impl AssetSource for GuiAssets {
    fn load(&self, path: &str) -> gpui_kit::Result<Option<Cow<'static, [u8]>>> {
        Ok(ASSETS
            .iter()
            .find(|(name, _)| *name == path)
            .map(|(_, bytes)| Cow::Borrowed(*bytes)))
    }

    fn list(&self, _: &str) -> gpui_kit::Result<Vec<SharedString>> {
        Ok(Vec::new())
    }
}

pub(super) fn icon(name: &'static str, size: f32, color: u32) -> Svg {
    svg().path(name).size(px(size)).text_color(rgb(color))
}
