use std::collections::HashMap;
use std::sync::LazyLock;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use nsh_core::element::{ImageSrc, Svg, svg};
use nsh_core::prelude::*;

use super::color;

macro_rules! embedded_assets {
    (icons: [$($icon:literal),* $(,)?], images: [$($image:literal),* $(,)?]) => {
        const ICONS: &[(&str, &str)] = &[
            $(($icon, include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/assets/icons/", $icon)))),*
        ];
        const IMAGES: &[(&str, &[u8])] = &[
            $(($image, include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/assets/", $image)))),*
        ];
    };
}

embedded_assets! {
    icons: [
        "play.svg", "calculator.svg", "lightbulb.svg", "menu.svg", "bot.svg",
        "user.svg", "clock.svg", "timer.svg", "stop-circle.svg", "arrow-left.svg",
        "loader-circle.svg", "chevron-left.svg", "chevron-right.svg",
        "chevron-first.svg", "chevron-last.svg", "copy.svg", "chart.svg",
        "activity.svg", "search.svg", "x.svg", "globe.svg", "hard-drive.svg",
        "info.svg", "trophy.svg", "check.svg", "chevron-down.svg", "chevron-up.svg",
        "square.svg", "grip.svg", "list.svg", "users.svg", "zap.svg",
    ],
    images: [
        "board/felt.png", "board/disc-black.png", "board/disc-white.png",
        "stones/black-16.png", "stones/white-16.png", "stones/black-21_6.png",
        "stones/white-21_6.png", "stones/black-40.png", "stones/white-40.png",
    ]
}

static IMAGE_SOURCES: LazyLock<HashMap<&'static str, ImageSrc>> = LazyLock::new(|| {
    IMAGES
        .iter()
        .map(|&(name, bytes)| {
            (
                name,
                ImageSrc::from(format!("data:image/png;base64,{}", STANDARD.encode(bytes))),
            )
        })
        .collect()
});

pub(super) fn image(name: &'static str) -> ImageSrc {
    IMAGE_SOURCES.get(name).expect("embedded GUI image").clone()
}

pub(super) fn icon(name: &'static str, side: f32, tint: u32) -> Svg {
    let markup = ICONS
        .iter()
        .find(|(path, _)| *path == name)
        .map(|(_, markup)| *markup)
        .expect("embedded GUI icon");
    svg(ImageSrc::svg_markup(markup))
        .size(px(side))
        .text_color(color(tint))
}
