use nsh_core::prelude::{Hsla, Points, Rgba, pixel_ratio, px};

pub(super) const BACKGROUND: u32 = 0x1c1c21;
pub(super) const PANEL: u32 = 0x252529;
pub(super) const SECONDARY: u32 = 0xa8a8aa;
pub(super) const ACCENT: u32 = 0x43a575;
pub(super) const GOLD: u32 = 0xb8956c;
pub(super) const BLUE: u32 = 0x7ba0c8;
pub(super) const RED: u32 = 0xc45c5c;
pub(super) const AI_COLOR: u32 = 0x5fb3c4;

pub(super) const TOKENS: &str = r##"{"cssVars":{"light":{"background":"#1c1c21","foreground":"#ededed","card":"#252529","popover":"#2a2a2f","primary":"#43a575","primary-foreground":"#ffffff","muted":"#2e2e32","muted-foreground":"#a8a8aa","border":"#3c3c40","ring":"#43a575","radius":"8px"},"dark":{"background":"#1c1c21","foreground":"#ededed","card":"#252529","popover":"#2a2a2f","primary":"#43a575","primary-foreground":"#ffffff","muted":"#2e2e32","muted-foreground":"#a8a8aa","border":"#3c3c40","ring":"#43a575","radius":"8px"}}}"##;

pub(super) fn layout_px(value: f32) -> Points {
    let ratio = pixel_ratio();
    let physical = value * ratio;
    // Half-pixel ties round toward zero.
    px((physical.abs() - 0.5).ceil().copysign(physical) / ratio)
}

pub(super) fn stroke_px(value: f32) -> Points {
    if value == 0. {
        px(0.)
    } else {
        px(layout_px(value.max(0.)).0.max(1. / pixel_ratio()))
    }
}

pub(super) fn color(rgb: u32) -> Hsla {
    Rgba {
        r: ((rgb >> 16) & 255) as f32 / 255.,
        g: ((rgb >> 8) & 255) as f32 / 255.,
        b: (rgb & 255) as f32 / 255.,
        a: 1.,
    }
    .into()
}

pub(super) fn rgba(rgba: u32) -> Hsla {
    Rgba {
        r: ((rgba >> 24) & 255) as f32 / 255.,
        g: ((rgba >> 16) & 255) as f32 / 255.,
        b: ((rgba >> 8) & 255) as f32 / 255.,
        a: (rgba & 255) as f32 / 255.,
    }
    .into()
}
