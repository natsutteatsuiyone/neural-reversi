#[derive(Clone, Copy)]
pub(super) enum ScorePrecision {
    Whole,
    Tenth,
    Raw,
}

// Rounds halves toward positive infinity, including negative half scores.
pub(super) fn whole_score(score: f32) -> i32 {
    (score + 0.5).floor() as i32
}

pub(super) fn display_score(score: f32, precision: ScorePrecision) -> String {
    let mut reduced = match precision {
        ScorePrecision::Whole => format!("{:+}", whole_score(score)),
        ScorePrecision::Tenth => format!("{score:+.1}"),
        ScorePrecision::Raw => format!("{score:+}"),
    };
    if matches!(reduced.as_str(), "+0" | "-0" | "+0.0" | "-0.0") {
        reduced.remove(0);
    }
    reduced
}

pub(super) fn display_depth(depth: u32, probability: i32) -> String {
    if probability == 100 {
        depth.to_string()
    } else {
        format!("{depth}@{probability}%")
    }
}

/// `value` with a K, M, or G suffix; M and G show `precision` decimals.
pub(super) fn compact_number(value: f64, precision: usize) -> String {
    if value >= 1_000_000_000. {
        format!("{:.*}G", precision, value / 1_000_000_000.)
    } else if value >= 1_000_000. {
        format!("{:.*}M", precision, value / 1_000_000.)
    } else if value >= 1_000. {
        format!("{:.1}K", value / 1_000.)
    } else {
        format!("{value:.0}")
    }
}
