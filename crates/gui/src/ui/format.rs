use reversi_core::score::Score;

#[derive(Clone, Copy)]
pub(super) enum ScorePrecision {
    Whole,
    Tenth,
    Raw,
}

// Rounds halves toward positive infinity, including negative half scores.
pub(super) fn whole_score(score: Score) -> i32 {
    (score.to_disc_diff_f32() + 0.5).floor() as i32
}

// Rounds halves away from zero; `{:.1}` alone would round x.x5 to even.
pub(super) fn tenth_score(score: Score) -> f32 {
    (score.to_disc_diff_f32() * 10.0).round() / 10.0
}

pub(super) fn display_score(score: Score, precision: ScorePrecision) -> String {
    let tenth = tenth_score(score);
    let mut reduced = match precision {
        ScorePrecision::Whole => format!("{:+}", whole_score(score)),
        ScorePrecision::Tenth => format!("{tenth:+.1}"),
        ScorePrecision::Raw => format!("{tenth:+}"),
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
