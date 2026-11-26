use std::io::{self, Write};

#[path = "../../../../../crates/reversi-core/src/search/counters.rs"]
mod counters;

use counters::{SearchCounters, SearchStats};
use flip_scalar_difftest::run_buffered;

const INCREMENTS: [fn(&mut SearchStats); 10] = [
    SearchStats::increment_tt_probe,
    SearchStats::increment_tt_hit,
    SearchStats::increment_tt_penalize,
    SearchStats::increment_probcut_attempt,
    SearchStats::increment_probcut_cut,
    SearchStats::increment_tt_probcut_attempt,
    SearchStats::increment_tt_probcut_cut,
    SearchStats::increment_etc_attempt,
    SearchStats::increment_etc_cut,
    SearchStats::increment_stability_cut,
];

fn with_incremented(bits: u64) -> SearchStats {
    let mut stats = SearchStats::default();
    for (bit, increment) in INCREMENTS.into_iter().enumerate() {
        if bits & (1 << bit) != 0 {
            increment(&mut stats);
        }
    }
    stats
}

fn render(counters: &SearchCounters) -> String {
    #[cfg(feature = "search-stats")]
    let fields = [
        counters.tt_probes,
        counters.tt_hits,
        counters.tt_penalizes,
        counters.probcut_attempts,
        counters.probcut_cuts,
        counters.tt_probcut_attempts,
        counters.tt_probcut_cuts,
        counters.etc_attempts,
        counters.etc_cuts,
        counters.stability_cuts,
    ];
    #[cfg(not(feature = "search-stats"))]
    let fields = [];
    std::iter::once(counters.n_nodes)
        .chain(fields)
        .map(|value| value.to_string())
        .collect::<Vec<_>>()
        .join(" ")
}

fn main() -> io::Result<()> {
    run_buffered(|out| {
        let enabled = u8::from(cfg!(feature = "search-stats"));
        let node_counts = [0, 1, 0xFE, u64::MAX / 2, u64::MAX - 1, u64::MAX];
        for (i, current) in node_counts.into_iter().enumerate() {
            for (j, previous) in node_counts.into_iter().enumerate() {
                for shift in 0..10 {
                    let first = (1 << shift) | (1 << ((shift + j) % 10));
                    let second = (1 << ((shift + i) % 10)) | (1 << ((shift + 1) % 10));
                    let mut left = with_incremented(first);
                    let right = with_incremented(second).with_nodes(previous);
                    let before = left.with_nodes(current);
                    let mut total = before.clone();
                    total.merge(&right);
                    left.merge(&right);
                    let updated = left.with_nodes(current);
                    #[cfg(feature = "search-stats")]
                    let delta = render(&total.delta_since(&before));
                    #[cfg(not(feature = "search-stats"))]
                    let delta = "-";
                    writeln!(
                        out,
                        "{enabled} {current} {previous} {first} {second}|{}|{}|{}|{delta}",
                        render(&before),
                        render(&total),
                        render(&updated),
                    )?;
                }
            }
        }
        Ok(())
    })
}
