// Used only by the concrete aspiration comparison in the temporary crate.
impl TranspositionTable {
    pub(super) fn serial_dump_image(&self, accepted: bool) {
        let mut fault = if accepted {
            std::env::var("REVERSI_TRACE_IMAGE_FAULT").ok()
        } else {
            None
        };
        for (i, entry) in self.entries.iter().enumerate() {
            if let Some((_, _, seq, data)) = entry.try_load_snapshot()
                && data.is_occupied()
                && let Some(which) = fault.take()
            {
                // The accepted search has returned. Change one physical field for
                // the negative control without affecting its moves or root score.
                match which.as_str() {
                    "move" => {
                        let changed = TTEntryData {
                            best_move: data.best_move ^ 1,
                            ..data
                        };
                        entry.data.store(changed.to_u64(), Ordering::Relaxed);
                    }
                    "sequence" => entry.seq.store(seq.wrapping_sub(2), Ordering::Relaxed),
                    _ => panic!("unknown table image mutation"),
                }
            }
            eprintln!("TTC {i} {}", entry.trace_words());
        }
    }
}
