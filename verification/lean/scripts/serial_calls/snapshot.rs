// Included inside search::context in the verification copy only.
thread_local! {
    static SERIAL_CONTEXT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

pub(super) fn watch_serial_context(active: bool) -> bool {
    SERIAL_CONTEXT.with(|flag| flag.replace(active))
}

impl SearchContext<'_> {
    pub(super) fn serial_snapshot(&self) -> String {
        use std::fmt::Write;
        let mut image = format!(
            "{},{},{},{}",
            self.empty_list.count(),
            self.empty_list.parity(),
            self.ply(),
            self.side_to_move as u8
        );
        for (next, prev) in self.empty_list.serial_links() {
            write!(&mut image, ",{next},{prev}").unwrap();
        }
        image
    }

    fn serial_before(&self) -> Option<String> {
        SERIAL_CONTEXT
            .with(|flag| flag.get())
            .then(|| self.serial_snapshot())
    }

    fn serial_change(&mut self, kind: &str, square: i32, before: Option<String>) {
        if let Some(before) = before {
            if kind == "undo" && std::env::var_os("REVERSI_TRACE_MUTATE_INACTIVE").is_some() {
                self.empty_list.serial_corrupt_inactive();
            }
            eprintln!("CTX {kind} {square} {before} {}", self.serial_snapshot());
        }
    }
}
