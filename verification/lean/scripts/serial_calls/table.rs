// Injected only in the temporary crate. Sample physical words separately from
// the production probe result and the independently reconstructed Python table.
impl TTEntry {
    fn trace_words(&self) -> String {
        format!(
            "{} {} {} {}",
            self.seq.load(Ordering::Relaxed),
            self.player.load(Ordering::Relaxed),
            self.opponent.load(Ordering::Relaxed),
            self.data.load(Ordering::Relaxed)
        )
    }
}

impl TTSlot<'_> {
    pub(crate) fn trace_addr(self) -> usize {
        self.0 as *const TTEntry as usize
    }
}

impl TTEntryData {
    pub(crate) fn trace_raw(self) -> u64 {
        self.to_u64()
    }
}

impl TranspositionTable {
    pub fn new(mb_size: usize) -> Self {
        let table = Self::trace_new(mb_size);
        assert!(table.entries.iter().all(|s| s.seq.load(Ordering::Relaxed)
            | s.player.load(Ordering::Relaxed)
            | s.opponent.load(Ordering::Relaxed)
            | s.data.load(Ordering::Relaxed)
            == 0));
        eprintln!(
            "ALLOC {} {} {}",
            table.entries.as_ptr() as usize,
            table.cluster_count,
            std::mem::size_of::<TTEntry>()
        );
        table
    }

    pub fn probe(&self, board: &Board, key: u64) -> TTProbeResult<'_> {
        let result = self.trace_probe(board, key);
        eprintln!(
            "PROBE {} {} {} {} {} {} {}",
            self.entries.as_ptr() as usize,
            board.player().bits(),
            board.opponent().bits(),
            key,
            self.generation(),
            result.slot().trace_addr(),
            result.data().map_or(0, TTEntryData::to_u64)
        );
        result
    }

    fn trace_lookup(&self, board: &Board, key: u64, physical: &TTEntry, data: Option<TTEntryData>) {
        eprintln!(
            "LOOK {} {} {} {} {} {}",
            self.entries.as_ptr() as usize,
            board.player().bits(),
            board.opponent().bits(),
            key,
            physical as *const _ as usize,
            data.map_or(0, TTEntryData::to_u64)
        );
    }
}
