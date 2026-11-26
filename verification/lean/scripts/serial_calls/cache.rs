// Included after the real cache implementation in the verification copy.
impl RawEntry {
    fn serial_words(self) -> String {
        let Self {
            player,
            opponent,
            lower,
            upper,
        } = self;
        format!("{player} {opponent} {lower} {upper}")
    }
}

impl<K> EndGameCache<K> {
    pub(super) fn serial_base(&self) -> usize {
        self.table.as_ptr() as usize
    }

    fn serial_kind() -> &'static str {
        std::any::type_name::<K>().rsplit("::").next().unwrap()
    }

    pub fn new(bytes: usize) -> Self {
        let cache = Self::serial_allocate(bytes);
        assert!(
            cache
                .table
                .iter()
                .all(|e| e.player == 0 && e.opponent == 0 && e.lower == -65 && e.upper == 65)
        );
        eprintln!(
            "ALLOC {} {} {} {}",
            Self::serial_kind(),
            cache.table.as_ptr() as usize,
            cache.table.len(),
            std::mem::size_of::<RawEntry>()
        );
        cache
    }

    // Input board and threshold, raw resident fields and actual Option discriminant
    // are separate operands. Read the cell even when the engine reports a miss.
    pub fn probe(&self, slot: CacheSlot<K>, board: &Board, alpha: DiscScore) -> Option<DiscScore> {
        let i = slot.0;
        let result = self.serial_probe(slot, board, alpha);
        let result = if result.is_some() && serial_fault("miss") {
            None
        } else {
            result
        };
        let raw = &self.table[i];
        eprintln!(
            "LOAD {} {} {} {} {} {} {} {} {} {}",
            Self::serial_kind(),
            self.table.as_ptr() as usize,
            raw as *const _ as usize,
            i,
            board.hash(),
            board.player().bits(),
            board.opponent().bits(),
            alpha.value(),
            result.map_or(i32::MIN, |v| v.value()),
            raw.serial_words()
        );
        result
    }

    pub fn store(&mut self, slot: CacheSlot<K>, board: &Board, alpha: DiscScore, score: DiscScore) {
        let i = slot.0;
        let before = self.table[i];
        self.serial_store(slot, board, alpha, score);
        if serial_fault("forget") {
            // A dropped bound can cost time while leaving the game result intact.
            self.table[i].lower = -65;
            self.table[i].upper = 65;
        }
        let raw = &self.table[i];
        eprintln!(
            "SAVE {} {} {} {} {} {} {} {} {} {} {}",
            Self::serial_kind(),
            self.table.as_ptr() as usize,
            raw as *const _ as usize,
            i,
            board.hash(),
            board.player().bits(),
            board.opponent().bits(),
            alpha.value(),
            score.value(),
            before.serial_words(),
            raw.serial_words()
        );
    }
}

impl<K> Drop for EndGameCache<K> {
    fn drop(&mut self) {
        eprintln!("FREE {}", self.table.as_ptr() as usize);
    }
}

fn serial_fault(name: &str) -> bool {
    static MODE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    static INJECTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    MODE.get_or_init(|| std::env::var("REVERSI_TRACE_CACHE_FAULT").unwrap_or_default()) == name
        && !INJECTED.swap(true, std::sync::atomic::Ordering::Relaxed)
}
