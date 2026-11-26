// Compiled inside the temporary, instrumented transposition_table module.
thread_local! {
    static WINDOW_TRACE: std::cell::Cell<Option<(usize, usize)>> = const { std::cell::Cell::new(None) };
    static HOLD_WRITE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static FIRST_LOAD: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
static FIRST_LOAD_BARRIER: std::sync::OnceLock<std::sync::Barrier> = std::sync::OnceLock::new();

static WRITER_GATE: std::sync::OnceLock<std::sync::Barrier> = std::sync::OnceLock::new();

fn window_hold_claim() {
    if HOLD_WRITE.replace(false) {
        let gate = WRITER_GATE.get().unwrap();
        gate.wait();
        gate.wait();
    }
}

fn window_op(slot: Option<&TTEntry>, opcode: &str, word: u64) {
    if let Some((worker, base)) = WINDOW_TRACE.get() {
        let index = slot.map_or(0, |s| {
            (s as *const _ as usize - base) / mem::size_of::<TTEntry>()
        });
        println!("op {worker} {opcode} {index} {word}");
        if FIRST_LOAD.replace(false) {
            FIRST_LOAD_BARRIER.get().unwrap().wait();
        }
    }
}

fn window_read(
    table: &TranspositionTable,
    b: &Board,
    key: u64,
    worker: usize,
    probe: bool,
    paired: bool,
) {
    let base = table.entries.as_ptr() as usize;
    let kind = if probe { "probe" } else { "lookup" };
    println!(
        "enter {worker} {kind} {} {} {} {key}",
        b.player().bits(),
        b.opponent().bits(),
        table.cluster_count
    );
    WINDOW_TRACE.set(Some((worker, base)));
    FIRST_LOAD.set(paired);
    if probe {
        let result = table.probe(b, key);
        let addr = result.slot().0 as *const TTEntry as usize;
        let slot = ((addr - base) / mem::size_of::<TTEntry>()) % CLUSTER_SIZE;
        println!(
            "leave {worker} probe {slot} {}",
            result.data().map_or(0, TTEntryData::to_u64)
        );
    } else {
        let data = table.lookup(b, key);
        println!(
            "leave {worker} lookup 0 {}",
            data.map_or(0, TTEntryData::to_u64)
        );
    }
    WINDOW_TRACE.set(None);
}

pub(crate) fn run_reader_window() {
    let boards = [
        Board::from_bitboards(33024u64, 66u64),
        Board::from_bitboards(33280u64, 66u64),
        Board::from_bitboards(33792u64, 66u64),
    ];
    let table = TranspositionTable::new(0);
    let key = u64::MAX;
    println!("endian {}", cfg!(target_endian = "big"));
    window_read(&table, &boards[0], key, 2, true, false);
    for (i, b) in boards[..2].iter().enumerate() {
        let slot = table.probe(b, key).slot();
        table.store(
            slot,
            b,
            Score::from_raw((i as i32 + 1) * 256),
            Bound::Exact,
            5,
            None,
            Selectivity::None,
            true,
            true,
        );
    }
    FIRST_LOAD_BARRIER.set(std::sync::Barrier::new(2)).unwrap();
    std::thread::scope(|scope| {
        for worker in 0..2 {
            let table = &table;
            let boards = &boards;
            scope.spawn(move || {
                for (index, probe) in [(0, true), (1, false), (2, true), (2, false), (0, true)] {
                    let b = &boards[(index + worker) % boards.len()];
                    window_read(table, b, key, worker, probe, true);
                }
            });
        }
    });

    let gate = WRITER_GATE.get_or_init(|| std::sync::Barrier::new(2));
    std::thread::scope(|scope| {
        scope.spawn(|| {
            let slot = table.probe(&boards[0], key).slot();
            HOLD_WRITE.set(true);
            table.store(
                slot,
                &boards[0],
                Score::from_raw(768),
                Bound::Exact,
                5,
                None,
                Selectivity::None,
                true,
                true,
            );
        });
        gate.wait();
        window_read(&table, &boards[0], key, 3, true, false);
        gate.wait();
    });
    window_read(&table, &boards[0], key, 4, true, false);
}
