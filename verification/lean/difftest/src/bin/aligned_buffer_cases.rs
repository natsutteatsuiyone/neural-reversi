use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::RefCell;
use std::io::{self, Write};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};

#[allow(dead_code)]
#[path = "../../../../../crates/reversi-core/src/constants.rs"]
mod constants;

#[allow(dead_code)]
#[path = "../../../../../crates/reversi-core/src/util/aligned_buffer.rs"]
mod aligned_buffer;

use aligned_buffer::AlignedBuffer;
use constants::CACHE_LINE_SIZE;

static BYTES: AtomicUsize = AtomicUsize::new(0);
static ALLOCATED: AtomicUsize = AtomicUsize::new(0);
static FREED: AtomicUsize = AtomicUsize::new(0);
static LARGE_FREED: AtomicUsize = AtomicUsize::new(0);
static ADDRESS: AtomicUsize = AtomicUsize::new(0);
static LARGE: AtomicBool = AtomicBool::new(false);

struct RecordingHeap;

#[global_allocator]
static HEAP: RecordingHeap = RecordingHeap;

unsafe impl GlobalAlloc for RecordingHeap {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !LARGE.load(Relaxed)
            && layout.align() == CACHE_LINE_SIZE
            && layout.size() == BYTES.load(Relaxed)
        {
            ADDRESS.store(ptr.addr(), Relaxed);
            ALLOCATED.fetch_add(1, Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if ptr.addr() == ADDRESS.load(Relaxed)
            && layout.size() == BYTES.load(Relaxed)
            && layout.align() == CACHE_LINE_SIZE
        {
            FREED.fetch_add(1, Relaxed);
        }
        unsafe { System.dealloc(ptr, layout) };
    }
}

mod util {
    pub mod large_pages {
        use std::alloc::{GlobalAlloc, Layout, System};
        use std::ptr::NonNull;

        use super::super::{
            ADDRESS, ALLOCATED, BYTES, CACHE_LINE_SIZE, LARGE, LARGE_FREED, Relaxed,
        };

        pub fn alloc_zeroed(size: usize) -> Option<NonNull<u8>> {
            if !LARGE.load(Relaxed) {
                return None;
            }
            let layout = Layout::from_size_align(size, CACHE_LINE_SIZE).unwrap();
            let ptr = NonNull::new(unsafe { System.alloc_zeroed(layout) })?;
            ADDRESS.store(ptr.as_ptr().addr(), Relaxed);
            ALLOCATED.fetch_add(1, Relaxed);
            Some(ptr)
        }

        pub unsafe fn free(ptr: NonNull<u8>) {
            assert_eq!(ptr.as_ptr().addr(), ADDRESS.load(Relaxed));
            let layout = Layout::from_size_align(BYTES.load(Relaxed), CACHE_LINE_SIZE).unwrap();
            LARGE_FREED.fetch_add(1, Relaxed);
            unsafe { System.dealloc(ptr.as_ptr(), layout) };
        }
    }
}

struct Element<'a> {
    id: usize,
    panic_at_drop: Option<usize>,
    drops: &'a RefCell<Vec<usize>>,
}

impl Drop for Element<'_> {
    fn drop(&mut self) {
        self.drops.borrow_mut().push(self.id);
        if self.panic_at_drop == Some(self.id) {
            panic!("fixture destructor panic");
        }
    }
}

#[derive(Clone, Copy)]
enum End {
    Complete,
    Short(usize),
    Panic(usize),
    DropPanic,
}

struct Source<'a> {
    len: usize,
    next: usize,
    end: End,
    panic_at_drop: Option<usize>,
    drops: &'a RefCell<Vec<usize>>,
}

impl<'a> Iterator for Source<'a> {
    type Item = Element<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.end {
            End::Short(i) if i == self.next => return None,
            End::Panic(i) if i == self.next => panic!("fixture iterator panic"),
            _ => {}
        }
        if self.next == self.len {
            return None;
        }
        let id = self.next;
        self.next += 1;
        Some(Element {
            id,
            panic_at_drop: self.panic_at_drop,
            drops: self.drops,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.len - self.next;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for Source<'_> {}

impl Drop for Source<'_> {
    fn drop(&mut self) {
        if matches!(self.end, End::DropPanic) {
            panic!("fixture iterator destructor panic");
        }
    }
}

fn fixture(
    out: &mut impl Write,
    large: bool,
    count: usize,
    end: End,
    panic_at_drop: Option<usize>,
) -> io::Result<()> {
    let drops = RefCell::new(Vec::with_capacity(count));
    let mut completed = false;
    let mut contents = Vec::with_capacity(count);
    let source = Source {
        len: count,
        next: 0,
        end,
        panic_at_drop,
        drops: &drops,
    };

    ALLOCATED.store(0, Relaxed);
    FREED.store(0, Relaxed);
    LARGE_FREED.store(0, Relaxed);
    ADDRESS.store(0, Relaxed);
    BYTES.store(count * size_of::<Element>(), Relaxed);
    LARGE.store(large, Relaxed);
    let result = catch_unwind(AssertUnwindSafe(|| {
        let buffer = AlignedBuffer::from_iter(source);
        completed = true;
        assert_eq!(buffer.as_ptr().addr() % CACHE_LINE_SIZE, 0);
        contents.extend(buffer.iter().map(|item| item.id));
        drop(buffer);
    }));
    LARGE.store(false, Relaxed);
    BYTES.store(0, Relaxed);
    let allocated = ALLOCATED.load(Relaxed);
    let freed = FREED.load(Relaxed);
    let large_freed = LARGE_FREED.load(Relaxed);
    assert_eq!(allocated, usize::from(count > 0), "one buffer allocation");
    assert_eq!(
        freed + large_freed,
        allocated,
        "buffer allocation was not released"
    );
    assert_eq!(
        large_freed,
        if large { allocated } else { 0 },
        "incorrect backing release"
    );

    let (kind, stop) = match end {
        End::Complete => ("all", count),
        End::Short(i) => ("short", i),
        End::Panic(i) => ("panic", i),
        End::DropPanic => ("drop-panic", count),
    };
    let as_list = |xs: &[usize]| {
        if xs.is_empty() {
            "-".into()
        } else {
            xs.iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(",")
        }
    };
    writeln!(
        out,
        "f {} {count} {kind} {stop} {} {} {} {allocated} {freed} {large_freed} {} {}",
        usize::from(large),
        panic_at_drop.unwrap_or(count),
        usize::from(completed),
        usize::from(result.is_err()),
        as_list(&contents),
        as_list(&drops.borrow())
    )
}

fn layouts(out: &mut impl Write) -> io::Result<()> {
    for item_size in [0, 1, 2, 3, 8, 32, 64, usize::MAX] {
        for count in [
            0,
            1,
            2,
            7,
            12,
            128,
            isize::MAX as usize - 64,
            isize::MAX as usize - 63,
            isize::MAX as usize - 62,
            isize::MAX as usize,
            usize::MAX - 1,
            usize::MAX,
        ] {
            let (tag, bytes) = match count.checked_mul(item_size) {
                None => ("overflow", 0),
                Some(0) => ("empty", 0),
                Some(bytes) => match Layout::from_size_align(bytes, CACHE_LINE_SIZE) {
                    Ok(layout) => ("alloc", layout.size()),
                    Err(_) => ("layout", 0),
                },
            };
            writeln!(out, "l {} {count} {item_size} {tag} {bytes}", usize::BITS)?;
        }
    }
    Ok(())
}

fn main() -> io::Result<()> {
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        if !info
            .payload()
            .downcast_ref::<&str>()
            .copied()
            .or_else(|| info.payload().downcast_ref::<String>().map(String::as_str))
            .is_some_and(|msg| {
                msg.starts_with("fixture") || msg.starts_with("AlignedBuffer::from_iter")
            })
        {
            hook(info);
        }
    }));
    let mut stdout = io::BufWriter::new(io::stdout().lock());
    layouts(&mut stdout)?;
    for large in [false, true] {
        for count in 0..=12 {
            fixture(&mut stdout, large, count, End::Complete, None)?;
            fixture(&mut stdout, large, count, End::DropPanic, None)?;
            for i in 0..count {
                fixture(&mut stdout, large, count, End::Short(i), None)?;
                fixture(&mut stdout, large, count, End::Panic(i), None)?;
                fixture(&mut stdout, large, count, End::Complete, Some(i))?;
            }
        }
    }
    stdout.flush()
}
