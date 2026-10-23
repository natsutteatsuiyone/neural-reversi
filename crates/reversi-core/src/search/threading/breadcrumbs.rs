//! Marks of the positions the search threads are inside.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Number of slots; a power of two.
const SLOTS: usize = 4096;

/// Shared marks of the positions that search threads are currently inside.
///
/// A thread entering a position it finds marked by another thread knows the
/// subtree is already being searched and can spend its effort elsewhere.
pub(super) struct Breadcrumbs {
    slots: Box<[Slot]>,
}

#[derive(Default)]
struct Slot {
    /// Pool index plus one of the owning thread, or 0 when free.
    owner: AtomicUsize,
    key: AtomicU64,
}

/// A thread's claim on a position, released on drop.
pub(in crate::search) struct Breadcrumb<'a> {
    owned: Option<&'a Slot>,
    marked: bool,
}

impl Breadcrumbs {
    pub(super) fn new() -> Self {
        Self {
            slots: (0..SLOTS).map(|_| Slot::default()).collect(),
        }
    }

    /// Marks `key` as searched by thread `idx`, or notes that another thread
    /// already has it.
    pub(super) fn enter(&self, idx: usize, key: u64) -> Breadcrumb<'_> {
        let slot = &self.slots[key as usize & (SLOTS - 1)];
        let owner = slot.owner.load(Ordering::Relaxed);
        if owner == 0 {
            slot.key.store(key, Ordering::Relaxed);
            slot.owner.store(idx + 1, Ordering::Relaxed);
            return Breadcrumb {
                owned: Some(slot),
                marked: false,
            };
        }
        Breadcrumb {
            owned: None,
            marked: owner != idx + 1 && slot.key.load(Ordering::Relaxed) == key,
        }
    }

    /// Returns whether a thread other than `idx` is inside `key`, without
    /// marking it.
    pub(super) fn is_marked(&self, idx: usize, key: u64) -> bool {
        let slot = &self.slots[key as usize & (SLOTS - 1)];
        let owner = slot.owner.load(Ordering::Relaxed);
        owner != 0 && owner != idx + 1 && slot.key.load(Ordering::Relaxed) == key
    }
}

impl Breadcrumb<'_> {
    /// Returns whether another thread was already inside the position.
    #[inline]
    pub(in crate::search) fn is_marked(&self) -> bool {
        self.marked
    }
}

impl Drop for Breadcrumb<'_> {
    fn drop(&mut self) {
        if let Some(slot) = self.owned {
            slot.owner.store(0, Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn another_thread_sees_the_mark_until_the_owner_leaves() {
        let crumbs = Breadcrumbs::new();
        let owner = crumbs.enter(0, 42);
        assert!(!owner.is_marked());
        assert!(crumbs.enter(1, 42).is_marked());
        assert!(!crumbs.enter(0, 42).is_marked());
        assert!(!crumbs.enter(1, 42 + SLOTS as u64).is_marked());
        drop(owner);
        assert!(!crumbs.enter(1, 42).is_marked());
    }
}
