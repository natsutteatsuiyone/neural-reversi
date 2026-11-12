//! Table of the endgame child positions being searched.
//!
//! A serial move loop searches a fresh sibling before a child another thread
//! is searching, as in ABDADA. A claim is only a scheduling hint, never a
//! score or a bound: a hash collision loses a hint, never search work.

use std::sync::atomic::{AtomicU64, Ordering};

const SLOTS: usize = 1 << 16;

/// Value of an unclaimed slot; keys set bit 0, so no key equals it.
const EMPTY: u64 = 0;

/// Hash-indexed slots holding the child positions that threads are searching.
pub(in crate::search) struct InFlightTable {
    slots: Box<[AtomicU64]>,
}

/// Result of claiming a child position; dropping it releases an owned slot.
pub(in crate::search) enum Claim<'a> {
    /// This thread advertises the child until the claim is dropped.
    Owned(&'a AtomicU64),
    /// Another thread is searching the same child.
    Busy,
    /// Nothing is advertised: claims are off, or another position holds the slot.
    Untracked,
}

impl InFlightTable {
    pub(super) fn new() -> Self {
        Self {
            slots: (0..SLOTS).map(|_| AtomicU64::new(EMPTY)).collect(),
        }
    }

    /// Claims the position with `hash` for the calling thread.
    #[inline]
    pub(super) fn claim(&self, hash: u64) -> Claim<'_> {
        let key = hash | 1;
        let slot = &self.slots[hash as usize & (SLOTS - 1)];
        match slot.compare_exchange(EMPTY, key, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => Claim::Owned(slot),
            Err(owner) if owner == key => Claim::Busy,
            Err(_) => Claim::Untracked,
        }
    }
}

impl Drop for Claim<'_> {
    #[inline]
    fn drop(&mut self) {
        if let Claim::Owned(slot) = self {
            slot.store(EMPTY, Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn colliding_claim_neither_reports_busy_nor_releases_the_owner() {
        let table = InFlightTable::new();
        let owner = table.claim(42);
        assert!(matches!(owner, Claim::Owned(_)));
        assert!(matches!(table.claim(42), Claim::Busy));
        drop(table.claim(42 + SLOTS as u64));
        assert!(matches!(table.claim(42), Claim::Busy));
        drop(owner);
        assert!(matches!(table.claim(42), Claim::Owned(_)));
    }
}
