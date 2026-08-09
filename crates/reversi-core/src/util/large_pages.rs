//! Large-page backing memory.
//!
//! Transposition-table probes are uniformly distributed over the whole table,
//! so every access is a dTLB miss candidate: a 1 GiB table spans 262144 4 KiB
//! pages but only 512 pages when the large-page minimum is 2 MiB. Windows
//! hands out large pages only while `SeLockMemoryPrivilege` ("Lock pages in
//! memory" in the local security policy) is enabled, so allocation is
//! best-effort and callers must keep a fallback path.

#[cfg(windows)]
mod imp {
    use std::mem;
    use std::ptr;
    use std::ptr::NonNull;
    use std::sync::Mutex;

    use windows_sys::Win32::Foundation::{CloseHandle, ERROR_SUCCESS, GetLastError, HANDLE, LUID};
    use windows_sys::Win32::Security::{
        AdjustTokenPrivileges, LUID_AND_ATTRIBUTES, LookupPrivilegeValueW, SE_LOCK_MEMORY_NAME,
        SE_PRIVILEGE_ENABLED, TOKEN_ADJUST_PRIVILEGES, TOKEN_PRIVILEGES, TOKEN_QUERY,
    };
    use windows_sys::Win32::System::Memory::{
        GetLargePageMinimum, MEM_COMMIT, MEM_LARGE_PAGES, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE,
        VirtualAlloc, VirtualFree,
    };
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, OpenProcessToken};

    /// Windows `BOOL` false, as returned by a failing Win32 call.
    const FALSE: windows_sys::core::BOOL = 0;

    /// Serializes changes to the process-wide lock-memory privilege.
    static PRIVILEGE_LOCK: Mutex<()> = Mutex::new(());

    /// Allocates at least `size` zeroed bytes backed by large pages.
    ///
    /// Returns [`None`] when `size` is smaller than one large page or the OS
    /// cannot provide the allocation. The block is large-page aligned and
    /// rounded up to whole pages, so it may exceed `size`.
    pub fn alloc_zeroed(size: usize) -> Option<NonNull<u8>> {
        // SAFETY: takes no arguments; returns 0 when large pages are unsupported.
        let page_size = unsafe { GetLargePageMinimum() };
        // Below one page the rounding would waste more memory than the saved
        // TLB pressure is worth.
        if page_size == 0 || size < page_size {
            return None;
        }

        let _lock = PRIVILEGE_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let _privilege = LockMemoryPrivilege::acquire()?;

        // SAFETY: a null base address lets the OS pick the region; committed
        // pages are zero-filled by Windows.
        let mem = unsafe {
            VirtualAlloc(
                ptr::null(),
                size.next_multiple_of(page_size),
                MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES,
                PAGE_READWRITE,
            )
        };
        NonNull::new(mem.cast::<u8>())
    }

    /// Releases a block obtained from [`alloc_zeroed`].
    ///
    /// # Safety
    ///
    /// `ptr` must come from [`alloc_zeroed`] and must not have been freed.
    pub unsafe fn free(ptr: NonNull<u8>) {
        // SAFETY: MEM_RELEASE frees the whole reservation and requires size 0.
        unsafe { VirtualFree(ptr.as_ptr().cast(), 0, MEM_RELEASE) };
    }

    /// Guard enabling `SeLockMemoryPrivilege` for as long as it is held.
    struct LockMemoryPrivilege {
        token: HANDLE,
        previous: TOKEN_PRIVILEGES,
    }

    impl LockMemoryPrivilege {
        fn acquire() -> Option<Self> {
            let mut token: HANDLE = ptr::null_mut();
            // SAFETY: `token` is a live out-parameter.
            let opened = unsafe {
                OpenProcessToken(
                    GetCurrentProcess(),
                    TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY,
                    &mut token,
                )
            };
            if opened == FALSE {
                return None;
            }

            // From here on the guard owns `token`, so every early return closes
            // it. An all-zero `previous` means "adjust nothing", which makes the
            // restore in `drop` a no-op until the real state is saved below.
            // SAFETY: TOKEN_PRIVILEGES and LUID are plain C structs.
            let mut guard = LockMemoryPrivilege {
                token,
                previous: unsafe { mem::zeroed() },
            };
            let mut luid: LUID = unsafe { mem::zeroed() };

            // SAFETY: a null system name looks the privilege up on this machine.
            let looked_up =
                unsafe { LookupPrivilegeValueW(ptr::null(), SE_LOCK_MEMORY_NAME, &mut luid) };
            if looked_up == FALSE {
                return None;
            }

            let requested = TOKEN_PRIVILEGES {
                PrivilegeCount: 1,
                Privileges: [LUID_AND_ATTRIBUTES {
                    Luid: luid,
                    Attributes: SE_PRIVILEGE_ENABLED,
                }],
            };
            let mut previous_len = 0u32;
            // SAFETY: the request and both out-parameters are live locals.
            let adjusted = unsafe {
                AdjustTokenPrivileges(
                    guard.token,
                    FALSE,
                    &requested,
                    mem::size_of::<TOKEN_PRIVILEGES>() as u32,
                    &mut guard.previous,
                    &mut previous_len,
                )
            };
            // A token that simply does not hold the privilege still reports
            // success, with ERROR_NOT_ALL_ASSIGNED, so the error code decides.
            // SAFETY: reads the calling thread's last-error value.
            if adjusted == FALSE || unsafe { GetLastError() } != ERROR_SUCCESS {
                return None;
            }

            Some(guard)
        }
    }

    impl Drop for LockMemoryPrivilege {
        fn drop(&mut self) {
            // SAFETY: `token` is a live handle owned by this guard.
            unsafe {
                AdjustTokenPrivileges(
                    self.token,
                    FALSE,
                    &self.previous,
                    0,
                    ptr::null_mut(),
                    ptr::null_mut(),
                );
                CloseHandle(self.token);
            }
        }
    }
}

#[cfg(not(windows))]
mod imp {
    use std::ptr::NonNull;

    /// Always [`None`]: large pages are only wired up for Windows.
    pub fn alloc_zeroed(_size: usize) -> Option<NonNull<u8>> {
        None
    }

    /// Unreachable: [`alloc_zeroed`] never hands out a block to release.
    ///
    /// # Safety
    ///
    /// `ptr` must come from [`alloc_zeroed`], which cannot happen here.
    pub unsafe fn free(_ptr: NonNull<u8>) {
        unreachable!("no large-page allocation on this platform")
    }
}

pub use imp::{alloc_zeroed, free};

#[cfg(test)]
mod tests {
    use super::*;

    /// Exercises the allocate/release pair, which on Windows runs the whole
    /// privilege-elevation path. Large pages are unavailable on machines
    /// without the "Lock pages in memory" right, so the test reports which
    /// path it took instead of requiring one.
    #[test]
    fn alloc_zeroed_round_trip() {
        const SIZE: usize = 4 << 20;

        match alloc_zeroed(SIZE) {
            Some(ptr) => {
                println!("large pages: available");
                assert_eq!(ptr.as_ptr() as usize % (2 << 20), 0);
                // SAFETY: the block is at least `SIZE` bytes and readable.
                let bytes = unsafe { std::slice::from_raw_parts(ptr.as_ptr(), SIZE) };
                assert!(bytes.iter().all(|&b| b == 0));
                // SAFETY: `ptr` comes from the `alloc_zeroed` call above.
                unsafe { free(ptr) };
            }
            None => println!("large pages: unavailable"),
        }
    }
}
