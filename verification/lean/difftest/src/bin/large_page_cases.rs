#![allow(non_camel_case_types, non_snake_case, clippy::upper_case_acronyms)]

use std::cell::RefCell;
use std::io::{self, Write};

use flip_scalar_difftest::{opt_or_dash, run_buffered};

extern crate self as windows_sys;

mod core {
    pub type BOOL = i32;
}

include!(concat!(env!("OUT_DIR"), "/windows_large_pages.rs"));

const OPENED: u8 = 1 << 0;
const FOUND: u8 = 1 << 1;
const ADJUSTED: u8 = 1 << 2;
const ASSIGNED: u8 = 1 << 3;
const ALLOCATED: u8 = 1 << 4;
const ALREADY_ENABLED: u8 = 1 << 5;
const ACQUIRED_PRIVILEGE: u8 = OPENED | FOUND | ADJUSTED | ASSIGNED;

#[derive(Default)]
struct Case {
    page: usize,
    flags: u8,
    trace: String,
    request: Option<usize>,
}

thread_local! {
    static CASE: RefCell<Case> = RefCell::default();
}

fn event(code: char, success_bit: u8) -> bool {
    CASE.with_borrow_mut(|case| {
        case.trace.push(code);
        case.flags & success_bit != 0
    })
}

mod Win32 {
    pub mod Foundation {
        use std::ffi::c_void;

        pub type HANDLE = *mut c_void;
        #[derive(Clone, Copy)]
        #[repr(C)]
        pub struct LUID {
            pub LowPart: u32,
            pub HighPart: i32,
        }
        pub const ERROR_SUCCESS: u32 = 0;

        pub unsafe fn CloseHandle(_: HANDLE) -> i32 {
            crate::event('c', 0);
            1
        }

        pub unsafe fn GetLastError() -> u32 {
            if crate::event('e', crate::ASSIGNED) {
                ERROR_SUCCESS
            } else {
                1300
            }
        }
    }

    pub mod Security {
        use super::Foundation::{HANDLE, LUID};

        #[repr(C)]
        pub struct LUID_AND_ATTRIBUTES {
            pub Luid: LUID,
            pub Attributes: u32,
        }

        #[repr(C)]
        pub struct TOKEN_PRIVILEGES {
            pub PrivilegeCount: u32,
            pub Privileges: [LUID_AND_ATTRIBUTES; 1],
        }

        pub const SE_LOCK_MEMORY_NAME: *const u16 = c"SeLockMemoryPrivilege".as_ptr().cast();
        pub const SE_PRIVILEGE_ENABLED: u32 = 2;
        pub const TOKEN_ADJUST_PRIVILEGES: u32 = 32;
        pub const TOKEN_QUERY: u32 = 8;

        pub unsafe fn LookupPrivilegeValueW(
            _: *const u16,
            name: *const u16,
            out: *mut LUID,
        ) -> i32 {
            assert_eq!(name, SE_LOCK_MEMORY_NAME);
            if crate::event('l', crate::FOUND) {
                unsafe {
                    out.write(LUID {
                        LowPart: 7,
                        HighPart: 0,
                    })
                };
                1
            } else {
                0
            }
        }

        pub unsafe fn AdjustTokenPrivileges(
            _: HANDLE,
            disable_all: i32,
            requested: *const TOKEN_PRIVILEGES,
            bytes: u32,
            previous: *mut TOKEN_PRIVILEGES,
            previous_bytes: *mut u32,
        ) -> i32 {
            assert_eq!(disable_all, 0);
            let requested = unsafe { &*requested };
            if previous.is_null() {
                assert_eq!(bytes, 0);
                assert!(previous_bytes.is_null());
                let changed = crate::CASE.with_borrow(|case| {
                    case.flags & (crate::ACQUIRED_PRIVILEGE | crate::ALREADY_ENABLED)
                        == crate::ACQUIRED_PRIVILEGE
                });
                assert_eq!(requested.PrivilegeCount, u32::from(changed));
                if requested.PrivilegeCount == 1 {
                    assert_eq!(requested.Privileges[0].Luid.LowPart, 7);
                    assert_eq!(requested.Privileges[0].Attributes, 0);
                }
                crate::event(if changed { 's' } else { 'r' }, 0);
                return 1;
            }

            assert_eq!(bytes as usize, std::mem::size_of::<TOKEN_PRIVILEGES>());
            assert_eq!(requested.PrivilegeCount, 1);
            let luid = requested.Privileges[0].Luid;
            assert_eq!((luid.LowPart, luid.HighPart), (7, 0));
            assert_eq!(requested.Privileges[0].Attributes, SE_PRIVILEGE_ENABLED);
            if !crate::event('a', crate::ADJUSTED) {
                return 0;
            }
            if crate::CASE.with_borrow(|case| {
                case.flags & (crate::ASSIGNED | crate::ALREADY_ENABLED) == crate::ASSIGNED
            }) {
                unsafe {
                    previous.write(TOKEN_PRIVILEGES {
                        PrivilegeCount: 1,
                        Privileges: [LUID_AND_ATTRIBUTES {
                            Luid: luid,
                            Attributes: 0,
                        }],
                    });
                    previous_bytes.write(bytes);
                }
            }
            1
        }
    }

    pub mod System {
        pub mod Memory {
            use std::{ffi::c_void, ptr};

            pub const MEM_COMMIT: u32 = 0x1000;
            pub const MEM_RESERVE: u32 = 0x2000;
            pub const MEM_RELEASE: u32 = 0x8000;
            pub const MEM_LARGE_PAGES: u32 = 0x2000_0000;
            pub const PAGE_READWRITE: u32 = 4;

            pub unsafe fn GetLargePageMinimum() -> usize {
                crate::CASE.with_borrow(|case| case.page)
            }

            pub unsafe fn VirtualAlloc(
                base: *const c_void,
                size: usize,
                flags: u32,
                protection: u32,
            ) -> *mut c_void {
                assert!(base.is_null());
                assert_eq!(flags, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES);
                assert_eq!(protection, PAGE_READWRITE);
                crate::CASE.with_borrow_mut(|case| case.request = Some(size));
                if crate::event('v', crate::ALLOCATED) {
                    ptr::dangling_mut::<u8>().cast()
                } else {
                    ptr::null_mut()
                }
            }

            pub unsafe fn VirtualFree(base: *mut c_void, size: usize, flags: u32) -> i32 {
                assert_eq!(base, ptr::dangling_mut::<u8>().cast());
                assert_eq!((size, flags), (0, MEM_RELEASE));
                crate::event('f', 0);
                1
            }
        }

        pub mod Threading {
            use std::ptr;

            use super::super::Foundation::HANDLE;
            use super::super::Security::{TOKEN_ADJUST_PRIVILEGES, TOKEN_QUERY};

            pub unsafe fn GetCurrentProcess() -> HANDLE {
                ptr::dangling_mut::<u8>().cast()
            }

            pub unsafe fn OpenProcessToken(process: HANDLE, access: u32, out: *mut HANDLE) -> i32 {
                assert_eq!(process, ptr::dangling_mut::<u8>().cast());
                assert_eq!(access, TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY);
                if crate::event('o', crate::OPENED) {
                    unsafe { out.write(process) };
                    1
                } else {
                    0
                }
            }
        }
    }
}

fn replay(size: usize, page: usize, flags: u8) -> (bool, Case) {
    CASE.set(Case {
        page,
        flags,
        ..Case::default()
    });
    let result = imp::alloc_zeroed(size);
    if let Some(memory) = result {
        unsafe { imp::free(memory) };
    }
    (result.is_some(), CASE.take())
}

fn main() -> io::Result<()> {
    let (allocated, case) = replay(usize::MAX, 2 << 20, ACQUIRED_PRIVILEGE | ALLOCATED);
    assert!(
        !allocated && case.trace.is_empty(),
        "overflow request reached Win32: {:?}, trace {}",
        case.request,
        case.trace
    );

    run_buffered(|out| {
        for page in [0, 1, 4096, 2 << 20, 6 << 20, 1 << (usize::BITS - 1)] {
            let last_multiple = if page == 0 {
                0
            } else {
                usize::MAX - (usize::MAX % page)
            };
            for size in [
                0,
                page.saturating_sub(1),
                page,
                page + 1,
                last_multiple,
                last_multiple.saturating_add(1),
                usize::MAX,
            ] {
                for flags in 0..64 {
                    let (allocated, case) = replay(size, page, flags);
                    let request = opt_or_dash(case.request);
                    let trace = if case.trace.is_empty() {
                        "-"
                    } else {
                        &case.trace
                    };
                    writeln!(
                        out,
                        "{} {size} {page} {flags} {} {request} {trace}",
                        usize::BITS,
                        u8::from(allocated)
                    )?;
                }
            }
        }
        Ok(())
    })
}
