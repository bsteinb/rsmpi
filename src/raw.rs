//! Bridge between rust types and raw values

use ffi::*;

use std;
use std::mem;
use std::os::raw::c_int;

/// Rust C bridge traits
pub mod traits {
    pub use super::{AsRaw, AsRawMut, Nullable};
}

/// A rust type than can identify as a raw value understood by the MPI C API.
pub unsafe trait AsRaw {
    /// The raw MPI C API type
    type Raw;
    /// The raw value
    fn as_raw(&self) -> Self::Raw;
}

unsafe impl<'a, T> AsRaw for &'a T
where
    T: 'a + AsRaw,
{
    type Raw = <T as AsRaw>::Raw;
    fn as_raw(&self) -> Self::Raw {
        (*self).as_raw()
    }
}

/// A rust type than can provide a mutable pointer to a raw value understood by the MPI C API.
pub unsafe trait AsRawMut: AsRaw {
    /// A mutable pointer to the raw value
    fn as_raw_mut(&mut self) -> *mut <Self as AsRaw>::Raw;
}

/// Applied to ffi handles that have a null value.
pub trait Nullable: Eq + Copy {
    /// Returns the null() value of this type.
    fn null() -> Self;

    /// Returns true if this value is null. Returns false if it is not null.
    fn is_handle_null(self) -> bool {
        self == Self::null()
    }
}

impl Nullable for MPI_Request {
    fn null() -> Self {
        unsafe_extern_static!(RSMPI_REQUEST_NULL)
    }
}

fn check_length<T>(arr: &[T]) {
    assert!(
        arr.len() <= std::i32::MAX as usize,
        "MPI can only index arrays up to size i32::MAX"
    );
}

fn check_statuses(requests: &[MPI_Request], statuses: &Option<&mut [MPI_Status]>) {
    if let &Some(ref statuses) = statuses {
        assert!(
            statuses.len() >= requests.len(),
            "The statuses array must be at least as large as the requests array."
        );
    }
}

fn check_indices(requests: &[MPI_Request], indices: &[i32]) {
    assert!(
        indices.len() >= requests.len(),
        "The indices array must be at least as large as the requests array."
    );
}

fn to_status_ptr_mut(status: Option<&mut MPI_Status>) -> *mut MPI_Status {
    match status {
        Some(status) => status,
        None => unsafe_extern_static!(RSMPI_STATUS_IGNORE),
    }
}

fn to_statuses_ptr_mut(statuses: Option<&mut [MPI_Status]>) -> (usize, *mut MPI_Status) {
    match statuses {
        Some(statuses) => (statuses.len(), statuses.as_mut_ptr()),
        None => (0, unsafe_extern_static!(RSMPI_STATUS_IGNORE)),
    }
}

/// Test to see if the request has been completed. If it has been, request
/// is set to RSMPI_REQUEST_NULL, and Some(status) is returned. If not, request
/// is not modified, and None is returned.
///
/// # Standard section(s)
///
/// 3.7.3
pub fn test(request: &mut MPI_Request) -> Option<MPI_Status> {
    unsafe {
        let mut status: MPI_Status = mem::uninitialized();
        let mut flag: c_int = mem::uninitialized();
        MPI_Test(request, &mut flag, &mut status);
        if flag != 0 {
            // persistent requests are not supported
            assert!(request.is_handle_null());

            Some(status)
        } else {
            None
        }
    }
}

/// Wait for the request to finish.
/// If provided, the status is written to the referent of the given reference.
/// The referent `MPI_Status` object is never read.
///
/// Prefer `Request::wait` or `Request::wait_without_status`.
///
/// # Standard section(s)
///
/// 3.7.3
pub fn wait(request: &mut MPI_Request, status: Option<&mut MPI_Status>) {
    unsafe {
        MPI_Wait(request, to_status_ptr_mut(status));

        // persistent requests are not supported
        assert!(request.is_handle_null());
    }
}

/// Wait for any request in the `requests` slice to complete. If there are
/// non-null requests in the `requests` slice, then `wait_any` returns
/// `Some(idx)`, where `idx` is the index of the request that was completed
/// in the slice. If there are no non-null requests in the `requests`
/// slice, it returns None.
///
/// Prefer `RequestCollection::wait_any`.
///
/// # Standard section(s)
///
/// 3.7.5
pub fn wait_any(requests: &mut [MPI_Request], status: Option<&mut MPI_Status>) -> Option<i32> {
    check_length(requests);

    unsafe {
        let mut idx = mem::uninitialized();

        MPI_Waitany(
            requests.len() as i32,
            requests.as_mut_ptr(),
            &mut idx,
            to_status_ptr_mut(status),
        );

        if idx == RSMPI_UNDEFINED {
            None
        } else {
            Some(idx)
        }
    }
}

/// Result type for `raw::test_any`.
#[derive(Clone, Copy)]
pub enum TestAny {
    /// Indicates that there are no active requests in the `requests` slice.
    NoneActive,
    /// Indicates that, while there are active requests in the `requests` slice, none of them were
    /// completed.
    NoneComplete,
    /// Indicates which request in the `requests` slice was completed.
    Completed(i32),
}

/// `test_any` is a safe, low-level interface to MPI_Testany.
///
/// `test_any` will check if any active requests in the `requests` slice are completed. If so, it
/// will deallocate the request and mark it as null in the `requests` slice. In all cases it will
/// return immediately, even if no request was completed.
///
/// If a request is completed and `Some(status)` is provided, `status` will contain the status of
/// the request that was completed.
///
/// See the documentation on `TestAny` for detailed information on the possible return values.
pub fn test_any(requests: &mut [MPI_Request], status: Option<&mut MPI_Status>) -> TestAny {
    check_length(requests);

    unsafe {
        let mut idx = mem::uninitialized();
        let mut flag = mem::uninitialized();

        MPI_Testany(
            requests.len() as i32,
            requests.as_mut_ptr(),
            &mut idx,
            &mut flag,
            to_status_ptr_mut(status),
        );

        if flag != 0 {
            if idx == RSMPI_UNDEFINED {
                TestAny::NoneActive
            } else {
                TestAny::Completed(idx)
            }
        } else {
            TestAny::NoneComplete
        }
    }
}

/// `wait_all` is a safe, low-level interface to MPI_Waitall.
///
/// `wait_all` will block until all requests in the `requests` slice are completed. When `wait_all`
/// returns, all requests in the `requests` slice will be MPI_REQUEST_NULL. If `Some(statuses)` is
/// provided, the slices[i] will contain the status for `requests[i]`.
///
/// Prefer `RequestCollection::wait_all` in typical code.
///
/// # Standard section(s)
///
/// 3.7.5
pub fn wait_all(requests: &mut [MPI_Request], statuses: Option<&mut [MPI_Status]>) {
    check_length(requests);
    check_statuses(requests, &statuses);

    let (_, statuses_ptr) = to_statuses_ptr_mut(statuses);

    unsafe {
        MPI_Waitall(requests.len() as i32, requests.as_mut_ptr(), statuses_ptr);
    }
}

/// `test_all` is a safe, low-level interface to MPI_Testall.
///
/// If all active `requests` are complete, it returns `true` immediately. `statuses` will contain
/// the status of each completed request, where `statuses[i]` is the completion `status` for
/// `requests[i]`. All `requests` are deallocated.
///
/// If not all `requests` are complete, `false` is returned. `requests` is unmodified and that value
/// of `statuses` is undefined.
///
/// Prefer `RequestCollection::test_all` in typical code.
///
/// # Standard section(s)
///
/// 3.7.5
pub fn test_all(requests: &mut [MPI_Request], statuses: Option<&mut [MPI_Status]>) -> bool {
    check_length(requests);
    check_statuses(requests, &statuses);

    let (_, statuses_ptr) = to_statuses_ptr_mut(statuses);

    let mut flag = unsafe { mem::uninitialized() };
    unsafe {
        MPI_Testall(
            requests.len() as i32,
            requests.as_mut_ptr(),
            &mut flag,
            statuses_ptr,
        );
    }

    flag != 0
}

/// `wait_some` is a safe, low-level interface to MPI_Waitsome.
///
/// `wait_some` blocks until at least 1 request is completed if any requests are active.
///
/// `Some(count)` is returned if there are oustanding active requests. `count`, which will be 1 or
/// greater, indicates the number of requests that are completed. `indices[0..count]` will contain
/// the indices of the completed requests, and `statuses[0..count]` will contain the completion
/// status of those requests. i.e. `statuses[i]` will contain the completion status of
/// `requests[indices[i]]`. `outstanding()` will be reduced by `count`.
///
/// `None` is returned if there are no active requests.
///
/// Prefer `RequestCollection::wait_some` in typical code.
///
/// # Standard section(s)
///
/// 3.7.5
pub fn wait_some(
    requests: &mut [MPI_Request],
    indices: &mut [i32],
    statuses: Option<&mut [MPI_Status]>,
) -> Option<i32> {
    check_length(requests);
    check_indices(requests, indices);
    check_statuses(requests, &statuses);

    let (_, statuses_ptr) = to_statuses_ptr_mut(statuses);

    let mut outcount = unsafe { mem::uninitialized() };
    unsafe {
        MPI_Waitsome(
            requests.len() as i32,
            requests.as_mut_ptr(),
            &mut outcount,
            indices.as_mut_ptr(),
            statuses_ptr,
        );
    }

    if outcount == unsafe_extern_static!(RSMPI_UNDEFINED) {
        None
    } else {
        Some(outcount)
    }
}

/// `test_some` is a safe, low-level interface to MPI_Waitsome.
///
/// `Some(count)` is returned if there are oustanding active requests. `count`, which will be 0 or
/// greater, indicates the number of requests that are completed. `indices[0..count]` will contain
/// the indices of the completed requests, and `statuses[0..count]` will contain the completion
/// status of those requests. i.e. `statuses[i]` will contain the completion status of
/// `requests[indices[i]]`. `outstanding()` will be reduced by `count`.
///
/// `None` is returned if there are no active requests.
///
/// Prefer `RequestCollection::test_some` in typical code.
///
/// # Standard section(s)
///
/// 3.7.5
pub fn test_some(
    requests: &mut [MPI_Request],
    indices: &mut [i32],
    statuses: Option<&mut [MPI_Status]>,
) -> Option<i32> {
    check_length(requests);
    check_indices(requests, indices);
    check_statuses(requests, &statuses);

    let (_, statuses_ptr) = to_statuses_ptr_mut(statuses);

    let mut outcount = unsafe { mem::uninitialized() };
    unsafe {
        MPI_Testsome(
            requests.len() as i32,
            requests.as_mut_ptr(),
            &mut outcount,
            indices.as_mut_ptr(),
            statuses_ptr,
        );
    }

    if outcount == unsafe_extern_static!(RSMPI_UNDEFINED) {
        None
    } else {
        Some(outcount)
    }
}
