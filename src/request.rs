//! Request objects for non-blocking operations
//!
//! Non-blocking operations such as `immediate_send()` return request objects that borrow any
//! buffers involved in the operation so as to ensure proper access restrictions. In order to
//! release the borrowed buffers from the request objects, a completion operation such as
//! [`wait()`](struct.Request.html#method.wait) or [`test()`](struct.Request.html#method.test) must
//! be used on the request object.
//!
//! **Note:** If the `Request` is dropped (as opposed to calling `wait` or `test` explicitly), the
//! program will panic.
//!
//! To enforce this rule, every request object must be registered to some pre-existing
//! [`Scope`](trait.Scope.html).  At the end of a `Scope`, all its remaining requests will be waited
//! for until completion.  Scopes can be created using either [`scope`](fn.scope.html) or
//! [`StaticScope`](struct.StaticScope.html).
//!
//! To handle request completion in an RAII style, a request can be wrapped in either
//! [`WaitGuard`](struct.WaitGuard.html) or [`CancelGuard`](struct.CancelGuard.html), which will
//! follow the respective policy for completing the operation.  When the guard is dropped, the
//! request will be automatically unregistered from its `Scope`.
//!
//! # Unfinished features
//!
//! - **3.7**: Nonblocking mode:
//!   - Completion, `MPI_Waitany()`, `MPI_Waitall()`, `MPI_Waitsome()`,
//!   `MPI_Testany()`, `MPI_Testall()`, `MPI_Testsome()`, `MPI_Request_get_status()`
//! - **3.8**:
//!   - Cancellation, `MPI_Test_cancelled()`

use std::cell::Cell;
use std::mem;
use std::marker::PhantomData;
use std::slice;

use ffi;
use ffi::{MPI_Request, MPI_Status};

use point_to_point::Status;
use raw::traits::*;
use raw;

/// Request traits
pub mod traits {
    pub use super::{AsyncRequest, CollectRequests};
}

/// A request object for a non-blocking operation registered with a `Scope` of lifetime `'a`
///
/// The `Scope` is needed to ensure that all buffers associated request will outlive the request
/// itself, even if the destructor of the request fails to run.
///
/// # Panics
///
/// Panics if the request object is dropped.  To prevent this, call `wait`, `wait_without_status`,
/// or `test`.  Alternatively, wrap the request inside a `WaitGuard` or `CancelGuard`.
///
/// # Examples
///
/// See `examples/immediate.rs`
///
/// # Standard section(s)
///
/// 3.7.1
pub trait AsyncRequest<'a, S: Scope<'a>>: AsRaw<Raw = MPI_Request> + Sized {
    /// Unregister the request object from its scope and deconstruct it into its raw parts.
    ///
    /// This is unsafe because the request may outlive its associated buffers.
    unsafe fn into_raw(self) -> (MPI_Request, S);

    /// Wait for an operation to finish.
    ///
    /// Will block execution of the calling thread until the associated operation has finished.
    ///
    /// # Examples
    ///
    /// See `examples/immediate.rs`
    ///
    /// # Standard section(s)
    ///
    /// 3.7.3
    fn wait(self) -> Status {
        let mut status: MPI_Status = unsafe { mem::uninitialized() };
        raw::wait(unsafe { &mut self.into_raw().0 }, Some(&mut status));
        Status::from_raw(status)
    }

    /// Wait for an operation to finish, but don’t bother retrieving the `Status` information.
    ///
    /// Will block execution of the calling thread until the associated operation has finished.
    ///
    /// # Standard section(s)
    ///
    /// 3.7.3
    fn wait_without_status(self) {
        raw::wait(unsafe { &mut self.into_raw().0 }, None)
    }

    /// Test whether an operation has finished.
    ///
    /// If the operation has finished, `Status` is returned.  Otherwise returns the unfinished
    /// `Request`.
    ///
    /// # Examples
    ///
    /// See `examples/immediate.rs`
    ///
    /// # Standard section(s)
    ///
    /// 3.7.3
    fn test(self) -> Result<Status, Self> {
        match raw::test(&mut self.as_raw()) {
            Some(status) => {
                unsafe { self.into_raw() };
                Ok(Status::from_raw(status))
            }
            None => Err(self),
        }
    }

    /// Initiate cancellation of the request.
    ///
    /// The MPI implementation is not guaranteed to fulfill this operation.  It may not even be
    /// valid for certain types of requests.  In the future, the MPI forum may [deprecate
    /// cancellation of sends][mpi26] entirely.
    ///
    /// [mpi26]: https://github.com/mpi-forum/mpi-issues/issues/26
    ///
    /// # Examples
    ///
    /// See `examples/immediate.rs`
    ///
    /// # Standard section(s)
    ///
    /// 3.8.4
    fn cancel(&self) {
        let mut request = self.as_raw();
        unsafe {
            ffi::MPI_Cancel(&mut request);
        }
    }

    /// Reduce the scope of a request.
    fn shrink_scope_to<'b, S2>(self, scope: S2) -> Request<'b, S2>
    where
        'a: 'b,
        S2: Scope<'b>,
    {
        unsafe {
            let (request, _) = self.into_raw();
            Request::from_raw(request, scope)
        }
    }
}

/// A request object for a non-blocking operation registered with a `Scope` of lifetime `'a`
///
/// The `Scope` is needed to ensure that all buffers associated request will outlive the request
/// itself, even if the destructor of the request fails to run.
///
/// # Panics
///
/// Panics if the request object is dropped.  To prevent this, call `wait`, `wait_without_status`,
/// or `test`.  Alternatively, wrap the request inside a `WaitGuard` or `CancelGuard`.
///
/// # Examples
///
/// See `examples/immediate.rs`
///
/// # Standard section(s)
///
/// 3.7.1
#[must_use]
#[derive(Debug)]
pub struct Request<'a, S: Scope<'a> = StaticScope> {
    request: MPI_Request,
    scope: S,
    phantom: PhantomData<Cell<&'a ()>>,
}

unsafe impl<'a, S: Scope<'a>> AsRaw for Request<'a, S> {
    type Raw = MPI_Request;
    fn as_raw(&self) -> Self::Raw {
        self.request
    }
}

impl<'a, S: Scope<'a>> Drop for Request<'a, S> {
    fn drop(&mut self) {
        panic!("request was dropped without being completed");
    }
}

impl<'a, S: Scope<'a>> Request<'a, S> {
    /// Construct a request object from the raw MPI type.
    ///
    /// # Requirements
    ///
    /// - The request is a valid, active request.  It must not be `MPI_REQUEST_NULL`.
    /// - The request must not be persistent.
    /// - All buffers associated with the request must outlive `'a`.
    /// - The request must not be registered with the given scope.
    ///
    pub unsafe fn from_raw(request: MPI_Request, scope: S) -> Self {
        debug_assert!(!request.is_handle_null());
        scope.register();
        Self {
            request,
            scope,
            phantom: Default::default(),
        }
    }
}

impl<'a, S: Scope<'a>> AsyncRequest<'a, S> for Request<'a, S> {
    unsafe fn into_raw(mut self) -> (MPI_Request, S) {
        let request = mem::replace(&mut self.as_raw(), mem::uninitialized());
        let scope = mem::replace(&mut self.scope, mem::uninitialized());
        let _ = mem::replace(&mut self.phantom, mem::uninitialized());
        mem::forget(self);
        scope.unregister();
        (request, scope)
    }
}

/// Collects an iterator of `Request` objects into a `RequestCollection` object
pub trait CollectRequests<'a, S: Scope<'a>>: IntoIterator<Item = Request<'a, S>> {
    /// Consumes and converts an iterator of `Requst` objects into a `RequestCollection` object.
    fn collect_requests<'b, S2: Scope<'b>>(self, scope: S2) -> RequestCollection<'b, S2>
    where
        'a: 'b;
}

impl<'a, S: Scope<'a>, T: IntoIterator<Item = Request<'a, S>>> CollectRequests<'a, S> for T {
    fn collect_requests<'b, S2: Scope<'b>>(self, scope: S2) -> RequestCollection<'b, S2>
    where
        'a: 'b,
    {
        RequestCollection::from_request_iter(self, scope)
    }
}

/// Result type for `RequestCollection::test_any`.
#[derive(Clone, Copy)]
pub enum TestAny {
    /// Indicates that there are no active requests in the `requests` slice.
    NoneActive,
    /// Indicates that, while there are active requests in the `requests` slice, none of them were
    /// completed.
    NoneComplete,
    /// Indicates which request in the `requests` slice was completed.
    Completed(i32, Status),
}

/// Result type for `RequestCollection::test_any_without_status`.
#[derive(Clone, Copy)]
pub enum TestAnyWithoutStatus {
    /// Indicates that there are no active requests in the `requests` slice.
    NoneActive,
    /// Indicates that, while there are active requests in the `requests` slice, none of them were
    /// completed.
    NoneComplete,
    /// Indicates which request in the `requests` slice was completed.
    Completed(i32),
}

/// A collection of request objects for a non-blocking operation registered with a `Scope` of
/// lifetime `'a`.
///
/// The `Scope` is needed to ensure that all buffers associated request will outlive the request
/// itself, even if the destructor of the request fails to run.
///
/// # Panics
///
/// Panics if the collection is dropped while it contains outstanding requests.
/// To prevent this, call `wait_all` or repeatedly call `wait_some`, `wait_any`, `test_any`,
/// `test_some`, or `test_all` until all requests are reported as complete.
///
/// # Examples
///
/// See `examples/immediate_wait_all.rs`
///
/// # Standard section(s)
///
/// 3.7.5
#[must_use]
#[derive(Debug)]
pub struct RequestCollection<'a, S: Scope<'a> = StaticScope> {
    // Tracks the number of request handles in `requests` are active.
    outstanding: usize,

    // NOTE: Once Rust supports some sort of "null pointer optimization" for custom types, this
    // could become Vec<Option<MPI_Request>>.
    requests: Vec<MPI_Request>,

    // The scope attached to the RequestCollection. All requests in the collection must be
    // deallocated when this scope exits.
    scope: S,
    phantom: PhantomData<Cell<&'a ()>>,
}

impl<'a, S: Scope<'a>> Drop for RequestCollection<'a, S> {
    fn drop(&mut self) {
        if self.outstanding != 0 {
            panic!("RequestCollection was dropped with outstanding requests not completed.");
        }
    }
}

impl<'a, S: Scope<'a>> RequestCollection<'a, S> {
    /// Constructs a `RequestCollection` from a Vec of `MPI_Request` handles and a scope object.
    /// `requests` are allowed to be null, but they must not be persistent requests.
    pub fn from_raw(requests: Vec<MPI_Request>, scope: S) -> Self {
        let outstanding = requests
            .iter()
            .filter(|&request| !request.is_handle_null())
            .count();

        scope.register_many(outstanding);
        Self {
            outstanding: outstanding,
            requests: requests,
            scope: scope,
            phantom: Default::default(),
        }
    }

    /// Constructs a new, empty `RequestCollection` object.
    pub fn new(scope: S) -> Self {
        Self::with_capacity(scope, 0)
    }

    /// Constructs a new, empty `RequestCollection` with reserved space for `capacity` requests.
    pub fn with_capacity(scope: S, capacity: usize) -> Self {
        RequestCollection {
            outstanding: 0,
            requests: Vec::with_capacity(capacity),
            scope: scope,
            phantom: Default::default(),
        }
    }

    /// Converts an iterator of request objects to a `RequestCollection`. The scope of each request
    /// must be larger than or equal of the new `RequestCollection`.
    fn from_request_iter<'b: 'a, T, S2: Scope<'b>>(iter: T, scope: S) -> Self
    where
        T: IntoIterator<Item = Request<'b, S2>>,
    {
        let iter = iter.into_iter();

        let (lbound, ubound) = iter.size_hint();
        let capacity = if let Some(ubound) = ubound {
            ubound
        } else {
            lbound
        };

        let mut collection = RequestCollection::with_capacity(scope, capacity);

        for request in iter {
            collection.push(request);
        }

        collection
    }

    /// Pushes a new request into the collection. The request is removed from its previous scope and
    /// attached to the new scope. Therefore, the request's scope must be greater than or equal to
    /// the collection's scope.
    pub fn push<'b: 'a, S2: Scope<'b>>(&mut self, request: Request<'b, S2>) {
        unsafe {
            let (request, _) = request.into_raw();
            assert!(
                !request.is_handle_null(),
                "Cannot add NULL requests to a RequestCollection."
            );
            self.requests.push(request);

            self.increase_outstanding(1);
        }
    }

    /// Reduce the scope of a request.
    pub fn shrink_scope_to<'b, S2>(self, scope: S2) -> RequestCollection<'b, S2>
    where
        'a: 'b,
        S2: Scope<'b>,
    {
        unsafe {
            let (requests, _) = self.into_raw();
            RequestCollection::from_raw(requests, scope)
        }
    }

    // Validates the number of outstanding requests.
    fn check_outstanding(&self) {
        debug_assert!(
            self.outstanding == self.requests.iter().filter(|&r| !r.is_handle_null()).count(),
            "Internal rsmpi error: the number of outstanding requests in the RequestCollection has \
            fallen out of sync with the tracking count.");
    }

    fn increase_outstanding(&mut self, new_outstanding: usize) {
        self.scope.register_many(new_outstanding);

        self.outstanding += new_outstanding;
        self.check_outstanding();
    }

    fn decrease_outstanding(&mut self, completed: usize) {
        unsafe { self.scope.unregister_many(completed) };

        self.outstanding -= completed;
        self.check_outstanding();
    }

    fn clear_outstanding(&mut self) {
        let outstanding = self.outstanding;
        self.decrease_outstanding(outstanding);
    }

    /// Called after a `wait_any` operation to validate that the request at idx is now null in DEBUG
    /// builds. This is to smoke out if the user is sneaking persistent requests into the
    /// collection.
    fn check_null(&self, idx: i32) {
        debug_assert!(
            self.requests[idx as usize].is_handle_null(),
            "Persistent requests are not allowed in RequestCollection."
        );
    }

    /// Called after a `wait_some` operation to validate that the requests at indices are now null
    /// in DEBUG builds. This is to smoke out if the user is sneaking persistent requests into the
    /// collection.
    fn check_some_null(&self, indices: &[i32]) {
        debug_assert!(
            indices
                .iter()
                .all(|&idx| self.requests[idx as usize].is_handle_null()),
            "Persistent requests are not allowed in RequestCollection."
        );
    }

    /// Called after a `wait_all` operations to validate that all requests are now null in DEBUG
    /// builds. This is to smoke out if the user is sneaking persistent requests into the
    /// collection.
    fn check_all_null(&self) {
        debug_assert!(
            self.requests.iter().all(|r| r.is_handle_null()),
            "Persistent requests are not allowed in RequestCollection."
        );
    }

    /// `outstanding` returns the number of requests in the collection that haven't been completed.
    pub fn outstanding(&self) -> usize {
        self.outstanding
    }

    /// Returns the number of request slots in the Collection.
    pub fn len(&self) -> usize {
        self.requests.len()
    }

    /// Returns the underlying array of MPI_Request objects and their attached
    /// scope.
    pub unsafe fn into_raw(mut self) -> (Vec<MPI_Request>, S) {
        let requests = mem::replace(&mut self.requests, mem::uninitialized());
        let scope = mem::replace(&mut self.scope, mem::uninitialized());
        let _ = mem::replace(&mut self.phantom, mem::uninitialized());
        mem::forget(self);
        scope.unregister();
        (requests, scope)
    }

    /// `shrink` removes all deallocated requests from the collection. It does not shrink the size
    /// of the underlying MPI_Request array, allowing the RequestCollection to be efficiently
    /// re-used for another set of requests without needing additional allocations.
    pub fn shrink(&mut self) {
        self.requests.retain(|&req| !req.is_handle_null())
    }

    /// `wait_any` blocks until any active request in the collection completes. It returns
    /// immediately if all requests in the collection are deallocated.
    ///
    /// If there are any active requests in the collection, then it returns `Some((idx, status))`,
    /// where `idx` is the index of the completed request in the collection and `status` is the
    /// status of the completed request. The request at `idx` will be set to None. `outstanding()`
    /// will be reduced by 1.
    ///
    /// Returns `None` if there are no active requests. `outstanding()` is 0.
    ///
    /// # Standard section(s)
    ///
    /// 3.7.5
    pub fn wait_any(&mut self) -> Option<(i32, Status)> {
        let mut status: MPI_Status = unsafe { mem::uninitialized() };
        if let Some(idx) = raw::wait_any(&mut self.requests, Some(&mut status)) {
            self.check_null(idx);
            self.decrease_outstanding(1);
            Some((idx, Status::from_raw(status)))
        } else {
            self.check_outstanding();
            None
        }
    }

    /// `wait_any_without_status` blocks until any active request in the collection completes. It
    /// returns immediately if all requests in the collection are deallocated.
    ///
    /// If there are any active requests in the collection, then it returns `Some(idx)`, where
    /// `idx` is the index of the completed request in the collection and `status` is the status of
    /// the completed request. The request at `idx` will be set to None. `outstanding()` will be
    /// reduced by 1.
    ///
    /// Returns `None` if there are no active requests. `outstanding()` is 0.
    ///
    /// # Standard section(s)
    ///
    /// 3.7.5
    pub fn wait_any_without_status(&mut self) -> Option<i32> {
        if let Some(idx) = raw::wait_any(&mut self.requests, None) {
            self.check_null(idx);
            self.decrease_outstanding(1);
            Some(idx)
        } else {
            self.check_outstanding();
            None
        }
    }

    /// `test_any` checks if any requests in the collection are completed. It does not block.
    ///
    /// If there are no active requests in the collection, it returns `TestAny::NoneActive`.
    /// `outstanding()` is 0.
    ///
    /// If none of the active requests in the collection are completed, it returns
    /// `TestAny::NoneComplete`. `outstanding()` is unchanged.
    ///
    /// Otherwise, `test_any` picks one request of the completed requests, deallocates it, and
    /// returns `Completed(idx, status)`, where `idx` is the index of the completed request and
    /// `status` is the status of the completed request. `outstanding()` will be reduced by 1.
    ///
    /// # Standard section(s)
    ///
    /// 3.7.5
    pub fn test_any(&mut self) -> TestAny {
        let mut status: MPI_Status = unsafe { mem::uninitialized() };
        let result = match raw::test_any(&mut self.requests, Some(&mut status)) {
            raw::TestAny::NoneActive => TestAny::NoneActive,
            raw::TestAny::NoneComplete => TestAny::NoneComplete,
            raw::TestAny::Completed(idx) => {
                self.check_null(idx);
                self.decrease_outstanding(1);
                TestAny::Completed(idx, Status::from_raw(status))
            }
        };
        self.check_outstanding();
        result
    }

    /// `test_any_without_status` checks if any requests in the collection are completed. It does
    /// not block.
    ///
    /// If there are no active requests in the collection, it returns `TestAny::NoneActive`.
    /// `outstanding()` is 0.
    ///
    /// If none of the active requests in the collection are completed, it returns
    /// `TestAny::NoneComplete`. `outstanding()` is unchanged.
    ///
    /// Otherwise, `test_any` picks one request of the completed requests, deallocates it, and
    /// returns `Completed(idx)`, where `idx` is the index of the completed request. `outstanding()`
    /// will be reduced by 1.
    ///
    /// # Standard section(s)
    ///
    /// 3.7.5
    pub fn test_any_without_status(&mut self) -> TestAnyWithoutStatus {
        let result = match raw::test_any(&mut self.requests, None) {
            raw::TestAny::NoneActive => TestAnyWithoutStatus::NoneActive,
            raw::TestAny::NoneComplete => TestAnyWithoutStatus::NoneComplete,
            raw::TestAny::Completed(idx) => {
                self.check_null(idx);
                self.decrease_outstanding(1);
                TestAnyWithoutStatus::Completed(idx)
            }
        };
        self.check_outstanding();
        result
    }

    /// `wait_all_into` blocks until all requests in the collection are deallocated. Upon return,
    /// all requests in the collection will be deallocated. `outstanding()` will be equal to 0.
    /// `statuses` will be updated with the status for each request that is completed by
    /// `wait_all_into` where each status will match the index of the completed request. The status
    /// for deallocated entries will be set to empty.
    ///
    /// Panics if `statuses.len()` is not >= `self.len()`.
    ///
    /// # Standard section(s)
    ///
    /// 3.7.5
    pub fn wait_all_into(&mut self, statuses: &mut [Status]) {
        // This code assumes that the representation of point_to_point::Status is the same as
        // ffi::MPI_Status.
        let raw_statuses =
            unsafe { slice::from_raw_parts_mut(statuses.as_mut_ptr() as *mut _, statuses.len()) };

        raw::wait_all(&mut self.requests, Some(raw_statuses));

        self.check_all_null();
        self.clear_outstanding();
    }

    /// `wait_all` blocks until all requests in the collection are deallocated. Upon return,
    /// all requests in the collection will be deallocated. `outstanding()` will be equal to 0.
    /// A vector of statuses is returned with the status for each request that is completed by
    /// `wait_all` where each status will match the index of the completed request. The status for
    /// deallocated entries will be set to empty.
    ///
    /// If you do not need the status of the completed requests, `wait_all_without_status` is
    /// slightly more efficient because it does not allocate memory.
    ///
    /// # Examples
    ///
    /// See `examples/immediate_wait_all.rs`
    ///
    /// # Standard section(s)
    ///
    /// 3.7.5
    pub fn wait_all(&mut self) -> Vec<Status> {
        let mut statuses = vec![unsafe { mem::uninitialized() }; self.requests.len()];
        self.wait_all_into(&mut statuses[..]);
        statuses
    }

    /// `wait_all_without_status` blocks until all requests in the collection are deallocated. Upon
    /// return, all requests in the collection will be deallocated. `outstanding()` will be equal to
    /// 0.
    ///
    /// # Standard section(s)
    ///
    /// 3.7.5
    pub fn wait_all_without_status(&mut self) {
        raw::wait_all(&mut self.requests[..], None);

        self.check_all_null();
        self.clear_outstanding();
    }

    /// `test_all_into` checks if all requests are completed.
    ///
    /// Returns `true` if all the requests are complete. The completed requests are deallocated.
    /// `statuses` will contain the status for each completed request, where `statuses[i]` is the
    /// status for `requests[i]`. `outstanding()` will be 0.
    ///
    /// Returns `false` if not all active requests are complete. The value of `statuses` is
    /// undefined. `requests` will be unchanged. `outstanding()` will be unchanged.
    ///
    /// # Standard section(s)
    ///
    /// 3.7.5
    pub fn test_all_into(&mut self, statuses: &mut [Status]) -> bool {
        // This code assumes that the representation of point_to_point::Status is the same as
        // ffi::MPI_Status.
        let raw_statuses =
            unsafe { slice::from_raw_parts_mut(statuses.as_mut_ptr() as *mut _, statuses.len()) };

        if raw::test_all(&mut self.requests, Some(raw_statuses)) {
            self.check_all_null();
            self.clear_outstanding();
            true
        } else {
            self.check_outstanding();
            false
        }
    }

    /// `test_all` checks if all requests are completed.
    ///
    /// Returns `Some(statuses)` if all the requests are complete. The completed requests are
    /// deallocated. `statuses` will contain the status for each completed request, where
    /// `statuses[i]` is the status for `requests[i]`. `outstanding()` will be 0.
    ///
    /// Returns `None` if not all active requests are complete. The value of `statuses` is
    /// undefined. `requests` will be unchanged. `outstanding()` will be unchanged.
    ///
    /// # Standard section(s)
    ///
    /// 3.7.5
    pub fn test_all(&mut self) -> Option<Vec<Status>> {
        let mut statuses = vec![unsafe { mem::uninitialized() }; self.requests.len()];
        if self.test_all_into(&mut statuses) {
            Some(statuses)
        } else {
            None
        }
    }

    /// `test_all_without_status` checks if all requests are completed.
    ///
    /// Returns `true` if all the requests are complete. The completed requests are deallocated.
    /// `outstanding()` will be 0.
    ///
    /// Returns `false` if not all active requests are complete. `requests` will be unchanged.
    /// `outstanding()` will be unchanged.
    ///
    /// # Standard section(s)
    ///
    /// 3.7.5
    pub fn test_all_without_status(&mut self) -> bool {
        if raw::test_all(&mut self.requests, None) {
            self.check_all_null();
            self.clear_outstanding();

            true
        } else {
            self.check_outstanding();

            false
        }
    }

    /// `wait_some_into` blocks until a request is completed.
    ///
    /// Returns `Some(count)` if there any active requests in the collection. `count` will be the
    /// number of requests that were completed. The indices of the completed requests will be
    /// written to `indices[0..count]`, and the status of each of those completed requests will be
    /// written to `statuses[0..count]`.
    ///
    /// Returns `None` if all requests in the collection have already been deallocated.
    ///
    /// # Standard section(s)
    ///
    /// 3.7.5
    pub fn wait_some_into(&mut self, indices: &mut [i32], statuses: &mut [Status]) -> Option<i32> {
        // This code assumes that the representation of point_to_point::Status is the same as
        // ffi::MPI_Status.
        let raw_statuses =
            unsafe { slice::from_raw_parts_mut(statuses.as_mut_ptr() as *mut _, statuses.len()) };

        match raw::wait_some(&mut self.requests, indices, Some(raw_statuses)) {
            Some(count) => {
                self.check_some_null(&indices[..count as usize]);
                self.decrease_outstanding(count as usize);

                Some(count)
            }
            None => None,
        }
    }

    /// `wait_some_into_without_status` blocks until a request is completed.
    ///
    /// Returns `Some(count)` if there any active requests in the collection. `count` will be the
    /// number of requests that were completed. The indices of the completed requests will be
    /// written to `indices[0..count]`.
    ///
    /// Returns `None` if all requests in the collection have already been deallocated.
    ///
    /// # Standard section(s)
    ///
    /// 3.7.5
    pub fn wait_some_into_without_status(&mut self, indices: &mut [i32]) -> Option<i32> {
        match raw::wait_some(&mut self.requests, indices, None) {
            Some(count) => {
                self.check_some_null(&indices[..count as usize]);
                self.decrease_outstanding(count as usize);

                Some(count)
            }
            None => None,
        }
    }

    /// `wait_some` blocks until a request is completed.
    ///
    /// Returns `Some((indices, statuses))` if there any active requests in the collection.
    /// `indices` and `statuses` will contain equal number of elements, where `indices` contains
    /// the indices of each completed request. `statuses` contains the completion status for each
    /// request.
    ///
    /// Returns `None` if all requests in the collection have already been deallocated.
    ///
    /// # Standard section(s)
    ///
    /// 3.7.5
    pub fn wait_some(&mut self) -> Option<(Vec<i32>, Vec<Status>)> {
        let mut indices = vec![unsafe { mem::uninitialized() }; self.requests.len()];
        let mut statuses = vec![unsafe { mem::uninitialized() }; self.requests.len()];

        match self.wait_some_into(&mut indices, &mut statuses) {
            Some(count) => {
                indices.resize(count as usize, unsafe { mem::uninitialized() });
                statuses.resize(count as usize, unsafe { mem::uninitialized() });

                Some((indices, statuses))
            }
            None => None,
        }
    }

    /// `wait_some_without_status` blocks until a request is completed.
    ///
    /// Returns `Some(indices)` if there any active requests in the collection.
    /// `indices` contains the indices of each completed request.
    ///
    /// Returns `None` if all requests in the collection have already been deallocated.
    ///
    /// # Standard section(s)
    ///
    /// 3.7.5
    pub fn wait_some_without_status(&mut self) -> Option<Vec<i32>> {
        let mut indices = vec![unsafe { mem::uninitialized() }; self.requests.len()];

        match self.wait_some_into_without_status(&mut indices) {
            Some(count) => {
                indices.resize(count as usize, unsafe { mem::uninitialized() });

                Some(indices)
            }
            None => None,
        }
    }

    /// `test_some_into` deallocates all active, completed requests.
    ///
    /// Returns `Some(count)` if there any active requests in the collection. `count` will be the
    /// number of requests that were completed. The indices of the completed requests will be
    /// written to `indices[0..count]`, and the status of each of those completed requests will be
    /// written to `statuses[0..count]`.
    ///
    /// Returns `None` if all requests in the collection have already been deallocated.
    ///
    /// # Standard section(s)
    ///
    /// 3.7.5
    pub fn test_some_into(&mut self, indices: &mut [i32], statuses: &mut [Status]) -> Option<i32> {
        // This code assumes that the representation of point_to_point::Status is the same as
        // ffi::MPI_Status.
        let raw_statuses =
            unsafe { slice::from_raw_parts_mut(statuses.as_mut_ptr() as *mut _, statuses.len()) };

        match raw::test_some(&mut self.requests, indices, Some(raw_statuses)) {
            Some(count) => {
                self.check_some_null(&indices[..count as usize]);
                self.decrease_outstanding(count as usize);

                Some(count)
            }
            None => None,
        }
    }

    /// `test_some_into_without_status` deallocates all active, completed requests.
    ///
    /// Returns `Some(count)` if there any active requests in the collection. `count` will be the
    /// number of requests that were completed. The indices of the completed requests will be
    /// written to `indices[0..count]`.
    ///
    /// Returns `None` if all requests in the collection have already been deallocated.
    ///
    /// # Standard section(s)
    ///
    /// 3.7.5
    pub fn test_some_into_without_status(&mut self, indices: &mut [i32]) -> Option<i32> {
        match raw::test_some(&mut self.requests, indices, None) {
            Some(count) => {
                self.check_some_null(&indices[..count as usize]);
                self.decrease_outstanding(count as usize);

                Some(count)
            }
            None => None,
        }
    }

    /// `test_some` deallocates all active, completed requests.
    ///
    /// Returns `Some((indices, statuses))` if there any active requests in the collection.
    /// `indices` and `statuses` will contain equal number of elements, where `indices` contains
    /// the indices of each completed request. `statuses` contains the completion status for each
    /// request.
    ///
    /// Returns `None` if all requests in the collection have already been deallocated.
    ///
    /// # Standard section(s)
    ///
    /// 3.7.5
    pub fn test_some(&mut self) -> Option<(Vec<i32>, Vec<Status>)> {
        let mut indices = vec![unsafe { mem::uninitialized() }; self.requests.len()];
        let mut statuses = vec![unsafe { mem::uninitialized() }; self.requests.len()];

        match self.test_some_into(&mut indices, &mut statuses) {
            Some(count) => {
                indices.resize(count as usize, unsafe { mem::uninitialized() });
                statuses.resize(count as usize, unsafe { mem::uninitialized() });

                Some((indices, statuses))
            }
            None => None,
        }
    }

    /// `test_some_without_status` deallocates all active, completed requests.
    ///
    /// Returns `Some(indices)` if there any active requests in the collection.
    /// `indices` contains the indices of each completed request.
    ///
    /// Returns `None` if all requests in the collection have already been deallocated.
    ///
    /// # Standard section(s)
    ///
    /// 3.7.5
    pub fn test_some_without_status(&mut self) -> Option<Vec<i32>> {
        let mut indices = vec![unsafe { mem::uninitialized() }; self.requests.len()];

        match self.test_some_into_without_status(&mut indices) {
            Some(count) => {
                indices.resize(count as usize, unsafe { mem::uninitialized() });

                Some(indices)
            }
            None => None,
        }
    }
}

/// Guard object that waits for the completion of an operation when it is dropped
///
/// The guard can be constructed or deconstructed using the `From` and `Into` traits.
///
/// # Examples
///
/// See `examples/immediate.rs`
#[derive(Debug)]
pub struct WaitGuard<'a, S: Scope<'a> = StaticScope>(Option<Request<'a, S>>);

impl<'a, S: Scope<'a>> Drop for WaitGuard<'a, S> {
    fn drop(&mut self) {
        self.0.take().expect("invalid WaitGuard").wait();
    }
}

unsafe impl<'a, S: Scope<'a>> AsRaw for WaitGuard<'a, S> {
    type Raw = MPI_Request;
    fn as_raw(&self) -> Self::Raw {
        self.0.as_ref().expect("invalid WaitGuard").as_raw()
    }
}

impl<'a, S: Scope<'a>> From<WaitGuard<'a, S>> for Request<'a, S> {
    fn from(mut guard: WaitGuard<'a, S>) -> Self {
        guard.0.take().expect("invalid WaitGuard")
    }
}

impl<'a, S: Scope<'a>> From<Request<'a, S>> for WaitGuard<'a, S> {
    fn from(req: Request<'a, S>) -> Self {
        WaitGuard(Some(req))
    }
}

impl<'a, S: Scope<'a>> WaitGuard<'a, S> {
    fn cancel(&self) {
        if let Some(ref req) = self.0 {
            req.cancel();
        }
    }
}

/// Guard object that tries to cancel and waits for the completion of an operation when it is
/// dropped
///
/// The guard can be constructed or deconstructed using the `From` and `Into` traits.
///
/// # Examples
///
/// See `examples/immediate.rs`
#[derive(Debug)]
pub struct CancelGuard<'a, S: Scope<'a> = StaticScope>(WaitGuard<'a, S>);

impl<'a, S: Scope<'a>> Drop for CancelGuard<'a, S> {
    fn drop(&mut self) {
        self.0.cancel();
    }
}

impl<'a, S: Scope<'a>> From<CancelGuard<'a, S>> for WaitGuard<'a, S> {
    fn from(mut guard: CancelGuard<'a, S>) -> Self {
        unsafe {
            let inner = mem::replace(&mut guard.0, mem::uninitialized());
            mem::forget(guard);
            inner
        }
    }
}

impl<'a, S: Scope<'a>> From<WaitGuard<'a, S>> for CancelGuard<'a, S> {
    fn from(guard: WaitGuard<'a, S>) -> Self {
        CancelGuard(guard)
    }
}

impl<'a, S: Scope<'a>> From<Request<'a, S>> for CancelGuard<'a, S> {
    fn from(req: Request<'a, S>) -> Self {
        CancelGuard(WaitGuard::from(req))
    }
}

/// A common interface for [`LocalScope`](struct.LocalScope.html) and
/// [`StaticScope`](struct.StaticScope.html) used internally by the `request` module.
///
/// This trait is an implementation detail.  You shouldn’t have to use or implement this trait.
pub unsafe trait Scope<'a> {
    /// Registers a request with the scope.
    fn register(&self) {
        self.register_many(1)
    }

    /// Registers multiple requests with the scope.
    fn register_many(&self, count: usize);

    /// Unregisters a request from the scope.
    unsafe fn unregister(&self) {
        self.unregister_many(1)
    }

    /// Unregisters multiple requests from the scope.
    unsafe fn unregister_many(&self, count: usize);
}

/// The scope that lasts as long as the entire execution of the program
///
/// Unlike `LocalScope<'a>`, `StaticScope` does not require any bookkeeping on the requests as every
/// request associated with a `StaticScope` can live as long as they please.
///
/// A `StaticScope` can be created simply by calling the `StaticScope` constructor.
///
/// # Invariant
///
/// For any `Request` registered with a `StaticScope`, its associated buffers must be `'static`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct StaticScope;

unsafe impl Scope<'static> for StaticScope {
    fn register_many(&self, _count: usize) {}
    unsafe fn unregister_many(&self, _count: usize) {}
}

/// A temporary scope that lasts no more than the lifetime `'a`
///
/// Use `LocalScope` for to perform requests with temporary buffers.
///
/// To obtain a `LocalScope`, use the [`scope`](fn.scope.html) function.
///
/// # Invariant
///
/// For any `Request` registered with a `LocalScope<'a>`, its associated buffers must outlive `'a`.
///
/// # Panics
///
/// When `LocalScope` is dropped, it will panic if there are any lingering `Requests` that have not
/// yet been completed.
#[derive(Debug)]
pub struct LocalScope<'a> {
    num_requests: Cell<usize>,
    phantom: PhantomData<Cell<&'a ()>>, // Cell needed to ensure 'a is invariant
}

impl<'a> Drop for LocalScope<'a> {
    fn drop(&mut self) {
        if self.num_requests.get() != 0 {
            panic!("at least one request was dropped without being completed");
        }
    }
}

unsafe impl<'a, 'b> Scope<'a> for &'b LocalScope<'a> {
    fn register_many(&self, count: usize) {
        self.num_requests.set(self.num_requests.get() + count)
    }

    unsafe fn unregister_many(&self, count: usize) {
        self.num_requests.set(
            self.num_requests
                .get()
                .checked_sub(count)
                .expect("unregister has been called more times than register"),
        )
    }
}

/// Used to create a [`LocalScope`](struct.LocalScope.html)
///
/// The function creates a `LocalScope` and then passes it into the given
/// closure as an argument.
///
/// For safety reasons, all variables and buffers associated with a request
/// must exist *outside* the scope with which the request is registered.
///
/// It is typically used like this:
///
/// ```
/// /* declare variables and buffers here ... */
/// mpi::request::scope(|scope| {
///     /* perform sends and/or receives using 'scope' */
/// });
/// /* at end of scope, panic if there are requests that have not yet completed */
/// ```
///
/// # Examples
///
/// See `examples/immediate.rs`
pub fn scope<'a, F, R>(f: F) -> R
where
    F: FnOnce(&LocalScope<'a>) -> R,
{
    f(&LocalScope {
        num_requests: Default::default(),
        phantom: Default::default(),
    })
}
