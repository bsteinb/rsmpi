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
//! To enforce this rule, every request object must be attached to some pre-existing
//! [`Scope`](trait.Scope.html), which is usually created with [`scope`](fn.scope.html).  At the end
//! of a `Scope`, all its remaining requests will be waited for until completion.
//!
//! To handle request completion in an RAII style, a request can be wrapped in either
//! [`WaitGuard`](struct.WaitGuard.html) or [`CancelGuard`](struct.CancelGuard.html), which will
//! follow the respective policy for completing the operation.  When the guard is dropped, the
//! request will be automatically detached from its `Scope`.
//!
//! # Unfinished features
//!
//! - **3.7**: Nonblocking mode:
//!   - Completion, `MPI_Waitany()`, `MPI_Waitall()`, `MPI_Waitsome()`,
//!   `MPI_Testany()`, `MPI_Testall()`, `MPI_Testsome()`, `MPI_Request_get_status()`
//! - **3.8**:
//!   - Cancellation, `MPI_Test_cancelled()`

use std::cell::RefCell;
use std::collections::HashSet;
use std::mem;
use std::marker::PhantomData;
use std::os::raw::c_int;

use ffi;
use ffi::{MPI_Request, MPI_Status};

use point_to_point::Status;
use raw::traits::*;

/// Check if the request is `MPI_REQUEST_NULL`.
fn is_null(request: MPI_Request) -> bool {
    request == unsafe_extern_static!(ffi::RSMPI_REQUEST_NULL)
}

/// A request object for a non-blocking operation attached to a `Scope` with lifetime `'a`
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
pub struct Request<'a, S: Scope<'a> = StaticScope>(WaitGuard<'a, S>);

unsafe impl<'a, S: Scope<'a>> AsRaw for Request<'a, S> {
    type Raw = MPI_Request;
    fn as_raw(&self) -> Self::Raw {
        self.0.as_raw()
    }
}

impl<'a, S: Scope<'a>> Drop for Request<'a, S> {
    fn drop(&mut self) {
        panic!("request was dropped without being completed");
    }
}

impl<'a, S: Scope<'a>> From<WaitGuard<'a, S>> for Request<'a, S> {
    fn from(guard: WaitGuard<'a, S>) -> Self {
        Request(guard)
    }
}

impl<'a, S: Scope<'a>> From<CancelGuard<'a, S>> for Request<'a, S> {
    fn from(mut guard: CancelGuard<'a, S>) -> Self {
        // unwrapping an object that implements Drop is tricky
        Request(unsafe {
            let inner = mem::replace(&mut guard.0, mem::uninitialized());
            mem::forget(guard);
            inner
        })
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
        Request(WaitGuard::from_raw(request, scope))
    }

    /// Unregister the request object from its scope and deconstruct it into its raw parts.
    ///
    /// This is unsafe because the request may outlive its associated buffers.
    pub unsafe fn into_raw(self) -> (MPI_Request, S) {
        WaitGuard::from(self).into_raw()
    }

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
    pub fn wait(self) -> Status {
        unsafe {
            let mut status: MPI_Status = mem::uninitialized();
            self.0.raw_wait(Some(&mut status));
            self.into_raw();
            Status::from_raw(status)
        }
    }

    /// Wait for an operation to finish, but don’t bother retrieving the `Status` information.
    ///
    /// Will block execution of the calling thread until the associated operation has finished.
    ///
    /// # Standard section(s)
    ///
    /// 3.7.3
    pub fn wait_without_status(self) {
        unsafe {
            self.0.raw_wait(None);
            self.into_raw();
        }
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
    pub fn test(self) -> Result<Status, Self> {
        unsafe {
            let mut status: MPI_Status = mem::uninitialized();
            let mut flag: c_int = mem::uninitialized();
            let mut request = self.as_raw();
            ffi::MPI_Test(&mut request, &mut flag, &mut status);
            if flag != 0 {
                assert!(is_null(request));  // persistent requests are not supported
                self.into_raw();
                Ok(Status::from_raw(status))
            } else {
                Err(self)
            }
        }
    }

    /// Cancel an operation.
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
    pub fn cancel(&self) {
        self.0.cancel();
    }

    /// Reduce the scope of a request.
    pub fn shrink_scope_to<'b, S2>(self, scope: S2) -> Request<'b, S2>
        where 'a: 'b, S2: Scope<'b>
    {
        unsafe {
            let (request, _) = self.into_raw();
            Request::from_raw(request, scope)
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
pub struct WaitGuard<'a, S: Scope<'a> = StaticScope> {
    request: MPI_Request,
    scope: S,
    phantom: PhantomData<RefCell<&'a ()>>,
}

impl<'a, S: Scope<'a>> Drop for WaitGuard<'a, S> {
    fn drop(&mut self) {
        unsafe {
            self.raw_wait(None);
            self.scope.unregister(&self.as_raw());
        }
    }
}

unsafe impl<'a, S: Scope<'a>> AsRaw for WaitGuard<'a, S> {
    type Raw = MPI_Request;
    fn as_raw(&self) -> Self::Raw {
        self.request
    }
}

impl<'a, S: Scope<'a>> From<Request<'a, S>> for WaitGuard<'a, S> {
    fn from(mut req: Request<'a, S>) -> Self {
        unsafe {
            let inner = mem::replace(&mut req.0, mem::uninitialized());
            mem::forget(req);
            inner
        }
    }
}

impl<'a, S: Scope<'a>> WaitGuard<'a, S> {
    /// Construct a request object from the raw MPI type.
    ///
    /// # Requirements
    ///
    /// - The request is a valid, active request.  It must not be `MPI_REQUEST_NULL`.
    /// - The request must not be persistent.
    /// - All buffers associated with the request must outlive `'a`.
    /// - The request must not be registered with the given scope.
    ///
    unsafe fn from_raw(request: MPI_Request, scope: S) -> Self {
        debug_assert!(!is_null(request));
        scope.register(request);
        WaitGuard { request: request, scope: scope, phantom: Default::default() }
    }

    /// Unregister the request object from its scope and deconstruct it into its raw parts.
    ///
    /// This is unsafe because the request may outlive its associated buffers.
    unsafe fn into_raw(mut self) -> (MPI_Request, S) {
        let request = self.as_raw();
        let scope = mem::replace(&mut self.scope, mem::uninitialized());
        mem::replace(&mut self.phantom, mem::uninitialized());
        mem::forget(self);
        scope.unregister(&request);
        (request, scope)
    }

    /// Wait for the request to finish.
    ///
    /// This is unsafe because `.request` is no longer valid afterwards.  This function should only
    /// be called if the caller calls `into_raw` afterward and forgets the request.
    unsafe fn raw_wait(&self, status: Option<&mut MPI_Status>) {
        let mut request = self.as_raw();
        let status = match status {
            Some(r) => r,
            None => ffi::RSMPI_STATUS_IGNORE,
        };
        ffi::MPI_Wait(&mut request, status);
        assert!(is_null(request));      // persistent requests are not supported
    }

    fn cancel(&self) {
        let mut request = self.as_raw();
        unsafe {
            ffi::MPI_Cancel(&mut request);
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

impl<'a, S: Scope<'a>> From<Request<'a, S>> for CancelGuard<'a, S> {
    fn from(req: Request<'a, S>) -> Self {
        CancelGuard(WaitGuard::from(req))
    }
}

/// A common interface for [`LocalScope`](struct.LocalScope.html) and
/// [`StaticScope`](struct.StaticScope.html)
///
/// This trait is an implementation detail.  You shouldn’t have to use or implement this trait.
pub unsafe trait Scope<'a> {
    /// Registers the request with the scope.
    unsafe fn register(&self, request: MPI_Request);

    /// Unregisters the request from the scope.
    unsafe fn unregister(&self, request: &MPI_Request);
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
/// For any `Request` attached to a `StaticScope`, its associated buffers must be `'static`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct StaticScope;

unsafe impl Scope<'static> for StaticScope {
    unsafe fn register(&self, _: MPI_Request) {}

    unsafe fn unregister(&self, _: &MPI_Request) {}
}

/// A temporary scope that lasts no more than the lifetime `'a`
///
/// Use `LocalScope` for to perform requests with temporary buffers.
///
/// To obtain a `LocalScope`, use the [`scope`](fn.scope.html) function.
///
/// # Invariant
///
/// For any `Request` attached to a `LocalScope<'a>`, its associated buffers must outlive `'a`.
#[derive(Debug)]
pub struct LocalScope<'a> {
    requests: RefCell<HashSet<MPI_Request>>,
    phantom: PhantomData<RefCell<&'a ()>>, // RefCell needed to ensure 'a is invariant
}

impl<'a> Drop for LocalScope<'a> {
    fn drop(&mut self) {
        for &request in &*self.requests.borrow() {
            unsafe {
                let _ = WaitGuard::from_raw(request, StaticScope);
            }
        }
    }
}

unsafe impl<'a, 'b> Scope<'a> for &'b LocalScope<'a> {
    unsafe fn register(&self, request: MPI_Request) {
        if !self.requests.borrow_mut().insert(request) {
            panic!("request already registered");
        }
    }

    unsafe fn unregister(&self, request: &MPI_Request) {
        if !self.requests.borrow_mut().remove(request) {
            panic!("can't unregister a request that wasn't registered");
        }
    }
}

/// Used to create a [`LocalScope`](struct.LocalScope.html)
///
/// The function creates a `LocalScope` and then passes it into the given
/// closure as an argument.
///
/// For safety reasons, all variables and buffers associated with a request
/// must exist *outside* the scope to which the request is attached.
///
/// It is typically used like this:
///
/// ```
/// /* declare variables and buffers here ... */
/// mpi::request::scope(|scope| {
///     /* perform sends and/or receives using 'scope' */
/// });
/// /* when scope ends, all associated requests are automatically waited for */
/// ```
///
/// # Examples
///
/// See `examples/immediate.rs`
pub fn scope<'a, F, R>(f: F) -> R
    where F: FnOnce(&LocalScope<'a>) -> R {
    f(&LocalScope {
        requests: Default::default(),
        phantom: Default::default(),
    })
}
