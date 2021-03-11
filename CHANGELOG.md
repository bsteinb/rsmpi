# RSMPI Release Notes

TODO: With release of 0.6.0, delete this comment and add a section devoted to v0.6.0

## Master Branch

**MSRV:** 1.40.0

### Notes

In order to use the `user-operations` optional feature (enabled by default), you may want to check 
that  you are using at least `libffi-sys` version 1.1.0. This can be done by running `cargo 
update`, but for most users this will likely be unnecessary. 

### New Features
- [[PR 52]](https://github.com/rsmpi/rsmpi/pull/52) Safely send and receive structs with arbitrary 
    data layout, including padding, using `#[derive(Equivalence)]`.
- [[PR 88]](https://github.com/rsmpi/rsmpi/pull/88) Support for `MPI_Waitany` using `mpi::wait_any`.
- [[PR 46]](https://github.com/rsmpi/rsmpi/pull/46)
    [MS-MPI](https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi) on Windows 
    is now supported.
- [[PR 58]](https://github.com/rsmpi/rsmpi/pull/58) Support for "cartesian" communicators using 
    `CartestianCommunicator`
- [[PR 51]](https://github.com/rsmpi/rsmpi/pull/51) Support `MPI_Pack` and `MPI_Unpack` using
    `Communicator::pack`, `Communicator::pack_into`, and `Communicator::unpack_into`.
- [[PR 49]](https://github.com/rsmpi/rsmpi/pull/49) Build compound datatypes without committing 
    intermediate types using `UncommittedUserDatatype`.
- [[PR 53]](https://github.com/rsmpi/rsmpi/pull/53) Construct a `UserCommunicator` from an 
    `MPI_Comm` with `UserCommunicator::from_raw`.
- [[PR 90]](https://github.com/rsmpi/rsmpi/pull/90) Support `MPI_Comm_split_type` using 
    `Communicator::split_shared`, which splits communicators into subcommunicators that are capable 
    of creating shared memory regions.
- [[PR 27]](https://github.com/rsmpi/rsmpi/pull/27) Support for `MPI_Comm_get_name` and 
    `MPI_Comm_set_name` using `Communicator::set_name` and `Communicator::get_name`.

### Changed APIs
- [[PR 48]](https://github.com/rsmpi/rsmpi/pull/48) `UserDatatype::structured` no longer takes a 
    `count` argument.

### Fixed Bugs
- [[PR 77]](https://github.com/rsmpi/rsmpi/pull/77) Failure to complete scoped requests now results
    in immediate process termination, rather than a panic.
- [[PR 53]](https://github.com/rsmpi/rsmpi/pull/96) Portability improved with newer libffi and 
    bindgen

### Deprecated APIs

### Removed Features

## Old Releases
Previous releases do not have discretized release notes. Please review the git commmit log to
understand which features were added in each version.