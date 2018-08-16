#![deny(warnings)]
extern crate mpi;

use mpi::traits::*;

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();

    let rank = world.rank();
    let size = world.size();

    let is_even = world.rank() % 2 == 0;

    // Each odd rank sends its rank to every even rank.
    // Each even rank sends its rank to every odd rank.
    let num_targets = size as usize / 2 + if size % 2 != 0 && !is_even { 1 } else { 0 };
    let lbound = if is_even { 1 } else { 0 };

    // The ranks will be received into this Vec<usize>
    let mut ranks = vec![0usize; num_targets];

    mpi::request::scope(|scope| {
        // Creates, initiates, and collects send requests into a RequestCollection.
        let mut send_requests = (0..num_targets)
            .map(|r| r * 2 + lbound)
            .map(|target| {
                world
                    .process_at_rank(target as i32)
                    .immediate_send(scope, &rank)
            })
            .collect_requests(scope);

        // Creates, initiates, and collects receive requests into a RequestCollection.
        let mut recv_requests = (0..num_targets)
            .map(|r| r * 2 + lbound)
            .zip(ranks.iter_mut())
            .map(|(target, rank)| {
                world
                    .process_at_rank(target as i32)
                    .immediate_receive_into(scope, rank)
            })
            .collect_requests(scope);

        // Calls wait_some on the RequestCollection until it is empty.
        let mut received_from = vec![false; num_targets];
        while let Some((indices, statuses)) = recv_requests.wait_some() {
            for (idx, status) in indices.into_iter().zip(statuses) {
                assert!(
                    !received_from[idx as usize],
                    "Received multiple messages from same rank."
                );
                received_from[idx as usize] = true;

                assert_eq!(
                    idx * 2 + lbound as i32,
                    status.source_rank(),
                    "Status does not match with the index of the request."
                )
            }
        }

        assert!(
            received_from.iter().all(|b| *b),
            "wait_some returned None even though not all requests were complete."
        );

        send_requests.wait_all_without_status();
    });

    let expected: Vec<_> = (0..num_targets).map(|rank| rank * 2 + lbound).collect();
    assert_eq!(expected, ranks, "Did not receive the expected ranks.");
}
