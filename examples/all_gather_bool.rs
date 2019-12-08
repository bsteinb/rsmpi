#![deny(warnings)]
extern crate mpi;

use mpi::traits::*;

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();

    let rank = world.rank();
    let count = world.size() as usize;

    let mut a = vec![0u8; count];
    world.all_gather_into(&(rank % 2 == 0), &mut a[..]);
    let a = safe_transmute::bool::transmute_bool_vec_pedantic(a).unwrap();

    let answer: Vec<_> = (0..count).map(|i| i % 2 == 0).collect();

    assert_eq!(answer, a);
}
