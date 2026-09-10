//! The sweep schedule, the pass-position bit spread and the tile block.

use super::super::sweep::{
    block_columns, spread, sweep_lengths, sweep_lengths_descending, STAGING_LEN,
};

#[test]
fn sweeps_take_full_lengths_first_then_the_remainder() {
    assert_eq!(sweep_lengths(1).collect::<Vec<_>>(), [1]);
    assert_eq!(sweep_lengths(4).collect::<Vec<_>>(), [4]);
    assert_eq!(sweep_lengths(7).collect::<Vec<_>>(), [4, 3]);
    assert_eq!(sweep_lengths(8).collect::<Vec<_>>(), [4, 4]);
    assert_eq!(sweep_lengths(9).collect::<Vec<_>>(), [4, 4, 1]);
}

#[test]
fn descending_sweeps_take_the_remainder_first() {
    use sweep_lengths_descending as descending;
    assert_eq!(descending(1).collect::<Vec<_>>(), [1]);
    assert_eq!(descending(4).collect::<Vec<_>>(), [4]);
    assert_eq!(descending(7).collect::<Vec<_>>(), [3, 4]);
    assert_eq!(descending(8).collect::<Vec<_>>(), [4, 4]);
    assert_eq!(descending(9).collect::<Vec<_>>(), [1, 4, 4]);
}

#[test]
fn spread_inserts_zero_bits_at_the_pass_position() {
    assert_eq!(spread(0b101, 0, 2), 0b10100);
    assert_eq!(spread(0b110, 1, 2), 0b11000);
    assert_eq!(spread(0b101, 2, 2), 0b10001);
    assert_eq!(spread(0b11, 1, 1), 0b101);
}

#[test]
fn block_columns_fill_the_tile_budget_in_whole_vectors() {
    assert_eq!(block_columns::<f64>(16, 4, 256), 64);
    assert_eq!(block_columns::<f32>(16, 8, 256), 128);
    assert_eq!(block_columns::<f64>(64, 4, 256), 16);
    assert_eq!(block_columns::<f64>(16, 4, 32), 32);
    assert_eq!(block_columns::<f32>(2, 8, 2), 2);
}

#[test]
fn a_tile_block_of_either_scalar_fits_the_staging_buffer() {
    for stages in 1..=4u32 {
        let rows = 1usize << stages;
        assert!(rows * block_columns::<f64>(rows, 4, 1 << 20) <= STAGING_LEN);
        assert!(rows * block_columns::<f32>(rows, 8, 1 << 20) <= STAGING_LEN);
    }
}
