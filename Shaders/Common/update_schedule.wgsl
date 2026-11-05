// A bijection over [0,count): masking gives a power-of-two permutation;
// cycle walking restricts it to non-power-of-two grids without collisions.
fn permute_update_index(index: u32, count: u32) -> u32 {
    var mask = count - 1u;
    mask |= mask >> 1u; mask |= mask >> 2u; mask |= mask >> 4u;
    mask |= mask >> 8u; mask |= mask >> 16u;
    var value = index;
    loop {
        value = (value * 1664525u + 1013904223u) & mask;
        value ^= value >> 10u;
        value = (value * 747796405u + 2891336453u) & mask;
        value ^= value >> 6u;
        if (value < count) { break; }
    }
    return value;
}
