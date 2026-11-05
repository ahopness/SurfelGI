fn hash1(x_in: u32) -> u32 {
    var x = x_in;
    x += (x << 10u);
    x ^= (x >> 6u);
    x += (x << 3u);
    x ^= (x >> 11u);
    x += (x << 15u);
    return x;
}

fn hash1_mut(h: ptr<function, u32>) -> u32 {
    let res = *h;
    *h = hash1(res);
    return res;
}

fn hashCombine2(x: u32, y: u32) -> u32 {
    let M = 1664525u;
    let C = 1013904223u;
    var seed = (x * M + y + C) * M;
    seed ^= (seed >> 11u);
    seed ^= (seed << 7u) & 0x9d2c5680u;
    seed ^= (seed << 15u) & 0xefc60000u;
    seed ^= (seed >> 18u);
    return seed;
}

fn uintToU01Float(h_in: u32) -> f32 {
    var h = h_in;
    let mantissaMask = 0x007fffffu;
    let one = 0x3f800000u;
    h &= mantissaMask;
    h |= one;
    let r2 = bitcast<f32>(h);
    return r2 - 1.0;
}
