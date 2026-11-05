@group(0) @binding(0) var<storage, read_write> poolMax: array<atomic<i32>>;
@group(0) @binding(1) var<storage, read_write> indirectArgs: array<u32>;

@compute @workgroup_size(1, 1, 1)
fn main() {
    let highMark = atomicLoad(&poolMax[0]);
    let groups = max(1, (highMark + 63) / 64);
    indirectArgs[0] = u32(groups);
    indirectArgs[1] = 1u;
    indirectArgs[2] = 1u;
}
