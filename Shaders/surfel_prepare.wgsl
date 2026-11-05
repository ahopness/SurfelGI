#include "Common/constants.wgsl"

@group(0) @binding(0) var<storage, read_write> surfels: array<SurfelPacked>;
@group(0) @binding(1) var<storage, read_write> poolBuf: array<i32>;
@group(0) @binding(2) var<storage, read_write> poolAlloc: array<atomic<i32>>;
@group(0) @binding(3) var<storage, read_write> poolMax: array<atomic<i32>>;
@group(0) @binding(4) var<storage, read_write> spawnCounter: array<atomic<i32>>;
@group(0) @binding(5) var<storage, read_write> retiredCounter: array<atomic<i32>>;

@compute @workgroup_size(64, 1, 1)
fn init_pool(@builtin(global_invocation_id) id: vec3u) {
    let idx = id.x;
    let cap = arrayLength(&surfels);
    if (idx < cap) {
        surfels[idx].age = SURFEL_LIFE_RECYCLED;
        poolBuf[idx] = i32(idx);
    }
    if (idx == 0u) {
        atomicStore(&poolAlloc[0], 0);
        atomicStore(&poolMax[0], 0);
        atomicStore(&spawnCounter[0], 0);
        atomicStore(&retiredCounter[0], 0);
    }
}

@compute @workgroup_size(1, 1, 1)
fn reset_counters() {
    atomicStore(&spawnCounter[0], 0);
    atomicStore(&retiredCounter[0], 0);
}
