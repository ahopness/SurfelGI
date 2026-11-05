#include "Common/update_schedule.wgsl"
struct ScheduleUniforms { tileCount: u32, frame: u32, budget: u32, pad: u32 };
@group(0) @binding(0) var<uniform> u: ScheduleUniforms;
@group(0) @binding(1) var<storage,read_write> tileAlloc: array<i32>;

@compute @workgroup_size(64)
fn select_candidates(@builtin(global_invocation_id) id: vec3u) {
    let lane = id.x;
    if (lane >= min(u.budget,u.tileCount)) { return; }
    // Disjoint interleaved sets in permutation space; each lane finds a live
    // candidate instead of spending its slot on an already covered screen tile.
    let count = (u.tileCount - 1u - lane) / u.budget + 1u;
    let start = (u.frame + lane * 17u) % count;
    for (var k = 0u; k < count; k++) {
        let rank = lane + ((start + k) % count) * u.budget;
        let tile = permute_update_index(rank,u.tileCount);
        if (tileAlloc[tile*2u] == 1) { tileAlloc[tile*2u] = 2; return; }
    }
}

@compute @workgroup_size(64)
fn finalize_candidates(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= u.tileCount) { return; }
    tileAlloc[id.x*2u] = select(0,1,tileAlloc[id.x*2u] == 2);
}
