#include "Common/constants.wgsl"
#include "Common/grid_helpers.wgsl"

struct AgeUniforms {
    prevCamPos: vec4f,
    prevGridOrigin: vec4f,
};

@group(0) @binding(0) var<uniform> u: AgeUniforms;
@group(0) @binding(1) var<storage, read_write> surfels: array<SurfelPacked>;
@group(0) @binding(2) var<storage, read_write> poolBuf: array<i32>;
@group(0) @binding(3) var<storage, read_write> poolAlloc: array<atomic<i32>>;
@group(0) @binding(4) var<storage, read_write> poolMax: array<atomic<i32>>;
@group(0) @binding(5) var<storage, read_write> touched: array<atomic<i32>>;
@group(0) @binding(6) var<storage, read> offsetsAndList: array<i32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let idx = i32(id.x);
    let total = atomicLoad(&poolMax[0]);
    if (idx >= total || u32(idx) >= arrayLength(&surfels)) {
        return;
    }

    var currentAge = surfels[idx].age;
    let income = atomicMin(&touched[idx], 0);

    if (currentAge < SURFEL_TTL) {
        if (income == SURFEL_KILL_SIGNAL) {
            currentAge = SURFEL_LIFE_RECYCLE;
            surfels[idx].age = SURFEL_LIFE_RECYCLE;
        } else {
            var delta: i32 = 1;

            let posb = surfels[idx].posb;
            let pRel = posb.xyz - u.prevGridOrigin.xyz;
            let gridCoord = surfel_pos_to_grid_coord(pRel);
            let c4 = surfel_grid_coord_to_c4(gridCoord);
            let hashVal = surfel_grid_c4_to_hash(c4);
            let cellIdx = i32(hashVal);

            let start = offsetsAndList[cellIdx];
            let end = offsetsAndList[cellIdx + 1];
            let count = end - start;

            let SAFE_CAP = MAX_SURFELS_PER_CELL_FOR_KEEP_ALIVE;
            let excess = max(0, count - SAFE_CAP);

            if (count > SAFE_CAP) {
                delta += excess / 10;
            }
            if (count > SAFE_CAP * 2) {
                delta += (excess * excess) / 32;
            }

            let nextAge = max(0, currentAge + max(min(delta, 8) - income, 0));
            currentAge = nextAge;
            surfels[idx].age = nextAge;
        }

        if (currentAge >= SURFEL_TTL) {
            surfels[idx].age = SURFEL_LIFE_RECYCLED;
            let surfelAllocCount = atomicAdd(&poolAlloc[0], -1);
            let freedSlot = surfelAllocCount - 1;
            if (freedSlot >= 0 && u32(freedSlot) < arrayLength(&poolBuf)) {
                poolBuf[freedSlot] = idx;
            }
        }
    }
}
