#include "Common/constants.wgsl"
#include "Common/grid_helpers.wgsl"

const WORKGROUP_SIZE: u32 = 512u;
const SEGMENT_SIZE: i32 = 1024;
const STEP_COUNT: u32 = 10u; // log2(512) + 1

struct GridUniforms {
    camPos: vec4f,
    gridOrigin: vec4f,
    totalCells: i32,
    segments: i32,
    listCapacity: i32,
    _pad0: i32,
};

@group(0) @binding(0) var<uniform> u: GridUniforms;
@group(0) @binding(1) var<storage, read> surfels: array<SurfelPacked>;
@group(0) @binding(2) var<storage, read_write> poolMax: array<atomic<i32>>;
@group(0) @binding(3) var<storage, read_write> offsetsAndList: array<atomic<i32>>;
@group(0) @binding(4) var<storage, read_write> segSums: array<i32>;

var<workgroup> sharedScan: array<i32, 1024>;

fn keep_grid_bindings() {
    _ = u.totalCells;
    _ = arrayLength(&surfels);
    _ = arrayLength(&poolMax);
    _ = arrayLength(&offsetsAndList);
    _ = arrayLength(&segSums);
}

@compute @workgroup_size(256, 1, 1)
fn grid_clear(@builtin(global_invocation_id) id: vec3u) {
    keep_grid_bindings();
    let i = i32(id.x);
    if (i <= u.totalCells) {
        atomicStore(&offsetsAndList[i], 0);
    }
}

@compute @workgroup_size(64, 1, 1)
fn grid_count(@builtin(global_invocation_id) id: vec3u) {
    keep_grid_bindings();
    let i = i32(id.x);
    let total = atomicLoad(&poolMax[0]);
    if (i >= total || id.x >= arrayLength(&surfels)) {
        return;
    }

    let surfel = surfels[i];
    if (surfel.age >= SURFEL_TTL) {
        return;
    }

    let worldPos = surfel.posb.xyz;
    let normal = surfel.normal;
    let pRel = worldPos - u.gridOrigin.xyz;
    if (any(abs(pRel) >= vec3f(SURFEL_GRID_CELL_DIAMETER * f32(SURFEL_CS) * 0.5 * exp2(f32(SURFEL_CASCADES-1))))) { return; }
    let radius = surfel_radius_for_pos(worldPos, u.camPos.xyz) * SURFEL_RADIUS_OVERSCALE;

    let box = get_surfel_grid_box_min_max(pRel);
    for (var ci: u32 = 0u; ci < box.cascadeCount; ci = ci + 1u) {
        let isC1 = ci == 1u;
        let currentMin = select(box.c0_min, box.c1_min, isC1);
        let currentMax = select(box.c0_max, box.c1_max, isC1);
        let cIdx = currentMin.w;

        for (var z = i32(currentMin.z); z <= i32(currentMax.z); z = z + 1) {
            for (var y = i32(currentMin.y); y <= i32(currentMax.y); y = y + 1) {
                for (var x = i32(currentMin.x); x <= i32(currentMax.x); x = x + 1) {
                    let c4 = vec4u(u32(x), u32(y), u32(z), cIdx);
                    if (surfel_intersects_grid_coord(worldPos, normal, radius, c4, u.gridOrigin.xyz)) {
                        let hash = i32(surfel_grid_c4_to_hash(c4));
                        atomicAdd(&offsetsAndList[hash], 1);
                    }
                }
            }
        }
    }
}

@compute @workgroup_size(512, 1, 1)
fn grid_scan_seg(
    @builtin(local_invocation_id) localId: vec3u,
    @builtin(workgroup_id) wgId: vec3u
) {
    keep_grid_bindings();
    let lid = i32(localId.x);
    let seg = i32(wgId.x);
    let base = seg * SEGMENT_SIZE;
    let idx2 = lid * 2;
    let g0 = base + idx2;
    let g1 = g0 + 1;

    let in0 = g0 <= u.totalCells;
    let in1 = g1 <= u.totalCells;

    if (in0) {
        sharedScan[idx2] = atomicLoad(&offsetsAndList[g0]);
    } else {
        sharedScan[idx2] = 0;
    }

    if (in1) {
        sharedScan[idx2 + 1] = atomicLoad(&offsetsAndList[g1]);
    } else {
        sharedScan[idx2 + 1] = 0;
    }

    workgroupBarrier();

    for (var step: u32 = 0u; step < STEP_COUNT; step = step + 1u) {
        let mask = (1 << step) - 1;
        let rd = ((lid >> step) << (step + 1u)) + mask;
        let wr = rd + 1 + (lid & mask);
        if (rd < SEGMENT_SIZE && wr < SEGMENT_SIZE) {
            let neighborVal = sharedScan[rd];
            sharedScan[wr] += neighborVal;
        }
        workgroupBarrier();
    }

    if (in0) {
        atomicStore(&offsetsAndList[g0], sharedScan[idx2]);
    }
    if (in1) {
        atomicStore(&offsetsAndList[g1], sharedScan[idx2 + 1]);
    }
}

@compute @workgroup_size(64, 1, 1)
fn grid_collect_seg(@builtin(global_invocation_id) id: vec3u) {
    keep_grid_bindings();
    let seg = i32(id.x);
    if (seg < u.segments) {
        let base = seg * SEGMENT_SIZE;
        let last = min(base + SEGMENT_SIZE - 1, u.totalCells);
        segSums[seg] = atomicLoad(&offsetsAndList[last]);
    }
}

@compute @workgroup_size(1, 1, 1)
fn grid_seg_prefix() {
    keep_grid_bindings();
    var accum: i32 = 0;
    for (var i: i32 = 0; i < u.segments; i = i + 1) {
        accum += segSums[i];
        segSums[i] = accum;
    }
}

@compute @workgroup_size(256, 1, 1)
fn grid_merge(@builtin(global_invocation_id) id: vec3u) {
    keep_grid_bindings();
    let idx = i32(id.x);
    if (idx <= u.totalCells) {
        let seg = idx / SEGMENT_SIZE;
        if (seg > 0) {
            let prevSum = segSums[seg - 1];
            let curr = atomicLoad(&offsetsAndList[idx]);
            atomicStore(&offsetsAndList[idx], curr + prevSum);
        }
    }
}

@compute @workgroup_size(64, 1, 1)
fn grid_slot(@builtin(global_invocation_id) id: vec3u) {
    keep_grid_bindings();
    let i = i32(id.x);
    let total = atomicLoad(&poolMax[0]);
    if (i >= total || id.x >= arrayLength(&surfels)) {
        return;
    }

    let surfel = surfels[i];
    if (surfel.age >= SURFEL_TTL) {
        return;
    }

    let worldPos = surfel.posb.xyz;
    let normal = surfel.normal;
    let pRel = worldPos - u.gridOrigin.xyz;
    if (any(abs(pRel) >= vec3f(SURFEL_GRID_CELL_DIAMETER * f32(SURFEL_CS) * 0.5 * exp2(f32(SURFEL_CASCADES-1))))) { return; }
    let radius = surfel_radius_for_pos(worldPos, u.camPos.xyz) * SURFEL_RADIUS_OVERSCALE;

    let box = get_surfel_grid_box_min_max(pRel);
    for (var ci: u32 = 0u; ci < box.cascadeCount; ci = ci + 1u) {
        let isC1 = ci == 1u;
        let currentMin = select(box.c0_min, box.c1_min, isC1);
        let currentMax = select(box.c0_max, box.c1_max, isC1);
        let cIdx = currentMin.w;

        for (var z = i32(currentMin.z); z <= i32(currentMax.z); z = z + 1) {
            for (var y = i32(currentMin.y); y <= i32(currentMax.y); y = y + 1) {
                for (var x = i32(currentMin.x); x <= i32(currentMax.x); x = x + 1) {
                    let c4 = vec4u(u32(x), u32(y), u32(z), cIdx);
                    if (surfel_intersects_grid_coord(worldPos, normal, radius, c4, u.gridOrigin.xyz)) {
                        let hash = i32(surfel_grid_c4_to_hash(c4));
                        let endPlusOne = atomicAdd(&offsetsAndList[hash], -1);
                        let writeIdx = endPlusOne - 1;
                        let absoluteIdx = OFFSETS_AND_LIST_START + writeIdx;
                        if (writeIdx >= 0 && absoluteIdx < u.listCapacity) {
                            atomicStore(&offsetsAndList[absoluteIdx], i);
                        }
                    }
                }
            }
        }
    }
}
